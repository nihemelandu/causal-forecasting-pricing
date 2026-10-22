# model_1_dml.py - Model 1: DML Forecaster (Schultz et al., Sec. 3, Eq. 11 additive head for synthetic data).
# Three transformer models: outcome (q~), treatment (d~), effect (psi). Two-fold cross-fitting by
# article parity. Hyperparameters are the paper's Table 10 (simulated data).
# Uses a GPU when PyTorch can see one, otherwise the CPU, and prints which.
# Quick CPU test:   python scripts/model_1_dml.py --smoke --windows w1 --seeds 0
# Full run:         python scripts/model_1_dml.py
# Writes (per run): <out-dir>/model_1_dml_results.csv   one row per fit
#                   <out-dir>/forecasts/model_1_dml_<window>_seed<seed>.parquet   forecasts for every scenario
# Any reduced run (--smoke, --n-articles, --epoch-scale) writes model_1_dml_test.csv and forecasts_test/.
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import splits as S

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Our assumptions (the paper is silent on these for synthetic data) ----
CONTEXT, HORIZON = 30, 5      # encoder sees 30 weeks, decoder predicts 5
D_MODEL, HEADS, BATCH = 32, 4, 256
EMB = 8                       # embedding size for categories d and k
# Learning-rate decay (gamma) is applied once per epoch. Targets are scaled per article
# (demand / mean demand of the context, floored at 1). Effect loss is L2. Effect is unconstrained.
# Ensemble of cross-fit and forecast paths is an arithmetic mean.

# ---- Paper, Table 10 (simulated data) ----
HP = {
    "outcome":   dict(layers=5, ffn=73, dropout=0.1,    lr=0.0088, wd=5.4619e-9,
                      betas=(0.8566, 0.9140), gamma=0.9388, epochs=44),
    "treatment": dict(layers=2, ffn=21, dropout=0.1497, lr=0.0162, wd=2.6e-9,
                      betas=(0.5977, 0.8740), gamma=0.9549, epochs=60),
    "effect":    dict(layers=6, ffn=43, dropout=0.1750, lr=0.0491, wd=5.75e-9,
                      betas=(0.6405, 0.6749), gamma=None,   epochs=20),
}


class Net(nn.Module):
    """Encoder-decoder transformer (outcome, treatment) or encoder + head (effect)."""

    def __init__(self, hp: dict, f_enc: int, f_dec: int, kind: str):
        super().__init__()
        self.kind = kind
        d = D_MODEL
        s = 2 * EMB + 1
        self.emb_d = nn.Embedding(64, EMB)
        self.emb_k = nn.Embedding(64, EMB)
        self.enc_in = nn.Linear(f_enc + s, d)
        self.pos = nn.Embedding(CONTEXT + HORIZON, d)
        enc_layer = nn.TransformerEncoderLayer(d, HEADS, hp["ffn"], hp["dropout"], batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, hp["layers"], enable_nested_tensor=False)
        if kind == "effect":
            self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        else:
            self.dec_in = nn.Linear(f_dec + s, d)
            dec_layer = nn.TransformerDecoderLayer(d, HEADS, hp["ffn"], hp["dropout"], batch_first=True)
            self.dec = nn.TransformerDecoder(dec_layer, hp["layers"])
            self.head = nn.Linear(d, 1)

    def forward(self, enc, dec, sd, sk, sp):
        dev = enc.device
        static = torch.cat([self.emb_d(sd), self.emb_k(sk), sp[:, None]], 1)

        def with_static(x):
            return torch.cat([x, static[:, None, :].expand(-1, x.shape[1], -1)], 2)

        h = self.enc_in(with_static(enc)) + self.pos(torch.arange(CONTEXT, device=dev))[None]
        mem = self.enc(h)
        if self.kind == "effect":
            return self.head(mem.mean(1)).squeeze(-1)                      # [N]
        g = self.dec_in(with_static(dec)) + self.pos(torch.arange(CONTEXT, CONTEXT + HORIZON, device=dev))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(HORIZON, device=dev)
        out = self.head(self.dec(g, mem, tgt_mask=mask)).squeeze(-1)        # [N, H]
        return nn.functional.softplus(out) if self.kind == "outcome" else out


def panel(df: pd.DataFrame) -> dict:
    df = df.sort_values(["article_id", "week"])
    A, T = df["article_id"].nunique(), df["week"].nunique()
    assert len(df) == A * T, "panel is not complete"
    g = lambda c: df[c].to_numpy().astype(float).reshape(A, T)
    return dict(ids=df["article_id"].to_numpy().reshape(A, T)[:, 0], dem=g("demand"), disc=g("discount"),
                wk=g("week"), promo=g("promotion"), d=g("d")[:, 0].astype(int),
                k=g("k")[:, 0].astype(int), p0=g("p0")[:, 0])


def make_batch(dem, disc, wk, promo, wk_f, promo_f, d, k, p0, dem_f=None, disc_f=None) -> dict:
    scale = np.maximum(dem.mean(1), 1.0)
    enc = np.stack([dem / scale[:, None], disc, wk / 100.0, promo], 2)
    dec = np.stack([wk_f / 100.0, promo_f], 2)
    B = dict(enc=enc, dec=dec, sd=d, sk=k, sp=p0 / scale, scale=scale)
    if dem_f is not None:
        B["tq"] = dem_f / scale[:, None]
        B["td"] = disc_f
    return {key: torch.as_tensor(v, dtype=torch.long if key in ("sd", "sk") else torch.float32, device=DEVICE)
            for key, v in B.items()}


def loss_fn(net, kind, b):
    out = net(b["enc"], b["dec"], b["sd"], b["sk"], b["sp"])
    if kind == "outcome":
        return ((out - b["tq"]) ** 2).mean()
    if kind == "treatment":
        return ((out - b["td"]) ** 2).mean()
    pred = b["qt"] + out[:, None] * (b["td"] - b["dt"])      # Eq. 11 in scaled units
    return ((pred - b["tq"]) ** 2).mean()


def train_net(net, kind, B, hp, epochs, gen):
    opt = torch.optim.RAdam(net.parameters(), lr=hp["lr"], betas=hp["betas"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=hp["gamma"]) if hp["gamma"] else None
    n = B["enc"].shape[0]
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, BATCH):
            b = {k: v[perm[i:i + BATCH]] for k, v in B.items()}
            opt.zero_grad()
            loss_fn(net, kind, b).backward()
            opt.step()
        if sched:
            sched.step()


@torch.no_grad()
def apply(net, B):
    net.eval()
    n = B["enc"].shape[0]
    outs = [net(B["enc"][i:i + 2048], B["dec"][i:i + 2048], B["sd"][i:i + 2048],
                B["sk"][i:i + 2048], B["sp"][i:i + 2048]) for i in range(0, n, 2048)]
    return torch.cat(outs)


class DMLForecaster:
    def __init__(self, epoch_scale: float = 1.0):
        self.epoch_scale = epoch_scale

    def _epochs(self, name):
        return max(1, round(HP[name]["epochs"] * self.epoch_scale))

    def fit(self, train: pd.DataFrame, seed: int = 0):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        P = panel(train)
        A, T = P["dem"].shape
        origins = list(range(CONTEXT, T - HORIZON + 1))
        parts = []
        for s in origins:
            c, f = slice(s - CONTEXT, s), slice(s, s + HORIZON)
            parts.append(make_batch(P["dem"][:, c], P["disc"][:, c], P["wk"][:, c], P["promo"][:, c],
                                    P["wk"][:, f], P["promo"][:, f], P["d"], P["k"], P["p0"],
                                    P["dem"][:, f], P["disc"][:, f]))
        B = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        self.parity = np.arange(A) % 2                       # split by article index
        par = torch.as_tensor(np.tile(self.parity, len(origins)), device=DEVICE)

        # Stage 1: nuisance models, one pair per half of the articles
        self.nets = {}
        for p in (0, 1):
            sub = {k: v[par == p] for k, v in B.items()}
            for kind in ("outcome", "treatment"):
                net = Net(HP[kind], 4, 2, kind).to(DEVICE)
                train_net(net, kind, sub, HP[kind], self._epochs(kind), gen)
                self.nets[(kind, p)] = net

        # Stage 2: effect model on cross-fitted (out-of-fold) nuisance predictions
        qt, dt = torch.zeros_like(B["tq"]), torch.zeros_like(B["td"])
        for p in (0, 1):
            m = par == p
            sub = {k: v[m] for k, v in B.items()}
            qt[m] = apply(self.nets[("outcome", 1 - p)], sub)
            dt[m] = apply(self.nets[("treatment", 1 - p)], sub)
        B["qt"], B["dt"] = qt, dt
        self.effect = Net(HP["effect"], 4, 2, "effect").to(DEVICE)
        train_net(self.effect, "effect", B, HP["effect"], self._epochs("effect"), gen)

        last = slice(T - CONTEXT, T)
        self.ctx = {k: (v[:, last] if v.ndim == 2 else v) for k, v in P.items()}
        self.last_week = int(P["wk"][0, -1])
        self._nuis = None
        return self

    def _prepare(self, inputs: pd.DataFrame):
        ids = self.ctx["ids"]
        A = len(ids)
        ai = pd.Series(np.arange(A), index=ids).reindex(inputs["article_id"]).to_numpy()
        wo = inputs["week"].to_numpy() - (self.last_week + 1)
        assert wo.min() >= 0 and wo.max() < HORIZON, "test weeks must follow the training window"
        wk_f = np.tile(self.last_week + 1 + np.arange(HORIZON), (A, 1)).astype(float)
        promo_f = np.zeros((A, HORIZON))
        promo_f[ai, wo] = inputs["promotion"].to_numpy()
        if self._nuis is None:        # nuisance outputs do not depend on the scenario's discount
            c = self.ctx
            B = make_batch(c["dem"], c["disc"], c["wk"], c["promo"], wk_f, promo_f, c["d"], c["k"], c["p0"])
            par = torch.as_tensor(self.parity, device=DEVICE)
            q_cf, d_cf, q_f, d_f = (torch.zeros(A, HORIZON, device=DEVICE) for _ in range(4))
            for p in (0, 1):
                m = par == p
                sub = {k: v[m] for k, v in B.items()}
                q_f[m], d_f[m] = apply(self.nets[("outcome", p)], sub), apply(self.nets[("treatment", p)], sub)
                q_cf[m] = apply(self.nets[("outcome", 1 - p)], sub)
                d_cf[m] = apply(self.nets[("treatment", 1 - p)], sub)
            self._nuis = (q_cf, d_cf, q_f, d_f, apply(self.effect, B), B["scale"])
        return ai, wo

    def predict(self, inputs: pd.DataFrame):
        ai, wo = self._prepare(inputs)
        q_cf, d_cf, q_f, d_f, psi, scale = self._nuis
        dd = torch.zeros(len(self.ctx["ids"]), HORIZON, device=DEVICE)
        ai_t, wo_t = torch.as_tensor(ai, device=DEVICE), torch.as_tensor(wo, device=DEVICE)
        dd[ai_t, wo_t] = torch.as_tensor(inputs["discount"].to_numpy(), dtype=torch.float32, device=DEVICE)
        qh_cf = q_cf + psi[:, None] * (dd - d_cf)
        qh_f = q_f + psi[:, None] * (dd - d_f)
        qhat = 0.5 * (qh_cf + qh_f) * scale[:, None]
        return qhat.cpu().numpy()[ai, wo]

    def effect_estimates(self) -> pd.Series:
        """Estimated effect of one unit of discount on weekly demand, per article."""
        _, _, _, _, psi, scale = self._nuis
        return pd.Series((psi * scale).cpu().numpy(), index=self.ctx["ids"])


def save_forecasts(model, test_full: pd.DataFrame, w, seed: int, path: Path):
    """One row per article, week and scenario: forecast and true demand. Feeds build_metrics.py."""
    frames = []
    for lv in (None,) + tuple(S.DISCOUNT_LEVELS):
        inp = S.test_inputs(test_full, lv)
        f = inp[["article_id", "week", "d", "k", "p0", "discount"]].copy()
        f["window"], f["seed"] = w.name, seed
        f["scenario"] = "on" if lv is None else f"off_{int(round(lv * 100))}"
        f["forecast"] = model.predict(inp)
        f["truth"] = S.truth_demand(test_full, lv).to_numpy()
        frames.append(f)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="w1,w2,w3,w4")
    ap.add_argument("--seeds", default=",".join(str(s) for s in S.SEEDS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="where results are written (use a Drive folder in Colab)")
    ap.add_argument("--smoke", action="store_true", help="200 articles, 5%% of the epochs")
    ap.add_argument("--n-articles", type=int, default=None, help="use only the first N articles")
    ap.add_argument("--epoch-scale", type=float, default=None, help="fraction of the paper's epochs, e.g. 0.05")
    args = ap.parse_args()

    n_art, scale = args.n_articles, args.epoch_scale
    if args.smoke:
        n_art = n_art or 200
        scale = scale or 0.05
    scale = 1.0 if scale is None else scale
    reduced = n_art is not None or scale != 1.0

    dev_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
    print(f"torch {torch.__version__} | device: {DEVICE} ({dev_name}) | "
          f"{'REDUCED test run' if reduced else 'full run'} | epoch scale {scale} | articles {n_art or 'all'}")

    df = S.load_data()
    if n_art is not None:
        keep = df["article_id"].drop_duplicates().iloc[:n_art]
        df = df[df["article_id"].isin(keep)].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / ("model_1_dml_test.csv" if reduced else "model_1_dml_results.csv")
    fdir = out_dir / ("forecasts_test" if reduced else "forecasts")

    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    wins = {w.name: w for w in S.WINDOWS}
    for wname in args.windows.split(","):
        for seed in [int(x) for x in args.seeds.split(",")]:
            if any(r["window"] == wname and r["seed"] == seed for r in rows):
                print(f"skip {wname} seed {seed} (already in {csv.name})")
                continue
            t0 = time.time()
            model = DMLForecaster(scale)
            res = S.run_window(model, df, wins[wname], seed)
            _, test_full = S.get_split(df, wins[wname])
            res.update(S.effect_errors(model.effect_estimates(), test_full))
            save_forecasts(model, test_full, wins[wname], seed, fdir / f"model_1_dml_{wname}_seed{seed}.parquet")
            res["secs"] = round(time.time() - t0, 1)
            res["device"], res["torch"] = str(DEVICE), torch.__version__
            rows.append(res)
            pd.DataFrame(rows).to_csv(csv, index=False)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                   if k in ("window", "seed", "on_mae", "off_mae", "on_mse", "off_mse",
                            "mae_effect", "mse_effect", "secs", "device")})


if __name__ == "__main__":
    main()