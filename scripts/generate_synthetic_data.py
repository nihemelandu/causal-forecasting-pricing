# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 03:22:39 2026

@author: ngozi

data_generating_process.py

Synthetic data for "Causal Forecasting for Pricing" (Schultz et al., arXiv 2312.15282),
following Appendix E, plus the stock sizing and pricing policy (Eqs. 26-30).

Run:  python scripts/data_generating_process.py [--seed 42] [--n-articles 4467]

DEVIATIONS FROM / ASSUMPTIONS BEYOND THE PAPER
 1. Eq. 13 is used with a minus sign (demand falls as price rises).
 2. Eq. 21 gives negative prices ~31% of the time, so p_i0 is redrawn until
    p_i0 > 0 and demand at p_i0 is non-negative in every week (replaces the
    paper's unspecified "additional filtering steps"). Only p_i0 is redrawn.
 3. Initial stock = (mean(q_b) - (1 - 0.14) * p_i0 * e_i) * 100, not Eq. 25 as printed.
 4. Eq. 27 uses weeks t-4..t-1 (the printed sum includes week t, which is circular).
 5. j = 0 in weeks 0-3 (inferred, not stated); j is clipped to 0..5.
 6. Sales = min(demand, stock); the true demand is kept in `demand`.
 7. Articles are assigned to d and k uniformly at random; the 15 k values are split
    into season groups of 3,3,3,2,2,2 (15 cannot be split evenly into 6).
 8. `promotion` is Bernoulli(0.5) per article, with no effect on demand.
 9. Season = sin(2*pi*(week + shift_k)/30); the sign convention of the shift is our choice.
10. If Eq. 27 cannot be computed (0/0 after a long stockout), j is left unchanged.
11. d and k names are cosmetic labels; they do not change the generation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_ARTICLES = 4467
N_WEEKS = 100
N_D, N_K, N_SEASON_GROUPS = 45, 15, 6
WINDOW = 4                    # weeks used in Eq. 27 (and constant-price weeks at the start)
TARGET_DISCOUNT = 0.14        # average discount the initial stock is sized for
MAX_PRICE_DRAWS = 1000        # safety cap on redraws of p_i0

D_NAMES = [
    "T-Shirts", "Polo Shirts", "Shirts", "Blouses", "Sweaters", "Hoodies", "Cardigans",
    "Jackets", "Coats", "Blazers", "Jeans", "Trousers", "Chinos", "Shorts", "Skirts",
    "Dresses", "Jumpsuits", "Leggings", "Swimwear", "Underwear", "Socks", "Pajamas",
    "Sports Tops", "Sports Bottoms", "Sneakers", "Boots", "Sandals", "Heels", "Flats",
    "Loafers", "Handbags", "Backpacks", "Wallets", "Belts", "Scarves", "Hats", "Gloves",
    "Sunglasses", "Watches", "Jewelry", "Ties", "Perfume", "Kids Tops", "Kids Bottoms",
    "Kids Shoes",
]
TIERS = ["Core", "Fashion", "Basics"]
assert len(D_NAMES) == N_D


@dataclass
class GlobalParams:
    alpha_d: np.ndarray        # level per d category (Eq. 17)
    beta_k: np.ndarray         # level per k category (Eq. 19)
    k_shift: np.ndarray        # season shift per k category
    season_groups: list        # the k values in each season group


def make_global_params(rng: np.random.Generator) -> GlobalParams:
    alpha_d = rng.normal(10, 3, N_D)                          # Eq. 17
    beta_k = rng.normal(300, 50, N_K)                         # Eq. 19
    season_groups = np.array_split(np.arange(N_K), N_SEASON_GROUPS)
    group_shift = rng.integers(-15, 16, N_SEASON_GROUPS)      # integers in [-15, 15]
    k_shift = np.zeros(N_K, dtype=int)
    for g, ks in enumerate(season_groups):
        k_shift[ks] = group_shift[g]
    return GlobalParams(alpha_d, beta_k, k_shift, season_groups)


def make_k_names(season_groups) -> list:
    return [f"Season {chr(65 + g)} - {TIERS[n]}"
            for g, ks in enumerate(season_groups) for n in range(len(ks))]


def make_article(rng: np.random.Generator, gp: GlobalParams, max_draws: int = MAX_PRICE_DRAWS):
    """One article: base demand (Eqs. 14-24), true effect e_i and a valid p_i0."""
    d = int(rng.integers(0, N_D))
    k = int(rng.integers(0, N_K))
    t = np.arange(N_WEEKS)

    a_it = gp.alpha_d[d] + rng.normal(0, 1, N_WEEKS)                  # Eq. 16
    b_it = gp.beta_k[k] + rng.normal(0, 5, N_WEEKS)                   # Eq. 18
    c_it = 0.05 * a_it**2 + 0.25 * a_it + 0.5 * b_it                  # Eq. 15

    gamma_i = rng.uniform(-0.02, 0.02)                                # Eq. 23
    sigma_tau_i = rng.uniform(0, 0.15)                                # Eq. 24
    tau_it = rng.normal(t * gamma_i, sigma_tau_i)                     # Eq. 22
    s_it = np.sin(2 * np.pi * (t + gp.k_shift[k]) / 30)               # season, shift tied to k

    q_b = (0.15 * tau_it + 0.25 * s_it + 1) * c_it                    # Eq. 14

    e_b = max(1.3, rng.lognormal(0.75, 0.125))                        # Eq. 20
    e_i = e_b * 0.15 * a_it.mean()

    q_b_bar = q_b.mean()
    for n_draws in range(1, max_draws + 1):                           # Eq. 21 + valid-price rule
        p_i0 = rng.normal(q_b_bar / 3, q_b_bar / 1.5)
        if p_i0 > 0 and (q_b - p_i0 * e_i).min() >= 0:
            return dict(d=d, k=k, q_b=q_b, e_i=e_i, p_i0=p_i0, n_draws=n_draws)
    return None                                                       # article dropped


def run_policy(art: dict, lam: np.ndarray, z0: float | None = None) -> dict:
    """Stock sizing and the pricing policy (Eqs. 26-30) for one article."""
    q_b, p0, e = art["q_b"], art["p_i0"], art["e_i"]
    if z0 is None:
        z0 = (q_b.mean() - (1 - TARGET_DISCOUNT) * p0 * e) * N_WEEKS
    price = np.zeros(N_WEEKS); demand = np.zeros(N_WEEKS); sales = np.zeros(N_WEEKS)
    stock = np.zeros(N_WEEKS + 1); j = np.zeros(N_WEEKS, dtype=int)
    stock[0] = z0
    for t in range(N_WEEKS):
        if t >= WINDOW:
            with np.errstate(divide="ignore", invalid="ignore"):
                m = WINDOW * stock[t] / sales[t - WINDOW:t].sum()     # Eq. 27 (weeks t-4..t-1)
            w = m / (N_WEEKS - t)                                     # Eq. 28
            j[t] = j[t - 1]                                           # Eq. 29
            if w > 1 and lam[t] > 1 / w:
                j[t] = j[t - 1] + 1
            elif w < 1 and lam[t] > w:
                j[t] = j[t - 1] - 1
            j[t] = min(max(j[t], 0), 5)
        price[t] = p0 * (1 - 0.1 * j[t])                              # Eq. 30
        demand[t] = q_b[t] - price[t] * e                             # Eq. 13 (sign fix)
        sales[t] = min(demand[t], stock[t])                           # censoring at zero stock
        stock[t + 1] = stock[t] - sales[t]                            # Eq. 26
    return dict(z0=z0, price=price, demand=demand, sales=sales, stock=stock, j=j)


def build_dataset(seed: int = 42, n_articles: int = N_ARTICLES):
    rng = np.random.default_rng(seed)
    gp = make_global_params(rng)
    d_lookup = dict(enumerate(D_NAMES))
    k_lookup = dict(enumerate(make_k_names(gp.season_groups)))

    frames, n_dropped = [], 0
    while len(frames) < n_articles:
        art = make_article(rng, gp)
        if art is None:
            n_dropped += 1
            continue
        r = run_policy(art, rng.uniform(0, 1, N_WEEKS))
        frames.append(pd.DataFrame({
            "article_id": len(frames),
            "week": np.arange(N_WEEKS),
            "d": art["d"], "d_name": d_lookup[art["d"]],
            "k": art["k"], "k_name": k_lookup[art["k"]],
            "promotion": int(rng.integers(0, 2)),                     # no effect on demand
            "p0": art["p_i0"],
            "price": r["price"],
            "discount": 1 - r["price"] / art["p_i0"],                 # 0.0 .. 0.5
            "demand": r["demand"],                                    # true demand
            "sales": r["sales"],                                      # demand capped by stock
            "stock": r["stock"][:N_WEEKS],                            # stock at start of week
            "true_base_demand": art["q_b"],                           # ground truth: never a model input
            "true_effect_e": art["e_i"],                              # ground truth: never a model input
        }))
    return pd.concat(frames, ignore_index=True), gp, n_dropped


def validate(df: pd.DataFrame) -> None:
    g = df.groupby("article_id")
    assert (g.size() == N_WEEKS).all()
    assert not df.duplicated(["article_id", "week"]).any()
    assert df[["price", "discount", "demand", "sales", "stock", "true_base_demand"]].notna().all().all()
    assert df["discount"].min() >= -1e-9 and df["discount"].max() <= 0.5 + 1e-9
    assert np.allclose(df["discount"] * 10, np.round(df["discount"] * 10), atol=1e-6)
    assert (df["price"] <= df["p0"] + 1e-9).all()
    assert (df["demand"] >= 0).all() and (df["stock"] >= -1e-9).all()
    assert (df["sales"] <= df["demand"] + 1e-9).all()
    assert np.allclose(df["demand"], df["true_base_demand"] - df["price"] * df["true_effect_e"])
    nxt = g["stock"].shift(-1)
    ok = nxt.notna()
    assert np.allclose(nxt[ok], (df["stock"] - df["sales"])[ok])
    print("validation passed: structure, ranges and identities")


def _within_corr(d: pd.DataFrame, a: str, b: str) -> pd.Series:
    aa = d[a] - d.groupby("article_id")[a].transform("mean")
    bb = d[b] - d.groupby("article_id")[b].transform("mean")
    num = (aa * bb).groupby(d["article_id"]).sum()
    den = np.sqrt((aa**2).groupby(d["article_id"]).sum() * (bb**2).groupby(d["article_id"]).sum())
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def report(df: pd.DataFrame, gp: GlobalParams) -> None:
    n = df["article_id"].nunique()
    stock_end = df["stock"] - df["sales"]
    zero_week = df[stock_end <= 1e-9].groupby("article_id")["week"].min()
    unmet = (df["demand"] - df["sales"]).groupby(df["article_id"]).sum() / df.groupby("article_id")["demand"].sum()
    disc = df[df["week"] >= WINDOW].groupby("article_id")["discount"].mean()
    q = lambda s, p: s.quantile(p)
    print(f"stocked out by week 99: {len(zero_week) / n:.0%}   stockout week p10/p50/p90: "
          f"{q(zero_week, .1):.0f} / {q(zero_week, .5):.0f} / {q(zero_week, .9):.0f}")
    print(f"unmet demand share p50/p90: {q(unmet, .5):.1%} / {q(unmet, .9):.1%}")
    print(f"avg discount weeks 4-99 p10/p50/p90: {q(disc, .1):.1%} / {q(disc, .5):.1%} / {q(disc, .9):.1%}")

    d = df.copy()                                         # does the discount follow the season with a lag?
    d["season"] = np.sin(2 * np.pi * (d["week"] + gp.k_shift[d["k"].to_numpy()]) / 30)
    d["season_lag8"] = d.groupby("article_id")["season"].shift(8)
    d = d[d["week"] >= 10]
    c = _within_corr(d, "discount", "season_lag8")
    print(f"corr(discount, season 8 weeks earlier), within article: mean {c.mean():+.2f}, "
          f"share negative {(c < 0).mean():.0%}")


def save(df: pd.DataFrame, out_dir: Path, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = out_dir / f"synthetic_pricing_seed{seed}.parquet"
        df.to_parquet(path, index=False)
    except Exception as e:                                # e.g. pyarrow not installed
        print("parquet not available:", e)
        path = out_dir / f"synthetic_pricing_seed{seed}.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
    return path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-articles", type=int, default=N_ARTICLES)
    ap.add_argument("--out-dir", type=Path, default=repo_root / "data")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    df, gp, n_dropped = build_dataset(args.seed, args.n_articles)
    print(f"articles: {df['article_id'].nunique()}   rows: {len(df)}   dropped during build: {n_dropped}")
    validate(df)
    if not args.no_report:
        report(df, gp)
    path = save(df, args.out_dir, args.seed)
    print("saved:", path.resolve(), f"({path.stat().st_size / 1e6:.1f} MB)")
    print(df.head(3).round(2).to_string())


if __name__ == "__main__":
    main()