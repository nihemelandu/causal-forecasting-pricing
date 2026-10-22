# splits.py - train/test windows and off-policy ground truth for the synthetic experiment.
# Shared by all seven model scripts so every model sees identical splits.
# Run the self-check:  python scripts/splits.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_pricing_seed42.parquet"

DISCOUNT_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)   # decision 1: six levels, 0-50%
SEEDS = (0, 1, 2, 3, 4)                            # decision 4: five model seeds, fixed data


@dataclass(frozen=True)
class Window:
    name: str
    train_start: int   # all bounds inclusive; weeks are 0..99
    train_end: int
    test_start: int
    test_end: int


# Hyperparameter tuning: first 45 weeks train, next 5 hold-out (paper: "weeks 46-50", mapped to 0-indexed).
TUNING = Window("tuning", 0, 44, 45, 49)

# The paper's four evaluation windows. The last is truncated to 96-99 because week 100 does not exist.
WINDOWS = (
    Window("w1", 20, 65, 66, 70),
    Window("w2", 30, 75, 76, 80),
    Window("w3", 40, 85, 86, 90),
    Window("w4", 50, 95, 96, 99),
)

INPUT_COLS = ["article_id", "week", "d", "k", "promotion", "p0"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values(["article_id", "week"]).reset_index(drop=True)
    return df


def get_split(df: pd.DataFrame, w: Window):
    """Return (train, test_full).

    train      : model-visible columns only (true_* columns removed), weeks train_start..train_end.
    test_full  : everything incl. true_* columns, weeks test_start..test_end. Used ONLY for scoring
                 and for building model inputs through test_inputs().
    """
    assert w.train_end < w.test_start, "train and test windows overlap"
    visible = df.drop(columns=[c for c in df.columns if c.startswith("true_")])
    train = visible[visible["week"].between(w.train_start, w.train_end)].reset_index(drop=True)
    test_full = df[df["week"].between(w.test_start, w.test_end)].reset_index(drop=True)
    return train, test_full


def test_inputs(test_full: pd.DataFrame, level: float | None = None) -> pd.DataFrame:
    """Model inputs for the test window. level=None -> realized prices (on-policy);
    level=0.3 -> a flat 30% discount off p0 in every test week (off-policy). No demand, sales or
    stock columns are included."""
    out = test_full[INPUT_COLS].copy()
    if level is None:
        out["price"] = test_full["price"].to_numpy()
        out["discount"] = test_full["discount"].to_numpy()
    else:
        out["discount"] = level
        out["price"] = test_full["p0"].to_numpy() * (1.0 - level)
    return out


def truth_demand(test_full: pd.DataFrame, level: float | None = None) -> pd.Series:
    """Ground-truth demand for the scenario. On-policy: realized demand. Off-policy: the same
    article-week base demand (which already holds the noise) with the price changed:
    q = true_base_demand - price * true_effect_e, price = p0 * (1 - level)."""
    if level is None:
        y = test_full["demand"].to_numpy()
    else:
        y = (test_full["true_base_demand"].to_numpy()
             - test_full["p0"].to_numpy() * (1.0 - level) * test_full["true_effect_e"].to_numpy())
    return pd.Series(y, index=test_full.index, name="demand")


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mse(y, yhat) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def effect_errors(e_hat: pd.Series, test_full: pd.DataFrame) -> dict:
    """e_hat: estimated price effect per article, per unit of discount, indexed by article_id.
    True effect per unit discount = p0 * e_i, because
    q = q_b - p0*(1-d)*e = (q_b - p0*e) + d*p0*e. (Our reading of the paper's Eq. 11.)"""
    g = test_full.groupby("article_id")
    e_true = g["p0"].first() * g["true_effect_e"].first()
    e_hat = e_hat.reindex(e_true.index)
    return {"mae_effect": mae(e_true, e_hat), "mse_effect": mse(e_true, e_hat)}

def run_window(model, df: pd.DataFrame, w: Window, seed: int = 0) -> dict:
    """Fit on the training window, then score on-policy and off-policy.

    The model must provide:  fit(train: DataFrame, seed: int)  and  predict(inputs: DataFrame) -> array
    of predicted demand, one value per row of inputs, in the same order."""
    train, test_full = get_split(df, w)
    model.fit(train, seed=seed)

    y_on = truth_demand(test_full)
    yhat_on = model.predict(test_inputs(test_full))
    res = {"window": w.name, "seed": seed,
           "on_mae": mae(y_on, yhat_on), "on_mse": mse(y_on, yhat_on)}

    off_mae, off_mse = [], []
    for lv in DISCOUNT_LEVELS:
        y = truth_demand(test_full, lv)
        yhat = model.predict(test_inputs(test_full, lv))
        res[f"off_mae_{int(round(lv * 100))}"] = mae(y, yhat)
        off_mae.append(mae(y, yhat))
        off_mse.append(mse(y, yhat))
    res["off_mae"] = float(np.mean(off_mae))   # average over the six discount levels
    res["off_mse"] = float(np.mean(off_mse))
    return res


class NaiveRecentMean:
    """Plumbing check and weakest baseline: predicts each article's mean demand over the last 4
    training weeks, ignoring price. Not one of the seven models."""

    def fit(self, train: pd.DataFrame, seed: int = 0):
        last = train["week"].max()
        recent = train[train["week"] > last - 4]
        self.mean_ = recent.groupby("article_id")["demand"].mean()
        return self

    def predict(self, inputs: pd.DataFrame):
        return inputs["article_id"].map(self.mean_).to_numpy()


def self_check():
    df = load_data()
    print(f"rows {len(df):,}  articles {df['article_id'].nunique():,}  "
          f"weeks {df['week'].min()}..{df['week'].max()}")

    # 1. identity behind the counterfactual: demand = true_base_demand - price * e
    ident = (df["demand"] - (df["true_base_demand"] - df["price"] * df["true_effect_e"])).abs().max()
    print(f"max |demand - (q_b - price*e)|            : {ident:.3e}")

    # 2. price = p0 * (1 - discount)
    pdiff = (df["price"] - df["p0"] * (1 - df["discount"])).abs().max()
    print(f"max |price - p0*(1-discount)|             : {pdiff:.3e}")

    # 3. counterfactual at the realized discount must reproduce realized demand
    for w in WINDOWS:
        _, test_full = get_split(df, w)
        worst = 0.0
        for lv in DISCOUNT_LEVELS:
            rows = np.isclose(test_full["discount"].to_numpy(), lv)
            if rows.any():
                cf = truth_demand(test_full, lv).to_numpy()[rows]
                worst = max(worst, float(np.abs(cf - test_full["demand"].to_numpy()[rows]).max()))
        print(f"{w.name}: train {w.train_start}-{w.train_end} test {w.test_start}-{w.test_end}  "
              f"rows(train/test) {len(get_split(df, w)[0]):,}/{len(test_full):,}  "
              f"max cf-vs-realized diff {worst:.1e}")

    # 4. no true_* column reaches the model
    train, test_full = get_split(df, WINDOWS[0])
    assert not any(c.startswith("true_") for c in train.columns)
    assert not any(c.startswith("true_") for c in test_inputs(test_full).columns)
    print("leakage guard: no true_* columns in train or test inputs - OK")

    # 5. mean true demand by scenario in window 1 (should rise with the discount)
    _, tf = get_split(df, WINDOWS[0])
    print("w1 mean demand  realized: %.2f" % truth_demand(tf).mean(),
          " | by level:", {int(lv * 100): round(float(truth_demand(tf, lv).mean()), 2)
                           for lv in DISCOUNT_LEVELS})

    # 6. plumbing run with the naive baseline
    rows = [run_window(NaiveRecentMean(), df, w) for w in WINDOWS]
    out = pd.DataFrame(rows)[["window", "on_mae", "off_mae", "on_mse", "off_mse"]]
    print("\nNaiveRecentMean (ignores price):")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    self_check()