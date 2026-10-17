# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:16:55 2026

@author: ngozi
"""

"""
Synthetic dataset generation - Causal Forecasting for Pricing (Appendix E)
Sprint 1 / Ticket 2: Category-level random effects (a_it, b_it)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

N_ARTICLES = 4467
N_WEEKS = 100
N_CATEGORIES_D = 45
N_CATEGORIES_K = 15

# --- Ticket 1 (re-run for continuity) ---
article_ids = np.arange(N_ARTICLES)
category_d = rng.integers(low=1, high=N_CATEGORIES_D + 1, size=N_ARTICLES)
category_k = rng.integers(low=1, high=N_CATEGORIES_K + 1, size=N_ARTICLES)

articles = pd.DataFrame({
    "article_id": article_ids,
    "category_d": category_d,
    "category_k": category_k,
})

# --- Ticket 2 ---
alpha_d = rng.normal(loc=10, scale=3, size=N_CATEGORIES_D + 1)
beta_k = rng.normal(loc=300, scale=50, size=N_CATEGORIES_K + 1)

weeks = np.arange(N_WEEKS)
panel = pd.DataFrame(
    [(a, t) for a in article_ids for t in weeks],
    columns=["article_id", "t"]
)
panel = panel.merge(articles, on="article_id", how="left")

n_rows = len(panel)
epsilon_it = rng.normal(loc=0, scale=1, size=n_rows)
psi_it = rng.normal(loc=0, scale=5, size=n_rows)

panel["a_it"] = alpha_d[panel["category_d"].values] + epsilon_it
panel["b_it"] = beta_k[panel["category_k"].values] + psi_it

# --- Ticket 3 ---
panel["c_it"] = 0.05 * panel["a_it"]**2 + 0.25 * panel["a_it"] + 0.5 * panel["b_it"]

# --- Ticket 4 ---
gamma_i = rng.uniform(low=-0.02, high=0.02, size=N_ARTICLES)
sigma_tau_i = rng.uniform(low=0, high=0.15, size=N_ARTICLES)

articles["gamma_i"] = gamma_i
articles["sigma_tau_i"] = sigma_tau_i

panel = panel.merge(articles[["article_id", "gamma_i", "sigma_tau_i"]], on="article_id", how="left")

tau_noise = rng.normal(loc=0, scale=1, size=len(panel))
panel["tau_it"] = panel["t"] * panel["gamma_i"] + panel["sigma_tau_i"] * tau_noise

# --- Ticket 5 ---
SEASON_PERIOD = 30
N_SEASON_TYPES = 6

k_values = np.arange(1, N_CATEGORIES_K + 1)
season_type_of_k = (rng.permutation(k_values) - 1) % N_SEASON_TYPES
k_to_season_type = dict(zip(k_values, season_type_of_k))

phase_shift_per_season_type = rng.integers(low=-15, high=16, size=N_SEASON_TYPES)

articles["season_type"] = articles["category_k"].map(k_to_season_type)
articles["phase_shift"] = articles["season_type"].map(
    lambda s: phase_shift_per_season_type[s]
)

panel = panel.merge(articles[["article_id", "season_type", "phase_shift"]], on="article_id", how="left")
panel["s_it"] = np.sin(2 * np.pi * (panel["t"] + panel["phase_shift"]) / SEASON_PERIOD)

# --- Ticket 6 ---
panel["q_b_it"] = (0.15 * panel["tau_it"] + 0.25 * panel["s_it"] + 1) * panel["c_it"]

# --- Ticket 7 ---
a_bar_i = panel.groupby("article_id")["a_it"].mean().rename("a_bar_i")
articles = articles.merge(a_bar_i, on="article_id", how="left")

e_b_i_raw = rng.lognormal(mean=0.75, sigma=0.125, size=N_ARTICLES)
e_b_i = np.maximum(1.3, e_b_i_raw)
articles["e_b_i"] = e_b_i

articles["e_i"] = articles["e_b_i"] * 0.15 * articles["a_bar_i"]

panel = panel.merge(articles[["article_id", "e_i"]], on="article_id", how="left")

# --- Ticket 8 ---
q_bar_i = panel.groupby("article_id")["q_b_it"].mean().rename("q_bar_i")
articles = articles.merge(q_bar_i, on="article_id", how="left")

# Initial price p_i0 - FIXED to use LogNormal (moment-matched to the paper's
# stated Normal mean/std) instead of Normal, since Normal produced ~31% negative draws.
# Target: mean = q_bar_i/3, std = q_bar_i/1.5 (i.e., std = 2*mean, ratio constant across articles)
# Moment-matching: sigma^2 = ln(1 + (std/mean)^2) = ln(5)  [same for every article]
#                  mu_i = ln(mean_i) - sigma^2/2
m_i = articles["q_bar_i"] / 3
sigma_ln = np.sqrt(np.log(5))
mu_i = np.log(m_i) - sigma_ln**2 / 2

articles["p_i0"] = rng.lognormal(mean=mu_i, sigma=sigma_ln, size=N_ARTICLES)

# z_i0 (Eq. 25, per clean PDF): plain multiplication, not exponentiation
panel = panel.merge(articles[["article_id", "p_i0"]], on="article_id", how="left")
panel["_clear_term"] = panel["q_b_it"] * (1 - 0.14) * panel["p_i0"] * panel["e_i"]
z_i0 = panel.groupby("article_id")["_clear_term"].mean().rename("z_i0")
articles = articles.merge(z_i0, on="article_id", how="left")
panel = panel.drop(columns=["_clear_term"])

# --- Tickets 9 & 10: pricing policy loop + weekly demand + stock updates ---
q_b_wide = panel.pivot(index="article_id", columns="t", values="q_b_it").values
e_i_arr = articles.set_index("article_id").loc[np.arange(N_ARTICLES), "e_i"].values
p_i0_arr = articles.set_index("article_id").loc[np.arange(N_ARTICLES), "p_i0"].values

price = np.zeros((N_ARTICLES, N_WEEKS))
demand_true = np.zeros((N_ARTICLES, N_WEEKS))
demand = np.zeros((N_ARTICLES, N_WEEKS))
stock = np.zeros((N_ARTICLES, N_WEEKS + 1))
discount_step = np.zeros(N_ARTICLES, dtype=int)

stock[:, 0] = z_i0.reindex(np.arange(N_ARTICLES)).values

def realize_and_deplete(t, price_t):
    # FIX (confirmed via independent third-party reimplementation): the paper's
    # literal Eq. 13 (q = q_b + p*e_i, with e_i > 0) implies higher price -> higher
    # demand, which is economically backwards and breaks the discount policy's
    # own intent. Switched to subtraction: higher price -> lower demand.
    dt_true = q_b_wide[:, t] - price_t * e_i_arr
    dt_true = np.maximum(dt_true, 0)                  # demand floor: subtraction can go negative
    dt_realized = np.minimum(dt_true, stock[:, t])    # censor at available stock
    dt_realized = np.maximum(dt_realized, 0)          # guard: stock[:,t] should be >=0, but be safe
    demand_true[:, t] = dt_true
    demand[:, t] = dt_realized
    stock[:, t + 1] = stock[:, t] - dt_realized

# Weeks 0-3: price held constant at p_i0 (Eq: p_i0 = p_i1 = p_i2 = p_i3)
for t in range(4):
    price[:, t] = p_i0_arr
    realize_and_deplete(t, price[:, t])

# Weeks 4-99: dynamic pricing policy
for t in range(4, N_WEEKS):
    trailing_demand = demand[:, t-4:t].sum(axis=1)
    trailing_demand_safe = np.where(trailing_demand <= 0, 1e-6, trailing_demand)
    m_t = 4 * stock[:, t] / trailing_demand_safe

    weeks_remaining = max(N_WEEKS - t, 1)
    w_t = m_t / weeks_remaining

    lambda_ti = rng.uniform(low=0, high=1, size=N_ARTICLES)
    step_up = (w_t > 1) & (lambda_ti > 1 / np.where(w_t == 0, 1e-6, w_t))
    step_down = (w_t < 1) & (lambda_ti > w_t)
    discount_step = discount_step + step_up.astype(int) - step_down.astype(int)
    discount_step = np.clip(discount_step, 0, 5)

    d_jt = discount_step * 0.1
    price[:, t] = p_i0_arr * (1 - d_jt)
    realize_and_deplete(t, price[:, t])

# --- Ticket 11 ---
price_long = price.flatten()
demand_long = demand.flatten()
demand_true_long = demand_true.flatten()
stock_long = stock[:, :N_WEEKS].flatten()

panel = panel.sort_values(["article_id", "t"]).reset_index(drop=True)
panel["price"] = price_long
panel["demand"] = demand_long
panel["demand_true"] = demand_true_long
panel["stock"] = stock_long

articles_with_negative_demand = panel.loc[panel["demand"] < 0, "article_id"].unique()
n_before = panel["article_id"].nunique()

panel_filtered = panel[~panel["article_id"].isin(articles_with_negative_demand)].copy()
articles_filtered = articles[~articles["article_id"].isin(articles_with_negative_demand)].copy()

n_after = panel_filtered["article_id"].nunique()
n_removed = n_before - n_after

if __name__ == "__main__":
    print(f"Articles before filtering: {n_before}")
    print(f"Articles with negative demand at some point: {n_removed} ({100*n_removed/n_before:.1f}%)")
    print(f"Articles remaining after filtering: {n_after}")

    panel_filtered.to_csv("data/panel_ticket11.csv", index=False)
    articles_filtered.to_csv("data/articles_ticket11.csv", index=False)
    print("\nSaved to data/panel_ticket11.csv and data/articles_ticket11.csv")