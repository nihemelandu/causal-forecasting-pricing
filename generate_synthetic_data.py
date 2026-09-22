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
# Step 1: draw ONE alpha per category d, ONE beta per category k (fixed, shared within category)
alpha_d = rng.normal(loc=10, scale=3, size=N_CATEGORIES_D + 1)   # index by category id (1..45)
beta_k = rng.normal(loc=300, scale=50, size=N_CATEGORIES_K + 1)  # index by category id (1..15)

# Step 2: build the long (article x week) panel
weeks = np.arange(N_WEEKS)
panel = pd.DataFrame(
    [(a, t) for a in article_ids for t in weeks],
    columns=["article_id", "t"]
)
panel = panel.merge(articles, on="article_id", how="left")

# Step 3: per (article, week) noise terms
n_rows = len(panel)
epsilon_it = rng.normal(loc=0, scale=1, size=n_rows)   # noise on a_it
psi_it = rng.normal(loc=0, scale=5, size=n_rows)       # noise on b_it

# Step 4: compute a_it, b_it using each row's category's fixed alpha/beta + weekly noise
panel["a_it"] = alpha_d[panel["category_d"].values] + epsilon_it
panel["b_it"] = beta_k[panel["category_k"].values] + psi_it

# --- Ticket 3 ---
# c_it: article-specific scaling factor, combining a_it and b_it with fixed weights
panel["c_it"] = 0.05 * panel["a_it"]**2 + 0.25 * panel["a_it"] + 0.5 * panel["b_it"]

# --- Ticket 4 ---
# Per-article trend parameters: slope gamma_i, noise scale sigma_tau_i (drawn once per article)
gamma_i = rng.uniform(low=-0.02, high=0.02, size=N_ARTICLES)
sigma_tau_i = rng.uniform(low=0, high=0.15, size=N_ARTICLES)

articles["gamma_i"] = gamma_i
articles["sigma_tau_i"] = sigma_tau_i

panel = panel.merge(articles[["article_id", "gamma_i", "sigma_tau_i"]], on="article_id", how="left")

# tau_it ~ N(t * gamma_i, sigma_tau_i^2), drawn per article-week
tau_noise = rng.normal(loc=0, scale=1, size=len(panel))  # standard normal draw
panel["tau_it"] = panel["t"] * panel["gamma_i"] + panel["sigma_tau_i"] * tau_noise

# --- Ticket 5 ---
# Seasonality: sine wave, 30-week period, with a phase shift per "season type"
# Season type: k-categories (1..15) split evenly into 6 subgroups; each subgroup gets
# a uniformly sampled integer phase shift in [-15, 15]
SEASON_PERIOD = 30
N_SEASON_TYPES = 6

# Map each k-category (1..15) to one of 6 season-type subgroups
k_values = np.arange(1, N_CATEGORIES_K + 1)
season_type_of_k = (rng.permutation(k_values) - 1) % N_SEASON_TYPES  # even split into 6 groups
k_to_season_type = dict(zip(k_values, season_type_of_k))

# Sample ONE phase shift per season type (not per article) - shared within the season type
phase_shift_per_season_type = rng.integers(low=-15, high=16, size=N_SEASON_TYPES)  # inclusive of 15

articles["season_type"] = articles["category_k"].map(k_to_season_type)
articles["phase_shift"] = articles["season_type"].map(
    lambda s: phase_shift_per_season_type[s]
)

panel = panel.merge(articles[["article_id", "season_type", "phase_shift"]], on="article_id", how="left")

# s_it = sin(2*pi*(t + phase_shift) / period)
panel["s_it"] = np.sin(2 * np.pi * (panel["t"] + panel["phase_shift"]) / SEASON_PERIOD)

# --- Ticket 6 ---
# Baseline demand: combine trend, seasonality, and the article-specific scale c_it
panel["q_b_it"] = (0.15 * panel["tau_it"] + 0.25 * panel["s_it"] + 1) * panel["c_it"]

# --- Ticket 7 ---
# Treatment effect e_i: the GROUND TRUTH causal parameter (fixed per article, constant over time)
# Step 1: a_bar_i = average of a_it over all 100 weeks, per article
a_bar_i = panel.groupby("article_id")["a_it"].mean().rename("a_bar_i")
articles = articles.merge(a_bar_i, on="article_id", how="left")

# Step 2: e_i^(b) ~ max(1.3, LogNormal(0.75, 0.125^2)), drawn once per article
e_b_i_raw = rng.lognormal(mean=0.75, sigma=0.125, size=N_ARTICLES)
e_b_i = np.maximum(1.3, e_b_i_raw)
articles["e_b_i"] = e_b_i

# Step 3: e_i = e_b_i * 0.15 * a_bar_i
articles["e_i"] = articles["e_b_i"] * 0.15 * articles["a_bar_i"]

# Merge ground-truth e_i back into the panel (constant across all weeks for a given article)
panel = panel.merge(articles[["article_id", "e_i"]], on="article_id", how="left")

# --- Ticket 8 ---
# Step 1: q_bar_i = average baseline demand over all 100 weeks, per article
q_bar_i = panel.groupby("article_id")["q_b_it"].mean().rename("q_bar_i")
articles = articles.merge(q_bar_i, on="article_id", how="left")

# Step 2: initial price p_i0 - FIXED to use LogNormal (moment-matched to the paper's
# stated Normal mean/std) instead of Normal, since Normal produced ~31% negative draws.
# Target: mean = q_bar_i/3, std = q_bar_i/1.5 (i.e., std = 2*mean, ratio constant across articles)
# Moment-matching: sigma^2 = ln(1 + (std/mean)^2) = ln(5)  [same for every article]
#                  mu_i = ln(mean_i) - sigma^2/2
m_i = articles["q_bar_i"] / 3
sigma_ln = np.sqrt(np.log(5))
mu_i = np.log(m_i) - sigma_ln**2 / 2

articles["p_i0"] = rng.lognormal(mean=mu_i, sigma=sigma_ln, size=N_ARTICLES)

# Step 3: initial stock z_i0 (Eq. 25, per clean PDF) - plain multiplication, not exponentiation
# z_i0 = (1/100) * sum_t [ q_b_it * (1 - 0.14) * p_i0 * e_i ]
panel = panel.merge(articles[["article_id", "p_i0"]], on="article_id", how="left")
panel["_clear_term"] = panel["q_b_it"] * (1 - 0.14) * panel["p_i0"] * panel["e_i"]
z_i0 = panel.groupby("article_id")["_clear_term"].mean().rename("z_i0")
articles = articles.merge(z_i0, on="article_id", how="left")
panel = panel.drop(columns=["_clear_term"])

# --- Tickets 9 & 10: pricing policy loop + weekly demand + stock updates ---
# These are implemented together because price at week t depends on the stock
# trajectory, which depends on realized demand in prior weeks - inherently sequential.

# Pivot q_b_it and e_i into (article x week) arrays for fast sequential computation
q_b_wide = panel.pivot(index="article_id", columns="t", values="q_b_it").values  # shape (N_ARTICLES, N_WEEKS)
e_i_arr = articles.set_index("article_id").loc[np.arange(N_ARTICLES), "e_i"].values
p_i0_arr = articles.set_index("article_id").loc[np.arange(N_ARTICLES), "p_i0"].values

# Allocate output arrays
price = np.zeros((N_ARTICLES, N_WEEKS))
demand_true = np.zeros((N_ARTICLES, N_WEEKS))   # underlying/uncensored demand from the model
demand = np.zeros((N_ARTICLES, N_WEEKS))        # realized/observed demand (censored by stock)
stock = np.zeros((N_ARTICLES, N_WEEKS + 1))  # stock[:, t] = stock AT START of week t
discount_step = np.zeros(N_ARTICLES, dtype=int)  # j_t, starts at 0 (no discount)

stock[:, 0] = z_i0.reindex(np.arange(N_ARTICLES)).values  # initial stock z_i0

# FIX: realized demand cannot exceed available stock (Table 3 states stock is Integer >= 0).
# demand_true is the model's underlying demand signal; demand (realized/recorded) is
# censored at whatever stock remains, so stock can never go negative.
def realize_and_deplete(t, price_t):
    dt_true = q_b_wide[:, t] + price_t * e_i_arr
    dt_realized = np.minimum(dt_true, stock[:, t])   # censor at available stock
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
    # m_t: linear extrapolation of weeks-to-clear, based on trailing 4-week REALIZED demand
    # (using realized demand since that's what's actually observable in practice)
    trailing_demand = demand[:, t-4:t].sum(axis=1)  # sum over (t-4,...,t-1)
    trailing_demand_safe = np.where(trailing_demand <= 0, 1e-6, trailing_demand)  # avoid div by 0
    m_t = 4 * stock[:, t] / trailing_demand_safe

    # w_t: stock coverage
    weeks_remaining = max(N_WEEKS - t, 1)
    w_t = m_t / weeks_remaining

    # Probabilistic discount step update (Eq. 29)
    lambda_ti = rng.uniform(low=0, high=1, size=N_ARTICLES)
    step_up = (w_t > 1) & (lambda_ti > 1 / np.where(w_t == 0, 1e-6, w_t))
    step_down = (w_t < 1) & (lambda_ti > w_t)
    discount_step = discount_step + step_up.astype(int) - step_down.astype(int)
    discount_step = np.clip(discount_step, 0, 5)  # j_t in {0,...,5}

    d_jt = discount_step * 0.1
    price[:, t] = p_i0_arr * (1 - d_jt)
    realize_and_deplete(t, price[:, t])

# Fold results back into the long panel
price_long = price.flatten()
demand_long = demand.flatten()
demand_true_long = demand_true.flatten()
stock_long = stock[:, :N_WEEKS].flatten()  # stock at start of each week

panel = panel.sort_values(["article_id", "t"]).reset_index(drop=True)
panel["price"] = price_long
panel["demand"] = demand_long
panel["demand_true"] = demand_true_long
panel["stock"] = stock_long

if __name__ == "__main__":
    print(panel[["article_id", "t", "q_b_it", "price", "demand", "stock"]].head(10))
    print(f"\nPanel shape: {panel.shape}")
    print(f"\nprice summary:\n{panel['price'].describe()}")
    print(f"\ndemand summary:\n{panel['demand'].describe()}")
    print(f"\ndemand negative count: {(panel['demand'] < 0).sum()}")
    print(f"\nstock negative count: {(panel['stock'] < 0).sum()}")

    # sanity: article 0's discount and price trajectory
    sample = panel[panel["article_id"] == 0][["t", "price", "demand", "stock"]]
    print(f"\nArticle 0, weeks 0-9:")
    print(sample.head(10))
    print(f"\nArticle 0, weeks 90-99:")
    print(sample.tail(10))

# --- Ticket 11 ---
# Filter out articles that have negative demand at ANY week (per paper's stated filtering rule)
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
    print(f"\n(Paper reports ~4,467 as their FINAL post-filter count, so a large removal here")
    print(f" would suggest we are still not matching their intended demand dynamics.)")

    panel_filtered.to_csv("/home/claude/panel_ticket11.csv", index=False)
    articles_filtered.to_csv("/home/claude/articles_ticket11.csv", index=False)
    print("\nSaved to panel_ticket11.csv and articles_ticket11.csv")
