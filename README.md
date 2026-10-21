# Causal Forecasting for Pricing

**Live dashboard: https://causal-forecasting-pricing.streamlit.app/**

> Can we forecast demand over the coming weeks for a price plan the business has never tried?

## The problem

Retailers decide discounts weeks ahead, and they need to know what each candidate plan would do to demand, stock and revenue. A standard demand forecaster learns from past prices and past sales. That works when next quarter's pricing looks like last quarter's. It breaks down when it does not, for example with a deeper discount than any the business has run, or a plan that discounts at a time of year when it never did before.

The reason is confounding. Past prices were not set at random: the pricing policy discounted when demand looked weak and held price when demand looked strong. A model trained on that history learns the policy's habits as much as customers' real response to price. Asked about a plan outside those habits, it extrapolates from patterns that no longer apply.

**This project asks:** how can we forecast demand over a given time horizon when the article's price plan has not been used in the past, and so was not in the training data?

## Why it matters

- **Discount decisions are made on forecasts.** A forecast that is wrong about untried prices makes the most valuable decisions, the new ones, the least reliable.
- **Errors are costly in both directions.** Over-forecasting leaves stock unsold at the end of the season; under-forecasting means lost sales and stockouts.
- **"Which price is best?" is a causal question.** Answering it needs an estimate of the price effect, not only a good fit to history.

## Approach

We replicate and assess the method in *Causal Forecasting for Pricing* (Schultz et al., Zalando SE, [arXiv:2312.15282](https://arxiv.org/abs/2312.15282)). It uses double machine learning (DML) to separate the effect of price from the pricing policy's habits, then forecasts demand under any chosen price plan.

The paper reports, on a closed-source retailer dataset, that its DML forecaster beat a transformer forecaster on off-policy Cyber Week forecasts in 2022 and 2020 and was similar in 2019, while trailing slightly on-policy. We have not verified these results ourselves; checking whether the approach holds up is the purpose of this repo.

**Plan**
1. Replicate the paper's synthetic-data experiment, where the true price effect is known.
2. Compare seven models under identical splits, each in its own script.
3. Test on a real dataset.

## How success is measured

- **Primary:** forecast error on new price plans (off-policy), alongside error under usual prices (on-policy), against a standard forecaster.
- **Also:** accuracy of the estimated price effect (possible on synthetic data only).
- **Targets:** to be agreed before results are produced, and recorded in the dashboard.

Synthetic data is used first because the true effect of price can never be observed in real data. Rankings on synthetic data do not automatically transfer to real data, which is why step 3 exists.

## Status

| Item | State |
|---|---|
| Synthetic data generator (4,467 articles, 100 weeks) | Done, validated |
| Dashboard with placeholders | Live |
| Train/test splits (paper's four windows, on- and off-policy) | In progress |
| Model 1 (DML) and six comparison models | Not started |
| Real dataset | Not started |

## Repository

```
scripts/generate_synthetic_data.py   synthetic data generator (seeded, reproducible)
scripts/dashboard.py                 Streamlit dashboard
outputs/metrics.json                 results file the dashboard reads
reference_papers/                    the paper
data/                                generated data (git-ignored, except small files)
```

## Run it

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py      # writes data/synthetic_pricing_seed42.parquet
streamlit run scripts/dashboard.py
```

## Where we depart from the paper

Everything here follows the paper unless stated. The generator's docstring lists every deviation and assumption (for example, the sign of the price term in Eq. 13, the initial-stock formula, and the Eq. 27 averaging window). The dashboard's Technical tab shows the same list.

## Reference

Schultz et al. *Causal Forecasting for Pricing.* Zalando SE. arXiv:2312.15282.
