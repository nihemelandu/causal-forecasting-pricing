# Causal Forecasting for Pricing

**Live dashboard: https://causal-forecasting-pricing.streamlit.app/**

> Can we forecast demand over the coming weeks for a price plan the business has never tried?

## The problem

Retailers decide discounts weeks ahead, and they need to know what each candidate plan would do to demand, stock and revenue. A standard demand forecaster learns from past prices and past sales. That works when next quarter's pricing looks like last quarter's. It breaks down when it does not, for example with a deeper discount than any the business has run.

The reason is confounding. Past prices were not set at random: the pricing policy discounted when demand looked weak and held price when demand looked strong. A model trained on that history learns the policy's habits as much as customers' real response to price.

**This project asks:** how can we forecast demand over a given time horizon when the article's price plan has not been used in the past?

## Why it matters

- **Discount decisions are made on forecasts.** A forecast that is wrong about untried prices makes the most valuable decisions, the new ones, the least reliable.
- **Errors are costly in both directions.** Over-forecasting leaves stock unsold at the end of the season; under-forecasting means lost sales and stockouts.
- **"Which price is best?" is a causal question.** It needs the price effect, not only a good fit to history.

## Result

**On new price plans, the DML forecaster's error is 24% lower than the standard forecaster's (12.4 vs 16.3), and under usual pricing it is 13% lower (10.0 vs 11.5).**

Success targets, set before the runs: at least 20% better on new prices, and usual-price error no more than 10% worse. Both are met.

| Model | New prices (MAE) | Usual prices (MAE) | Price-effect error (MAE) |
|---|---|---|---|
| **DML forecaster** | **12.4** | **10.0** | **25.0** |
| Standard forecaster (TF) | 16.3 | 11.5 | 45.8 |
| sDML (no treatment model) | 20.5 | 11.0 | 89.1 |
| DML without cross-fitting | 12.4 | 10.1 | 22.9 |
| sDML without cross-fitting | 20.5 | 11.0 | 89.5 |
| SARIMAX | — | — | — |
| TWFE elasticities | — | — | — |

MAE is the mean absolute error of weekly demand; lower is better. Values are averages over 4 test windows and 5 seeds.

## How we tested it

The data is synthetic: 4,467 articles over 100 weeks, where the true price effect is known. Each model is trained on one of four windows (weeks 20–65, 30–75, 40–85, 50–95) and forecasts the 5 weeks that follow. It is scored two ways:

- **Usual prices:** the prices actually used in those weeks.
- **New prices:** flat discounts of 0, 10, 20, 30, 40 and 50%, scored against the true demand at those prices.

Each model is trained 5 times per window with different seeds, giving 20 fits per model.

## Limits of these results

- The data is synthetic, built so that causal claims can be checked. Rankings may not carry over to real retail data.
- The spread across seeds measures training noise, not uncertainty about real-world accuracy.
- The four test windows overlap in training weeks and use the same articles, so they are not independent evidence.

## Next step

Validate the approach on a real retail dataset.

## Repository

```
scripts/generate_synthetic_data.py   synthetic data generator (seeded, reproducible)
scripts/splits.py                    train/test windows and off-policy ground truth
scripts/model_1_dml.py ...           one script per model
scripts/build_metrics.py             fills outputs/metrics.json for the dashboard
scripts/dashboard.py                 Streamlit dashboard
outputs/                             results files
```

## Run it

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python scripts/model_1_dml.py
python scripts/build_metrics.py
streamlit run scripts/dashboard.py
```

## Where we depart from the paper

The generator's docstring lists every deviation and assumption. The dashboard's Technical tab shows the same list.

## Reference

Schultz et al. *Causal Forecasting for Pricing.* Zalando SE. arXiv:2312.15282.
