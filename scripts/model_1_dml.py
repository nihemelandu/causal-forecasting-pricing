# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:02:33 2026

@author: ngozi
"""

"""
model_1_dml.py

DML Forecaster - bare minimum implementation for Sprint 2.

Architecture (per the paper, transformer treated as a black box -
LightGBM used as the stand-in regressor):
    - Outcome model:   predicts demand_true from covariates z
    - Treatment model: predicts price from covariates z
    - Effect model:    bare minimum = single global slope theta,
                        recovered via residual-on-residual regression
                        (Frisch-Waugh-Lovell). The paper's actual
                        effect model is a flexible psi(z) that varies
                        per article/context - that's an incremental
                        improvement for a later sprint.

Forecast formula - bare minimum (additive, not the paper's multiplicative
elasticity form):
    q_hat = q_tilde + theta * (d - d_tilde)
"""
from lightgbm import LGBMRegressor
import numpy as np

from setup_utils import load_data, get_train_test_split, Z_COLS, TARGET, PRICE


def train_dml(train):
    """Train the outcome model, treatment model, and recover theta."""
    outcome_model = LGBMRegressor().fit(train[Z_COLS], train[TARGET])
    treatment_model = LGBMRegressor().fit(train[Z_COLS], train[PRICE])

    q_tilde_train = outcome_model.predict(train[Z_COLS])
    d_tilde_train = treatment_model.predict(train[Z_COLS])

    u = train[TARGET].values - q_tilde_train   # outcome residual
    v = train[PRICE].values - d_tilde_train     # treatment residual

    theta = np.sum(u * v) / np.sum(v ** 2)      # residual-on-residual slope

    return outcome_model, treatment_model, theta


def predict_dml(df, outcome_model, treatment_model, theta):
    """Forecast demand for a given dataframe of covariates + price."""
    q_tilde = outcome_model.predict(df[Z_COLS])
    d_tilde = treatment_model.predict(df[Z_COLS])
    return q_tilde + theta * (df[PRICE].values - d_tilde)


if __name__ == "__main__":
    panel = load_data()
    train, test = get_train_test_split(panel)

    outcome_model, treatment_model, theta = train_dml(train)
    print(f"Recovered theta (global price effect estimate): {theta:.4f}")

    preds = predict_dml(test, outcome_model, treatment_model, theta)
    print("\nFirst 10 predictions vs actual (test set):")
    print("Predicted:", np.round(preds[:10], 2))
    print("Actual:   ", test[TARGET].values[:10].round(2))