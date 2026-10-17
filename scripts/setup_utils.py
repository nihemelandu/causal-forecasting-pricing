# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:48:13 2026

@author: ngozi
"""

"""
setup_utils.py

Shared utilities for the model comparison sprint: data loading,
train/test split, and the covariate/target/treatment column names
used consistently across all 7 model scripts.
"""
import pandas as pd


# ============================================================
# Shared column definitions
# ============================================================
Z_COLS = ['category_d', 'category_k', 't', 's_it', 'tau_it', 'stock']  # covariates
TARGET = 'demand_true'   # uncensored demand - the actual modeling target
PRICE  = 'price'         # the treatment variable


def load_data(path="data/panel_ticket11.csv"):
    """Load the synthetic panel dataset generated in Sprint 1."""
    return pd.read_csv(path)


def get_train_test_split(panel, train_start=20, train_end=65, test_start=66, test_end=70):
    """
    Bare-minimum train/test split: one rolling window, matching the
    first of the paper's four windows (20-65 train / 66-70 test).

    Later sprints: extend to all four rolling windows, and repeat
    with multiple seeds, per the paper's actual evaluation design.
    """
    train = panel[(panel['t'] >= train_start) & (panel['t'] <= train_end)].copy()
    test = panel[(panel['t'] >= test_start) & (panel['t'] <= test_end)].copy()
    return train, test


if __name__ == "__main__":
    # Quick sanity check when this file is run directly
    panel = load_data()
    train, test = get_train_test_split(panel)
    print(f"Panel shape: {panel.shape}")
    print(f"Train shape: {train.shape}  (weeks {train['t'].min()}-{train['t'].max()})")
    print(f"Test shape:  {test.shape}  (weeks {test['t'].min()}-{test['t'].max()})")
    print(f"Z_COLS: {Z_COLS}")
    print(f"TARGET: {TARGET}")
    print(f"PRICE: {PRICE}")