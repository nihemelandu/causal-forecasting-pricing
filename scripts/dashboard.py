# dashboard.py - Streamlit dashboard for "Causal Forecasting for Pricing".
# Reads outputs/metrics.json (small, committed to GitHub). Null values are shown as
# placeholders such as [XX.X]; charts without data show a labelled placeholder shape.
# Run locally:  streamlit run scripts/dashboard.py
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# set_page_config must be the first Streamlit command in the script.
st.set_page_config(page_title="Causal Forecasting for Pricing", layout="wide")
#st.markdown("<style>.block-container {padding-top: 1rem; padding-bottom: 0.5rem;}</style>",
#            unsafe_allow_html=True)


METRICS_PATH = Path(__file__).resolve().parent.parent / "outputs" / "metrics.json"

if not METRICS_PATH.exists():
    st.error(f"Missing {METRICS_PATH}. Commit outputs/metrics.json.")
    st.stop()


def load_metrics(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


m = load_metrics(METRICS_PATH)
meta, targets, h = m["meta"], m["targets"], m["headline"]
models, ex = m["models"], m["article_example"]


def fmt(x, spec=".1f", placeholder="[XX.X]"):
    return placeholder if x is None else format(x, spec)


def pct_change(value, baseline):
    if value is None or not baseline:
        return None
    return (value - baseline) / baseline * 100


def placeholder_bars(labels):
    return pd.DataFrame({"value": [4, 5, 6, 5, 7, 6][:len(labels)]}, index=labels)


DEF_OFF = ("Average gap between forecast and actual weekly demand when the forecast is for a price "
           "plan the model never saw in training. Lower is better.")
DEF_ON = ("The same gap under usual pricing, the kind of prices the model did see in training. "
          "Lower is better.")
DEF_EFFECT = ("How far the model's estimate of the price effect is from the true value. Only possible "
              "on synthetic data, where the true effect is known. Lower is better.")
DEF_STATUS = "Whether the results meet the success targets agreed for the project."
DEFINITIONS = (
    f"**New prices (off-policy error).** {DEF_OFF}\n\n"
    f"**Usual prices (on-policy error).** {DEF_ON}\n\n"
    f"**Price-effect accuracy.** {DEF_EFFECT}\n\n"
    f"**Status vs target.** {DEF_STATUS}\n\n"
    "**Standard forecaster.** A forecasting model that learns from past prices and demand "
    "without correcting for why those prices were chosen."
)

DEVIATIONS = [
    "Eq. 13 uses a minus sign, so demand falls as price rises.",
    "Eq. 21 gives negative prices about 31% of the time; p_i0 is redrawn until it is positive and "
    "demand at p_i0 is non-negative in every week.",
    "Initial stock = (mean(q_b) - 0.86 * p_i0 * e_i) * 100, not Eq. 25 as printed.",
    "Eq. 27 uses weeks t-4..t-1 (the printed sum includes week t).",
    "j = 0 in weeks 0-3 (inferred), and j is clipped to 0..5.",
    "Sales = min(demand, stock); the true demand is kept.",
    "Articles are assigned to d and k at random; the 15 k values form season groups of 3,3,3,2,2,2.",
    "promotion is Bernoulli(0.5) per article, with no effect on demand.",
    "Season = sin(2*pi*(week + shift_k)/30); the sign convention of the shift is our choice.",
    "If Eq. 27 cannot be computed (0/0 after a long stockout), j is left unchanged.",
    "d and k names are cosmetic labels.",
]

# ---------------- Header ----------------
head_l, head_r = st.columns([3, 2])
#head_l.markdown("### Causal Forecasting for Pricing")
head_l.markdown(
    "<h1 style='margin:0; padding:0; font-size:2.2rem; font-weight:700; line-height:1.15;'>"
    "Causal Forecasting for Pricing</h1>",
    unsafe_allow_html=True,
)
head_l.caption("Can we forecast demand at prices we have never used?")
head_r.caption(f"Data: {meta['data_version']}  |  Run: {meta['run_date'] or '[date]'}  |  "
               f"Error metric: {meta['metric_name'] or '[metric TBD]'}")
with head_r.popover("? Definitions"):
    st.markdown(DEFINITIONS)

tab_overview, tab_whatif, tab_tech = st.tabs(["Overview", "What-if", "Technical"])

# ---------------- Overview (one screen) ----------------
with tab_overview:
    st.caption(
        "**Problem:** forecast demand at prices never used   |   "
        "**Decision:** pick a discount plan for an article   |   "
        f"**Success:** off-policy at least {fmt(targets['off_policy_improvement_pct'], '.0f', '[TBD]')}% "
        f"better than standard; usual pricing within {fmt(targets['on_policy_tolerance_pct'], '.0f', '[TBD]')}%")

    off_ch = pct_change(h["off_policy_error"], h["off_policy_baseline"])
    on_ch = pct_change(h["on_policy_error"], h["on_policy_baseline"])
    if off_ch is None:
        st.info("[Model 1 forecasts demand at new prices XX% more accurately than a standard model.]")
    else:
        word = "more" if off_ch < 0 else "less"
        st.info(f"Model 1 forecasts demand at new prices {abs(off_ch):.0f}% {word} accurately "
                "than a standard model.")

    def tile(col, label, value, baseline, change, help_text):
        delta = None if change is None else f"{change:+.0f}% vs standard"
        col.metric(label, fmt(value), delta=delta, delta_color="inverse", help=help_text)
        col.caption(f"vs standard {fmt(baseline)}" + ("  |  change [-XX%]" if change is None else ""))

    c1, c2, c3, c4 = st.columns(4)
    tile(c1, "New prices (off-policy error)", h["off_policy_error"], h["off_policy_baseline"], off_ch, DEF_OFF)
    tile(c2, "Usual prices (on-policy error)", h["on_policy_error"], h["on_policy_baseline"], on_ch, DEF_ON)
    c3.metric("Price-effect accuracy", fmt(h["effect_error"], ".2f", "[X.XX]"), help=DEF_EFFECT)
    c3.caption("error vs true effect (synthetic data only)")
    c4.metric("Status vs target", h["status"] or "[PASS / NOT YET]", help=DEF_STATUS)
    c4.caption("targets shown above")

    left, right = st.columns(2)
    with left:
        view = st.radio("Error by model (lower is better)", ["New prices", "Usual prices"], horizontal=True)
        key = "off_policy" if view == "New prices" else "on_policy"
        names = [mm["name"] for mm in models]
        values = [mm[key] for mm in models]
        if all(v is None for v in values):
            st.caption("PLACEHOLDER shape only - no results yet")
            values = [9, 8, 7, 7, 6, 6, 5]
        else:
            values = [0 if v is None else v for v in values]
        st.bar_chart(pd.DataFrame({"error": values}, index=names), height=230)
    with right:
        st.markdown("**One article, one test window**")
        if ex["actual"]:
            df_ex = pd.DataFrame({"actual": ex["actual"], "forecast": ex["forecast"]}, index=ex["weeks"])
        else:
            st.caption("PLACEHOLDER shape only - no results yet")
            df_ex = pd.DataFrame({"actual": [60, 66, 70, 68, 63], "forecast": [58, 64, 69, 69, 65]},
                                 index=[1, 2, 3, 4, 5])
        st.line_chart(df_ex, height=250)

# ---------------- What-if ----------------
with tab_whatif:
    ctrl, out = st.columns([1, 3])
    with ctrl:
        st.selectbox("Article", ["[pick an article once results exist]"], disabled=True)
        st.caption("Discount per week of the test window")
        defaults = [0, 10, 20, 30, 30]
        plan = [st.select_slider(f"Week {i}", options=[0, 10, 20, 30, 40, 50], value=defaults[i - 1],
                                 format_func=lambda v: f"{v}%", key=f"wk{i}") for i in range(1, 6)]
    with out:
        st.line_chart(pd.DataFrame({"current plan": [60, 66, 70, 68, 63], "your plan": [62, 70, 75, 76, 72]},
                                   index=[1, 2, 3, 4, 5]), height=250)
        st.caption("PLACEHOLDER shape only - no results yet. The true-demand line "
                   "(synthetic data only) is added with results.")
        avg = sum(plan) / len(plan)
        st.table(pd.DataFrame({
            "Scenario": ["Current plan", "Your plan", "Difference"],
            "Avg discount": ["[XX%]", f"{avg:.0f}%", ""],
            "Units": ["[XXXX]", "[XXXX]", "[+XX%]"],
            "Revenue": ["[XXXX]", "[XXXX]", "[+XX%]"],
            "Range (80%)": ["[XXXX-XXXX]", "[XXXX-XXXX]", ""],
        }).set_index("Scenario"))
        st.warning("A forecast is not a guarantee. It was validated on synthetic data, "
                   "where the true effect is known.")

# ---------------- Technical ----------------
with tab_tech:
    top = st.columns(3)
    bottom = st.columns(3)
    with top[0].container(border=True):
        st.markdown("**Error by horizon week**")
        st.bar_chart(placeholder_bars(["wk1", "wk2", "wk3", "wk4", "wk5"]), height=140)
        st.caption("PLACEHOLDER shape only - no results yet")
    with top[1].container(border=True):
        st.markdown("**Error by category (d x k)**")
        st.caption("[heatmap appears when results exist]")
    with top[2].container(border=True):
        st.markdown("**Error by discount level**")
        st.bar_chart(placeholder_bars(["0%", "10%", "20%", "30%", "40%", "50%"]), height=140)
        st.caption("PLACEHOLDER shape only - no results yet")
    with bottom[0].container(border=True):
        st.markdown("**Bias** (forecast minus actual; positive = over-forecast)")
        st.bar_chart(placeholder_bars(["wk1", "wk2", "wk3", "wk4", "wk5"]), height=110)
        st.caption("PLACEHOLDER shape only - no results yet")
    with bottom[1].container(border=True):
        st.markdown("**Health checks**")
        st.table(pd.DataFrame({
            "Check": ["Leakage (true_* columns unused)", "Test prices outside training range", "Calibration"],
            "Status": ["[ OK ]", "[XX%]", "[ OK ]"],
        }).set_index("Check"))
    with bottom[2].container(border=True):
        st.markdown("**Run and lineage**")
        st.caption(f"Data: {meta['data_version']}")
        st.caption("Generator: scripts/generate_synthetic_data.py")
        with st.expander("Deviations from the paper (11)"):
            for i, d in enumerate(DEVIATIONS, 1):
                st.markdown(f"{i}. {d}")