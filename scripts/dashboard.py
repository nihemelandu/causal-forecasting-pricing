# dashboard.py - Streamlit dashboard for "Causal Forecasting for Pricing".
# Reads outputs/metrics.json and merges it onto an all-empty layout, so every panel shows a
# labelled placeholder until its slot in the file is filled.
# Run locally:  streamlit run scripts/dashboard.py
import copy
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# set_page_config must be the first Streamlit command in the script.
st.set_page_config(page_title="Causal Forecasting for Pricing", layout="wide")

METRICS_PATH = Path(__file__).resolve().parent.parent / "outputs" / "metrics.json"
LEVELS = [0, 10, 20, 30, 40, 50]
PH = "PLACEHOLDER shape only - no results yet"

EMPTY = {
    "meta": {"data_version": None, "run_date": None, "metric_name": None,
             "baseline_name": None, "coverage": None},
    "decision": None,
    "targets": {"off_policy_improvement_pct": None, "on_policy_tolerance_pct": None},
    "headline": {"off_policy_error": None, "off_policy_baseline": None,
                 "on_policy_error": None, "on_policy_baseline": None,
                 "deep_discount_error": None, "deep_discount_baseline": None},
    "models": [{"name": n, "off_policy": None, "on_policy": None}
               for n in ["M1 DML", "M2 [name]", "M3 [name]", "M4 [name]",
                         "M5 [name]", "M6 [name]", "M7 TWFE"]],
    "article_example": {"weeks": [], "actual": [], "forecast": []},
    "by_horizon": {"labels": [], "error": []},
    "by_discount": {"labels": [], "error": []},
    "by_category": {"rows": [], "cols": [], "values": []},
    "bias": {"labels": [], "value": []},
    "health": [],
    "whatif": {"articles": []},
}


def deep_merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def load():
    m = copy.deepcopy(EMPTY)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            deep_merge(m, json.load(f))
    return m


m = load()
meta, targets, h = m["meta"], m["targets"], m["headline"]


def fmt(x, spec=".1f", placeholder="[XX.X]"):
    return placeholder if x is None else format(x, spec)


def pct_change(value, baseline):
    if value is None or not baseline:
        return None
    return (value - baseline) / baseline * 100


DEF_OFF = ("Average gap between forecast and actual weekly demand when the forecast is for a price "
           "plan the model never saw in training. Lower is better.")
DEF_ON = ("The same gap under usual pricing, the kind of prices the model did see in training. "
          "Lower is better.")
DEF_DEEP = ("The gap for a 50% discount, the deepest plan we test. Deep discounts are the plans a "
            "business has least often tried. Lower is better.")
DEF_STATUS = "PASS when both success targets are met."
DEFINITIONS = (
    f"**New prices (off-policy error).** {DEF_OFF}\n\n"
    f"**Usual prices (on-policy error).** {DEF_ON}\n\n"
    f"**Deepest discount (50%).** {DEF_DEEP}\n\n"
    f"**Status vs target.** {DEF_STATUS}\n\n"
    "**Standard forecaster.** The comparison model, named under the tiles. It learns from past "
    "prices and demand without correcting for why those prices were chosen.\n\n"
    "**Typical miss.** The average error at that discount level in testing. It is not a "
    "probability range."
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
MODEL_ASSUMPTIONS = [
    "Six discount levels (0-50%) instead of the paper's 'five'; last test window is weeks 96-99.",
    "Context of 30 weeks, horizon 5; per-article demand scaling; learning-rate decay per epoch.",
    "Effect loss is L2 and the effect is unconstrained; the two inference paths are averaged.",
    "Forecasts are not clamped at zero.",
]

# ---------------- Header ----------------
head_l, head_r = st.columns([3, 2])
head_l.markdown(
    "<h1 style='margin:0; padding:0; font-size:2.2rem; font-weight:700; line-height:1.15;'>"
    "Causal Forecasting for Pricing</h1>", unsafe_allow_html=True)
head_l.caption("Can we forecast demand at prices we have never used?")
head_r.caption(f"Data: {meta['data_version'] or '[data]'}  |  Run: {meta['run_date'] or '[date]'}  |  "
               f"Error metric: {meta['metric_name'] or '[metric TBD]'}")
with head_r.popover("? Definitions"):
    st.markdown(DEFINITIONS)

tab_overview, tab_whatif, tab_tech = st.tabs(["Overview", "What-if", "Technical"])

# ---------------- Overview (one screen) ----------------
with tab_overview:
    t_off, t_on = targets["off_policy_improvement_pct"], targets["on_policy_tolerance_pct"]
    st.caption(
        "**Problem:** forecast demand at prices never used   |   "
        f"**Decision:** {m['decision'] or '[decision TBD]'}   |   "
        f"**Success:** new prices at least {fmt(t_off, '.0f', '[TBD]')}% better than standard; "
        f"usual prices no more than {fmt(t_on, '.0f', '[TBD]')}% worse")

    off_ch = pct_change(h["off_policy_error"], h["off_policy_baseline"])
    on_ch = pct_change(h["on_policy_error"], h["on_policy_baseline"])
    deep_ch = pct_change(h["deep_discount_error"], h["deep_discount_baseline"])
    bl = meta["baseline_name"] or "[baseline]"

    if None in (off_ch, on_ch, t_off, t_on):
        status = "[PASS / NOT YET]"
    else:
        status = "PASS" if (-off_ch >= t_off and on_ch <= t_on) else "NOT YET"

    if off_ch is None:
        st.info(f"[Model 1 forecasts demand at new prices XX% more accurately than the standard forecaster ({bl}).]")
    else:
        word = "more" if off_ch < 0 else "less"
        st.info(f"Model 1 forecasts demand at new prices {abs(off_ch):.0f}% {word} accurately "
                f"than the standard forecaster ({bl}).")

    def tile(col, label, value, baseline, change, help_text):
        delta = None if change is None else f"{change:+.0f}% vs standard"
        col.metric(label, fmt(value), delta=delta, delta_color="inverse", help=help_text)
        col.caption(f"standard ({bl}): {fmt(baseline)}")

    c1, c2, c3, c4 = st.columns(4)
    tile(c1, "New prices (off-policy error)", h["off_policy_error"], h["off_policy_baseline"], off_ch, DEF_OFF)
    tile(c2, "Usual prices (on-policy error)", h["on_policy_error"], h["on_policy_baseline"], on_ch, DEF_ON)
    tile(c3, "Deepest discount (50%) error", h["deep_discount_error"], h["deep_discount_baseline"], deep_ch, DEF_DEEP)
    c4.metric("Status vs target", status, help=DEF_STATUS)
    c4.caption("targets shown above")

    left, right = st.columns(2)
    with left:
        view = st.radio("Error by model (lower is better)", ["New prices", "Usual prices"], horizontal=True)
        key = "off_policy" if view == "New prices" else "on_policy"
        names = [mm["name"] for mm in m["models"]]
        values = [mm[key] for mm in m["models"]]
        if all(v is None for v in values):
            st.caption(PH)
            values = [9, 8, 7, 7, 6, 6, 5]
        else:
            values = [0 if v is None else v for v in values]
        st.bar_chart(pd.DataFrame({"error": values}, index=names), height=170)
        st.caption("Models without a result show no bar.")
    with right:
        st.markdown("**One article, one test window**")
        ex = m["article_example"]
        if ex["actual"]:
            df_ex = pd.DataFrame({"actual": ex["actual"], "forecast": ex["forecast"]}, index=ex["weeks"])
        else:
            st.caption(PH)
            df_ex = pd.DataFrame({"actual": [60, 66, 70, 68, 63], "forecast": [58, 64, 69, 69, 65]},
                                 index=[1, 2, 3, 4, 5])
        st.line_chart(df_ex, height=190)

# ---------------- What-if ----------------
with tab_whatif:
    arts = m["whatif"]["articles"]
    bd = dict(zip(m["by_discount"]["labels"], m["by_discount"]["error"]))
    ctrl, out = st.columns([1, 3])
    with ctrl:
        if arts:
            labels = [a["label"] for a in arts]
            sel = st.selectbox("Article", labels)
            a = arts[labels.index(sel)]
            st.caption("Discount per week of the test window")
            plan = []
            for i, wk in enumerate(a["weeks"]):
                cur = int(a["current_discount"][i])
                plan.append(st.select_slider(f"Week {wk}", options=LEVELS, value=cur if cur in LEVELS else 0,
                                             format_func=lambda v: f"{v}%", key=f"{sel}_{wk}"))
        else:
            st.selectbox("Article", ["[pick an article once results exist]"], disabled=True)
            st.caption("Discount per week of the test window")
            defaults = [0, 10, 20, 30, 30]
            plan = [st.select_slider(f"Week {i}", options=LEVELS, value=defaults[i - 1],
                                     format_func=lambda v: f"{v}%", key=f"wk{i}") for i in range(1, 6)]
    with out:
        if arts:
            cur_plan = [int(d) for d in a["current_discount"]]
            units = lambda p: [a["units"][str(d)][i] for i, d in enumerate(p)]
            revenue = lambda p: sum(u * a["list_price"] * (1 - d / 100) for u, d in zip(units(p), p))
            u_cur, u_you = units(cur_plan), units(plan)
            su_cur, su_you = sum(u_cur), sum(u_you)
            sr_cur, sr_you = revenue(cur_plan), revenue(plan)

            def miss(p):
                vals = [bd.get(f"{d}%") for d in p]
                return "[X.X]" if any(v is None for v in vals) else f"±{sum(vals) / len(vals):.1f} / week"

            def change(new, old):
                return f"{(new - old) / old * 100:+.0f}%" if old else ""

            st.line_chart(pd.DataFrame({"current plan": u_cur, "your plan": u_you}, index=a["weeks"]), height=220)
            st.table(pd.DataFrame({
                "Scenario": ["Current plan", "Your plan", "Difference"],
                "Avg discount": [f"{sum(cur_plan) / len(cur_plan):.0f}%", f"{sum(plan) / len(plan):.0f}%", ""],
                "Units": [f"{su_cur:,.0f}", f"{su_you:,.0f}", change(su_you, su_cur)],
                "Revenue": [f"{sr_cur:,.0f}", f"{sr_you:,.0f}", change(sr_you, sr_cur)],
                "Typical miss": [miss(cur_plan), miss(plan), ""],
            }).set_index("Scenario"))
            st.caption("Forecast by Model 1 (DML). Revenue = units x list price x (1 - discount).")
        else:
            st.caption(PH)
            st.line_chart(pd.DataFrame({"current plan": [60, 66, 70, 68, 63], "your plan": [62, 70, 75, 76, 72]},
                                       index=[1, 2, 3, 4, 5]), height=220)
            avg = sum(plan) / len(plan)
            st.table(pd.DataFrame({
                "Scenario": ["Current plan", "Your plan", "Difference"],
                "Avg discount": ["[XX%]", f"{avg:.0f}%", ""],
                "Units": ["[XXXX]", "[XXXX]", "[+XX%]"],
                "Revenue": ["[XXXX]", "[XXXX]", "[+XX%]"],
                "Typical miss": ["[X.X]", "[X.X]", ""],
            }).set_index("Scenario"))
        st.warning("A forecast is not a guarantee. It was validated on synthetic data, "
                   "where the true effect is known.")

# ---------------- Technical ----------------
def bar_panel(container, title, block, value_key, ph_labels):
    with container.container(border=True):
        st.markdown(f"**{title}**")
        if block["labels"]:
            st.bar_chart(pd.DataFrame({"v": block[value_key]}, index=block["labels"]), height=140)
        else:
            st.bar_chart(pd.DataFrame({"v": [4, 5, 6, 5, 7, 6][:len(ph_labels)]}, index=ph_labels), height=140)
            st.caption(PH)


with tab_tech:
    top = st.columns(3)
    bottom = st.columns(3)
    bar_panel(top[0], "Error by horizon week (new prices)", m["by_horizon"], "error",
              ["wk1", "wk2", "wk3", "wk4", "wk5"])
    with top[1].container(border=True):
        st.markdown("**Error by category (d x k)**")
        bc = m["by_category"]
        if bc["rows"]:
            long = pd.DataFrame([{"d": r, "k": c, "error": bc["values"][i][j]}
                                 for i, r in enumerate(bc["rows"]) for j, c in enumerate(bc["cols"])])
            chart = alt.Chart(long).mark_rect().encode(
                x=alt.X("k:N", title=None, sort=bc["cols"]),
                y=alt.Y("d:N", title=None, sort=bc["rows"]),
                color=alt.Color("error:Q", legend=None),
                tooltip=["d", "k", "error"]).properties(height=140)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("[heatmap appears when results exist]")
    bar_panel(top[2], "Error by discount level", m["by_discount"], "error",
              ["0%", "10%", "20%", "30%", "40%", "50%"])
    bar_panel(bottom[0], "Bias (forecast minus actual; positive = over-forecast)", m["bias"], "value",
              ["wk1", "wk2", "wk3", "wk4", "wk5"])
    with bottom[1].container(border=True):
        st.markdown("**Health checks**")
        if m["health"]:
            st.table(pd.DataFrame({"Check": [x["check"] for x in m["health"]],
                                   "Status": [x["status"] for x in m["health"]]}).set_index("Check"))
        else:
            st.table(pd.DataFrame({
                "Check": ["Leakage (true_* columns unused)", "Test discounts rarely seen in training",
                          "Average bias within 5% of demand"],
                "Status": ["[ OK ]", "[XX%]", "[ OK ]"]}).set_index("Check"))
    with bottom[2].container(border=True):
        st.markdown("**Run and lineage**")
        st.caption(f"Data: {meta['data_version'] or '[data]'}  |  {meta['coverage'] or '[coverage]'}")
        st.caption("Generator: scripts/generate_synthetic_data.py")
        with st.expander("Deviations from the paper (11)"):
            for i, d in enumerate(DEVIATIONS, 1):
                st.markdown(f"{i}. {d}")
        with st.expander("Model 1 assumptions (paper is silent)"):
            for i, d in enumerate(MODEL_ASSUMPTIONS, 1):
                st.markdown(f"{i}. {d}")