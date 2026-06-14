"""Interactive credit-default risk dashboard.

Scores a loan applicant's probability of serious delinquency and shows where they
fall in the risk ranking, plus a review-capacity view on the versioned sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(config.SAMPLE_PATH) if config.SAMPLE_PATH.exists() else pd.DataFrame()


@st.cache_data
def sample_scores() -> np.ndarray:
    df = load_sample()
    return load_predictor().score(df.sample(min(20000, len(df)), random_state=42)) if not df.empty else np.array([])


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.yaxis.label.set_color(D["foreground"])
    ax.grid(True, color=D["current_line"], linestyle="--", alpha=0.4)


def gains_chart(scores, y, capacity_pct):
    order = np.argsort(scores)[::-1]
    ys = y[order]
    cum = np.cumsum(ys) / ys.sum()
    pct = np.arange(1, len(ys) + 1) / len(ys)
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=D["background"])
    ax.plot(pct * 100, cum * 100, color=D["green"], linewidth=2, label="Model")
    ax.plot([0, 100], [0, 100], color=D["comment"], linestyle="--", linewidth=1.5, label="Random")
    ax.axvline(capacity_pct, color=D["pink"], linestyle=":", linewidth=1.5)
    ax.set_xlabel("Applicants reviewed (%)")
    ax.set_ylabel("Defaulters captured (%)")
    ax.legend(facecolor=D["current_line"], edgecolor=D["comment"], labelcolor=D["foreground"], fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    st.title("MLflow Credit Risk — Default Scoring")
    st.markdown(
        "Scores a loan applicant's probability of serious delinquency in the next two years to "
        "support approve / review decisions and to prioritize a limited underwriting team."
    )

    with st.sidebar:
        st.header("Applicant")
        age = st.slider("Age", 18, 95, 45)
        income = st.number_input("Monthly income", 0.0, 50000.0, 5000.0, 100.0)
        utilization = st.slider("Revolving utilization", 0.0, 1.5, 0.4, 0.05)
        debt_ratio = st.slider("Debt ratio", 0.0, 2.0, 0.3, 0.05)
        open_lines = st.slider("Open credit lines", 0, 40, 8)
        real_estate = st.slider("Real-estate loans", 0, 10, 1)
        dependents = st.slider("Dependents", 0, 10, 2)
        pd30 = st.slider("Times 30-59 days late", 0, 10, 0)
        pd60 = st.slider("Times 60-89 days late", 0, 10, 0)
        pd90 = st.slider("Times 90+ days late", 0, 10, 0)
        run = st.button("Score applicant", type="primary")

    applicant = {"RevolvingUtilization": utilization, "Age": age, "DebtRatio": debt_ratio,
                 "MonthlyIncome": income, "OpenCreditLines": open_lines, "RealEstateLoans": real_estate,
                 "Dependents": dependents, "PastDue30_59": pd30, "PastDue60_89": pd60, "PastDue90": pd90}

    if run:
        score = predictor.score_one(applicant)
        pct = predictor.rank_percentile(score, sample_scores())
        st.subheader("Default risk")
        c = st.columns(3)
        c[0].metric("Default probability", f"{score*100:.1f}%")
        c[1].metric("Decision", "Decline / review" if score >= config.DEFAULT_THRESHOLD else "Approve")
        c[2].metric("Portfolio default rate", f"{predictor.base_rate*100:.1f}%")
        color = D["red"] if score >= config.DEFAULT_THRESHOLD else D["green"]
        st.markdown(
            f"<span style='color:{color}'>{predictor.decision(score).capitalize()}.</span> "
            f"Riskier than {pct:.0f}% of the portfolio.", unsafe_allow_html=True)
        st.subheader("Most influential features (model-wide)")
        imp = pd.DataFrame(predictor.top_features(6)).rename(
            columns={"feature": "Feature", "importance": "Permutation importance (PR-AUC drop)"})
        st.dataframe(imp, hide_index=True, width="stretch")

    df = load_sample()
    if not df.empty and config.TARGET in df.columns:
        st.subheader("Underwriting capacity view (reference sample)")
        capacity = st.slider("Review capacity (% of applicants)", 5, 100, 20, 5)
        sub = df.sample(min(20000, len(df)), random_state=42)
        scores = sample_scores()
        y = sub[config.TARGET].to_numpy()
        order = np.argsort(scores)[::-1]
        k = int(len(y) * capacity / 100)
        captured = y[order][:k].sum() / y.sum() if y.sum() else 0
        lift = (y[order][:k].mean() / y.mean()) if (k and y.mean()) else 0
        left, right = st.columns([2, 1])
        with left:
            st.pyplot(gains_chart(scores, y, capacity))
        with right:
            st.metric("Defaulters captured", f"{captured*100:.0f}%")
            st.metric("Lift vs random", f"{lift:.2f}x")


if __name__ == "__main__":
    main()
