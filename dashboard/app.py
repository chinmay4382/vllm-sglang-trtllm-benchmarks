"""
Streamlit dashboard for LLM benchmarking results.

Sections:
  1. Overview KPIs
  2. Performance graphs (TPS/TTFT vs concurrency, RPS bar chart)
  3. Quality metrics (accuracy by dataset)
  4. Tradeoff scatter (TPS vs TTFT)
  5. Interactive controls + CSV export
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root when running: streamlit run dashboard/app.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.data_store import DataStore

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
  .metric-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-label { color: #cdd6f4; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-value { color: #89b4fa; font-size: 2rem; font-weight: 700; margin: 0.25rem 0; }
  .metric-unit  { color: #6c7086; font-size: 0.75rem; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def get_store() -> DataStore:
    store_path = Path(__file__).parent.parent / "data" / "benchmark_results.parquet"
    return DataStore(str(store_path))


store = get_store()


def render_metric_card(label: str, value: str, unit: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-unit">{unit}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("⚡ LLM Benchmarks")
st.sidebar.markdown("---")

df_perf = store.load_performance()
df_eval = store.load_evaluations()

available_models = sorted(df_perf["model_name"].unique().tolist()) if not df_perf.empty else []
selected_models = st.sidebar.multiselect(
    "Models",
    options=available_models,
    default=available_models,
)

available_datasets = sorted(df_perf["dataset_name"].unique().tolist()) if not df_perf.empty else []
selected_datasets = st.sidebar.multiselect(
    "Benchmark Datasets",
    options=available_datasets,
    default=available_datasets,
)

concurrency_levels = sorted(df_perf["concurrency"].unique().tolist()) if not df_perf.empty else [1]
min_c, max_c = int(min(concurrency_levels)), int(max(concurrency_levels))
if min_c == max_c:
    selected_concurrency = min_c
    st.sidebar.info(f"Single concurrency level: {min_c}")
else:
    selected_concurrency = st.sidebar.slider(
        "Highlight concurrency level",
        min_value=min_c,
        max_value=max_c,
        value=min_c,
        step=1,
    )

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_resource.clear()
    st.rerun()

# Apply filters
mask = pd.Series(True, index=df_perf.index)
if selected_models:
    mask &= df_perf["model_name"].isin(selected_models)
if selected_datasets:
    mask &= df_perf["dataset_name"].isin(selected_datasets)
df_filtered = df_perf[mask].copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("LLM Inference Benchmark Dashboard")
st.caption("Real-time performance and quality metrics powered by vLLM + GuideLLM")

if df_filtered.empty:
    st.warning(
        "No benchmark data found. Run `python main.py` to generate benchmark results, "
        "then refresh this page."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Section 1 — Overview KPIs
# ---------------------------------------------------------------------------

st.subheader("Overview")

snapshot = df_filtered[df_filtered["concurrency"] == selected_concurrency]
if snapshot.empty:
    snapshot = df_filtered

agg = snapshot.agg(
    {
        "rps": "mean",
        "tps": "mean",
        "avg_ttft": "mean",
        "avg_itl": "mean",
        "p95_ttft": "mean",
        "p95_latency": "mean",
    }
)

cols = st.columns(6)
kpis = [
    ("RPS", f"{agg['rps']:.1f}", "req / sec"),
    ("TPS", f"{agg['tps']:.0f}", "tokens / sec"),
    ("TTFT avg", f"{agg['avg_ttft'] * 1000:.0f}", "ms"),
    ("TTFT p95", f"{agg['p95_ttft'] * 1000:.0f}", "ms"),
    ("ITL avg", f"{agg['avg_itl'] * 1000:.1f}", "ms / token"),
    ("Latency p95", f"{agg['p95_latency'] * 1000:.0f}", "ms"),
]
for col, (label, value, unit) in zip(cols, kpis):
    col.markdown(render_metric_card(label, value, unit), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2 — Performance Graphs
# ---------------------------------------------------------------------------

st.subheader("Performance vs Concurrency")

tab_tps, tab_ttft, tab_rps = st.tabs(["TPS vs Concurrency", "TTFT vs Concurrency", "RPS per Model"])

with tab_tps:
    fig = px.line(
        df_filtered.sort_values("concurrency"),
        x="concurrency",
        y="tps",
        color="model_name",
        markers=True,
        labels={"tps": "Tokens per Second", "concurrency": "Concurrent Users", "model_name": "Model"},
        title="Throughput (TPS) vs Concurrency",
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab_ttft:
    fig_ttft = go.Figure()
    for model in df_filtered["model_name"].unique():
        mdf = df_filtered[df_filtered["model_name"] == model].sort_values("concurrency")
        fig_ttft.add_trace(
            go.Scatter(
                x=mdf["concurrency"],
                y=mdf["avg_ttft"] * 1000,
                mode="lines+markers",
                name=f"{model} avg",
                line=dict(dash="solid"),
            )
        )
        fig_ttft.add_trace(
            go.Scatter(
                x=mdf["concurrency"],
                y=mdf["p95_ttft"] * 1000,
                mode="lines+markers",
                name=f"{model} p95",
                line=dict(dash="dash"),
            )
        )
    fig_ttft.update_layout(
        template="plotly_dark",
        height=400,
        title="Time to First Token (TTFT) vs Concurrency",
        xaxis_title="Concurrent Users",
        yaxis_title="TTFT (ms)",
    )
    st.plotly_chart(fig_ttft, use_container_width=True)

with tab_rps:
    rps_df = (
        df_filtered.groupby("model_name")["rps"]
        .mean()
        .reset_index()
        .sort_values("rps", ascending=False)
    )
    fig_rps = px.bar(
        rps_df,
        x="model_name",
        y="rps",
        color="model_name",
        labels={"rps": "Requests per Second", "model_name": "Model"},
        title="Average RPS per Model",
    )
    fig_rps.update_layout(template="plotly_dark", height=400, showlegend=False)
    st.plotly_chart(fig_rps, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3 — Quality Metrics
# ---------------------------------------------------------------------------

st.subheader("Quality Metrics")

if not df_eval.empty:
    eval_filtered = df_eval[df_eval["model_name"].isin(selected_models)] if selected_models else df_eval

    col_acc, col_em = st.columns(2)

    with col_acc:
        acc_df = eval_filtered.dropna(subset=["accuracy"])
        if not acc_df.empty:
            fig_acc = px.bar(
                acc_df,
                x="dataset_name",
                y="accuracy",
                color="model_name",
                barmode="group",
                labels={"accuracy": "Accuracy", "dataset_name": "Dataset", "model_name": "Model"},
                title="Accuracy by Dataset",
                range_y=[0, 1],
            )
            fig_acc.update_layout(template="plotly_dark", height=380)
            fig_acc.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("No accuracy data available.")

    with col_em:
        # pass@k for HumanEval
        he_df = eval_filtered[eval_filtered["dataset_name"] == "HumanEval"].dropna(subset=["pass_at_k"])
        if not he_df.empty:
            fig_he = px.bar(
                he_df,
                x="model_name",
                y="pass_at_k",
                color="model_name",
                labels={"pass_at_k": "pass@1", "model_name": "Model"},
                title="HumanEval pass@1 by Model",
                range_y=[0, 1],
            )
            fig_he.update_layout(template="plotly_dark", height=380, showlegend=False)
            fig_he.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_he, use_container_width=True)
        else:
            em_df = eval_filtered.dropna(subset=["exact_match"])
            if not em_df.empty:
                fig_em = px.bar(
                    em_df,
                    x="dataset_name",
                    y="exact_match",
                    color="model_name",
                    barmode="group",
                    labels={"exact_match": "Exact Match", "dataset_name": "Dataset"},
                    title="Exact Match by Dataset",
                    range_y=[0, 1],
                )
                fig_em.update_layout(template="plotly_dark", height=380)
                fig_em.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_em, use_container_width=True)
            else:
                st.info("No exact-match data available.")
else:
    st.info("No evaluation data. Run `python main.py --eval` to collect quality metrics.")

# ---------------------------------------------------------------------------
# Section 4 — Tradeoff Scatter (TPS vs TTFT)
# ---------------------------------------------------------------------------

st.subheader("Throughput / Latency Tradeoff")

tradeoff_df = df_filtered.dropna(subset=["tps", "avg_ttft"])
if not tradeoff_df.empty:
    fig_scatter = px.scatter(
        tradeoff_df,
        x="avg_ttft",
        y="tps",
        color="model_name",
        size="concurrency",
        hover_data=["concurrency", "rps", "avg_itl"],
        labels={
            "avg_ttft": "Avg TTFT (s)",
            "tps": "Tokens per Second",
            "model_name": "Model",
            "concurrency": "Concurrency",
        },
        title="TPS vs TTFT — bubble size = concurrency level",
    )
    fig_scatter.update_layout(template="plotly_dark", height=440)
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough data for tradeoff scatter.")

# ---------------------------------------------------------------------------
# Section 5 — Raw Data Table + CSV Export
# ---------------------------------------------------------------------------

with st.expander("Raw Benchmark Data"):
    st.dataframe(df_filtered, use_container_width=True)

    csv_bytes = df_filtered.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Download CSV",
        data=csv_bytes,
        file_name="benchmark_results.csv",
        mime="text/csv",
    )

if not df_eval.empty:
    with st.expander("Raw Evaluation Data"):
        st.dataframe(df_eval, use_container_width=True)
        eval_csv = df_eval.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download Evaluation CSV",
            data=eval_csv,
            file_name="evaluation_results.csv",
            mime="text/csv",
        )
