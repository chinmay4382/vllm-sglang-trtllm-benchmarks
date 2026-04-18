import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from dashboard.data_store import DataStore

st.set_page_config(page_title="LLM Benchmark Dashboard", page_icon="⚡", layout="wide")


@st.cache_data(show_spinner=False)
def load_perf():
    store = DataStore(str(Path(__file__).parent.parent / "data" / "benchmark_results.parquet"))
    return store.load_performance()


@st.cache_data(show_spinner=False)
def load_eval():
    store = DataStore(str(Path(__file__).parent.parent / "data" / "benchmark_results.parquet"))
    return store.load_evaluations()


st.title("LLM Inference Benchmark Dashboard")

if st.button("Refresh data", key="refresh"):
    st.cache_data.clear()
    st.rerun()

df = load_perf()
df_eval = load_eval()

if df.empty:
    st.warning("No benchmark data found. Run `python main.py` to generate results.")
    st.stop()

# ── Sidebar filters ──────────────────────────────────────────────────────────
models = sorted(df["model_name"].unique().tolist())
datasets = sorted(df["dataset_name"].unique().tolist())

sel_models = st.sidebar.multiselect("Models", models, default=models, key="sel_models")
sel_datasets = st.sidebar.multiselect("Datasets", datasets, default=datasets, key="sel_datasets")

mask = pd.Series(True, index=df.index)
if sel_models:
    mask &= df["model_name"].isin(sel_models)
if sel_datasets:
    mask &= df["dataset_name"].isin(sel_datasets)
dff = df[mask].copy()

if dff.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────
st.subheader("Overview (all concurrency levels, mean)")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("RPS",          f"{dff['rps'].mean():.1f}")
c2.metric("TPS",          f"{dff['tps'].mean():.0f}")
c3.metric("TTFT avg",     f"{dff['avg_ttft'].mean()*1000:.0f} ms")
c4.metric("TTFT p95",     f"{dff['p95_ttft'].mean()*1000:.0f} ms")
c5.metric("ITL avg",      f"{dff['avg_itl'].mean()*1000:.1f} ms")
c6.metric("Latency p95",  f"{dff['p95_latency'].mean()*1000:.0f} ms")

st.markdown("---")

# ── Performance charts ────────────────────────────────────────────────────────
st.subheader("Performance vs Concurrency")

col_l, col_r = st.columns(2)

with col_l:
    fig = px.line(
        dff.sort_values("concurrency"), x="concurrency", y="tps",
        color="model_name", markers=True,
        title="Throughput (TPS) vs Concurrency",
        labels={"tps": "Tokens/sec", "concurrency": "Concurrency", "model_name": "Model"},
    )
    fig.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig, key="tps_chart")

with col_r:
    fig2 = px.line(
        dff.sort_values("concurrency"), x="concurrency", y="avg_ttft",
        color="model_name", markers=True,
        title="Time to First Token (TTFT) vs Concurrency",
        labels={"avg_ttft": "TTFT (s)", "concurrency": "Concurrency", "model_name": "Model"},
    )
    fig2.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig2, key="ttft_chart")

# ── Tradeoff scatter ──────────────────────────────────────────────────────────
st.subheader("Throughput / Latency Tradeoff")
fig3 = px.scatter(
    dff, x="avg_ttft", y="tps", color="model_name", size="concurrency",
    hover_data=["concurrency", "rps"],
    title="TPS vs TTFT (bubble = concurrency)",
    labels={"avg_ttft": "Avg TTFT (s)", "tps": "Tokens/sec", "model_name": "Model"},
)
fig3.update_layout(template="plotly_dark", height=420)
st.plotly_chart(fig3, key="scatter_chart")

st.markdown("---")

# ── Quality metrics ───────────────────────────────────────────────────────────
st.subheader("Quality Metrics")
if not df_eval.empty:
    acc_df = df_eval.dropna(subset=["accuracy"])
    if not acc_df.empty:
        fig4 = px.bar(
            acc_df, x="dataset_name", y="accuracy", color="model_name", barmode="group",
            title="Accuracy by Dataset", range_y=[0, 1],
        )
        fig4.update_layout(template="plotly_dark", height=360)
        fig4.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig4, key="acc_chart")
else:
    st.info("No evaluation data. Run `python main.py` (without --no-eval) to collect quality metrics.")

st.markdown("---")

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("Raw Benchmark Data"):
    st.dataframe(dff, key="raw_table")
    st.download_button(
        "Download CSV", dff.to_csv(index=False).encode(),
        file_name="benchmark_results.csv", mime="text/csv", key="dl_csv",
    )
