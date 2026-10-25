"""Plotly chart helpers built from real dataset values."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_utils import LANGUAGE_COLUMN, TARGET_COLUMN


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#a5b0a7"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,26,22,0.55)",
        font={"color": "#e8efe9", "family": "DM Sans, sans-serif"},
        margin=dict(l=40, r=20, t=50, b=40),
        height=420,
    )
    return fig


def language_counts_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or LANGUAGE_COLUMN not in df.columns:
        return empty_figure("No language data available.")

    counts = (
        df[LANGUAGE_COLUMN]
        .value_counts()
        .rename_axis("language")
        .reset_index(name="songs")
    )
    fig = px.bar(
        counts,
        x="language",
        y="songs",
        color="songs",
        color_continuous_scale=["#143d28", "#1ed760"],
    )
    fig.update_coloraxes(showscale=False)
    return _base_layout(fig, "Song counts by language")


def popularity_distribution_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or TARGET_COLUMN not in df.columns:
        return empty_figure("No popularity data available.")

    fig = px.histogram(
        df,
        x=TARGET_COLUMN,
        nbins=40,
        color_discrete_sequence=["#1ed760"],
    )
    fig.update_layout(bargap=0.05)
    return _base_layout(fig, "Popularity distribution")


def average_popularity_by_language_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("No data available.")

    grouped = (
        df.groupby(LANGUAGE_COLUMN, dropna=False)[TARGET_COLUMN]
        .agg(avg_popularity="mean", songs="count")
        .reset_index()
        .sort_values("avg_popularity", ascending=False)
    )
    grouped["label"] = grouped.apply(
        lambda r: f"{r[LANGUAGE_COLUMN]} (n={int(r['songs'])})",
        axis=1,
    )
    fig = px.bar(
        grouped,
        x="label",
        y="avg_popularity",
        color="avg_popularity",
        color_continuous_scale=["#2a4a36", "#1ed760"],
        hover_data={"songs": True, LANGUAGE_COLUMN: True},
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Language (with sample count)")
    fig.update_yaxes(title="Average popularity")
    return _base_layout(fig, "Average popularity by language")


def popularity_by_year_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or "year" not in df.columns:
        return empty_figure("No year data available.")

    yearly = (
        df.groupby("year", dropna=False)[TARGET_COLUMN]
        .agg(avg_popularity="mean", songs="count")
        .reset_index()
        .sort_values("year")
    )
    # Avoid noisy one-song years dominating the narrative — still show them
    fig = px.line(
        yearly,
        x="year",
        y="avg_popularity",
        markers=True,
        hover_data={"songs": True},
    )
    fig.update_traces(line_color="#1ed760")
    return _base_layout(fig, "Average popularity by year")


def feature_vs_popularity_chart(
    df: pd.DataFrame,
    feature: str,
    *,
    sample_size: int = 4000,
) -> go.Figure:
    if df.empty or feature not in df.columns:
        return empty_figure(f"Feature `{feature}` is unavailable.")

    sample = df if len(df) <= sample_size else df.sample(sample_size, random_state=42)
    fig = px.scatter(
        sample,
        x=feature,
        y=TARGET_COLUMN,
        color=LANGUAGE_COLUMN if LANGUAGE_COLUMN in sample.columns else None,
        opacity=0.45,
        hover_data=[c for c in ("track_name", "artist_name") if c in sample.columns],
    )
    return _base_layout(fig, f"{feature} vs popularity")


def audio_feature_distribution_chart(df: pd.DataFrame, feature: str) -> go.Figure:
    if df.empty or feature not in df.columns:
        return empty_figure(f"Feature `{feature}` is unavailable.")

    fig = px.histogram(
        df,
        x=feature,
        nbins=40,
        color_discrete_sequence=["#7CFFB2"],
    )
    return _base_layout(fig, f"Distribution of {feature}")


def metrics_comparison_chart(metrics: dict[str, Any] | None) -> go.Figure:
    """Compare baseline vs model MAE/RMSE when metrics are available."""
    if not metrics:
        return empty_figure("Train the popularity model to see evaluation metrics.")

    baseline = metrics.get("baseline_median") or {}
    model = metrics.get("random_forest_random_split") or {}
    if not baseline or not model:
        return empty_figure("Evaluation metrics are incomplete.")

    rows = [
        {"model": "Median baseline", "metric": "MAE", "value": baseline.get("mae")},
        {"model": "Median baseline", "metric": "RMSE", "value": baseline.get("rmse")},
        {"model": "Random Forest", "metric": "MAE", "value": model.get("mae")},
        {"model": "Random Forest", "metric": "RMSE", "value": model.get("rmse")},
    ]
    frame = pd.DataFrame(rows).dropna()
    if frame.empty:
        return empty_figure("No numeric metrics available.")

    fig = px.bar(
        frame,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        color_discrete_sequence=["#6b7c70", "#1ed760"],
    )
    return _base_layout(fig, "Error metrics: baseline vs Random Forest (lower is better)")
