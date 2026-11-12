"""Reusable presentation helpers for SaansCheck's Streamlit interface."""

from __future__ import annotations

from html import escape
from typing import Iterable

import streamlit as st


def render_badge(label: str, tone: str = "neutral") -> None:
    """Render a compact status badge with one of the shared visual tones."""
    safe_label = escape(str(label))
    st.markdown(
        f'<span class="sc-badge sc-badge-{escape(tone)}">{safe_label}</span>',
        unsafe_allow_html=True,
    )


def render_breadcrumb(items: Iterable[str]) -> None:
    """Render a concise breadcrumb trail above a page heading."""
    trail = ' <span aria-hidden="true">/</span> '.join(escape(str(item)) for item in items)
    st.markdown(f'<div class="sc-breadcrumb">{trail}</div>', unsafe_allow_html=True)


def render_page_header(
    breadcrumbs: Iterable[str],
    title: str,
    subtitle: str,
    badges: Iterable[tuple[str, str]] = (),
) -> None:
    """Render the standard breadcrumb, title, subtitle, and optional badges."""
    render_breadcrumb(breadcrumbs)
    st.markdown(f'<h1 class="sc-page-title">{escape(title)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sc-page-subtitle">{escape(subtitle)}</p>', unsafe_allow_html=True)
    badges = list(badges)
    if badges:
        badge_markup = "".join(
            f'<span class="sc-badge sc-badge-{escape(str(tone))}">{escape(str(label))}</span>'
            for label, tone in badges
        )
        st.markdown(f'<div class="sc-badge-row">{badge_markup}</div>', unsafe_allow_html=True)


def render_kpi_cards(metrics: Iterable[tuple[str, str, str]]) -> None:
    """Render actual calculated KPI values in a responsive Streamlit row."""
    metrics = list(metrics)
    if not metrics:
        return
    columns = st.columns(len(metrics))
    for column, (label, value, detail) in zip(columns, metrics):
        with column:
            st.markdown(
                '<div class="sc-kpi">'
                f'<div class="sc-kpi-label">{escape(str(label))}</div>'
                f'<div class="sc-kpi-value">{escape(str(value))}</div>'
                f'<div class="sc-kpi-detail">{escape(str(detail))}</div>'
                '</div>',
                unsafe_allow_html=True,
            )


def render_empty_state(
    title: str = "Recent data unavailable",
    detail: str = (
        "SaansCheck could not verify sufficiently recent compatible observations "
        "from the connected source. Historical ML forecasting remains available."
    ),
) -> None:
    """Render the shared, factual empty state without substituting sample values."""
    st.markdown(
        '<div class="sc-empty-state">'
        '<div class="sc-empty-mark" aria-hidden="true">◌</div>'
        f'<div class="sc-empty-title">{escape(title)}</div>'
        f'<div class="sc-empty-detail">{escape(detail)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )