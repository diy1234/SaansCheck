"""
===============================================================================
SaansCheck: A Personalized Air Quality Risk Advisor and AQI Forecasting System
Main Streamlit Application Entry Point
Phase 5A: Backend Data Integration & Health Layer
===============================================================================

This application is built with Streamlit. In Phase 5A, it connects directly
to the processed datasets and serialized machine learning models created in
earlier phases, validating all artifacts without using any placeholder/fake numbers.
===============================================================================
"""

import base64
import calendar
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from src.ui_components import (
    render_badge,
    render_empty_state,
    render_kpi_cards,
    render_page_header,
)
from src.waqi_data import (
    AQI_SCALE_LABEL,
    AQI_SCALE_URL,
    POLLUTANT_FIELDS,
    SOURCE_LABEL,
    load_city_stations,
    load_nearby_station_observation,
    load_station_observation,
    merge_station_search_results,
    refresh_observation_freshness,
)
from src.browser_location import request_browser_location
from src.health_guidance import build_health_guidance
from src.cpcb_national_aqi import (
    build_cpcb_health_guidance,
    build_cpcb_observed_alert,
    load_cpcb_national_aqi,
)
from src.aqi_risk_scale import (
    INDIAN_CPCB_SCALE,
    US_AQI_SCALE,
    aqi_category,
    current_risk_observation,
)
from src.aqi_alerts import (
    HIGH_AQI_THRESHOLD,
    build_observed_aqi_alert,
    cloud_alert_component_html,
    dismiss_alert_session_state,
    update_alert_session_state,
)
from src.live_forecast import build_live_forecast_status
from src.forecast_features import build_direct_forecasts
from src.dashboard_insights import (
    available_history_years,
    build_key_findings,
    build_location_leaderboard,
    build_month_calendar,
    summarize_walk_forward_results,
)
from src.category_evaluation import evaluate_regression_categories
from src.theme import apply_chart_theme, chart_palette, theme_css

# Import our reusable, cached data loading functions from src/data_loader.py
from src.data_loader import (
    load_daily_air_quality,
    load_ml_ready_dataset,
    load_model_results,
    load_test_predictions,
    load_feature_importance,
    load_trained_models,
    validate_all_system_artifacts
)

STATIONS = ["Delhi", "Faridabad", "Ghaziabad", "Gurgaon", "Noida"]
CITY_SEARCH_OPTIONS = [
    "Delhi / New Delhi",
    "Noida",
    "Ghaziabad",
    "Gurugram / Gurgaon",
    "Faridabad",
]
CITY_SEARCH_STATIONS = {
    "Delhi / New Delhi": "Delhi",
    "Noida": "Noida",
    "Ghaziabad": "Ghaziabad",
    "Gurugram / Gurgaon": "Gurgaon",
    "Faridabad": "Faridabad",
}
AQI_CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]
PERSISTENCE_BASELINE = {
    "Day+1": {"MAE": 9.2799, "RMSE": 12.1533, "R2": 0.9570},
    "Day+2": {"MAE": 9.2945, "RMSE": 12.2927, "R2": 0.9561},
    "Day+3": {"MAE": 9.5604, "RMSE": 12.5760, "R2": 0.9542},
}
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
HISTORICAL_MODE = "Historical Intelligence"
RECENT_MODE = "Recent Air Quality"
PRIMARY_NAV_ITEMS = ["Overview", "Intelligence", "Insights", "About"]
HISTORICAL_PAGES = [
    "Overview", "AQI Forecast", "Location Intelligence", "Pollution Analytics",
    "Daily AQI Calendar", "Seasonal Risk", "Health Advisor", "Model Insights",
    "Learn About AQI",
]
RECENT_PAGES = [
    "Recent Overview", "Pollutant Trends", "Location Data", "Data Freshness",
    "Forecast",
]
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = next(
    (
        path
        for path in (
            APP_DIR / "assets" / "saanscheck_logo.png",
            APP_DIR / "assets" / "Saanscheck Delhi Air Quality Emblem.png",
        )
        if path.is_file()
    ),
    None,
)


def _logo_markup(class_name, alt_text="SaansCheck logo"):
    """Return a responsive local logo image, or a text fallback if unavailable."""
    if LOGO_PATH is None:
        return f'<span class="{class_name} sc-logo-fallback">SaansCheck</span>'
    encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return (
        f'<img class="{class_name}" src="data:image/png;base64,{encoded_logo}" '
        f'alt="{alt_text}">'
    )


def _page_for_primary_nav(primary_nav, mode):
    """Resolve a top-level navigation choice to a mode-specific page."""
    page_map = {
        "Overview": {HISTORICAL_MODE: "Overview", RECENT_MODE: "Recent Overview"},
        "Intelligence": {HISTORICAL_MODE: "AQI Forecast", RECENT_MODE: "Pollutant Trends"},
        "Insights": {HISTORICAL_MODE: "Seasonal Risk", RECENT_MODE: "Data Freshness"},
        "About": {HISTORICAL_MODE: "Learn About AQI", RECENT_MODE: "Learn About AQI"},
    }
    return page_map[primary_nav][mode]


def _primary_nav_for_page(page):
    """Keep the primary navigation aligned with its selected mode page."""
    if page in {"Overview", "Recent Overview"}:
        return "Overview"
    if page in {"AQI Forecast", "Location Intelligence", "Pollution Analytics", "Daily AQI Calendar", "Pollutant Trends", "Location Data", "Forecast"}:
        return "Intelligence"
    if page in {"Seasonal Risk", "Health Advisor", "Model Insights", "Data Freshness"}:
        return "Insights"
    if page == "Learn About AQI":
        return "About"
    return "Overview"


def _on_primary_nav_change():
    st.session_state["active_page"] = _page_for_primary_nav(
        st.session_state["primary_nav"], st.session_state["workspace_mode"]
    )


def _on_mode_change():
    st.session_state["active_page"] = _page_for_primary_nav(
        "Overview", st.session_state["workspace_mode"]
    )
    st.session_state["primary_nav"] = "Overview"


def _on_switch_workspace():
    """Return to the selected workspace overview without changing its mode."""
    st.session_state["primary_nav"] = "Overview"
    st.session_state["active_page"] = _page_for_primary_nav(
        "Overview", st.session_state["workspace_mode"]
    )


def _on_page_change():
    st.session_state["primary_nav"] = _primary_nav_for_page(
        st.session_state["active_page"]
    )


def _on_location_bar_change():
    st.session_state["historical_station"] = st.session_state["location_bar_station"]
    st.session_state["workspace_mode"] = RECENT_MODE
    st.session_state["primary_nav"] = "Overview"
    st.session_state["active_page"] = "Recent Overview"
    st.session_state["recent_location_mode"] = "Manual city/station"
    st.session_state["recent_station_query"] = ""
    st.session_state["recent_station_query_applied"] = ""
    st.session_state["recent_station_uid"] = None


def _on_city_search_change():
    selected_city = st.session_state.get("navbar_city_search")
    if selected_city not in CITY_SEARCH_STATIONS:
        return
    st.session_state["location_bar_station"] = CITY_SEARCH_STATIONS[selected_city]
    st.session_state["historical_station"] = CITY_SEARCH_STATIONS[selected_city]
    _on_location_bar_change()
    st.session_state["navbar_city_search"] = None


def _on_home_navigation():
    """Return to the recent overview without changing the selected city."""
    st.session_state["workspace_mode"] = RECENT_MODE
    st.session_state["primary_nav"] = "Overview"
    st.session_state["active_page"] = "Recent Overview"
    st.session_state["recent_location_mode"] = "Manual city/station"


def _on_historical_station_change():
    st.session_state["location_bar_station"] = st.session_state["historical_station"]


def _on_locality_search():
    st.session_state["recent_station_query_applied"] = st.session_state.get(
        "recent_station_query", ""
    ).strip()
    st.session_state["recent_station_uid"] = None
    st.session_state["recent_location_mode"] = "Manual city/station"


def _on_manual_station_change():
    st.session_state["recent_location_mode"] = "Manual city/station"


def _retry_browser_location():
    st.session_state["browser_location_status"] = "not_requested"
    st.session_state["browser_location_request_id"] = (
        int(st.session_state.get("browser_location_request_id", 0)) + 1
    )
    st.session_state.pop("nearby_station_observation", None)
    st.session_state.pop("browser_location_lookup_done", None)


def _on_risk_scale_change():
    st.session_state["risk_scale_user_selected"] = True


def _render_historical_category_evaluation(predictions_df):
    st.markdown("---")
    st.subheader("Historical AQI Category Classification Evaluation")
    st.caption(
        "Derived by categorizing the held-out regression predictions; these are "
        "not metrics from separately trained classifiers. Evaluation uses the "
        "combined Delhi-NCR test predictions, not only the selected station."
    )
    evaluation_columns = st.columns(2)
    with evaluation_columns[0]:
        evaluation_model = st.selectbox(
            "Regression model",
            ["Linear Regression", "Random Forest"],
            key="category_evaluation_model",
        )
    with evaluation_columns[1]:
        evaluation_horizon = st.selectbox(
            "Forecast horizon",
            ["Day+1", "Day+2", "Day+3"],
            key="category_evaluation_horizon",
        )

    try:
        category_evaluation = evaluate_regression_categories(
            predictions_df,
            evaluation_model,
            evaluation_horizon,
        )
    except ValueError as error:
        st.info(f"Category evaluation unavailable: {error}")
        return

    metric_fields = [
        ("Macro Precision", category_evaluation["report"]["macro avg"]["precision"]),
        ("Macro Recall", category_evaluation["report"]["macro avg"]["recall"]),
        ("Macro F1-score", category_evaluation["report"]["macro avg"]["f1-score"]),
        ("Weighted F1-score", category_evaluation["report"]["weighted avg"]["f1-score"]),
    ]
    metric_columns = st.columns(4)
    for metric_column, (label, value) in zip(metric_columns, metric_fields):
        metric_column.metric(label, f"{value:.3f}")

    st.caption(
        f"Forecast origins: {category_evaluation['origin_start']:%Y-%m-%d} to "
        f"{category_evaluation['origin_end']:%Y-%m-%d} · AQI target dates: "
        f"{category_evaluation['target_start']:%Y-%m-%d} to "
        f"{category_evaluation['target_end']:%Y-%m-%d} · "
        f"{category_evaluation['sample_count']:,} scored samples · "
        f"{category_evaluation['excluded_count']:,} rows excluded for missing "
        "or non-finite actual/predicted values."
    )
    st.markdown(
        "Precision measures how often predictions of a category are correct; "
        "recall measures how many actual cases of that category are found; "
        "F1 is their harmonic mean. Macro scores weight each of the six "
        "categories equally, while weighted F1 weights categories by actual "
        "support. Unsupported categories are retained in the six-label macro "
        "average with `zero_division=0`."
    )
    st.warning(
        "Scale caveat: this uses the exact category thresholds implemented by "
        "historical preprocessing (Good ≤50, Satisfactory ≤100, Moderate ≤200, "
        "Poor ≤300, Very Poor ≤400, Severe >400). The historical AQI source, "
        "calculation method, and scale provenance are unverified; these scores "
        "must not be described as verified CPCB performance. The stored split "
        "also has a boundary overlap: final 2022 training origins have targets "
        "in 2023 (up to three days), so this is not a fully isolated temporal "
        "holdout."
    )

    matrix = category_evaluation["confusion_matrix"]
    categories = category_evaluation["categories"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(categories)), labels=categories, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(categories)), labels=categories)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title(f"{evaluation_model} · {evaluation_horizon} confusion matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "#17232d",
                fontsize=9,
            )
    theme_name = st.session_state["appearance_theme"]
    apply_chart_theme(fig, theme_name)
    for text in ax.texts:
        value = matrix[int(round(text.get_position()[1])), int(round(text.get_position()[0]))]
        text.set_color("white" if value > threshold else "#17232d")
    colorbar.ax.tick_params(
        colors="#586a78" if theme_name == "Light" else "#a8b7ca"
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("#### Per-class classification report and support")
    st.dataframe(
        category_evaluation["per_class"].reset_index(),
        width="stretch",
        hide_index=True,
        column_config={
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1-score": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "Actual and predicted support counts are shown for every category. "
        "The original MAE, RMSE, and R² regression metrics above remain unchanged."
    )


def _render_health_guidance(observation):
    guidance = build_health_guidance(observation)
    st.markdown("### WAQI — US AQI health guidance")
    if guidance is None:
        st.info(
            "Current health guidance is unavailable because a valid, fresh, "
            "timestamped AQI observation is not available for this station."
        )
        return

    with st.container(border=True):
        st.markdown(f"**{guidance['category']}** · {guidance['scale']}")
        st.write(guidance["effects"])
        st.markdown("**Precautions**")
        st.write(guidance["precautions"])
        st.caption(
            "General public-health information, not medical advice. "
            f"Category thresholds follow the "
            f"[WAQI-published US EPA 2016 scale]({guidance['scale_url']})."
        )


def _render_cpcb_health_guidance(observation):
    guidance = build_cpcb_health_guidance(observation)
    if guidance is None:
        return
    st.markdown("### CPCB National AQI health guidance")
    with st.container(border=True):
        st.markdown(f"**{guidance['category']}** · {guidance['scale']}")
        st.write(guidance["effects"])
        st.markdown("**Precautions**")
        st.write(guidance["precautions"])
        st.caption(
            "General public-health information, not medical advice. "
            f"See the [CPCB National AQI portal]({guidance['scale_url']})."
        )


def _render_aqi_risk_scale(cpcb_observation, waqi_observation):
    """Render a scale-specific educational legend and only a verified fresh marker."""
    current = current_risk_observation(cpcb_observation, waqi_observation)
    options = [INDIAN_CPCB_SCALE, US_AQI_SCALE]
    default_scale = current["scale"] if current else INDIAN_CPCB_SCALE
    if (
        not st.session_state.get("risk_scale_user_selected")
        or st.session_state.get("recent_aqi_risk_scale") not in options
    ):
        st.session_state["recent_aqi_risk_scale"] = default_scale
    selected_scale = st.radio(
        "AQI scale",
        options,
        horizontal=True,
        key="recent_aqi_risk_scale",
        on_change=_on_risk_scale_change,
    )

    cpcb_levels = [
        ("Good", "0–50", "🌿", "#27AE60"),
        ("Satisfactory", "51–100", "≋", "#75B84A"),
        ("Moderate", "101–200", "⚠", "#D5B82E"),
        ("Poor", "201–300", "‼", "#E98632"),
        ("Very Poor", "301–400", "⛔", "#CF4548"),
        ("Severe", "401–500", "▲", "#77293A"),
    ]
    us_levels = [
        ("Good", "0–50", "🌿", "#27AE60"),
        ("Moderate", "51–100", "≋", "#D5B82E"),
        ("Unhealthy for Sensitive Groups", "101–150", "⚠", "#E98632"),
        ("Unhealthy", "151–200", "‼", "#CF4548"),
        ("Very Unhealthy", "201–300", "⛔", "#8B4BA8"),
        ("Hazardous", "301–500", "▲", "#77293A"),
    ]
    cpcb_advice = {
        "Good": "Air quality is generally satisfactory.",
        "Satisfactory": "Air quality is acceptable; sensitive people may experience minor discomfort.",
        "Moderate": "People with lung or heart disease, children, and older adults may experience breathing discomfort.",
        "Poor": "Consider reducing prolonged outdoor exertion.",
        "Very Poor": "Prolonged exposure may pose significant respiratory risks; reduce outdoor exertion.",
        "Severe": "Avoid unnecessary outdoor exertion and follow authoritative health guidance.",
    }
    us_advice = {
        "Good": "Air quality is generally satisfactory.",
        "Moderate": "Unusually sensitive people may experience minor effects.",
        "Unhealthy for Sensitive Groups": "Sensitive individuals should reduce prolonged or heavy outdoor exertion.",
        "Unhealthy": "Consider reducing prolonged outdoor exertion; sensitive groups should take extra care.",
        "Very Unhealthy": "Air pollution may pose significant health risks; limit prolonged outdoor activity.",
        "Hazardous": "Avoid unnecessary outdoor exertion and follow authoritative health guidance.",
    }
    levels = cpcb_levels if selected_scale == INDIAN_CPCB_SCALE else us_levels
    active_scale = current is not None and current["scale"] == selected_scale
    current_category = current["category"] if active_scale else None
    beyond_range = bool(active_scale and current.get("beyond_range"))
    gradient = (
        "linear-gradient(90deg, #27AE60 0%, #75B84A 10%, #D5B82E 20%, "
        "#E98632 40%, #CF4548 60%, #77293A 80%, #77293A 100%)"
        if selected_scale == INDIAN_CPCB_SCALE
        else
        "linear-gradient(90deg, #27AE60 0%, #D5B82E 10%, #E98632 20%, "
        "#CF4548 30%, #8B4BA8 40%, #77293A 60%, #77293A 100%)"
    )

    marker = ""
    if active_scale and current_category and not beyond_range:
        position = min(max(float(current["aqi"]) / 5, 1.5), 98.5)
        marker = (
            f'<span class="sc-risk-marker" role="img" '
            f'aria-label="Current {html.escape(selected_scale)} AQI '
            f'{current["aqi"]:g}, {html.escape(current_category)}" '
            f'style="left:{position:.2f}%"></span>'
        )

    if active_scale and beyond_range:
        status = (
            f'Beyond AQI · {current["aqi"]:g} — outside the 0–500 legend; '
            "no clipped marker is shown."
        )
        advice = "This WAQI value is beyond the displayed US AQI range; follow authoritative health guidance."
    elif active_scale and current_category:
        status = (
            f'Current: {current_category} · AQI {current["aqi"]:g} · '
            f'{current["station"]}'
        )
        advice = (
            cpcb_advice[current_category]
            if selected_scale == INDIAN_CPCB_SCALE
            else us_advice[current_category]
        )
    elif current:
        status = (
            f'Fresh {current["scale"]} observation available; switch to that '
            "scale to see its current risk marker."
        )
        advice = "The marker appears only on the AQI scale used by the active observation."
    elif selected_scale == INDIAN_CPCB_SCALE:
        freshness = (
            cpcb_observation.get("freshness")
            if isinstance(cpcb_observation, dict)
            else None
        )
        status = (
            f"CPCB observation is {str(freshness).lower()} — no current marker is shown."
            if freshness and str(freshness).upper() != "FRESH"
            else "CPCB National AQI unavailable or unverified — no current marker is shown."
        )
        advice = "Use the category bands as general guidance only; no current CPCB value is being classified."
    else:
        freshness = (
            waqi_observation.get("freshness")
            if isinstance(waqi_observation, dict)
            else None
        )
        status = (
            f"WAQI observation is {str(freshness).lower()} — no current marker is shown."
            if freshness and str(freshness).upper() != "FRESH"
            else "No fresh, scale-verified WAQI observation — no current marker is shown."
        )
        advice = "Use the category bands as general guidance only; no current US AQI value is being classified."

    cards = []
    for name, value_range, icon, color in levels:
        is_current = name == current_category and active_scale and not beyond_range
        cards.append(
            f'<article class="sc-risk-level{" is-current" if is_current else ""}" role="listitem" '
            f'style="--sc-risk-color:{color}" '
            f'aria-label="{html.escape(name)}, AQI {value_range}">'
            f'<span class="sc-risk-icon" aria-hidden="true">{icon}</span>'
            f'<span class="sc-risk-name">{html.escape(name)}</span>'
            f'<span class="sc-risk-range">AQI {value_range}</span>'
            "</article>"
        )
    station_detail = (
        f' · {html.escape(current["source"])}'
        if active_scale and current and not beyond_range
        else ""
    )
    animation_class = (
        "" if st.session_state.get("risk_scale_animation_seen") else " sc-risk-enter"
    )
    st.session_state["risk_scale_animation_seen"] = True
    st.markdown(
        f"""
<style>
    .sc-risk-scale {{
        --sc-risk-surface: var(--sc-surface, #111C2E);
        --sc-risk-elevated: var(--sc-elevated, #17243A);
        --sc-risk-text: var(--sc-text, #F1F5F9);
        --sc-risk-muted: var(--sc-muted, #A8B7CA);
        --sc-risk-line: var(--sc-line, #2A3A50);
        margin: 0.1rem 0 1.2rem;
        padding: clamp(1rem, 2.4vw, 1.5rem);
        color: var(--sc-risk-text);
        background: var(--sc-risk-surface);
        border: 1px solid var(--sc-risk-line);
        border-radius: 20px;
        box-shadow: 0 10px 28px #00000014;
    }}
    .sc-risk-scale.sc-risk-enter {{ animation: sc-risk-fade-in 480ms ease-out both; }}
    @keyframes sc-risk-fade-in {{
        from {{ opacity: 0; transform: translateY(7px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .sc-risk-scale .sc-risk-heading {{ margin: 0 !important; color: var(--sc-risk-text) !important; font-size: clamp(1.05rem, 2vw, 1.35rem) !important; font-weight: 750; line-height: 1.25 !important; }}
    .sc-risk-subtitle {{ margin: 0.25rem 0 1.1rem; color: var(--sc-risk-muted); font-size: 0.92rem; }}
    .sc-risk-levels {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.55rem; }}
    .sc-risk-level {{
        display: flex; min-height: 102px; flex-direction: column; align-items: flex-start;
        justify-content: center; gap: 0.3rem; padding: 0.68rem;
        color: var(--sc-risk-text); background: var(--sc-risk-elevated);
        border: 1px solid var(--sc-risk-line); border-top: 3px solid var(--sc-risk-color);
        border-radius: 13px; transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
    }}
    .sc-risk-level.is-current {{
        border-color: var(--sc-risk-color);
        box-shadow: 0 0 0 1px var(--sc-risk-color), 0 0 18px color-mix(in srgb, var(--sc-risk-color) 28%, transparent);
        transform: translateY(-2px);
    }}
    .sc-risk-icon {{ color: var(--sc-risk-color); font-size: 1.25rem; line-height: 1.1; }}
    .sc-risk-enter .sc-risk-icon {{ animation: sc-risk-breathe 3.8s ease-in-out infinite; }}
    @keyframes sc-risk-breathe {{ 0%, 100% {{ transform: scale(1); opacity: 0.92; }} 50% {{ transform: scale(1.07); opacity: 1; }} }}
    .sc-risk-name {{ font-size: clamp(0.69rem, 0.9vw, 0.83rem); font-weight: 700; line-height: 1.22; }}
    .sc-risk-range {{ color: var(--sc-risk-muted); font-size: 0.71rem; white-space: nowrap; }}
    .sc-risk-meter {{ position: relative; margin: 1.15rem 0 0.45rem; padding-top: 0.68rem; }}
    .sc-risk-gradient {{ height: 9px; background: {gradient}; border-radius: 999px; box-shadow: inset 0 1px 2px #0003; }}
    .sc-risk-marker {{
        position: absolute; top: 0; width: 3px; height: 23px; border-radius: 4px;
        background: var(--sc-risk-text); border: 1px solid var(--sc-risk-surface);
        box-shadow: 0 0 0 2px #0b122044, 0 2px 6px #0005;
        transform: translateX(-50%); transition: left 700ms cubic-bezier(.2,.8,.2,1);
    }}
    .sc-risk-marker::before {{
        content: ""; position: absolute; left: 50%; top: -5px; width: 8px; height: 8px;
        background: var(--sc-risk-text); border: 1px solid var(--sc-risk-surface);
        border-radius: 50%; transform: translateX(-50%);
    }}
    .sc-risk-scale-labels {{ display: flex; justify-content: space-between; color: var(--sc-risk-muted); font-size: 0.68rem; }}
    .sc-risk-status {{ margin: 0.8rem 0 0; color: var(--sc-risk-text); font-size: 0.86rem; font-weight: 650; }}
    .sc-risk-guidance {{ margin: 0.25rem 0 0; color: var(--sc-risk-muted); font-size: 0.83rem; line-height: 1.45; }}
    .sc-risk-footnote {{ margin: 0.45rem 0 0; color: var(--sc-risk-muted); font-size: 0.7rem; }}
    @media (max-width: 880px) {{ .sc-risk-levels {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
    @media (max-width: 520px) {{
        .sc-risk-scale {{ padding: 0.9rem; border-radius: 16px; }}
        .sc-risk-levels {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.45rem; }}
        .sc-risk-level {{ min-height: 88px; }}
        .sc-risk-name {{ font-size: 0.77rem; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        .sc-risk-scale *, .sc-risk-scale *::before, .sc-risk-scale *::after {{
            animation: none !important; transition: none !important; scroll-behavior: auto !important;
        }}
    }}
</style>
<section class="sc-risk-scale{animation_class}" aria-labelledby="sc-risk-title">
    <h2 class="sc-risk-heading" id="sc-risk-title">Air Quality Index — Know Your Risk</h2>
    <p class="sc-risk-subtitle">Understand when the air is safe, unhealthy, or dangerous.</p>
    <div class="sc-risk-levels" role="list" aria-label="{html.escape(selected_scale)} category levels">
        {"".join(cards)}
    </div>
    <div class="sc-risk-meter" aria-label="AQI scale from 0 to 500">
        <div class="sc-risk-gradient"></div>{marker}
    </div>
    <div class="sc-risk-scale-labels"><span>0</span><span>100</span><span>200</span><span>300</span><span>400</span><span>500</span></div>
    <p class="sc-risk-status">{html.escape(status)}{station_detail}</p>
    <p class="sc-risk-guidance">{html.escape(advice)}</p>
    <p class="sc-risk-footnote">Legend: {html.escape(selected_scale)} · General health guidance only, not a medical diagnosis.</p>
</section>
""",
        unsafe_allow_html=True,
    )


def _choose_mode(mode):
    """Switch modes from the overview cards without mixing their navigation."""
    st.session_state["workspace_mode"] = mode
    st.session_state["primary_nav"] = "Overview"
    st.session_state["active_page"] = _page_for_primary_nav("Overview", mode)


def _measurement_columns(frame):
    """Find source-provided time/value fields without deriving or renaming data."""
    lowered = {str(column).casefold(): column for column in frame.columns}
    time_column = next(
        (lowered[name] for name in ("period_start", "collected_at", "observed_at", "timestamp", "datetime", "date", "time") if name in lowered),
        None,
    )
    value_column = next(
        (lowered[name] for name in ("value", "mean", "measurement", "reading", "concentration") if name in lowered),
        None,
    )
    return time_column, value_column


def _render_forecast_reliability_note():
    """Show the stored pooled walk-forward metrics without retraining models."""
    results_path = APP_DIR / "data" / "processed" / "walk_forward_results.csv"
    try:
        results = pd.read_csv(results_path)
    except (OSError, pd.errors.ParserError) as error:
        st.warning(
            "Stored walk-forward reliability results are unavailable "
            f"({type(error).__name__}); no metric is shown."
        )
        return

    try:
        summary = summarize_walk_forward_results(results)
    except ValueError as error:
        st.warning(f"Stored reliability results could not be validated: {error}")
        return

    horizon_order = ["Day+1", "Day+2", "Day+3"]
    if not all(horizon in summary for horizon in horizon_order):
        st.info(
            "Three complete pooled Random Forest validation windows are required "
            "for each forecast horizon; a reliability metric is not available."
        )
        return

    st.markdown("---")
    st.subheader("Forecast Error & Reliability")
    start = min(summary[horizon]["validation_start"] for horizon in horizon_order)
    end = max(summary[horizon]["validation_end"] for horizon in horizon_order)
    render_kpi_cards([
        (
            f"{horizon} mean MAE",
            f"{summary[horizon]['mae']:.3f}",
            f"AQI points · {summary[horizon]['window_count']} windows",
        )
        for horizon in horizon_order
    ])
    st.caption(
        "Source: stored walk_forward_results.csv. Values are means across three "
        f"expanding chronological windows ({start:%Y-%m-%d} to {end:%Y-%m-%d}), "
        "pooled across dataset locations. The target is dataset-defined historical "
        "AQI; MAE is average error in AQI points, not an accuracy percentage or "
        "a guarantee for an individual forecast."
    )


def _render_mode_cards():
    """Show the two explicit workspaces without implying recent AQI is live."""
    st.markdown(
        '<div class="sc-landing-brand">'
        f'{_logo_markup("sc-landing-logo")}'
        '<h1>SaansCheck</h1>'
        '<p>Delhi-NCR Air Quality Intelligence</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sc-tagline">Breathe smarter. Understand better.</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="sc-hero-title">Understand the air. Make smarter decisions.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sc-hero-copy">SaansCheck combines historical air-quality analytics, machine-learning forecasting, location insights and personalized risk insights for Delhi-NCR.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown('<div class="sc-mode-card-title">📊 Historical Intelligence</div>', unsafe_allow_html=True)
            st.markdown('<div class="sc-mode-card-copy">Explore historical air-quality patterns, machine-learning forecasts and model performance.</div>', unsafe_allow_html=True)
            render_badge("2019–2023", "teal")
            st.button(
                "Explore Historical →",
                key="open_historical_mode",
                on_click=_choose_mode,
                args=(HISTORICAL_MODE,),
                use_container_width=True,
            )
    with right:
        with st.container(border=True):
            st.markdown('<div class="sc-mode-card-title">◌ Recent Air Quality</div>', unsafe_allow_html=True)
            st.markdown('<div class="sc-mode-card-copy">Explore external air-quality observations, station data and source freshness when available.</div>', unsafe_allow_html=True)
            render_badge("External Data", "amber")
            st.button(
                "Explore Recent →",
                key="open_recent_mode",
                on_click=_choose_mode,
                args=(RECENT_MODE,),
                use_container_width=True,
            )


def _render_recent_location_picker(city_options, key):
    """Offer only project locations actually present in the source registry."""
    if not city_options:
        return None, None
    labels = list(city_options)
    label = st.selectbox("Available location", labels, key=key)
    return label, city_options[label]


def _format_source_timestamp(value):
    """Format a source-provided timestamp, or return an honest empty label."""
    if value is None or str(value).strip() == "":
        return "Not provided"
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return "Unparseable source timestamp"
    if pd.isna(parsed):
        return "Unparseable source timestamp"
    suffix = f" {parsed.strftime('%Z')}" if parsed.tzinfo is not None else " (source timezone unspecified)"
    return parsed.strftime("%Y-%m-%d %H:%M") + suffix


def _latest_measurement_timestamp(frame):
    """Find the latest timestamp in returned measurement rows, if one exists."""
    time_column, _ = _measurement_columns(frame)
    if time_column is None or frame.empty:
        return None
    parsed = pd.to_datetime(frame[time_column], errors="coerce").dropna()
    return frame.loc[parsed.idxmax(), time_column] if not parsed.empty else None


def _source_parameters(station):
    """Display only parameter names attached to a real station registry entry."""
    values = station.get("parameters", [])
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError:
            values = [values]
    if isinstance(values, (list, tuple, set)):
        return [str(value) for value in values]
    return []

def aqi_category_from_value(value):
    if pd.isna(value):
        return "Unknown"
    val = float(value)
    if val <= 50:
        return "Good"
    if val <= 100:
        return "Satisfactory"
    if val <= 200:
        return "Moderate"
    if val <= 300:
        return "Poor"
    if val <= 400:
        return "Very Poor"
    return "Severe"


def get_station_records(daily_df, station_name):
    station_daily = daily_df[daily_df["location_name"] == station_name].sort_values("date").copy()
    if station_daily.empty:
        raise ValueError(f"No records available for station: {station_name}")
    station_daily["date"] = pd.to_datetime(station_daily["date"])
    return station_daily


def format_display_value(value, digits=1):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _render_chart(figure, **kwargs):
    """Apply the selected appearance to a chart before displaying it."""
    apply_chart_theme(figure, st.session_state["appearance_theme"])
    st.pyplot(figure, **kwargs)


def build_forecast_for_station(daily_df, station_name, models_dict):
    forecast_rows = build_direct_forecasts(daily_df, station_name, models_dict)
    for row in forecast_rows:
        prediction = row["predicted_aqi"]
        row["category"] = aqi_category_from_value(prediction)
        row["predicted_aqi"] = round(prediction, 2)
    return forecast_rows


def _render_observed_aqi_alert(alert):
    """Render the transient cloud and persistent, dismissible alert dock."""
    st.iframe(
        cloud_alert_component_html(alert),
        height=1,
        tab_index=-1,
        alt="Transient high AQI health alert",
    )
    if alert is None:
        return

    observed_label = pd.Timestamp(alert["observed_at"]).tz_convert(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%Y-%m-%d %H:%M %Z")
    with st.container(key="waqi_alert_dock"):
        st.markdown("**⚠ High AQI Alert**")
        st.caption(
            f"Observed AQI {alert['aqi']:g} · {alert['station_name']}"
        )
        with st.expander("Alert details", expanded=False):
            st.write(f"Risk band: {alert['severity']}")
            st.write(f"Observed: {observed_label}")
            st.write(alert["precaution"])
            st.caption(
                f"WAQI US EPA 2016 scale; alert threshold AQI ≥ {HIGH_AQI_THRESHOLD}. "
                "Observed data only—not a forecast or diagnosis."
            )
        st.button(
            "Dismiss alert",
            key="dismiss_waqi_aqi_alert",
            on_click=_dismiss_current_aqi_alert,
            use_container_width=True,
        )


def _dismiss_current_aqi_alert():
    dismiss_alert_session_state(st.session_state)


# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="SaansCheck – Delhi-NCR AQI Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "appearance_theme" not in st.session_state:
    st.session_state["appearance_theme"] = "Dark"

# Base layout styles; the selected appearance is applied after this stylesheet.
st.markdown("""
<style>
    :root {
        --bg: #f5f8fa;
        --bg-soft: #edf4f7;
        --card: #ffffff;
        --card-soft: #f9fbfc;
        --line: #e2e8ec;
        --text: #17232d;
        --muted: #687983;
        --teal: #0f8b8d;
        --teal-soft: #ddf4f2;
        --green: #2e9d68;
        --green-soft: #e5f5ec;
        --amber: #d99a2b;
        --amber-soft: #fff3d6;
        --red: #d9534f;
        --red-soft: #fde8e7;
        --shadow: 0 10px 24px rgba(18, 36, 46, 0.08);
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    .brand-box {
        padding: 0.4rem 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.3rem 0.1rem;
    }
    .brand-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: var(--teal-soft);
        color: var(--teal);
        display: grid;
        place-items: center;
        font-size: 17px;
        font-weight: 700;
        border: 1px solid rgba(15, 139, 141, 0.12);
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: var(--text);
    }
    .brand-sub {
        color: var(--muted);
        font-size: 0.7rem;
        line-height: 1.5;
        margin-top: 0.2rem;
    }
    .sidebar-meta {
        color: var(--muted);
        font-size: 0.72rem;
        line-height: 1.7;
        padding: 0 0.3rem;
    }
    .stRadio > div {
        gap: 8px;
        margin-top: 8px;
    }
    .stRadio label {
        background: #f3f6f8;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 9px 12px;
        margin-bottom: 6px;
        color: var(--text);
        transition: all 0.2s ease;
        box-shadow: none;
    }
    .stRadio label:hover {
        background: #edf8f8;
        border-color: rgba(15, 139, 141, 0.3);
    }
    .stRadio [data-baseweb="radio"] {
        margin-right: 8px;
    }
    .stRadio input:checked + div {
        color: var(--text) !important;
    }
    .stRadio label:has(input:checked) {
        background: var(--teal-soft) !important;
        border-color: rgba(15, 139, 141, 0.35) !important;
        border-left: 3px solid var(--teal) !important;
    }
    .section-kicker {
        color: var(--teal);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .hero-card {
        background: #f0f8f8;
        border: 1px solid rgba(15, 139, 141, 0.12);
        border-radius: 18px;
        padding: 1.2rem 1.3rem 1rem;
        margin-bottom: 1.1rem;
        box-shadow: var(--shadow);
    }
    .hero-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .hero-title {
        margin: 0.5rem 0 0;
        font-size: clamp(1.8rem, 2vw, 2.4rem);
        line-height: 1.15;
        color: var(--text);
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 0.98rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    .badge-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 0.45rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #ffffff;
        border: 1px solid var(--line);
        color: var(--muted);
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .status-pill .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--green);
        display: inline-block;
    }
    .section-title {
        color: var(--text);
        font-size: clamp(1.5rem, 1.8vw, 2rem);
        font-weight: 700;
        margin: 0.4rem 0 0.2rem;
        letter-spacing: -0.03em;
    }
    .subtle-note {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.6;
    }
    .premium-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: var(--shadow);
        padding: 1rem 1rem 0.9rem;
        height: 100%;
    }
    .kpi-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: var(--shadow);
        padding: 1rem 1rem 0.8rem;
        height: 100%;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-header {
        color: var(--muted);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.68rem;
        font-weight: 700;
    }
    .kpi-value {
        font-size: clamp(1.6rem, 2vw, 2.4rem);
        font-weight: 800;
        color: var(--text);
        line-height: 1.15;
        margin-top: 0.55rem;
    }
    .kpi-meta {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }
    .kpi-meta strong {
        color: var(--text);
    }
    .metric-box {
        background: var(--card-soft);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1rem 0.85rem;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .insight-card {
        background: var(--card-soft);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    .insight-card h4 {
        color: var(--muted);
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0 0 0.5rem;
        font-weight: 700;
    }
    .insight-card .value {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text);
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: var(--green-soft);
        border: 1px solid rgba(46, 157, 104, 0.12);
        color: var(--green);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .status-box {
        background: var(--card-soft);
        border: 1px solid rgba(15, 139, 141, 0.15);
        border-left: 4px solid var(--teal);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        color: var(--text);
        font-size: 0.9rem;
        line-height: 1.55;
        margin: 1rem 0;
    }
    .page-section {
        margin-top: 1.2rem;
    }
    .chart-frame {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.8rem;
        box-shadow: var(--shadow);
    }
    .comparison-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .selection-pill {
        background: var(--card-soft);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--text);
    }
    .selection-pill.active {
        background: var(--teal-soft);
        border-color: rgba(15, 139, 141, 0.2);
        color: var(--teal);
    }
    .small-note {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.55;
    }
    .footer-text {
        text-align: center;
        color: var(--muted);
        font-size: 0.76rem;
        padding: 2rem 0 0.5rem;
    }
    .stTabs [role="tablist"] {
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .stTabs [role="tab"] {
        border-radius: 10px;
        background: #f1f5f6;
        border: 1px solid var(--line);
        color: var(--text);
        padding: 0.55rem 0.8rem;
        font-weight: 600;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: var(--teal-soft);
        border-color: rgba(15, 139, 141, 0.2);
        color: var(--teal);
    }
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--line);
    }
    .stDataFrame table {
        border-collapse: collapse;
    }
    .stDataFrame th {
        background: #f7fafb;
        color: var(--text);
        font-weight: 700;
    }
    .stDataFrame td {
        color: var(--text);
    }
    .main .block-container {
        max-width: 1480px;
        padding-top: 0.6rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stSidebar"] {
        display: block;
        background: #ffffff;
        border-right: 1px solid #e2e8ec;
    }
    [data-testid="stSidebar"] > div:first-child { background: #ffffff; }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1rem 0.85rem 1.2rem; }
    .sc-sidebar-kicker {
        margin: 0.35rem 0 0.55rem;
        color: #64748b;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .sc-sidebar-title {
        margin: 1.1rem 0 0.45rem;
        color: #64748b;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .sc-sidebar-badge {
        margin: 0.3rem 0 0.65rem;
        padding: 0.62rem 0.7rem;
        border: 1px solid #d5ecea;
        border-radius: 10px;
        background: #eef8f7;
        color: #0f7375;
        font-size: 0.68rem;
        font-weight: 750;
        line-height: 1.4;
    }
    .sc-sidebar-badge.recent { border-color: #f2e4c2; background: #fff8e9; color: #8a5b11; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 0.35rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.55rem 0.6rem !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
        border-radius: 9px !important;
        background: transparent !important;
        color: #334155 !important;
        font-size: 0.82rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #f3f8f8 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        border-color: #d5ecea !important;
        border-left: 3px solid #0f8b8d !important;
        background: #eaf6f5 !important;
        color: #0f7375 !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div {
        gap: 0 !important;
    }
    .sc-sidebar-bottom { margin-top: 2.25rem; border-top: 1px solid #e2e8ec; padding-top: 0.85rem; }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) {
        position: sticky;
        top: 0;
        z-index: 1000;
        align-items: center;
        background: rgba(255, 255, 255, 0.98);
        border-bottom: 1px solid #e2e8ec;
        padding: 0.55rem 0;
    }
    .sc-topnav-marker, .sc-page-nav-marker { display: none; }
    .sc-brand { display: flex; align-items: center; gap: 0.65rem; min-height: 3rem; }
    .sc-brand-logo {
        display: block;
        flex: 0 0 auto;
        width: 2.8rem;
        height: 2.8rem;
        object-fit: contain;
        border-radius: 10px;
    }
    .sc-logo-fallback {
        display: grid;
        place-items: center;
        flex: 0 0 auto;
        width: 2.8rem;
        height: 2.8rem;
        overflow: hidden;
        border-radius: 10px;
        color: #0f8b8d;
        background: #e4f3f1;
        border: 1px solid #d4eae7;
        font-size: 0.65rem;
        font-weight: 800;
        line-height: 1.1;
        text-align: center;
    }
    .sc-brand-name { color: #17232d; font-size: 0.95rem; font-weight: 800; letter-spacing: 0.02em; line-height: 1.2; }
    .sc-brand-subtitle { color: #64748b; font-size: 0.68rem; line-height: 1.35; margin-top: 0.16rem; }
    .sc-sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 0.1rem 0 1.15rem;
        padding: 0 0.15rem 0.9rem;
        border-bottom: 1px solid #e2e8ec;
    }
    .sc-sidebar-logo {
        display: block;
        flex: 0 0 auto;
        width: 3.15rem;
        height: 3.15rem;
        object-fit: contain;
        border-radius: 9px;
    }
    .sc-sidebar-brand .sc-logo-fallback {
        width: 3.15rem;
        height: 3.15rem;
    }
    .sc-sidebar-brand-name { color: #17232d; font-size: 0.98rem; font-weight: 800; line-height: 1.25; }
    .sc-sidebar-brand-subtitle { color: #64748b; font-size: 0.72rem; line-height: 1.4; margin-top: 0.18rem; }
    .sc-dashboard-status {
        white-space: nowrap;
        color: #475569;
        background: #f8fafb;
        border: 1px solid #e2e8ec;
        border-radius: 999px;
        padding: 0.42rem 0.68rem;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .sc-location-marker { display: none; }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(2) [data-testid="stSelectbox"] {
        min-width: 0;
        padding: 0.12rem 0.35rem;
        border: 1px solid #cde7e4;
        border-radius: 12px;
        background: linear-gradient(135deg, #f2fbfa, #e6f5f3);
        box-shadow: 0 3px 10px rgba(15, 111, 115, 0.07);
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(2) [data-testid="stSelectbox"] > label {
        display: none;
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(2) [data-testid="stSelectbox"] [data-baseweb="select"] {
        min-height: 2.1rem;
        border: 0;
        background: transparent;
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(2) [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-height: 2.1rem;
        border: 0;
        background: transparent;
        box-shadow: none;
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(1) [data-testid="stTooltipHoverTarget"] button {
        min-height: 2.8rem;
        justify-content: flex-start;
        color: var(--sc-text) !important;
        background: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
        font-weight: 800;
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(1) [data-testid="stTooltipHoverTarget"] button:hover {
        color: var(--sc-accent) !important;
        background: var(--sc-accent-soft) !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(3) [data-testid="stTooltipHoverTarget"] button {
        min-height: 2.7rem;
        white-space: nowrap;
        color: var(--sc-accent) !important;
        background: var(--sc-accent-soft) !important;
        border-color: var(--sc-line) !important;
        font-weight: 700;
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(3) [data-testid="stTooltipHoverTarget"] button:hover {
        color: var(--sc-accent) !important;
        border-color: var(--sc-accent) !important;
        box-shadow: 0 3px 10px rgba(15, 111, 115, 0.12) !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(2) input[role="combobox"]::placeholder {
        color: var(--sc-muted) !important;
        opacity: 1;
    }
    .sc-location-pin {
        position: relative;
        display: block;
        flex: 0 0 auto;
        width: 1rem;
        height: 1rem;
        border-radius: 50% 50% 50% 0;
        background: linear-gradient(145deg, #22a6a0, #0e7479);
        box-shadow: 0 2px 5px rgba(15, 111, 115, 0.22);
        transform: rotate(-45deg);
    }
    .sc-location-pin::after {
        position: absolute;
        top: 0.3rem;
        left: 0.3rem;
        width: 0.4rem;
        height: 0.4rem;
        border-radius: 50%;
        background: #effaf8;
        content: "";
    }
    .sc-mode-row { margin: 1.15rem 0 0.35rem; padding-bottom: 0.8rem; border-bottom: 1px solid #e2e8ec; }
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) [data-testid="stSegmentedControl"],
    div[data-testid="stHorizontalBlock"]:has(.sc-mode-marker) [data-testid="stSegmentedControl"],
    div[data-testid="stHorizontalBlock"]:has(.sc-page-nav-marker) [data-testid="stSegmentedControl"] {
        min-width: 0;
        overflow-x: auto;
    }
    [role="radiogroup"] button[data-variant="segmented_control"] {
        color: #475569 !important;
        border-color: #e2e8ec !important;
        background: #ffffff !important;
        box-shadow: none !important;
    }
    [role="radiogroup"] button[data-variant="segmented_control"] p {
        margin: 0 !important;
        white-space: nowrap;
        font-size: 0.83rem;
    }
    [role="radiogroup"][aria-label="Mode navigation"] button[data-variant="segmented_control"] {
        padding-left: 0.38rem !important;
        padding-right: 0.38rem !important;
    }
    [role="radiogroup"][aria-label="Mode navigation"] button[data-variant="segmented_control"] p {
        font-size: 0.75rem;
    }
    [role="radiogroup"] button[data-variant="segmented_control"][data-selected="true"] {
        color: #0f7375 !important;
        background: #e7f5f4 !important;
        border-color: #d5ecea !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.sc-page-nav-marker) { overflow-x: auto; flex-wrap: nowrap; }
    .sc-breadcrumb { color: #64748b; font-size: 0.75rem; margin: 1.15rem 0 0.55rem; }
    .sc-breadcrumb span { color: #a0acb6; padding: 0 0.35rem; }
    .sc-page-title { margin: 0; color: #17232d; font-size: 1.8rem; font-weight: 750; line-height: 1.2; letter-spacing: 0; }
    .sc-page-subtitle { margin: 0.5rem 0 1rem; color: #64748b; font-size: 0.96rem; line-height: 1.55; }
    .sc-badge-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.3rem 0 1rem; }
    .sc-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.32rem 0.62rem;
        border: 1px solid #e2e8ec;
        border-radius: 999px;
        color: #475569;
        background: #f8fafb;
        font-size: 0.68rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .sc-badge-teal { color: #0f7375; background: #e7f5f4; border-color: #d5ecea; }
    .sc-badge-green { color: #287b55; background: #eaf5ef; border-color: #d8ecdf; }
    .sc-badge-amber { color: #8a5b11; background: #fff5df; border-color: #f2e4c2; }
    .sc-kpi { height: 100%; min-height: 128px; padding: 1rem; border: 1px solid #e2e8ec; border-radius: 14px; background: #fff; box-shadow: 0 3px 12px rgba(23,35,45,0.035); }
    .sc-kpi-label { color: #64748b; font-size: 0.72rem; font-weight: 700; line-height: 1.4; }
    .sc-kpi-value { color: #17232d; font-size: 1.55rem; font-weight: 750; line-height: 1.2; margin-top: 0.7rem; overflow-wrap: anywhere; }
    .sc-kpi-detail { color: #64748b; font-size: 0.72rem; line-height: 1.45; margin-top: 0.5rem; }
    .sc-historical-aqi-value { font-size: 2.2rem; font-weight: 800; line-height: 1.1; margin-top: 0.35rem; }
    .sc-empty-state { display: flex; flex-direction: column; align-items: flex-start; gap: 0.4rem; margin: 1rem 0; padding: 1.2rem; border: 1px solid #e2e8ec; border-radius: 14px; background: #fff; }
    .sc-empty-mark { color: #0f8b8d; font-size: 1.15rem; }
    .sc-empty-title { color: #17232d; font-size: 1rem; font-weight: 700; }
    .sc-empty-detail { max-width: 58rem; color: #64748b; font-size: 0.86rem; line-height: 1.55; }
    .sc-mode-card-title { color: #17232d; font-size: 1.05rem; font-weight: 750; }
    .sc-mode-card-copy { min-height: 3.25rem; color: #64748b; font-size: 0.84rem; line-height: 1.55; }
    .sc-hero-title { margin: 0.3rem 0; color: #17232d; font-size: 2rem; font-weight: 780; line-height: 1.18; letter-spacing: 0; }
    .sc-tagline { margin: 0; color: #0f7375; font-size: 0.78rem; font-weight: 700; }
    .sc-hero-copy { max-width: 54rem; margin: 0 0 1.2rem; color: #64748b; font-size: 0.98rem; line-height: 1.6; }
    .sc-landing-brand { text-align: center; margin: 0.15rem 0 1rem; }
    .sc-landing-logo {
        display: block;
        width: min(11rem, 42vw);
        height: auto;
        aspect-ratio: 1;
        object-fit: contain;
        margin: 0 auto 0.2rem;
    }
    .sc-landing-brand .sc-logo-fallback {
        display: inline-grid;
        width: auto;
        min-width: 9rem;
        height: auto;
        padding: 0.7rem 1rem;
        margin: 0 auto 0.5rem;
        font-size: 1rem;
    }
    .sc-landing-brand h1 { margin: 0.1rem 0 0; color: #17232d; font-size: 1.45rem; font-weight: 800; }
    .sc-landing-brand p { margin: 0.25rem 0 0; color: #64748b; font-size: 0.88rem; }
    .stRadio label { letter-spacing: 0; }
    @media (max-width: 760px) {
        div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) { position: static; flex-wrap: wrap; }
        .sc-brand-logo, .sc-brand .sc-logo-fallback { width: 2.4rem; height: 2.4rem; }
        .sc-brand { gap: 0.45rem; }
        .sc-brand-subtitle { font-size: 0.62rem; }
        .sc-hero-title { font-size: 1.65rem; }
        .sc-page-title { font-size: 1.5rem; }
        .sc-landing-logo { width: min(9rem, 48vw); }
        div[data-testid="stHorizontalBlock"]:has(.sc-page-nav-marker) { flex-wrap: wrap; }
    }
    @media (max-width: 1200px) {
        div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) {
            display: grid !important;
            grid-template-columns: minmax(0, 1fr) auto;
            grid-template-rows: auto auto;
            column-gap: 0.45rem;
            row-gap: 0.45rem;
        }
        div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"] {
            width: auto !important;
            min-width: 0 !important;
            flex: initial !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(1) { grid-column: 1; grid-row: 1; }
        div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(2) { grid-column: 1 / -1; grid-row: 2; }
        div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) > [data-testid="stColumn"]:nth-child(3) { grid-column: 2; grid-row: 1; }
        div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) [data-testid="stSegmentedControl"] { width: 100%; }
    }
    @media (max-width: 900px) {
        .insight-grid {
            grid-template-columns: 1fr;
        }
        .hero-row {
            flex-direction: column;
            align-items: flex-start;
        }
    }
    .sc-sidebar-intro {
        position: relative;
        isolation: isolate;
        display: grid;
        place-items: center;
        min-height: 7.3rem;
        margin: -0.45rem 0 0.75rem;
        padding: 0.7rem;
        overflow: hidden;
        border: 1px solid #d5ecea;
        border-radius: 12px;
        background:
            radial-gradient(circle at 50% 50%, rgba(15, 139, 141, 0.13), transparent 58%),
            linear-gradient(135deg, #f6fbfb, #edf7f6);
    }
    .sc-sidebar-intro::before {
        position: absolute;
        inset: 0;
        z-index: -1;
        background: radial-gradient(circle, rgba(94, 220, 205, 0.28), transparent 68%);
        content: "";
        animation: sc-sidebar-glow 5s ease-out both;
    }
    .sc-sidebar-intro-content {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.65rem;
    }
    .sc-sidebar-intro-symbol {
        display: block;
        flex: 0 0 auto;
        width: 3.35rem;
        height: 3.35rem;
        object-fit: contain;
        border-radius: 50%;
        filter: drop-shadow(0 0 11px rgba(15, 139, 141, 0.38));
        animation: sc-sidebar-symbol 5s cubic-bezier(0.2, 0.75, 0.25, 1) both;
    }
    .sc-sidebar-intro-copy {
        min-width: 0;
        text-align: left;
        animation: sc-sidebar-copy 5s ease-out both;
    }
    .sc-sidebar-intro-title {
        margin: 0;
        color: #123642;
        font-size: 0.88rem;
        font-weight: 800;
        letter-spacing: 0.025em;
        line-height: 1.2;
    }
    .sc-sidebar-intro-tagline {
        margin: 0.22rem 0 0;
        color: #527079;
        font-size: 0.62rem;
        line-height: 1.35;
    }
    .sc-sidebar-firework {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 4.4rem;
        aspect-ratio: 1;
        border-radius: 50%;
        opacity: 0;
        background: repeating-conic-gradient(
            from 8deg,
            transparent 0deg 13deg,
            rgba(15, 139, 141, 0.52) 14deg 15deg,
            transparent 16deg 45deg
        );
        mask-image: radial-gradient(circle, transparent 0 18%, #000 30% 47%, transparent 68%);
        animation: sc-sidebar-firework 5s ease-out both;
    }
    .sc-sidebar-firework.one { margin: -2.2rem 0 0 -3.1rem; }
    .sc-sidebar-firework.two {
        width: 3.6rem;
        margin: -1.7rem 0 0 1.2rem;
        animation-delay: 0.12s;
    }
    #sc-startup-intro {
        position: fixed;
        inset: 0;
        z-index: 2147483647;
        display: grid;
        place-items: center;
        overflow: hidden;
        color: #f4fbfd;
        background:
            radial-gradient(ellipse at 50% 46%, rgba(17, 132, 143, 0.34), transparent 38%),
            linear-gradient(140deg, #0b1d2b 0%, #103342 52%, #0b2433 100%);
        animation: sc-startup-lifecycle 4s ease-in-out forwards;
    }
    .sc-startup-content {
        position: relative;
        z-index: 2;
        width: min(34rem, calc(100vw - 2rem));
        padding: 2rem;
        text-align: center;
        animation: sc-startup-reveal 0.9s ease-out both;
    }
    .sc-startup-logo {
        display: block;
        width: min(12rem, 30vh, 42vw);
        height: min(12rem, 30vh, 42vw);
        margin: 0 auto 1.15rem;
        object-fit: contain;
        border-radius: 50%;
        filter: drop-shadow(0 0 25px rgba(103, 223, 211, 0.42));
        animation: sc-startup-pulse 2.4s ease-in-out infinite;
    }
    .sc-startup-title {
        margin: 0;
        color: #f4fbfd;
        font-size: clamp(2rem, 6vw, 3.5rem);
        font-weight: 800;
        letter-spacing: 0.025em;
        text-shadow: 0 0 28px rgba(103, 223, 211, 0.24);
    }
    .sc-startup-tagline {
        margin: 0.7rem 0 0;
        color: #c7e4e8;
        font-size: clamp(0.9rem, 2.5vw, 1.12rem);
        letter-spacing: 0.035em;
    }
    .sc-startup-particle {
        position: absolute;
        top: 50%;
        left: 50%;
        width: clamp(8rem, 19vw, 15rem);
        aspect-ratio: 1;
        border-radius: 50%;
        opacity: 0;
        background: repeating-conic-gradient(
            from 8deg,
            transparent 0deg 12deg,
            rgba(122, 233, 221, 0.7) 13deg 14deg,
            transparent 15deg 36deg
        );
        mask-image: radial-gradient(circle, transparent 0 18%, #000 30% 47%, transparent 69%);
        animation: sc-startup-firework 2.4s ease-out infinite;
    }
    .sc-startup-particle.one { margin: -11rem 0 0 -17rem; }
    .sc-startup-particle.two {
        width: clamp(6rem, 14vw, 11rem);
        margin: 2rem 0 0 8rem;
        animation-delay: 0.75s;
    }
    .sc-startup-particle.three {
        width: clamp(5rem, 11vw, 8rem);
        margin: 5rem 0 0 -13rem;
        animation-delay: 1.3s;
    }
    section[data-testid="stMain"]:has(#sc-startup-intro) {
        overflow: visible;
        animation: sc-startup-scroll-release 4s steps(1, end) forwards;
    }
    @keyframes sc-startup-lifecycle {
        0%, 82% { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; pointer-events: none; }
    }
    @keyframes sc-startup-scroll-release {
        0%, 99.99% { overflow: visible; }
        100% { overflow: auto; }
    }
    @keyframes sc-startup-reveal {
        from { opacity: 0; transform: translateY(14px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes sc-startup-pulse {
        0%, 100% { transform: scale(0.97); }
        50% { transform: scale(1.03); }
    }
    @keyframes sc-startup-firework {
        0% { opacity: 0; transform: translate(-50%, -50%) scale(0.42) rotate(0deg); }
        22% { opacity: 0.66; }
        100% { opacity: 0; transform: translate(-50%, -50%) scale(1.2) rotate(22deg); }
    }
    @keyframes sc-sidebar-glow {
        0% { opacity: 0; transform: scale(0.65); }
        22% { opacity: 1; }
        100% { opacity: 0.35; transform: scale(1.15); }
    }
    @keyframes sc-sidebar-symbol {
        0% { opacity: 0.55; transform: scale(0.78); }
        25% { opacity: 1; transform: scale(1.08); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes sc-sidebar-copy {
        0%, 18% { opacity: 0; transform: translateX(-7px); }
        42% { opacity: 1; transform: translateX(0); }
        100% { opacity: 1; transform: translateX(0); }
    }
    @keyframes sc-sidebar-firework {
        0% { opacity: 0; transform: translate(-50%, -50%) scale(0.3) rotate(0deg); }
        18% { opacity: 0.7; }
        100% { opacity: 0; transform: translate(-50%, -50%) scale(1.25) rotate(18deg); }
    }
    @media (prefers-reduced-motion: reduce) {
        #sc-startup-intro { animation-duration: 0.35s; }
        section[data-testid="stMain"]:has(#sc-startup-intro) {
            animation-duration: 0.35s;
        }
        .sc-startup-content,
        .sc-startup-logo,
        .sc-startup-particle { animation: none; }
        .sc-sidebar-intro::before,
        .sc-sidebar-intro-symbol,
        .sc-sidebar-intro-copy { animation: none; }
        .sc-sidebar-firework { display: none; }
    }
</style>
""", unsafe_allow_html=True)
st.markdown(theme_css(st.session_state["appearance_theme"]), unsafe_allow_html=True)

if not st.session_state.get("saanscheck_startup_intro_v1_seen", False):
    st.session_state["saanscheck_startup_intro_v1_seen"] = True
    st.markdown(
        '<div id="sc-startup-intro" role="status" '
        'aria-label="SaansCheck startup introduction">'
        '<span class="sc-startup-particle one" aria-hidden="true"></span>'
        '<span class="sc-startup-particle two" aria-hidden="true"></span>'
        '<span class="sc-startup-particle three" aria-hidden="true"></span>'
        '<div class="sc-startup-content">'
        f'{_logo_markup("sc-startup-logo", "SaansCheck emblem")}'
        '<h1 class="sc-startup-title">SaansCheck</h1>'
        '<p class="sc-startup-tagline">Delhi-NCR Air Quality Intelligence</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 2. BACKEND ARTIFACT INGESTION & DIAGNOSTICS
# =============================================================================

if "workspace_mode" not in st.session_state:
    st.session_state["workspace_mode"] = RECENT_MODE
if "primary_nav" not in st.session_state:
    st.session_state["primary_nav"] = "Overview"
if "active_page" not in st.session_state:
    st.session_state["active_page"] = _page_for_primary_nav(
        st.session_state["primary_nav"], st.session_state["workspace_mode"]
    )
if "recent_location_mode" not in st.session_state:
    st.session_state["recent_location_mode"] = "Automatic nearby"
if "recent_station_query" not in st.session_state:
    st.session_state["recent_station_query"] = ""
if "recent_station_query_applied" not in st.session_state:
    st.session_state["recent_station_query_applied"] = ""
if "browser_location_status" not in st.session_state:
    st.session_state["browser_location_status"] = "not_requested"
if "browser_location_request_id" not in st.session_state:
    st.session_state["browser_location_request_id"] = 0
if "location_bar_station" not in st.session_state:
    current_station = st.session_state.get("historical_station", "Delhi")
    st.session_state["location_bar_station"] = (
        current_station if current_station in STATIONS else "Delhi"
    )
if st.session_state["primary_nav"] != "About":
    st.session_state["primary_nav"] = _primary_nav_for_page(
        st.session_state["active_page"]
    )

data_load_error = None
system_status = {"all_valid": False}
daily_df = ml_df = results_df = predictions_df = importance_df = models_dict = None


# =============================================================================
# 3. RECENT AIR QUALITY NAVIGATION & SYSTEM STATUS
# =============================================================================

if st.session_state["workspace_mode"] == RECENT_MODE:
    top_columns = st.columns([2.0, 5.2, 1.0], vertical_alignment="center")
    with top_columns[0]:
        st.markdown('<span class="sc-topnav-marker"></span>', unsafe_allow_html=True)
        st.button(
            "SaansCheck",
            icon="🌿",
            key="navbar_brand_home",
            help="Go to the Recent Air Quality homepage",
            use_container_width=True,
            on_click=_on_home_navigation,
        )
    with top_columns[1]:
        st.selectbox(
            "Search Delhi-NCR city...",
            CITY_SEARCH_OPTIONS,
            index=None,
            placeholder="🔍 Search Delhi-NCR city...",
            key="navbar_city_search",
            on_change=_on_city_search_change,
            label_visibility="collapsed",
            help="Search Delhi, New Delhi, Noida, Ghaziabad, Gurugram/Gurgaon, or Faridabad.",
        )
        st.caption(f"Selected city: {st.session_state['location_bar_station']}")
    with top_columns[2]:
        st.button(
            "Go to homepage",
            icon="🏠",
            key="navbar_home",
            help="Open Recent Air Quality → Recent Overview",
            use_container_width=True,
            on_click=_on_home_navigation,
        )

    st.segmented_control(
        "Primary navigation",
        PRIMARY_NAV_ITEMS,
        label_visibility="collapsed",
        key="primary_nav",
        on_change=_on_primary_nav_change,
        width="stretch",
        wrap=True,
    )

with st.sidebar:
    st.markdown(
        '<div class="sc-sidebar-brand">'
        f'{_logo_markup("sc-sidebar-logo")}'
        '<div><div class="sc-sidebar-brand-name">SaansCheck</div>'
        '<div class="sc-sidebar-brand-subtitle">Air Quality Intelligence</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.segmented_control(
        "Appearance",
        ["Dark", "Light"],
        key="appearance_theme",
        width="stretch",
        help="Switch between the dark and light dashboard themes.",
    )
    if not st.session_state.get("saanscheck_intro_v3_seen", False):
        st.session_state["saanscheck_intro_v3_seen"] = True
        st.markdown(
            '<div class="sc-sidebar-intro" '
            'aria-label="SaansCheck five-second introduction">'
            '<span class="sc-sidebar-firework one" aria-hidden="true"></span>'
            '<span class="sc-sidebar-firework two" aria-hidden="true"></span>'
            '<div class="sc-sidebar-intro-content">'
            f'{_logo_markup("sc-sidebar-intro-symbol", "SaansCheck symbol")}'
            '<div class="sc-sidebar-intro-copy">'
            '<div class="sc-sidebar-intro-title">SaansCheck</div>'
            '<div class="sc-sidebar-intro-tagline">Delhi-NCR Air Quality Intelligence</div>'
            '</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sc-sidebar-kicker">DATA WORKSPACE</div>', unsafe_allow_html=True)
    st.radio(
        "Data workspace",
        [HISTORICAL_MODE, RECENT_MODE],
        key="workspace_mode",
        on_change=_on_mode_change,
        label_visibility="collapsed",
    )

    if st.session_state["workspace_mode"] == HISTORICAL_MODE:
        page_options = HISTORICAL_PAGES
        st.markdown(
            '<div class="sc-sidebar-title">HISTORICAL INTELLIGENCE</div>'
            '<div class="sc-sidebar-badge">● HISTORICAL INTELLIGENCE<br>Historical dataset · 2019–2023</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Historical AQI is dataset-defined; its original calculation method "
            "and the historical pollutant units have not been verified. "
            "Historical values are not live observations."
        )
    else:
        page_options = RECENT_PAGES
        st.markdown(
            '<div class="sc-sidebar-title">RECENT AIR QUALITY</div>'
            '<div class="sc-sidebar-badge recent">● RECENT AIR QUALITY<br>External observational data</div>',
            unsafe_allow_html=True,
        )
        st.radio(
            "Location lookup",
            ["Automatic nearby", "Manual city/station"],
            key="recent_location_mode",
            help=(
                "Nearby lookup asks the browser for approximate location. "
                "Manual lookup uses the selected city and a WAQI station you choose."
            ),
        )
        recent_location = st.session_state["location_bar_station"]
        city_station_search = load_city_stations(recent_location)
        locality_query = st.session_state["recent_station_query_applied"]
        locality_station_search = (
            load_city_stations(recent_location, query=locality_query)
            if locality_query
            else None
        )
        recent_station_search = merge_station_search_results(
            city_station_search,
            locality_station_search,
        )
        recent_stations = recent_station_search["stations"]
        recent_station_ids = [station["station_id"] for station in recent_stations]
        current_station_id = st.session_state.get("recent_station_uid")
        if current_station_id not in recent_station_ids:
            st.session_state["recent_station_uid"] = (
                recent_station_ids[0] if len(recent_station_ids) == 1 else None
            )
        if st.session_state["recent_location_mode"] == "Manual city/station":
            st.caption(
                f"City: {recent_location}. Change the City selector above to "
                "search another supported area."
            )
            with st.form("waqi_station_search_form", clear_on_submit=False):
                st.text_input(
                    "Search locality or station",
                    key="recent_station_query",
                    placeholder="e.g. Rohini, Dwarka, R.K. Puram",
                    help=(
                        "Search is sent to WAQI using its station-name search. "
                        "A locality is a query, not a guarantee of a station there."
                    ),
                )
                st.form_submit_button(
                    "Search WAQI stations",
                    on_click=_on_locality_search,
                    use_container_width=True,
                )
            if locality_query and locality_station_search is not None:
                if (
                    locality_station_search["api_available"]
                    and not locality_station_search["stations"]
                ):
                    st.info("No exact monitoring station found for this locality.")
                    if recent_stations:
                        st.caption(
                            "Other available stations in the selected city are "
                            "listed below; they are not represented as stations "
                            "in the searched locality."
                        )
                elif not locality_station_search["api_available"]:
                    st.warning(locality_station_search["error"])
            with st.expander("Station search results and validation"):
                st.caption(
                    f"WAQI city search returned "
                    f"{city_station_search.get('raw_result_count', 'unknown')} "
                    f"records; {city_station_search.get('station_count', len(city_station_search['stations']))} unique station "
                    "IDs remain after station-ID deduplication and exact city validation."
                )
                if locality_station_search is not None:
                    st.caption(
                        f"WAQI locality search returned "
                        f"{locality_station_search.get('raw_result_count', 'unknown')} records; "
                        f"{locality_station_search.get('station_count', len(locality_station_search['stations']))} exact locality "
                        "matches remain after city validation."
                    )
                st.caption(
                    f"{len(recent_stations)} unique stations are available in "
                    "the selection list. No result limit is applied."
                )
        if recent_stations:
            st.selectbox(
                "Available WAQI monitoring stations",
                recent_station_ids,
                format_func=lambda station_id: next(
                    station["station_name"]
                    for station in recent_stations
                    if station["station_id"] == station_id
                ),
                index=None,
                key="recent_station_uid",
                on_change=_on_manual_station_change,
                help=(
                    "Choose a real WAQI station returned for the selected city "
                    "or locality search. The feed is checked against its selected "
                    "station ID and city before AQI is displayed."
                ),
            )
            st.caption(
                f"{len(recent_stations)} WAQI monitoring stations available for "
                f"{recent_location}. Names are shown as returned by WAQI."
            )
        elif recent_station_search["error"]:
            st.warning(recent_station_search["error"])

    if st.session_state["primary_nav"] != "About":
        st.radio(
            "Workspace pages",
            page_options,
            key="active_page",
            on_change=_on_page_change,
            label_visibility="collapsed",
        )

    st.markdown('<div class="sc-sidebar-bottom"></div>', unsafe_allow_html=True)
    st.button(
        "⇄ Switch Workspace",
        key="switch_workspace_home",
        use_container_width=True,
        on_click=_on_switch_workspace,
    )
    with st.expander("⚙ Settings"):
        st.caption("External data credentials are stored in Streamlit secrets.")
        st.caption("Recent observations are never merged into historical training data.")

workspace_mode = st.session_state["workspace_mode"]
location_lookup_auto = (
    workspace_mode == RECENT_MODE
    and st.session_state["recent_location_mode"] == "Automatic nearby"
)
browser_location_status = st.session_state["browser_location_status"]
browser_location_request_id = str(
    st.session_state["browser_location_request_id"]
)
browser_location_result = request_browser_location(
    request_id=browser_location_request_id,
    enabled=(
        location_lookup_auto
        and browser_location_status == "not_requested"
    ),
    clear_result=browser_location_status in {
        "granted", "denied", "unsupported", "unavailable",
        "timeout", "invalid", "error", "lookup_failed",
    },
)
if isinstance(browser_location_result, dict):
    location_result_status = browser_location_result.get("status")
    if location_result_status == "requesting":
        st.session_state["browser_location_status"] = "requesting"
    elif location_result_status in {
        "denied", "unsupported", "unavailable", "timeout", "invalid", "error",
    }:
        st.session_state["browser_location_status"] = location_result_status
    elif (
        location_result_status == "granted"
        and not st.session_state.get("browser_location_lookup_done", False)
    ):
        nearby_observation = load_nearby_station_observation(
            browser_location_result.get("latitude"),
            browser_location_result.get("longitude"),
        )
        st.session_state["browser_location_lookup_done"] = True
        st.session_state["nearby_station_observation"] = nearby_observation
        st.session_state["browser_location_status"] = (
            "granted" if nearby_observation["available"] else "lookup_failed"
        )

if workspace_mode == HISTORICAL_MODE:
    system_status = validate_all_system_artifacts()
    try:
        daily_df = load_daily_air_quality()
        ml_df = load_ml_ready_dataset()
        results_df = load_model_results()
        predictions_df = load_test_predictions()
        importance_df = load_feature_importance()
        models_dict = load_trained_models()
    except Exception as e:
        data_load_error = str(e)

page = st.session_state["active_page"]
selected_station = None
if workspace_mode == HISTORICAL_MODE and data_load_error is None:
    available_stations = sorted(daily_df["location_name"].unique())
    selected_station = st.selectbox(
        "Historical location",
        available_stations,
        key="historical_station",
        on_change=_on_historical_station_change,
    )


# =============================================================================
# 4. MODE-SPECIFIC PAGE CONTENT
# =============================================================================

if st.session_state["primary_nav"] != "About" and workspace_mode == HISTORICAL_MODE and data_load_error:
    st.error(f"Historical data could not be loaded: {data_load_error}")
    st.info("The Recent Air Quality mode remains separate and does not depend on the historical model artifacts.")
    st.stop()

if st.session_state["primary_nav"] == "About":
    render_page_header(
        ["SaansCheck", "Learn"],
        "Learn About AQI",
        "A beginner-friendly guide to the historical records, recent source readings, and model forecasts shown in SaansCheck.",
    )
    st.markdown("### What AQI means")
    st.write(
        "An Air Quality Index (AQI) is a summary scale that makes air-quality "
        "information easier to communicate and compare at a glance. Air pollution "
        "can affect health, but this page does not assess any individual's exposure. "
        "SaansCheck's historical AQI is the value supplied by its dataset; "
        "the source's calculation method has not "
        "been verified as CPCB."
    )
    st.markdown("### Pollutants in the historical dataset")
    st.write(
        "**PM2.5** and **PM10** refer to particle-size groups. **NO2** (nitrogen "
        "dioxide), **SO2** (sulfur dioxide), **CO** (carbon monoxide), and **O3** "
        "(ozone) are pollutant gases. These labels describe the fields available "
        "in the dataset; pollutant units are not verified and are not inferred here."
    )
    st.markdown("### How to read SaansCheck's historical categories")
    st.caption(
        "These are the dashboard's existing category labels and display ranges, "
        "not a claim that the dataset uses an official CPCB AQI method."
    )
    st.dataframe(
        pd.DataFrame({
            "Dashboard label": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
            "Dashboard AQI range": ["0–50", "51–100", "101–200", "201–300", "301–400", "Above 400"],
        }),
        width="stretch",
        hide_index=True,
    )
    st.markdown("### Historical, recent, and forecast values are different")
    st.write(
        "**Historical observations** are daily rows in the project's 2019–2023 "
        "dataset and use its dataset-defined AQI. **Recent Air Quality** shows "
        "values reported by WAQI when a valid source observation is available; "
        "the app does not relabel that AQI as CPCB. **Model predictions** are "
        "estimates generated by saved models from historical features; they are "
        "not measurements or current conditions."
    )
    st.markdown("### What the 1–3 day forecast and MAE mean")
    st.write(
        "The historical model was trained to predict the dataset-defined AQI at "
        "one, two, and three days ahead from historical inputs. MAE (mean absolute "
        "error) is the average absolute difference (distance, ignoring whether a "
        "prediction was high or low) between predicted and observed target values, "
        "expressed in the target's AQI points. It is not "
        "an accuracy percentage and does not guarantee the error for a particular day."
    )
    st.info(
        "The Recent Air Quality Forecast page currently shows predictions as unavailable: "
        "a single WAQI reading cannot supply the model's historical lags, and the saved "
        "model's AQI scale and pollutant units have not been verified as compatible. "
        "Fresh WAQI observations at AQI 101 or above may trigger an observed-data alert "
        "on the provider's US EPA 2016 scale; forecasts never trigger that alert."
    )
    st.info(
        "The historical dataset's AQI provenance and pollutant units are not "
        "verified. WAQI documents its reported AQI on the US EPA 2016 scale; "
        "that is not India's CPCB/NAQI scale, and WAQI pollutant units are shown "
        "only when explicitly provided. Forecasts are estimates, coverage is "
        "limited to the available historical records, and SaansCheck is not medical advice."
    )
    st.markdown('<div class="footer-text">SaansCheck · Breathe smarter. Understand better.</div>', unsafe_allow_html=True)
    st.stop()

if workspace_mode == RECENT_MODE:
    automatic_location = (
        st.session_state["recent_location_mode"] == "Automatic nearby"
    )
    requested_location = (
        "Nearby area"
        if automatic_location
        else st.session_state["location_bar_station"]
    )
    station_search = recent_station_search
    selected_station_id = st.session_state.get("recent_station_uid")
    selected_station = next(
        (
            station for station in station_search["stations"]
            if station["station_id"] == selected_station_id
        ),
        None,
    )
    nearby_observation = st.session_state.get("nearby_station_observation")
    if automatic_location and isinstance(nearby_observation, dict):
        observation = nearby_observation
        selected_station_id = observation.get("station_id")
    elif not automatic_location and selected_station is not None:
        observation = load_station_observation(
            requested_location,
            selected_station["station_id"],
        )
    else:
        station_search_error = station_search["error"]
        if automatic_location:
            geo_status = st.session_state["browser_location_status"]
            if geo_status in {"not_requested", "requesting"}:
                station_search_error = (
                    "Waiting for browser location permission. Your approximate "
                    "area is used only to look up a nearby WAQI station."
                )
            elif geo_status == "denied":
                station_search_error = (
                    "Location permission was denied. Select Manual city/station "
                    "in the sidebar to continue with a city you choose."
                )
            elif geo_status == "unsupported":
                station_search_error = (
                    "This browser does not support location access. Select Manual "
                    "city/station in the sidebar to choose a city and station."
                )
            elif geo_status in {"unavailable", "timeout", "invalid", "error"}:
                station_search_error = (
                    "Browser location could not be verified. Select Manual "
                    "city/station in the sidebar to choose a city and station."
                )
            elif isinstance(nearby_observation, dict):
                station_search_error = nearby_observation.get("error")
        elif selected_station is None and station_search["stations"]:
            station_search_error = (
                f"Choose one of the {len(station_search['stations'])} matching "
                f"WAQI stations for {requested_location} in the sidebar to load "
                "its live AQI and observation timestamp."
            )
        observation = {
            "requested_location": requested_location,
            "api_available": station_search["api_available"],
            "status_code": station_search["status_code"],
            "available": False,
            "error": station_search_error or (
                f"Choose a WAQI monitoring station for {requested_location}."
            ),
            "source": SOURCE_LABEL,
            "aqi_label": AQI_SCALE_LABEL,
            "aqi_scale_url": AQI_SCALE_URL,
            "aqi": None,
            "aqi_source_value": None,
            "aqi_display": "Not provided by source",
            "station_name": None,
            "station_id": None,
            "attributions": [],
            "observed_at": None,
            "fetched_at": None,
            "cached": False,
            "age_minutes": None,
            "freshness": None,
            "pollutants": {
                pollutant: {"value": None, "unit": "Unit not specified by source"}
                for pollutant in POLLUTANT_FIELDS
            },
            "warnings": [],
        }
    observation = refresh_observation_freshness(observation)
    cpcb_observation = (
        load_cpcb_national_aqi(
            requested_location,
            observation.get("station_name"),
        )
        if page == "Recent Overview"
        else None
    )
    live_forecast_status = build_live_forecast_status(
        observation,
        expected_station_id=selected_station_id,
    )
    active_alert, _ = update_alert_session_state(
        st.session_state,
        build_observed_aqi_alert(observation),
    )
    _render_observed_aqi_alert(active_alert)
    freshness_tone = {
        "FRESH": "green",
        "DELAYED": "teal",
        "STALE": "amber",
    }.get(observation["freshness"], "neutral")
    observed_at = observation["observed_at"]
    timestamp_label = (
        observed_at.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(
            sep=" ", timespec="minutes"
        )
        if observed_at is not None
        else "Not provided"
    )
    fetched_at = observation["fetched_at"]
    fetched_label = (
        fetched_at.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(
            sep=" ", timespec="minutes"
        )
        if fetched_at is not None
        else "Not available"
    )
    freshness_label = observation["freshness"] or (
        "TIMESTAMP UNAVAILABLE" if observation["available"] else "UNAVAILABLE"
    )
    if observation["cached"]:
        freshness_label = "CACHED · " + freshness_label
    age_minutes = observation["age_minutes"]
    age_label = (
        f"{age_minutes:.0f} minutes"
        if age_minutes is not None and age_minutes < 60
        else f"{age_minutes / 60:.1f} hours"
        if age_minutes is not None
        else "Not available"
    )
    api_status = (
        f"Available · HTTP {observation['status_code']}"
        if observation["api_available"] and observation["status_code"] is not None
        else "Unavailable"
    )
    if observation["available"] and age_minutes is not None:
        freshness_detail = f"Observed {age_label} ago"
    elif observation["available"]:
        freshness_detail = "WAQI observation timestamp unavailable"
    else:
        freshness_detail = (
            observation["error"] or "No valid station observation returned."
        )
    if observation["cached"]:
        st.info(
            "Showing a previously retrieved WAQI response from the short-term "
            f"cache (retrieved {fetched_label}); this rerun did not fetch a new reading."
        )
    if observation["available"] and observation["freshness"] == "STALE":
        st.warning(
            "Fresh WAQI data is unavailable for this station. Current conditions "
            "are not confirmed because the returned observation is stale."
        )
    if observation["available"] and observed_at is None:
        st.warning(
            "WAQI did not provide a usable observation timestamp; the reading's age is unknown."
        )
    if observation["available"] and observation["aqi"] is None:
        st.warning(
            "WAQI returned this station feed without a valid AQI value. "
            "No AQI has been calculated from pollutant readings."
        )

    if page == "Recent Overview":
        render_page_header(
            ["SaansCheck", RECENT_MODE, "Overview"],
            "Recent Air Quality",
            (
                f"WAQI-reported observation for {requested_location}"
                + (
                    f" at {observation['station_name']}"
                    if observation["station_name"]
                    else ""
                )
                + ". Separate from historical ML forecasting."
            ),
            [(freshness_label, freshness_tone)],
        )
        _render_aqi_risk_scale(cpcb_observation, observation)
        if automatic_location:
            if observation["available"]:
                distance_km = observation.get("distance_km")
                distance_detail = (
                    f"WAQI geolocalized station · {distance_km:g} km away"
                    if distance_km is not None
                    else "WAQI geolocalized station"
                )
                st.caption(
                    f"Nearby station verified from WAQI station identity and "
                    f"coordinates · {distance_detail}. Browser coordinates are "
                    "not retained after this lookup."
                )
            else:
                st.info(
                    "SaansCheck requests approximate browser location to find a "
                    "nearby WAQI monitoring station. Your browser controls this "
                    "permission; location lookup is optional."
                )
                if st.session_state["browser_location_status"] == "denied":
                    st.warning(
                        "Browser location permission was denied. Recent Air "
                        "Quality remains available through Manual city/station."
                    )
                elif st.session_state["browser_location_status"] == "unsupported":
                    st.warning(
                        "This browser does not support geolocation. Use Manual "
                        "city/station to choose an area."
                    )
                elif st.session_state["browser_location_status"] in {
                    "unavailable", "timeout", "invalid", "error", "lookup_failed",
                }:
                    st.warning(
                        "Automatic location lookup could not verify a nearby "
                        "WAQI station. Use Manual city/station to continue."
                    )
                if st.session_state["browser_location_status"] in {
                    "denied", "unsupported", "unavailable", "timeout",
                    "invalid", "error", "lookup_failed",
                }:
                    st.button(
                        "Try location again",
                        key="retry_browser_location",
                        on_click=_retry_browser_location,
                    )
                    st.caption(
                        "Or select Manual city/station in the sidebar and choose "
                        "a city and WAQI station. The current city selection is "
                        "preserved as the manual fallback."
                    )
        st.markdown("### Indian National AQI — CPCB")
        if cpcb_observation and cpcb_observation["available"]:
            cpcb_observed_at = cpcb_observation["observed_at"].astimezone(
                ZoneInfo("Asia/Kolkata")
            ).isoformat(sep=" ", timespec="minutes")
            render_kpi_cards([
                (
                    "Indian National AQI",
                    cpcb_observation["aqi_display"],
                    f"CPCB category: {cpcb_observation['category']}",
                ),
                (
                    "Monitoring station",
                    cpcb_observation["station"],
                    cpcb_observation["location"],
                ),
                ("AQI scale", cpcb_observation["scale"], cpcb_observation["source"]),
                (
                    "Observed at",
                    cpcb_observed_at,
                    f"Freshness: {cpcb_observation['freshness']}",
                ),
            ])
            cpcb_alert = build_cpcb_observed_alert(cpcb_observation)
            if cpcb_alert:
                st.warning(
                    f"CPCB National AQI Alert · {cpcb_alert['category']} "
                    f"(AQI {cpcb_alert['aqi']}) at {cpcb_alert['station']}. "
                    "Reduce prolonged outdoor exertion and follow local "
                    "public-health advice."
                )
            _render_cpcb_health_guidance(cpcb_observation)
        else:
            cpcb_error = (
                cpcb_observation["error"]
                if cpcb_observation
                else "No CPCB lookup was made on this page."
            )
            st.info(f"CPCB National AQI unavailable. {cpcb_error}")
            st.markdown(
                "[Open the official CPCB National AQI dashboard]"
                "(https://airquality.cpcb.gov.in/AQI/)"
            )
        st.markdown("### WAQI — US AQI (alternative source)")
        location_detail = (
            "Approximate browser location · WAQI-reported observation"
            if automatic_location
            else "WAQI exact-city station match"
        )
        render_kpi_cards([
            ("Selected location", requested_location, location_detail),
            (
                "Monitoring station",
                observation["station_name"] or "Not available",
                (
                    (
                        f"WAQI station · {observation['distance_km']:g} km away"
                        if observation.get("distance_km") is not None
                        else "Actual station name returned by WAQI"
                    )
                ),
            ),
            (
                observation["aqi_label"],
                observation["aqi_display"],
                "Direct WAQI station AQI; not calculated from pollutants",
            ),
            ("Observed at", timestamp_label, freshness_detail),
        ])
        render_kpi_cards([
            ("Data source", SOURCE_LABEL, "Station feed and provider attribution"),
            (
                "Retrieved at",
                fetched_label,
                "Cached response" if observation["cached"] else "WAQI response retrieval time",
            ),
        ])
        for pollutant_row in (("PM2.5", "PM10", "NO2"), ("SO2", "CO", "O3")):
            metrics = []
            for pollutant in pollutant_row:
                reading = observation["pollutants"][pollutant]
                value = reading["value"]
                value_label = (
                    f"{value:g}"
                    if value is not None
                    else "Not provided by source"
                )
                metrics.append((pollutant, value_label, reading["unit"]))
            render_kpi_cards(metrics)
        with st.expander("Compare Air Quality Sources", expanded=False):
            st.markdown(
                "AQI values can differ because providers may use different "
                "measurements, aggregation methods, reporting times or station mappings."
            )
            cpcb_column, waqi_column, aqi_in_column = st.columns(3)
            with cpcb_column:
                with st.container(border=True):
                    st.markdown("#### Indian National AQI — CPCB")
                    if cpcb_observation and cpcb_observation["available"]:
                        st.markdown(
                            f"**AQI: {cpcb_observation['aqi_display']} · "
                            f"{cpcb_observation['category']}**"
                        )
                        st.caption(
                            f"Station: {cpcb_observation['station']} · "
                            f"{cpcb_observation['location']}"
                        )
                        st.caption(
                            f"Observed at: {cpcb_observation['observed_at'].astimezone(ZoneInfo('Asia/Kolkata')).isoformat(sep=' ', timespec='minutes')}"
                        )
                        st.caption(f"Freshness: {cpcb_observation['freshness']}")
                    else:
                        reason = (
                            cpcb_observation["error"]
                            if cpcb_observation
                            else "Not queried on this page."
                        )
                        st.markdown("**CPCB National AQI unavailable**")
                        st.caption(reason)
                    st.caption(f"Scale: Indian National AQI · {SOURCE_LABEL}")
            with waqi_column:
                with st.container(border=True):
                    st.markdown("#### WAQI — US AQI")
                    st.markdown(f"**AQI: {observation['aqi_display']}**")
                    st.caption(f"Provider: {SOURCE_LABEL}")
                    st.caption(f"Scale: {AQI_SCALE_LABEL}")
                    st.caption(
                        "Provider-reported category: Not provided by this feed"
                    )
                    st.caption(
                        f"Station: {observation['station_name'] or 'Not available'}"
                    )
                    st.caption(f"Observed at: {timestamp_label}")
                    st.caption(f"Freshness: {freshness_label}")
            with aqi_in_column:
                with st.container(border=True):
                    st.markdown("#### AQI.in — AQI-US")
                    st.markdown("**AQI.in live data not connected**")
                    comparison_station = (
                        observation["station_name"] or requested_location
                    )
                    st.caption(f"Selected station: {comparison_station}")
                    st.caption(
                        "SaansCheck does not retrieve AQI.in values, category, "
                        "observation time or freshness."
                    )
                    normalized_station = comparison_station.casefold()
                    if (
                        "anand vihar" in normalized_station
                        and "delhi" in normalized_station
                        and requested_location.casefold() in {"delhi", "nearby area"}
                    ):
                        aqi_in_link = (
                            "https://www.aqi.in/in/dashboard/india/delhi/"
                            "new-delhi/anand-vihar"
                        )
                        link_label = "Open AQI.in Anand Vihar page"
                    else:
                        aqi_in_link = "https://www.aqi.in/"
                        link_label = "Open AQI.in to find this station"
                    st.markdown(
                        f"[{link_label}]({aqi_in_link})"
                    )
                    st.caption(
                        "Compare the provider's current reading there; it is not "
                        "automatically matched or displayed here."
                    )
        _render_health_guidance(observation)
        if not observation["available"]:
            render_empty_state(
                f"Fresh {requested_location} data unavailable",
                observation["error"] or "WAQI did not return a valid station observation.",
            )
        st.caption(
            f"AQI scale: [{AQI_SCALE_LABEL}]({AQI_SCALE_URL}). "
            "WAQI's documented scale is US EPA 2016, not India's CPCB/NAQI scale. "
            "Pollutant units are shown only when supplied by WAQI."
        )

    elif page == "Pollutant Trends":
        render_page_header(
            ["SaansCheck", RECENT_MODE, page],
            "Pollutant Trends",
            (
                f"Latest WAQI station snapshot for {requested_location}; "
                "this is not a historical time series."
            ),
            [("Source-provided readings", "teal")],
        )
        if observation["available"]:
            metrics = []
            for pollutant in POLLUTANT_FIELDS:
                reading = observation["pollutants"][pollutant]
                value = reading["value"]
                metrics.append((
                    pollutant,
                    f"{value:g}" if value is not None else "Not provided by source",
                    reading["unit"],
                ))
            render_kpi_cards(metrics[:3])
            render_kpi_cards(metrics[3:])
            st.caption(
                f"Station: {observation['station_name'] or 'Not provided by source'} · "
                f"Observed: {timestamp_label} · Retrieved: {fetched_label}. "
                "No values are filled, converted, or inferred."
            )
        else:
            render_empty_state(
                "Pollutant readings unavailable",
                observation["error"] or "WAQI did not return a valid station observation.",
            )

    elif page == "Location Data":
        render_page_header(
            ["SaansCheck", RECENT_MODE, page],
            "Recent Location Data",
            f"WAQI station selected for {requested_location}; its returned name is shown without relabeling.",
            [("WAQI station feed", "teal")],
        )
        if observation["available"]:
            render_kpi_cards([
                ("Requested location", requested_location, "City matched against station name"),
                ("Station", observation["station_name"] or "Not provided by source", "Actual WAQI name"),
                ("Station identifier", observation["station_id"] or "Not provided by source", "Public feed identifier"),
                ("Observed at", timestamp_label, freshness_label),
                ("Data source", SOURCE_LABEL, "WAQI station feed"),
                (
                    "Retrieved at",
                    fetched_label,
                    "Cached response" if observation["cached"] else "Response retrieval time",
                ),
            ])
            attributions = observation["attributions"]
            st.markdown("#### Source attribution")
            if attributions:
                for attribution in attributions:
                    name = attribution["name"]
                    url = attribution["url"]
                    st.write(f"{name} ({url})" if url else name)
            else:
                st.write("Attribution details not provided by source.")
        else:
            render_empty_state(
                "Location data unavailable",
                observation["error"] or f"No station details were returned for {requested_location}.",
            )

    elif page == "Data Freshness":
        render_page_header(
            ["SaansCheck", RECENT_MODE, page],
            "Data Freshness",
            (
                "Freshness uses the observation timestamp returned by WAQI; "
                "retrieval time is shown separately."
            ),
            [(freshness_label, freshness_tone)],
        )
        render_kpi_cards([
            ("Requested location", requested_location, observation["station_name"] or "Station not selected"),
            ("Last observation time", timestamp_label, "WAQI source timestamp"),
            ("Observation age", age_label, "Calculated from WAQI timestamp"),
            ("Freshness status", freshness_label, "FRESH ≤60 min · DELAYED ≤6 hr · STALE >6 hr"),
            (
                "Data retrieved",
                fetched_label,
                "Cached response" if observation["cached"] else "WAQI response retrieval time",
            ),
            ("API availability", api_status, observation["error"] or SOURCE_LABEL),
        ])
        if not observation["available"]:
            render_empty_state(
                "Fresh data unavailable",
                observation["error"] or f"WAQI did not return a valid {requested_location} observation.",
            )

    elif page == "Forecast":
        render_page_header(
            ["SaansCheck", RECENT_MODE, "Forecast"],
            "Station Forecast",
            "Observed WAQI values and forecast estimates are separate. Predictions remain unavailable until compatible authorized history and a validated model exist.",
            [("Forecast unavailable", "amber"), ("WAQI observation", "teal")],
        )
        observed_aqi = observation.get("aqi")
        observed_value = (
            f"{float(observed_aqi):g}"
            if observation.get("available") and observed_aqi is not None
            else "Unavailable"
        )
        observed_label = (
            "Today's observed AQI"
            if live_forecast_status["observation_is_today"]
            else "Latest observed AQI"
        )
        observed_date_detail = (
            "Observation timestamp falls on today's Asia/Kolkata date"
            if live_forecast_status["observation_is_today"]
            else "Not confirmed as today's observation; see source timestamp"
        )

        def _target_label(value):
            return value.strftime("%Y-%m-%d") if value is not None else "Date unavailable"

        target_dates = live_forecast_status["target_dates"]
        render_kpi_cards([
            (
                observed_label,
                observed_value,
                f"{observation.get('aqi_label', AQI_SCALE_LABEL)} · "
                f"{timestamp_label} · {observed_date_detail}",
            ),
            (
                "Today's estimated AQI",
                "Unavailable",
                live_forecast_status["today_estimate_reason"],
            ),
            (
                "Tomorrow's predicted AQI",
                "Unavailable",
                _target_label(target_dates["tomorrow"]),
            ),
            (
                "Day after tomorrow's predicted AQI",
                "Unavailable",
                _target_label(target_dates["day_after"]),
            ),
        ])
        forecast_origin = live_forecast_status["forecast_origin"]
        origin_label = (
            forecast_origin.strftime("%Y-%m-%d %H:%M %Z")
            if forecast_origin is not None
            else "Unavailable: no valid timestamped observation"
        )
        render_kpi_cards([
            ("Selected station", observation.get("station_name") or "Not selected", requested_location),
            ("Forecast origin", origin_label, "Latest source observation · Asia/Kolkata"),
            ("Data source", observation.get("source") or SOURCE_LABEL, "Direct selected-station feed"),
            ("Data freshness", freshness_label, "Based on provider observation timestamp"),
        ])
        st.warning(live_forecast_status["reason"])
        st.info(live_forecast_status["compatibility_reason"])
        if observation.get("freshness") in {"STALE", "DELAYED"}:
            st.warning(
                f"The selected WAQI observation is {observation['freshness'].lower()}; "
                "it is shown as an observation, not as confirmed current conditions."
            )
        st.caption(
            "WAQI AQI is reported on the US EPA 2016 scale, not India's CPCB/NAQI scale. "
            "The saved Random Forest was trained on historical CSV features whose AQI provenance "
            "and pollutant units/averaging are not verified against WAQI. No prediction or same-day "
            "estimate is computed from those unverified inputs. Stored 2023 scores also have "
            "horizon-target overlap across the training/test year boundary and are not accepted "
            "as live forecast validation."
        )

    st.markdown('<div class="footer-text">SaansCheck · Breathe smarter. Understand better.</div>', unsafe_allow_html=True)
    st.stop()

if page == "Overview":
    _render_mode_cards()
    render_page_header(
        ["SaansCheck", HISTORICAL_MODE, "Overview"],
        "Historical Air Quality Overview",
        "Explore Delhi-NCR air-quality patterns from the historical dataset.",
        [("Historical data", "teal"), ("2019–2023", "green")],
    )
    station_daily = get_station_records(daily_df, selected_station)
    latest_record = station_daily.iloc[-1]
    latest_date = pd.to_datetime(latest_record["date"]).strftime("%Y-%m-%d")
    station_averages = daily_df.groupby("location_name")["aqi"].mean().sort_values()
    best_station = station_averages.index[0]
    highest_risk_station = station_averages.index[-1]
    render_kpi_cards([
        ("Latest Available Historical AQI", f"{float(latest_record['aqi']):.1f}", f"{latest_record['aqi_category']} · {latest_date} · {selected_station}"),
        ("Average AQI", f"{station_daily['aqi'].mean():.1f}", f"Selected location · {selected_station}"),
        ("Lowest mean AQI location", best_station, f"Historical mean · {station_averages.loc[best_station]:.1f}"),
        ("Highest mean AQI location", highest_risk_station, f"Historical mean · {station_averages.loc[highest_risk_station]:.1f}"),
    ])

    st.markdown("---")
    st.subheader("Key Findings")
    years = available_history_years(daily_df)
    period_options = ["All available years"] + [str(year) for year in years]
    selected_period = st.selectbox(
        "Key Findings period",
        period_options,
        key="overview_findings_year",
        help="The findings below use only the selected location and period.",
    )
    findings_year = int(selected_period) if selected_period != "All available years" else None
    findings = build_key_findings(daily_df, selected_station, year=findings_year)
    period_detail = (
        selected_period
        if findings_year is not None
        else f"{daily_df['date'].min()} to {daily_df['date'].max()}"
    )
    st.caption(
        f"Dataset-defined historical AQI · {selected_station} · {period_detail} · "
        f"{findings['count']} valid daily AQI observations. This is not a live reading."
    )
    if findings["count"] == 0:
        st.info("No valid historical AQI observations are available for this location and period.")
    else:
        trend_value = (
            f"{findings['trend_change']:+.1f}"
            if findings["trend_change"] is not None
            else "Not enough dates"
        )
        trend_detail = (
            (
                f"{findings['first_date']:%Y-%m-%d} to "
                f"{findings['latest_date']:%Y-%m-%d}; endpoint difference only"
            )
            if findings["trend_change"] is not None
            else "At least two distinct observation dates are needed"
        )
        render_kpi_cards([
            (
                "Highest observed historical AQI",
                f"{findings['maximum_aqi']:.1f}",
                f"{findings['maximum_date']:%Y-%m-%d}",
            ),
            (
                "Lowest observed historical AQI",
                f"{findings['minimum_aqi']:.1f}",
                f"{findings['minimum_date']:%Y-%m-%d}",
            ),
            (
                "Latest observation in selection",
                f"{findings['latest_aqi']:.1f}",
                f"Historical date · {findings['latest_date']:%Y-%m-%d}",
            ),
            ("First-to-last change", trend_value, trend_detail),
        ])
        if findings["most_common_category"]:
            st.caption(
                "Most frequent stored category in this selection: "
                f"{findings['most_common_category']}. Categories are dataset labels."
            )

    st.markdown("---")
    st.markdown('<div class="section-kicker">AQI status</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">AQI Status — {selected_station}</div>', unsafe_allow_html=True)

    scale = [
        ("Good", 0, 50),
        ("Satisfactory", 51, 100),
        ("Moderate", 101, 200),
        ("Poor", 201, 300),
        ("Very Poor", 301, 400),
        ("Severe", 401, 999)
    ]
    aqi_value = float(latest_record["aqi"])
    current_category = latest_record["aqi_category"]
    status_color = {"Good": "#2e9d68", "Satisfactory": "#0f8b8d", "Moderate": "#d99a2b", "Poor": "#d9534f", "Very Poor": "#b7443e", "Severe": "#7a2d2b"}[current_category]

    st.markdown(f"""
    <div class="chart-frame">
      <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap;">
        <div>
          <div style="font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:#687983;font-weight:700;">AQI Status</div>
          <div class="sc-historical-aqi-value">{aqi_value:.1f}</div>
        </div>
        <div style="padding:0.65rem 0.9rem;border-radius:999px;background:{status_color}1a;border:1px solid {status_color}33;color:{status_color};font-weight:800;letter-spacing:0.08em;text-transform:uppercase;">{current_category}</div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:1rem;">
      {''.join(f'<span style="display:inline-block;padding:0.38rem 0.7rem;border-radius:999px;border:1px solid rgba(23,35,45,0.08);background:{"#e5f5ec" if label == current_category else "#f8fafb"};color:{"#2e9d68" if label == current_category else "#687983"};font-size:0.7rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;">{label}</span>' for label, _, _ in scale)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-kicker">AQI trend</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Historical AQI Trend</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle-note">Daily dataset AQI • Selected Location • 2019–2023</div>', unsafe_allow_html=True)

    trend_df = station_daily.set_index("date")[["aqi"]].copy()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(trend_df.index, trend_df["aqi"], color="#0f8b8d", linewidth=2.6)
    ax.fill_between(trend_df.index, trend_df["aqi"], 0, alpha=0.1, color="#0f8b8d")
    ax.set_title(f"{selected_station} AQI History", color="#17232d", fontsize=11, pad=12)
    ax.set_ylabel("AQI", color="#475d6b")
    ax.set_xlabel("Date", color="#475d6b")
    ymax = max(300.0, float(trend_df["aqi"].max()) * 1.15)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
    ax.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#dfe7eb")
    ax.tick_params(colors="#475d6b")
    fig.tight_layout()
    _render_chart(fig)

    st.markdown("---")
    insight = station_daily["aqi"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="insight-card"><h4>Average AQI</h4><div class="value">%.1f</div></div>' % insight.mean(), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="insight-card"><h4>Median AQI</h4><div class="value">%.1f</div></div>' % insight.median(), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="insight-card"><h4>Peak AQI</h4><div class="value">%.1f</div></div>' % insight.max(), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-kicker">Pollutant summary</div>', unsafe_allow_html=True)
    st.caption("Historical source units are not verified; values are shown as supplied.")
    pollutant_summary = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
        "Latest value": [
            float(latest_record["pm25"]), float(latest_record["pm10"]),
            float(latest_record["no2"]), float(latest_record["so2"]),
            float(latest_record["co"]), float(latest_record["o3"])
        ],
        "Unit": ["Unverified"] * 6
    })
    st.dataframe(pollutant_summary.assign(Trend=["-","-","-","-","-","-"]), width="stretch", hide_index=True)

    overview_columns = st.columns(2)
    with overview_columns[0]:
        st.markdown("#### Historical location comparison")
        station_compare = station_averages.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.barh(station_compare.index, station_compare.values, color="#0f8b8d")
        ax.invert_yaxis()
        ax.set_xlabel("Mean historical AQI")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)
        fig.tight_layout()
        _render_chart(fig)
    with overview_columns[1]:
        st.markdown("#### AQI category distribution")
        category_counts = daily_df["aqi_category"].value_counts().reindex(AQI_CATEGORY_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.bar(category_counts.index, category_counts.values, color="#2e9d68")
        ax.set_ylabel("Daily records")
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        _render_chart(fig)

    st.warning("Historical dataset available through 31 December 2023. Current/live AQI is not represented in this section.")

elif page == "Daily AQI Calendar":
    render_page_header(
        ["SaansCheck", HISTORICAL_MODE, page],
        "Daily Historical AQI Calendar",
        "Explore observed daily dataset AQI by location and month. Missing days are shown separately from low AQI values.",
        [("Historical dataset", "teal")],
    )
    calendar_locations = sorted(daily_df["location_name"].dropna().astype(str).unique())
    calendar_years = available_history_years(daily_df)
    control_columns = st.columns(3)
    with control_columns[0]:
        calendar_location = st.selectbox(
            "Location",
            calendar_locations,
            key="calendar_location",
        )
    with control_columns[1]:
        calendar_year = st.selectbox(
            "Year",
            calendar_years,
            key="calendar_year",
        )
    with control_columns[2]:
        calendar_month = st.selectbox(
            "Month",
            list(range(1, 13)),
            format_func=lambda month: calendar.month_name[month],
            key="calendar_month",
        )

    month_data = build_month_calendar(
        daily_df,
        calendar_location,
        calendar_year,
        calendar_month,
    )
    category_colors = {
        "Good": "#2e9d68",
        "Satisfactory": "#0f8b8d",
        "Moderate": "#d99a2b",
        "Poor": "#e58259",
        "Very Poor": "#d9534f",
        "Severe": "#7a2d2b",
    }
    missing_color = "#edf1f4"
    fig, ax = plt.subplots(figsize=(11, 5.2))
    weeks = month_data["weeks"]
    for week_index, week in enumerate(weeks):
        y_position = len(weeks) - week_index - 1
        for weekday, day in enumerate(week):
            if day == 0:
                continue
            value = month_data["values"].get(day)
            has_reading = day in month_data["observed_days"] and value is not None
            color = (
                category_colors[aqi_category_from_value(value)]
                if has_reading
                else missing_color
            )
            ax.add_patch(
                Rectangle(
                    (weekday, y_position),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="#ffffff",
                    linewidth=2,
                )
            )
            reading_label = f"{value:.1f}" if has_reading else ("—" if day in month_data["observed_days"] else "·")
            ax.text(
                weekday + 0.5,
                y_position + 0.5,
                f"{day}\n{reading_label}",
                ha="center",
                va="center",
                fontsize=8,
                color="#17232d",
                fontweight="medium",
            )
    ax.set_xlim(0, 7)
    ax.set_ylim(0, len(weeks))
    ax.set_xticks(np.arange(7) + 0.5)
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_yticks(np.arange(len(weeks)) + 0.5)
    ax.set_yticklabels([f"Week {index + 1}" for index in reversed(range(len(weeks)))])
    ax.set_title(
        f"{calendar_location} · {calendar.month_name[calendar_month]} {calendar_year}",
        color="#17232d",
    )
    ax.tick_params(length=0, colors="#475d6b")
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend_items = [
        Patch(facecolor=color, label=label)
        for label, color in category_colors.items()
    ]
    legend_items.append(Patch(facecolor=missing_color, label="No AQI reading"))
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    _render_chart(fig, width="stretch")
    st.caption(
        "The color bands use SaansCheck's existing dataset category mapping; "
        "they do not verify CPCB methodology. Gray cells indicate no numeric "
        "AQI reading for that date. Exact dates and values are listed below."
    )
    if month_data["records"].empty:
        st.info("No historical rows are available for this location and month.")
    else:
        exact_values = month_data["records"].copy()
        exact_values["date"] = exact_values["date"].dt.strftime("%Y-%m-%d")
        exact_values = exact_values.rename(columns={"date": "Date", "aqi": "Dataset AQI"})
        st.dataframe(exact_values, width="stretch", hide_index=True)

elif page == "AQI Forecast":
    render_page_header(
        ["SaansCheck", HISTORICAL_MODE, "AQI Forecast"],
        "AQI Forecast",
        "Historical forecast demonstration using the saved Random Forest models.",
        [("Historical demonstration", "amber"), ("Random Forest", "teal")],
    )

    station_daily = get_station_records(daily_df, selected_station)
    forecast_rows = build_forecast_for_station(daily_df, selected_station, models_dict)
    historical_data_end = pd.to_datetime(daily_df["date"]).max()
    forecast_origin = pd.to_datetime(station_daily["date"]).max()
    target_start = min(row["date"] for row in forecast_rows)
    target_end = max(row["date"] for row in forecast_rows)
    st.warning(
        "Historical forecast demonstration — not a current forecast. "
        f"The historical dataset ends {historical_data_end:%B %d, %Y}. "
        f"This forecast uses observations through {forecast_origin:%B %d, %Y} "
        f"and targets {target_start:%B %d}–{target_end:%B %d, %Y}."
    )
    st.caption(
        "AQI is dataset-defined; the source AQI calculation method has not been "
        "verified. These historical model outputs are not an official forecast."
    )
    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["predicted_aqi"] = forecast_df["predicted_aqi"].round(2)
    forecast_summary = forecast_df[["horizon", "date", "predicted_aqi", "category"]].copy()
    forecast_summary["date"] = forecast_summary["date"].dt.strftime("%Y-%m-%d")

    cards = st.columns(3)
    for idx, horizon in enumerate(["Day +1", "Day +2", "Day +3"]):
        row = forecast_df[forecast_df["horizon"] == horizon].iloc[0]
        with cards[idx]:
            st.markdown(f"""
            <div class="kpi-card sc-forecast-card">
                <div class="kpi-header">{horizon}</div>
                <div class="kpi-value">{row['predicted_aqi']:.1f}</div>
                <div class="kpi-meta"><strong>{row['category']}</strong> • {row['date'].strftime('%Y-%m-%d')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-kicker">Forecast trend</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(forecast_df))
    y = forecast_df["predicted_aqi"].to_numpy()
    hist_line = np.array([float(station_daily["aqi"].iloc[-1])] * len(forecast_df))
    ax.plot(x, hist_line, color="#a2b7bf", linestyle="--", linewidth=1.8, alpha=0.9, label="Last observed historical AQI")
    ax.plot(x, y, color="#0f8b8d", linewidth=2.8, marker='o', markersize=6, label="Forecast AQI")
    ax.set_xticks(x)
    ax.set_xticklabels([row["horizon"] for _, row in forecast_df.iterrows()])
    ax.set_ylabel("Predicted AQI", color="#475d6b")
    ax.set_ylim(0, max(300.0, float(y.max()) * 1.2, 100.0))
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
    ax.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    for spine in ax.spines.values():
        spine.set_color("#dfe7eb")
    ax.tick_params(colors="#475d6b")
    fig.tight_layout()
    _render_chart(fig)

    trend_delta = y[-1] - y[0]
    if abs(trend_delta) < 2:
        trend_text = "Stable"
    elif trend_delta > 0:
        trend_text = "Increasing"
    else:
        trend_text = "Decreasing"

    st.markdown("---")
    st.markdown('<div class="section-kicker">Forecast interpretation</div>', unsafe_allow_html=True)
    interpretation_columns = st.columns(3)
    for column, label, value in zip(
        interpretation_columns,
        ("Forecast trend", "Forecast range", "Location"),
        (trend_text, f"{y.min():.1f} to {y.max():.1f}", selected_station),
    ):
        with column:
            st.markdown(
                '<div class="metric-box">'
                f'<div class="sc-kpi-label">{label}</div>'
                f'<div class="sc-kpi-value">{value}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-kicker">Model note</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-box">
        <div style="display:flex;flex-wrap:wrap;gap:0.7rem;align-items:center;">
            <span class="pill">Model used</span>
            <span class="sc-forecast-model-name">Random Forest Regressor</span>
        </div>
        <div class="sc-forecast-model-detail">
            Forecast horizon: 1–3 days<br>
            Evaluation: 2023 historical test origins; temporal target-boundary overlap is a known limitation.<br>
            Forecast origin: {forecast_origin:%Y-%m-%d}<br>
            This forecast is generated from the trained historical ML model and is not an official government forecast.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Forecast table")
    st.dataframe(forecast_summary, width="stretch", hide_index=True)
    _render_forecast_reliability_note()

elif page == "Location Intelligence":
    render_page_header(
        ["SaansCheck", HISTORICAL_MODE, "Location Intelligence"],
        "Location Intelligence",
        "Compare historical AQI characteristics across the Delhi-NCR location labels in this dataset.",
        [("Historical location data", "teal")],
    )

    comparison_controls = st.columns(2)
    with comparison_controls[0]:
        year_options = ["All available years"] + [
            str(year) for year in available_history_years(daily_df)
        ]
        selected_year_label = st.selectbox(
            "Comparison period",
            year_options,
            key="location_comparison_year",
        )
    with comparison_controls[1]:
        metric_label = st.selectbox(
            "Leaderboard metric",
            ["Average AQI", "Maximum AQI"],
            key="location_leaderboard_metric",
            help="Rankings change when the selected period or metric changes.",
        )
    comparison_year = (
        int(selected_year_label)
        if selected_year_label != "All available years"
        else None
    )
    ranking_metric = "mean" if metric_label == "Average AQI" else "max"
    ranking_column = "Average_AQI" if ranking_metric == "mean" else "Maximum_AQI"
    station_summary = build_location_leaderboard(
        daily_df,
        year=comparison_year,
        metric=ranking_metric,
    )
    comparison_data = daily_df.copy()
    comparison_data["date"] = pd.to_datetime(comparison_data["date"], errors="coerce")
    comparison_data["aqi"] = pd.to_numeric(comparison_data["aqi"], errors="coerce")
    if comparison_year is not None:
        comparison_data = comparison_data.loc[
            comparison_data["date"].dt.year.eq(comparison_year)
        ]

    period_text = selected_year_label
    st.caption(
        f"Historical dataset locations · {period_text} · ranking by {metric_label}. "
        "Locations are dataset labels; a ranking is not a universal pollution claim."
    )
    metric_cols = st.columns(max(1, min(5, len(station_summary))))
    for idx, row in station_summary.iterrows():
        with metric_cols[idx % len(metric_cols)]:
            rank_label = f"#{int(row['Rank'])}" if pd.notna(row["Rank"]) else "N/A"
            value_label = f"{row[ranking_column]:.1f}" if pd.notna(row[ranking_column]) else "N/A"
            st.markdown(f"""
            <div class="kpi-card sc-location-rank-card">
                <div class="kpi-header">{row['location_name']}</div>
                <div class="kpi-value" style="font-size:1.5rem;">{value_label}</div>
                <div class="kpi-meta"><strong>Rank</strong> {rank_label} · {int(row['Days_Recorded'])} valid days</div>
            </div>
            """, unsafe_allow_html=True)

    leaderboard = station_summary[[
        "Rank",
        "location_name",
        ranking_column,
        "Days_Recorded",
        "Coverage_Start",
        "Coverage_End",
    ]].copy()
    leaderboard = leaderboard.rename(columns={
        "Rank": "Rank",
        "location_name": "Dataset location",
        ranking_column: metric_label,
        "Days_Recorded": "Valid AQI days",
        "Coverage_Start": "Coverage start",
        "Coverage_End": "Coverage end",
    })
    for coverage_column in ("Coverage start", "Coverage end"):
        leaderboard[coverage_column] = pd.to_datetime(
            leaderboard[coverage_column], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    leaderboard["Rank"] = leaderboard["Rank"].map(
        lambda rank: int(rank) if pd.notna(rank) else "N/A"
    )
    st.dataframe(leaderboard, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-kicker">Location comparison</div>', unsafe_allow_html=True)
    available_comparison_locations = sorted(
        comparison_data["location_name"].dropna().astype(str).unique()
    )
    compare_default = (
        [selected_station]
        if selected_station in available_comparison_locations
        else []
    )
    compare_stations = st.multiselect(
        "Choose locations to compare",
        available_comparison_locations,
        default=compare_default,
        key="location_comparison_locations",
        help="Choose dataset locations within the selected comparison period.",
    )
    if compare_stations:
        compare_df = station_summary.loc[
            station_summary["location_name"].isin(compare_stations)
        ].sort_values(ranking_column, ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        chart_colors = chart_palette(st.session_state["appearance_theme"])
        bars = ax.barh(
            compare_df["location_name"],
            compare_df[ranking_column],
            color="#0f8b8d",
            edgecolor=chart_colors["line"],
        )
        ax.bar_label(
            bars,
            fmt="%.1f",
            padding=4,
            color=chart_colors["text"],
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel(metric_label, color=chart_colors["muted"])
        ax.set_ylabel("Dataset location", color=chart_colors["muted"])
        maximum_value = pd.to_numeric(
            compare_df[ranking_column], errors="coerce"
        ).max()
        ax.set_xlim(0, float(maximum_value) * 1.14 if pd.notna(maximum_value) and maximum_value > 0 else 1)
        ax.grid(True, alpha=0.12, axis='x')
        for spine in ax.spines.values():
            spine.set_color(chart_colors["line"])
        ax.tick_params(colors=chart_colors["muted"])
        fig.tight_layout()
        _render_chart(fig)

    st.markdown("---")
    selected_rows = station_summary.loc[
        station_summary["location_name"].eq(selected_station)
    ]
    if selected_rows.empty or int(selected_rows.iloc[0]["Days_Recorded"]) == 0:
        st.info(
            f"No historical AQI observations are available for {selected_station} "
            f"during {period_text}."
        )
    else:
        selected_station_summary = selected_rows.iloc[0]
        metrics_cols = st.columns(5)
        metrics = [
            ("Average AQI", selected_station_summary["Average_AQI"]),
            ("Median AQI", selected_station_summary["Median_AQI"]),
            ("Maximum AQI", selected_station_summary["Maximum_AQI"]),
            ("Valid AQI days", selected_station_summary["Days_Recorded"]),
            ("Leaderboard rank", selected_station_summary["Rank"]),
        ]
        for column, (label, value) in zip(metrics_cols, metrics):
            with column:
                shown = (
                    f"#{int(value)}"
                    if label == "Leaderboard rank"
                    else f"{int(value)}"
                    if label == "Valid AQI days"
                    else f"{value:.1f}"
                )
                st.metric(label, shown)

    st.markdown("---")
    st.subheader("AQI category distribution")
    if compare_stations:
        category_df = comparison_data.loc[
            comparison_data["location_name"].isin(compare_stations),
            ["location_name", "aqi_category"],
        ]
        category_counts = category_df.groupby(
            ["location_name", "aqi_category"]
        ).size().unstack(fill_value=0)
        st.dataframe(category_counts, width="stretch", hide_index=True)
    else:
        st.info("Choose at least one location to display category counts.")

elif page == "Pollution Analytics":
    render_page_header(
        ["SaansCheck", HISTORICAL_MODE, "Pollution Analytics"],
        "Pollution Analytics",
        "Pollutant distributions, relationships and monthly patterns from the historical dataset.",
        [("PM2.5 · PM10 · NO2 · SO2 · CO · O3", "teal")],
    )
    station_daily = get_station_records(daily_df, selected_station)
    pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]

    st.markdown('<div class="section-kicker">Pollution Overview</div>', unsafe_allow_html=True)
    pollutant_cards = st.columns(6)
    pollutant_labels = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3"}
    for idx, pollutant in enumerate(pollutant_cols):
        latest_value = float(station_daily[pollutant].iloc[-1])
        unit = "Historical source unit unverified"
        with pollutant_cards[idx]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">{pollutant_labels[pollutant]}</div>
                <div class="kpi-value" style="font-size:1.5rem;">{latest_value:.1f}</div>
                <div class="kpi-meta">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(
        "Historical pollutant source units are not documented in the available "
        "dataset metadata. Values are shown as supplied; magnitudes across "
        "different pollutants are not directly comparable."
    )
    tabs = st.tabs(["Descriptive Statistics", "AQI Distribution", "Pollutant Comparison", "Correlation", "Monthly Patterns"])
    with tabs[0]:
        desc_df = station_daily[pollutant_cols].describe().T.reset_index().rename(columns={"index": "Pollutant", "mean": "Mean", "std": "Std Dev", "min": "Min", "50%": "Median", "max": "Max"})
        st.dataframe(desc_df, width="stretch", hide_index=True)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.hist(station_daily["aqi"], bins=24, color="#0f8b8d", edgecolor="#ffffff")
        ax.set_title(f"{selected_station} AQI Distribution", color="#17232d")
        ax.set_xlabel("AQI", color="#475d6b")
        ax.set_ylabel("Days", color="#475d6b")
        ax.grid(True, alpha=0.12)
        for spine in ax.spines.values():
            spine.set_color("#dfe7eb")
        fig.tight_layout()
        _render_chart(fig)
        most_common = station_daily["aqi_category"].mode().iloc[0]
        st.caption(f"Most observations fall within the {most_common} category for {selected_station}.")

    with tabs[2]:
        means = station_daily[pollutant_cols].mean().sort_values(ascending=False)
        means_df = pd.DataFrame({"Pollutant": [pollutant_labels[p] for p in means.index], "Mean concentration": means.values})
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.barh(means_df["Pollutant"], means_df["Mean concentration"], color="#2e9d68", edgecolor="#ffffff")
        ax.invert_yaxis()
        ax.set_xlabel("Mean source value (unit unverified)", color="#475d6b")
        ax.set_ylabel("Pollutant", color="#475d6b")
        ax.grid(True, alpha=0.12, axis='x')
        for spine in ax.spines.values():
            spine.set_color("#dfe7eb")
        fig.tight_layout()
        _render_chart(fig)

    with tabs[3]:
        corr = station_daily[["aqi"] + pollutant_cols].corr().round(3)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(corr, cmap='YlGnBu', aspect='auto')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='left')
        ax.set_yticklabels(corr.index)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black' if abs(corr.iloc[i, j]) < 0.7 else 'white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("AQI and pollutant correlation", color="#17232d", fontsize=11)
        fig.tight_layout()
        _render_chart(fig)
        st.caption("Correlation indicates statistical association and does not establish causation.")

    with tabs[4]:
        monthly_pattern = station_daily.groupby("month")[pollutant_cols].mean().sort_index()
        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        for ax, pollutant in zip(axes.flat, pollutant_cols):
            ax.plot(
                monthly_pattern.index.map(MONTH_NAMES),
                monthly_pattern[pollutant],
                color="#0f8b8d",
                linewidth=1.8,
                marker="o",
            )
            ax.set_title(pollutant_labels[pollutant], color="#17232d")
            ax.set_ylabel("Source value (unit unverified)", color="#475d6b")
            ax.grid(True, alpha=0.12)
            ax.tick_params(colors="#475d6b", rotation=35)
        fig.tight_layout()
        _render_chart(fig)

elif page == "Health Advisor":
    render_page_header(
        ["SaansCheck", HISTORICAL_MODE, "Health Advisor"],
        "Personalized Health Advisor",
        "Historical AQI context and general exposure guidance by selected profile.",
        [("Informational only", "amber")],
    )
    station_daily = get_station_records(daily_df, selected_station)
    latest_record = station_daily.iloc[-1]
    aqi_value = float(latest_record["aqi"])
    aqi_category = latest_record["aqi_category"]

    profiles = {
        "General User": {"outdoor": "Maintain normal outdoor activity but stay aware of air quality changes.", "mask": "Use a well-fitted mask if the air feels irritating or if you are outdoors for extended time."},
        "Asthma / Respiratory": {"outdoor": "Reduce prolonged outdoor exposure and avoid strenuous activity in higher AQI periods.", "mask": "Consider using a mask during outdoor exposure and monitor breathing comfort."},
        "Elderly": {"outdoor": "Limit extended outdoor time and take breaks indoors when air quality worsens.", "mask": "Use protective measures during active outdoor periods if necessary."},
        "Child": {"outdoor": "Keep outdoor sessions shorter and prefer indoor breaks to reduce cumulative exposure.", "mask": "Monitor symptoms and choose lower-exertion activities when AQI is moderate or worse."},
        "Outdoor Athlete": {"outdoor": "Reduce intense outdoor training as AQI worsens and plan sessions carefully.", "mask": "Use a mask and consider indoor alternatives when conditions are poor."},
    }

    profile = st.selectbox("Select user profile", list(profiles.keys()))
    profile_col, info_col = st.columns([1, 2])
    with profile_col:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:0.68rem;color:#687983;letter-spacing:0.12em;text-transform:uppercase;font-weight:700;">Historical AQI (dataset-defined)</div>
            <div style="font-size:2.2rem;font-weight:800;color:#17232d;line-height:1.1;margin-top:0.4rem;">{aqi_value:.1f}</div>
            <div style="margin-top:0.6rem;padding:0.4rem 0.65rem;border-radius:999px;background:#f0f8f8;border:1px solid rgba(15,139,141,0.13);display:inline-block;color:#0f8b8d;font-size:0.7rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;">{aqi_category}</div>
            <div style="margin-top:0.9rem;color:#687983;font-size:0.8rem;line-height:1.6;">Historical date: <strong style="color:#17232d;">{pd.to_datetime(latest_record['date']):%Y-%m-%d}</strong><br>Selected location: <strong style="color:#17232d;">{selected_station}</strong><br>Profile: <strong style="color:#17232d;">{profile}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    with info_col:
        risk_guidance = []
        if aqi_category in ["Good", "Satisfactory"]:
            risk_guidance.extend([
                "Outdoor activity can usually continue with normal awareness.",
                "Keep an eye on AQI changes if you are sensitive to air pollution.",
                "Ventilation is generally acceptable; indoor air quality can still vary."
            ])
        elif aqi_category == "Moderate":
            risk_guidance.extend([
                "Consider reducing prolonged outdoor exposure for sensitive people.",
                "If outdoors for long periods, take breaks and monitor comfort and symptoms.",
                "Ventilation may need extra attention in enclosed spaces."
            ])
        else:
            risk_guidance.extend([
                "Reduce prolonged outdoor exposure and avoid strenuous activity where possible.",
                "Keep windows and ventilation choices in mind, especially in polluted conditions.",
                "Sensitive individuals should monitor AQI conditions closely."
            ])

        risk_guidance.extend([profiles[profile]["outdoor"], profiles[profile]["mask"], "Monitor AQI conditions during outdoor activity and plan alternatives if conditions worsen."])

        st.markdown('<div class="section-kicker">Personalized guidance</div>', unsafe_allow_html=True)
        for item in risk_guidance:
            st.write("• " + item)

    st.warning(
        "This is historical, informational guidance—not a live assessment of "
        "your current exposure. SaansCheck does not provide medical diagnosis "
        "or treatment."
    )

elif page == "Seasonal Risk":
    render_page_header(
        ["SaansCheck", HISTORICAL_MODE, "Seasonal Risk"],
        "Seasonal Risk Intelligence",
        "Historical monthly AQI patterns across Delhi-NCR.",
        [("Jan–Dec calendar order", "teal")],
    )

    monthly = daily_df.groupby("month").agg(
        avg_aqi=("aqi", "mean"),
        median_aqi=("aqi", "median"),
        max_aqi=("aqi", "max"),
        std_aqi=("aqi", "std")
    ).reset_index().sort_values("month")
    monthly["Month"] = monthly["month"].map(MONTH_NAMES)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(monthly["Month"], monthly["avg_aqi"], color="#0f8b8d", linewidth=2.5, marker='o', label="Average AQI")
    ax.plot(monthly["Month"], monthly["median_aqi"], color="#88c7c8", linewidth=2, linestyle='--', label="Median AQI")
    ax.set_title("Monthly Average AQI", color="#17232d")
    ax.set_ylabel("AQI", color="#475d6b")
    ax.grid(True, alpha=0.12)
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_color("#dfe7eb")
    ax.tick_params(colors="#475d6b", rotation=35)
    fig.tight_layout()
    _render_chart(fig)

    highest_month = monthly.loc[monthly["avg_aqi"].idxmax()]
    lowest_month = monthly.loc[monthly["avg_aqi"].idxmin()]
    most_variable = monthly.loc[monthly["std_aqi"].idxmax()]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="insight-card"><h4>Highest historical AQI month</h4><div class="value">{highest_month["Month"]}</div><div style="color:#687983;font-size:0.8rem;">Avg AQI {highest_month["avg_aqi"]:.1f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="insight-card"><h4>Lowest historical AQI month</h4><div class="value">{lowest_month["Month"]}</div><div style="color:#687983;font-size:0.8rem;">Avg AQI {lowest_month["avg_aqi"]:.1f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="insight-card"><h4>Most variable month</h4><div class="value">{most_variable["Month"]}</div><div style="color:#687983;font-size:0.8rem;">Std Dev {most_variable["std_aqi"]:.1f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Monthly variability and peak months")
    st.dataframe(monthly[["Month", "avg_aqi", "median_aqi", "max_aqi", "std_aqi"]].sort_values("avg_aqi", ascending=False), width="stretch", hide_index=True)

    st.caption("These findings are observations from the available dataset and are based only on calculated monthly patterns from this historical record.")

    monthly_category = daily_df.groupby(["month", "aqi_category"]).size().unstack(fill_value=0)
    st.subheader("AQI category distribution by month")
    st.dataframe(monthly_category, width="stretch")

else:
    render_page_header(
        ["SaansCheck", HISTORICAL_MODE, "Model Insights"],
        "Model Insights",
        "Linear Regression and Random Forest performance on chronological 2023 test origins.",
        [("Stored evaluation results", "teal")],
    )

    rf_df = results_df[results_df["model"] == "Random Forest"].copy()
    rf_df["MAE"] = rf_df["MAE"].astype(float)
    rf_df["RMSE"] = rf_df["RMSE"].astype(float)
    rf_df["R2"] = rf_df["R2"].astype(float)

    model_cols = st.columns(2)
    with model_cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">Linear Regression</div>
            <div class="kpi-value" style="font-size:1.4rem;">{results_df[results_df['model']=='Linear Regression']['R2'].mean():.3f}</div>
            <div class="kpi-meta">Average R² across horizons</div>
        </div>
        """, unsafe_allow_html=True)
    with model_cols[1]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">Random Forest</div>
            <div class="kpi-value" style="font-size:1.4rem;">{results_df[results_df['model']=='Random Forest']['R2'].mean():.3f}</div>
            <div class="kpi-meta">Average R² across horizons</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Performance comparison")
    st.dataframe(results_df, width="stretch", hide_index=True)

    comparison_df = results_df.pivot(index="horizon", columns="model", values="MAE")
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(comparison_df.index))
    width = 0.32
    ax.bar(x - width / 2, comparison_df["Linear Regression"], width=width, label="Linear Regression", color="#9fc6d8")
    ax.bar(x + width / 2, comparison_df["Random Forest"], width=width, label="Random Forest", color="#0f8b8d")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index)
    ax.set_ylabel("MAE", color="#475d6b")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.12, axis='y')
    for spine in ax.spines.values():
        spine.set_color("#dfe7eb")
    fig.tight_layout(); _render_chart(fig)

    r2_df = results_df.pivot(index="horizon", columns="model", values="R2")
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(r2_df.index))
    width = 0.32
    ax.bar(x - width / 2, r2_df["Linear Regression"], width=width, label="Linear Regression", color="#9fc6d8")
    ax.bar(x + width / 2, r2_df["Random Forest"], width=width, label="Random Forest", color="#2e9d68")
    ax.set_xticks(x)
    ax.set_xticklabels(r2_df.index)
    ax.set_ylabel("R²", color="#475d6b")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.12, axis='y')
    for spine in ax.spines.values():
        spine.set_color("#dfe7eb")
    fig.tight_layout(); _render_chart(fig)

    st.markdown("---")
    st.markdown('<div class="section-kicker">Selected model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-box">
      <div style="display:flex;flex-wrap:wrap;gap:0.8rem;align-items:center;">
        <span class="pill">Selected Model</span>
        <span style="font-weight:800;color:#17232d;font-size:1.1rem;">Random Forest Regressor</span>
      </div>
      <div style="margin-top:0.9rem;color:#687983;line-height:1.7;">
        Why? Lower MAE and stronger R² across all three horizons on the chronological 2023 test origins; see the evaluation caveat below.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Comparison against persistence baseline")
    baseline_rows = []
    for horizon, baseline in PERSISTENCE_BASELINE.items():
        rf_metric = rf_df[rf_df["horizon"] == horizon].iloc[0]
        improvement = ((baseline["MAE"] - rf_metric["MAE"]) / baseline["MAE"]) * 100
        baseline_rows.append({
            "Horizon": horizon,
            "Persistence MAE": baseline["MAE"],
            "Random Forest MAE": rf_metric["MAE"],
            "MAE improvement (%)": round(improvement, 2),
        })
    st.dataframe(pd.DataFrame(baseline_rows), width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Top Predictive Features")
    top_features = importance_df.head(10).sort_values("mean_importance", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(top_features["feature"], top_features["mean_importance"], color="#0f8b8d", edgecolor="#ffffff")
    ax.invert_yaxis()
    ax.set_xlabel("Mean importance", color="#475d6b")
    ax.set_ylabel("Feature", color="#475d6b")
    ax.grid(True, alpha=0.12, axis='x')
    for spine in ax.spines.values():
        spine.set_color("#dfe7eb")
    fig.tight_layout(); _render_chart(fig)
    st.caption("Feature importance reflects predictive contribution within the trained Random Forest model and does not imply causation.")

    st.markdown("---")
    st.subheader("Actual vs predicted")
    pred_target = st.selectbox("Actual vs predicted chart", ["Day +1", "Day +2", "Day +3"])
    target_index = {"Day +1": 1, "Day +2": 2, "Day +3": 3}[pred_target]
    actual_col = f"actual_AQI_t+{target_index}"
    pred_col = f"pred_rf_AQI_t+{target_index}"
    if actual_col in predictions_df.columns and pred_col in predictions_df.columns:
        station_predictions = predictions_df.loc[
            predictions_df["location_name"].eq(selected_station),
            ["date", actual_col, pred_col],
        ].copy()
        station_predictions["date"] = pd.to_datetime(
            station_predictions["date"], errors="coerce"
        )
        chart_df = (
            station_predictions.dropna()
            .sort_values("date")
            .tail(60)
        )
        if chart_df.empty:
            st.info(f"No actual and predicted values are available for {selected_station}.")
        else:
            date_range = (
                f"{chart_df['date'].min():%Y-%m-%d} to "
                f"{chart_df['date'].max():%Y-%m-%d}"
            )
            st.caption(
                f"{selected_station} · forecast origins {date_range} · "
                "actual vs predicted"
            )
            st.line_chart(chart_df.set_index("date"), height=260)
    else:
        st.info(f"Actual and predicted columns are unavailable for {selected_station}.")

    _render_historical_category_evaluation(predictions_df)

    st.markdown("---")
    st.markdown("""
    <div class="metric-box">
        <div style="font-size:0.68rem;color:#687983;letter-spacing:0.12em;text-transform:uppercase;font-weight:700;">Data & Model Integrity</div>
        <div style="margin-top:0.8rem;color:#687983;line-height:1.8;">
            ✓ Chronological train/test split<br>
            ✓ Leakage audit completed<br>
            ✓ Persistence baseline evaluated<br>
            ✓ Saved models verified<br>
            ✓ Historical data validated
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 5. FOOTER
# =============================================================================
st.markdown("""
<div class="footer-text">
    SaansCheck · Breathe smarter. Understand better.<br>
    Historical Intelligence and external observations are presented separately.
</div>
""", unsafe_allow_html=True)
