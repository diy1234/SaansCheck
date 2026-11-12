"""Theme tokens and presentation helpers shared by the dashboard."""

from __future__ import annotations

from matplotlib.figure import Figure

THEME_PALETTES = {
    "Dark": {
        "background": "#0B1220",
        "surface": "#111C2E",
        "elevated": "#17243A",
        "text": "#F1F5F9",
        "muted": "#A8B7CA",
        "line": "#2A3A50",
        "accent": "#2DD4BF",
        "accent_soft": "#153C42",
        "grid": "#2A3A50",
        "scheme": "dark",
    },
    "Light": {
        "background": "#F5F8FA",
        "surface": "#FFFFFF",
        "elevated": "#F9FBFC",
        "text": "#17232D",
        "muted": "#586A78",
        "line": "#D8E1E8",
        "accent": "#0F8B8D",
        "accent_soft": "#E7F5F4",
        "grid": "#D8E1E8",
        "scheme": "light",
    },
}


def theme_css(theme: str) -> str:
    """Build the page and widget styles for a supported appearance theme."""
    try:
        palette = THEME_PALETTES[theme]
    except KeyError as error:
        raise ValueError(f"Unsupported theme: {theme!r}") from error

    return f"""
<style>
    :root {{
        color-scheme: {palette["scheme"]};
        --sc-bg: {palette["background"]};
        --sc-surface: {palette["surface"]};
        --sc-elevated: {palette["elevated"]};
        --sc-text: {palette["text"]};
        --sc-muted: {palette["muted"]};
        --sc-line: {palette["line"]};
        --sc-accent: {palette["accent"]};
        --sc-accent-soft: {palette["accent_soft"]};
        --sc-grid: {palette["grid"]};
    }}
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], main {{
        background: var(--sc-bg) !important;
        color: var(--sc-text) !important;
    }}
    [data-testid="stHeader"] {{
        background: var(--sc-bg) !important;
    }}
    [data-testid="stSidebar"] {{
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
        background: var(--sc-surface) !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.sc-topnav-marker) {{
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    .stApp :is(h1, h2, h3, h4, h5, h6, p, li, label, small, code),
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMarkdownContainer"] {{
        color: var(--sc-text);
    }}
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMarkdownContainer"] p {{
        color: var(--sc-muted);
    }}
    .sc-brand-name, .sc-sidebar-brand-name, .sc-page-title,
    .sc-kpi-value, .sc-empty-title, .sc-mode-card-title,
    .sc-hero-title, .sc-historical-aqi-value, .insight-card .value,
    .kpi-value, .section-title, .hero-title {{
        color: var(--sc-text) !important;
    }}
    .sc-brand-subtitle, .sc-sidebar-brand-subtitle, .sc-page-subtitle,
    .sc-breadcrumb, .sc-kpi-label, .sc-kpi-detail, .sc-empty-detail,
    .sc-mode-card-copy, .sc-hero-copy, .sc-sidebar-title,
    .subtle-note, .small-note, .kpi-header, .kpi-meta, .hero-sub {{
        color: var(--sc-muted) !important;
    }}
    .sc-forecast-card .kpi-meta strong,
    .sc-forecast-model-name {{
        color: var(--sc-text) !important;
    }}
    .sc-location-rank-card .kpi-meta {{
        color: var(--sc-muted) !important;
    }}
    .sc-location-rank-card .kpi-meta strong {{
        color: var(--sc-text) !important;
    }}
    .sc-forecast-model-detail {{
        color: var(--sc-muted) !important;
        line-height: 1.6;
    }}
    .sc-tagline, .sc-empty-mark, .section-kicker {{
        color: var(--sc-accent) !important;
    }}
    .premium-card, .kpi-card, .metric-box, .insight-card, .chart-frame,
    .sc-kpi, .sc-empty-state, .sc-dashboard-status,
    [data-testid="stVerticalBlockBorderWrapper"] {{
        color: var(--sc-text) !important;
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    .hero-card {{
        color: var(--sc-text) !important;
        background: var(--sc-elevated) !important;
        border-color: var(--sc-line) !important;
    }}
    .sc-sidebar-brand, .sc-sidebar-bottom, .sc-mode-row {{
        border-color: var(--sc-line) !important;
    }}
    .sc-sidebar-intro {{
        color: var(--sc-text) !important;
        background: var(--sc-elevated) !important;
        border-color: var(--sc-line) !important;
    }}
    .sc-sidebar-intro-title {{
        color: var(--sc-text) !important;
    }}
    .sc-sidebar-intro-tagline {{
        color: var(--sc-muted) !important;
    }}
    .sc-sidebar-badge {{
        color: var(--sc-accent) !important;
        background: var(--sc-accent-soft) !important;
        border-color: var(--sc-line) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        color: var(--sc-muted) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
        color: var(--sc-accent) !important;
        background: var(--sc-accent-soft) !important;
        border-color: var(--sc-line) !important;
    }}
    [role="radiogroup"] button[data-variant="segmented_control"],
    .stTabs [role="tab"] {{
        color: var(--sc-muted) !important;
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    [role="radiogroup"] button[data-variant="segmented_control"][data-selected="true"],
    .stTabs [role="tab"][aria-selected="true"] {{
        color: var(--sc-accent) !important;
        background: var(--sc-accent-soft) !important;
        border-color: var(--sc-accent) !important;
    }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="textarea"], .stTextInput input, .stNumberInput input,
    .stDateInput input, .stTimeInput input {{
        color: var(--sc-text) !important;
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    .stApp [data-testid="stSelectbox"] {{
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    .stApp [data-testid="stSelectbox"] [role="group"][data-rac],
    .stApp [data-testid="stSelectbox"] input[role="combobox"],
    .stApp [data-testid="stSelectbox"] button {{
        color: var(--sc-text) !important;
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {{
        color: var(--sc-text) !important;
        background: var(--sc-surface) !important;
    }}
    [role="option"]:hover, [role="option"][aria-selected="true"] {{
        background: var(--sc-accent-soft) !important;
    }}
    .st-key-location_comparison_locations [data-tag] {{
        color: var(--sc-text) !important;
        background: var(--sc-accent-soft) !important;
        border: 1px solid var(--sc-accent) !important;
    }}
    .st-key-location_comparison_locations [data-tag] span,
    .st-key-location_comparison_locations [data-tag] button {{
        color: var(--sc-text) !important;
    }}
    .st-key-location_comparison_locations [data-tag] svg {{
        color: var(--sc-text) !important;
        stroke: var(--sc-text) !important;
    }}
    .st-key-location_comparison_locations [role="group"][data-rac] {{
        color: var(--sc-text) !important;
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    .st-key-location_comparison_locations [data-baseweb="select"] > div {{
        border-color: var(--sc-line) !important;
    }}
    .st-key-location_comparison_locations [data-baseweb="select"]:focus-within > div {{
        border-color: var(--sc-accent) !important;
        box-shadow: 0 0 0 1px var(--sc-accent) !important;
    }}
    .stButton > button {{
        color: var(--sc-text) !important;
        background: var(--sc-elevated) !important;
        border-color: var(--sc-line) !important;
    }}
    .stButton > button:hover {{
        color: var(--sc-accent) !important;
        border-color: var(--sc-accent) !important;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border-color: var(--sc-line) !important;
    }}
    [data-testid="stExpander"] {{
        color: var(--sc-text) !important;
        background: var(--sc-surface) !important;
        border-color: var(--sc-line) !important;
    }}
    .sc-startup-title, .sc-startup-tagline {{
        color: #F1F5F9 !important;
    }}
    .st-key-waqi_alert_dock {{
        position: fixed;
        right: 18px;
        bottom: 16px;
        z-index: 1000;
        width: min(270px, calc(100vw - 34px));
        max-height: 38vh;
        overflow-y: auto;
        padding: 0.55rem 0.7rem;
        border: 1px solid #f59e0b;
        border-radius: 16px;
        background: var(--sc-surface);
        box-shadow: 0 8px 28px #0004, 0 0 18px #f59e0b33;
    }}
    .st-key-waqi_alert_dock [data-testid="stExpander"] {{
        border: 0;
        background: transparent;
    }}
    .st-key-waqi_alert_dock [data-testid="stExpander"] summary {{
        font-weight: 700;
    }}
    @media (max-width: 640px) {{
        .st-key-waqi_alert_dock {{
            right: 8px;
            bottom: 8px;
            width: min(235px, calc(100vw - 20px));
            max-height: 32vh;
            padding: 0.35rem 0.5rem;
        }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            scroll-behavior: auto !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
</style>
"""


def chart_palette(theme: str) -> dict[str, str]:
    """Return Matplotlib surface and text colors for a dashboard theme."""
    try:
        palette = THEME_PALETTES[theme]
    except KeyError as error:
        raise ValueError(f"Unsupported theme: {theme!r}") from error
    return {
        "background": palette["background"],
        "surface": palette["surface"],
        "text": palette["text"],
        "muted": palette["muted"],
        "line": palette["line"],
        "accent": palette["accent"],
    }


def apply_chart_theme(figure: Figure, theme: str) -> None:
    """Style a completed Matplotlib figure without changing global rcParams."""
    palette = chart_palette(theme)
    figure.set_facecolor(palette["background"])
    for axis in figure.axes:
        axis.set_facecolor(palette["surface"])
        axis.tick_params(colors=palette["muted"])
        axis.xaxis.label.set_color(palette["muted"])
        axis.yaxis.label.set_color(palette["muted"])
        axis.title.set_color(palette["text"])
        axis.grid(color=palette["line"], alpha=0.55)
        for spine in axis.spines.values():
            spine.set_color(palette["line"])
        for text in axis.texts:
            text.set_color(palette["text"])
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(palette["surface"])
            legend.get_frame().set_edgecolor(palette["line"])
            for text in legend.get_texts():
                text.set_color(palette["muted"])
