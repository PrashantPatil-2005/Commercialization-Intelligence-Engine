"""
Theme System — Light + Dark

Premium enterprise color palettes inspired by GitHub, Azure, Linear, and Microsoft Fabric.
Single source of truth for every color, font, and spacing value.

Usage:
    from app.theme import get_theme
    t = get_theme()
    bg = t["bg_primary"]
"""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# ============================================================
# OUTCOME COLORS — identical across both themes (never change)
# Muted, professional tones inspired by enterprise design systems
# ============================================================

COLOR_MVP = "#107c10"        # Forest green
COLOR_PILOT = "#0078d4"      # Microsoft blue
COLOR_ASSET = "#8764b8"      # Muted purple
COLOR_INCUBATE = "#ca5010"   # Burnt orange
COLOR_ARCHIVE = "#d13438"    # Soft red

OUTCOME_META = {
    "MVP Build":      {"color": COLOR_MVP,    "label": "MVP Build"},
    "Customer Pilot": {"color": COLOR_PILOT,   "label": "Customer Pilot"},
    "Reusable Asset": {"color": COLOR_ASSET,   "label": "Reusable Asset"},
    "Incubate":       {"color": COLOR_INCUBATE, "label": "Incubate"},
    "Archive":        {"color": COLOR_ARCHIVE,  "label": "Archive"},
}

CLUSTER_COLORS = ["#0078d4", "#107c10", "#ca5010", "#d13438", "#8764b8", "#e16f24"]

# ============================================================
# DARK THEME — Neutral grays, inspired by GitHub + Linear
# ============================================================

DARK = {
    # Backgrounds
    "bg_primary":    "#111111",
    "bg_secondary":  "#1a1a1a",
    "bg_tertiary":   "#242424",
    "bg_sidebar":    "#0a0a0a",
    "bg_input":      "#1a1a1a",

    # Borders
    "border":        "#2a2a2a",
    "border_light":  "#1f1f1f",
    "border_focus":  "#5b9bd5",

    # Text
    "text_primary":   "#e5e5e5",
    "text_secondary": "#a0a0a0",
    "text_muted":     "#777777",
    "text_link":      "#5b9bd5",

    # Semantic
    "success":        "#4caf50",
    "success_subtle": "#0d2818",
    "warning":        "#f0ad4e",
    "warning_subtle": "#3d2e00",
    "danger":         "#e06060",
    "danger_subtle":  "#3d1114",
    "info":           "#5b9bd5",
    "info_subtle":    "#0d1f3c",

    # Sidebar
    "sidebar_bg":          "#0a0a0a",
    "sidebar_text":        "#e5e5e5",
    "sidebar_text_sec":    "#a0a0a0",
    "sidebar_border":      "#2a2a2a",
    "sidebar_hover":       "#1a1a1a",
    "sidebar_active_bg":   "#5b9bd525",
    "sidebar_active_text": "#5b9bd5",

    # Table
    "table_header_bg":  "#1a1a1a",
    "table_row_bg":     "#111111",
    "table_row_alt":    "#161616",
    "table_row_hover":  "#242424",
    "table_border":     "#2a2a2a",
    "table_selected":   "#5b9bd515",

    # KPI
    "kpi_bg":     "#1a1a1a",
    "kpi_border": "#2a2a2a",

    # Progress
    "progress_bg":       "#2a2a2a",
    "progress_high":     "#4caf50",
    "progress_moderate": "#f0ad4e",
    "progress_low":      "#e06060",

    # Charts
    "chart_bg":      "#1a1a1a",
    "chart_grid":    "#2a2a2a",
    "chart_text":    "#a0a0a0",
    "chart_title":   "#e5e5e5",
    "chart_edge":    "#2a2a2a",
    "chart_bar_edge": "#111111",
    "chart_positive": "#4caf50",
    "chart_negative": "#e06060",

    # Callout
    "callout_info_bg":    "#0d1f3c",
    "callout_info_border":"#5b9bd5",
    "callout_warn_bg":    "#3d2e00",
    "callout_warn_border":"#f0ad4e",
    "callout_muted_bg":   "#1a1a1a",
    "callout_muted_border":"#2a2a2a",

    # Misc
    "scrollbar":     "#444444",
    "scrollbar_hover":"#666666",
}

# ============================================================
# LIGHT THEME — Warm whites, inspired by Microsoft Fabric
# ============================================================

LIGHT = {
    # Backgrounds
    "bg_primary":    "#f5f5f5",
    "bg_secondary":  "#ffffff",
    "bg_tertiary":   "#f0f0f0",
    "bg_sidebar":    "#f5f5f5",
    "bg_input":      "#ffffff",

    # Borders
    "border":        "#e0e0e0",
    "border_light":  "#eeeeee",
    "border_focus":  "#0078d4",

    # Text
    "text_primary":   "#1a1a1a",
    "text_secondary": "#555555",
    "text_muted":     "#767676",
    "text_link":      "#0078d4",

    # Semantic
    "success":        "#107c10",
    "success_subtle": "#dff6dd",
    "warning":        "#ca5010",
    "warning_subtle": "#fed9cc",
    "danger":         "#d13438",
    "danger_subtle":  "#fde7e9",
    "info":           "#0078d4",
    "info_subtle":    "#deecf9",

    # Sidebar
    "sidebar_bg":          "#ffffff",
    "sidebar_text":        "#1a1a1a",
    "sidebar_text_sec":    "#555555",
    "sidebar_border":      "#e0e0e0",
    "sidebar_hover":       "#f0f0f0",
    "sidebar_active_bg":   "#0078d420",
    "sidebar_active_text": "#0078d4",

    # Table
    "table_header_bg":  "#f5f5f5",
    "table_row_bg":     "#ffffff",
    "table_row_alt":    "#fafafa",
    "table_row_hover":  "#f0f0f0",
    "table_border":     "#e0e0e0",
    "table_selected":   "#0078d412",

    # KPI
    "kpi_bg":     "#ffffff",
    "kpi_border": "#e0e0e0",

    # Progress
    "progress_bg":       "#e0e0e0",
    "progress_high":     "#107c10",
    "progress_moderate": "#ca5010",
    "progress_low":      "#d13438",

    # Charts
    "chart_bg":      "#ffffff",
    "chart_grid":    "#e8e8e8",
    "chart_text":    "#666666",
    "chart_title":   "#1a1a1a",
    "chart_edge":    "#e0e0e0",
    "chart_bar_edge": "#ffffff",
    "chart_positive": "#107c10",
    "chart_negative": "#d13438",

    # Callout
    "callout_info_bg":    "#deecf9",
    "callout_info_border":"#0078d4",
    "callout_warn_bg":    "#fed9cc",
    "callout_warn_border":"#ca5010",
    "callout_muted_bg":   "#f5f5f5",
    "callout_muted_border":"#e0e0e0",

    # Misc
    "scrollbar":     "#c1c1c1",
    "scrollbar_hover":"#a0a0a0",
}

# ============================================================
# THEME REGISTRY
# ============================================================

THEMES = {
    "light": LIGHT,
    "dark": DARK,
}


def get_theme() -> dict:
    """Return the active theme dict. Defaults to dark."""
    try:
        mode = st.session_state.get("theme_mode", "dark")
    except RuntimeError:
        mode = "dark"
    return THEMES.get(mode, DARK)


def set_theme(mode: str):
    """Set the active theme."""
    st.session_state["theme_mode"] = mode


def toggle_theme():
    """Toggle between light and dark themes."""
    current = st.session_state.get("theme_mode", "dark")
    new_mode = "light" if current == "dark" else "dark"
    set_theme(new_mode)


def is_dark() -> bool:
    """Check if current theme is dark."""
    return st.session_state.get("theme_mode", "dark") == "dark"


# ============================================================
# TYPOGRAPHY (shared across themes)
# ============================================================

FONT_FAMILY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'SFMono-Regular', 'Consolas', 'Liberation Mono', monospace"

FONT_XS = "11px"
FONT_SM = "12px"
FONT_BASE = "13px"
FONT_MD = "14px"
FONT_LG = "15px"
FONT_XL = "18px"
FONT_2XL = "20px"
FONT_3XL = "24px"

FONT_XS_F = 11
FONT_SM_F = 12
FONT_BASE_F = 13
FONT_MD_F = 14
FONT_LG_F = 15

WEIGHT_REGULAR = "400"
WEIGHT_MEDIUM = "500"
WEIGHT_SEMIBOLD = "600"
WEIGHT_BOLD = "700"

LEADING_TIGHT = "1.2"
LEADING_NORMAL = "1.5"
LEADING_RELAXED = "1.6"

# ============================================================
# SPACING / RADIUS
# ============================================================

RADIUS_SM = "4px"
RADIUS_MD = "6px"
RADIUS_LG = "8px"

BADGE_RADIUS = "10px"

# ============================================================
# PAGE CONFIG
# ============================================================

PAGE_CONFIG = {
    "page_title": "CIE - Commercialization Intelligence Engine",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
