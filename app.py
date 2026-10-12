"""CineMatch 3D | Futuristic Movie & TV Series Recommendation System."""

import random
import re
from typing import List, Dict, Any
import streamlit as st
import pandas as pd

from src.recommender import MovieRecommender
from src.series_recommender import SeriesRecommender

# Set Page Configuration
st.set_page_config(
    page_title="CineMatch 3D | Neural Entertainment Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-profile curated picks for instant surprise
MOVIE_PICKS = [
    "Avatar", "The Dark Knight", "Inception", "Interstellar", "Iron Man",
    "The Matrix", "Pulp Fiction", "Forrest Gump", "Gladiator", "Finding Nemo",
    "Titanic", "The Avengers", "Jurassic Park", "Star Wars", "The Godfather",
    "The Lion King", "Back to the Future", "Fight Club", "Toy Story"
]

SERIES_PICKS = [
    "Game of Thrones", "Breaking Bad", "The Walking Dead", "The Big Bang Theory",
    "Dexter", "Friends", "Sherlock", "Lost", "Prison Break", "Fargo",
    "Black Mirror", "Vikings", "Suits", "Daredevil", "House of Cards", "Peaky Blinders"
]

# Advanced Futuristic 3D Cyber CSS Styling
FUTURISTIC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&display=swap');

    /* Scoped Typography without overriding Streamlit Icon fonts */
    h1, h2, h3, .cyber-glitch {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1.2px;
    }
    .stApp, .stApp p, .stApp label, .stApp input {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.05rem;
    }

    /* Preserve Material Icon Fonts across all Streamlit components */
    [data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined, [data-testid="stExpander"] summary span:first-child, [data-testid="stExpanderToggleIcon"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
        font-feature-settings: "liga" 1 !important;
        font-style: normal !important;
    }

    /* Ambient Background Glow */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f1422 0%, #07090e 55%, #040507 100%) !important;
        color: #e2e8f0;
    }

    /* Futuristic 3D Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090c14 0%, #05070a 100%) !important;
        border-right: 1px solid rgba(0, 242, 254, 0.25) !important;
        box-shadow: 6px 0 30px rgba(0, 242, 254, 0.12) !important;
    }

    /* Futuristic HUD Cards in Sidebar */
    .cyber-panel {
        background: linear-gradient(135deg, rgba(16, 24, 39, 0.85) 0%, rgba(9, 13, 22, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-top: 2px solid rgba(0, 242, 254, 0.6);
        border-bottom: 3px solid rgba(0, 0, 0, 0.8);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
    }

    /* Cyber Brand Logo */
    .cyber-logo-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #030712 100%);
        border: 1px solid rgba(0, 242, 254, 0.5);
        border-top: 2px solid #00f2fe;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.3), inset 0 0 15px rgba(0, 242, 254, 0.15);
    }
    .cyber-logo-banner h2 {
        margin: 0;
        font-weight: 900;
        color: #00f2fe !important;
        text-shadow: 0 0 12px rgba(0, 242, 254, 0.8), 0 0 24px rgba(0, 242, 254, 0.4);
    }
    .cyber-logo-banner p {
        margin: 4px 0 0 0;
        font-size: 0.82rem;
        color: #94a3b8;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Futuristic Hero Banner */
    .cyber-hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(20, 15, 38, 0.9) 100%);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-left: 5px solid #00f2fe;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 242, 254, 0.15);
        position: relative;
        overflow: hidden;
    }
    .cyber-hero::after {
        content: "";
        position: absolute;
        top: 0; right: 0; width: 120px; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.08));
        pointer-events: none;
    }

    /* 3D Cyber Movie/Series Cards */
    .cyber-card {
        background: linear-gradient(145deg, rgba(18, 24, 38, 0.95) 0%, rgba(10, 14, 23, 0.98) 100%);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.18);
        border-left: 2px solid rgba(0, 242, 254, 0.5);
        border-bottom: 4px solid #05070c;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.28s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .cyber-card:hover {
        transform: translateY(-6px) scale(1.01);
        border-left: 3px solid #00f2fe;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.8), 0 0 25px rgba(0, 242, 254, 0.25);
    }
    .cyber-card-series {
        border-left: 2px solid rgba(168, 85, 247, 0.6) !important;
    }
    .cyber-card-series:hover {
        border-left: 3px solid #a855f7 !important;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.8), 0 0 25px rgba(168, 85, 247, 0.3) !important;
    }

    /* 3D Holographic Badges */
    .badge-holo-match {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #030712 !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.85rem;
        box-shadow: 0 0 14px rgba(0, 242, 254, 0.6);
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-holo-purple {
        background: linear-gradient(135deg, #c084fc 0%, #a855f7 100%);
        color: #030712 !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.85rem;
        box-shadow: 0 0 14px rgba(168, 85, 247, 0.6);
        display: inline-block;
    }
    .badge-holo-star {
        background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
        color: #111 !important;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 900;
        font-size: 0.88rem;
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.5);
        display: inline-block;
    }
    .badge-holo-genre {
        background: rgba(0, 242, 254, 0.08);
        color: #00f2fe !important;
        border: 1px solid rgba(0, 242, 254, 0.35);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
    }
    .badge-genre-purple {
        background: rgba(168, 85, 247, 0.1) !important;
        color: #c084fc !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
    }
    .badge-holo-meta {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.82rem;
        color: #cbd5e1;
        margin-right: 8px;
        display: inline-block;
    }

    /* 3D Push Tactile Buttons */
    div.stButton > button {
        background: linear-gradient(145deg, #1e293b, #0f172a) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-bottom: 3px solid #020617 !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-weight: 700 !important;
        letter-spacing: 0.8px !important;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.5) !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover {
        border-color: #00f2fe !important;
        box-shadow: 0 0 16px rgba(0, 242, 254, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    div.stButton > button:active {
        transform: translateY(2px) !important;
        border-bottom-width: 1px !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00f2fe 0%, #0284c7 100%) !important;
        color: #020617 !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 18px rgba(0, 242, 254, 0.6) !important;
    }

    /* Futuristic HUD Inset Box */
    .hud-box {
        background: rgba(10, 15, 26, 0.9);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.8);
    }
    .hud-box h4 {
        margin: 0;
        color: #00f2fe;
        font-family: 'Orbitron', sans-serif;
    }
    .hud-box p {
        margin: 2px 0 0 0;
        font-size: 0.72rem;
        color: #64748b;
        letter-spacing: 1px;
    }
</style>
"""
st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)


# Cache Recommender Models
@st.cache_resource(show_spinner="Booting Movie Neural Matrix (4,760 records)...")
def get_movie_engine() -> MovieRecommender:
    return MovieRecommender(include_overview=True)


@st.cache_resource(show_spinner="Booting TV Series Cyber Database (1,926 records)...")
def get_series_engine() -> SeriesRecommender:
    return SeriesRecommender()


# Initialize Session State
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "🎬 Movies Dimension"
if "current_movie" not in st.session_state:
    st.session_state.current_movie = "Avatar"
if "current_series" not in st.session_state:
    st.session_state.current_series = "Breaking Bad"
if "movie_sort" not in st.session_state:
    st.session_state.movie_sort = "🎯 Best Match"
if "series_sort" not in st.session_state:
    st.session_state.series_sort = "🎯 Best Match"


def surprise_movie_callback():
    picked = random.choice(MOVIE_PICKS)
    st.session_state.current_movie = picked
    st.session_state["movie_drop_key"] = picked
    st.session_state["movie_search_key"] = picked


def surprise_series_callback():
    picked = random.choice(SERIES_PICKS)
    st.session_state.current_series = picked
    st.session_state["series_drop_key"] = picked
    st.session_state["series_search_key"] = picked


def apply_sorting_and_tier_filter(
    items: List[Dict[str, Any]],
    sort_choice: str,
    star_tier: str,
    min_stars: float,
    selected_genres: List[str],
) -> List[Dict[str, Any]]:
    filtered = []
    for it in items:
        vote = float(it.get("vote_average", 0.0))

        if star_tier == "⭐⭐⭐⭐⭐ Masterpieces (8.0+ ⭐)" and vote < 8.0:
            continue
        elif star_tier == "⭐⭐⭐⭐ Highly Rated (7.0+ ⭐)" and vote < 7.0:
            continue
        elif star_tier == "⭐⭐⭐ Good (6.0+ ⭐)" and vote < 6.0:
            continue
        elif star_tier == "⭐⭐ Moderate (5.0 - 5.9 ⭐)" and not (5.0 <= vote < 6.0):
            continue
        elif star_tier == "⭐ Below 5.0 (< 5.0 ⭐)" and vote >= 5.0:
            continue

        if vote < min_stars:
            continue

        if selected_genres:
            genre_str = it.get("genre", "")
            if not any(g.lower() in genre_str.lower() for g in selected_genres):
                continue

        filtered.append(it)

    sorted_list = list(filtered)
    if sort_choice == "⭐ Stars: High → Low":
        sorted_list.sort(
            key=lambda x: (float(x.get("vote_average", 0.0)), float(x.get("similarity_score", 0.0))),
            reverse=True,
        )
    elif sort_choice == "⭐ Stars: Low → High":
        sorted_list.sort(key=lambda x: float(x.get("vote_average", 0.0)), reverse=False)
    elif sort_choice == "🔥 Most Popular":
        sorted_list.sort(key=lambda x: int(x.get("vote_count", 0)), reverse=True)
    elif sort_choice == "📅 Newest First":
        def extract_yr(item):
            raw = str(item.get("release_date", item.get("year", "")))
            parts = [p for p in raw.replace("–", "-").split("-") if p.isdigit()]
            return int(parts[-1]) if parts else 0
        sorted_list.sort(key=extract_yr, reverse=True)
    elif sort_choice == "🔤 Title (A → Z)":
        sorted_list.sort(key=lambda x: x.get("title", "").lower())
    else:
        # Default: Best Match
        sorted_list.sort(key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)

    for i, it in enumerate(sorted_list, 1):
        it["display_rank"] = i

    return sorted_list


def main():
    # Load Engines
    movie_engine = get_movie_engine()
    series_engine = get_series_engine()

    movie_titles = movie_engine.get_all_titles()
    series_titles = series_engine.get_all_titles()

    # -------------------------------------------------------------
    # FUTURISTIC CYBER SIDEBAR HUD
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div class="cyber-logo-banner">
                <h2>⚡ CINEMATCH 3D</h2>
                <p>Neural Entertainment Engine</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dimension / Realm Selector
        st.markdown(
            """
            <div class="cyber-panel">
                <h4 style="margin:0 0 6px 0; color:#00f2fe;">🛰️ Entertainment Dimension</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        dimension = st.radio(
            "Select Dimension:",
            ["🎬 Movies Dimension", "📺 TV Series Dimension"],
            index=0 if st.session_state.app_mode == "🎬 Movies Dimension" else 1,
            label_visibility="collapsed",
        )
        st.session_state.app_mode = dimension
        is_movie_mode = "Movies" in dimension

        # Cyber Toggle Switches
        st.markdown(
            """
            <div class="cyber-panel">
                <h4 style="margin:0 0 8px 0; color:#fff;">⚙️ Neural Calibrations</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        synopsis_toggle = st.toggle("🔮 Neural Plot Synopsis", value=True, help="Semantic understanding of storylines")
        anti_self_toggle = st.toggle("🛡️ Anti-Self Match Safeguard", value=True, help="Excludes query title from output")
        elite_only_toggle = st.toggle("💎 Elite 8.0+ Only", value=False, help="Restricts exclusively to masterworks")

        st.markdown("<br>", unsafe_allow_html=True)

        # Star Rating Filter Card
        st.markdown(
            """
            <div class="cyber-panel">
                <h4 style="margin:0 0 6px 0; color:#fbbf24;">⭐ Star Quality Spectrum</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        star_tiers = [
            "All Ratings",
            "⭐⭐⭐⭐⭐ Masterpieces (8.0+ ⭐)",
            "⭐⭐⭐⭐ Highly Rated (7.0+ ⭐)",
            "⭐⭐⭐ Good (6.0+ ⭐)",
            "⭐⭐ Moderate (5.0 - 5.9 ⭐)",
            "⭐ Below 5.0 (< 5.0 ⭐)",
        ]
        chosen_star_tier = "⭐⭐⭐⭐⭐ Masterpieces (8.0+ ⭐)" if elite_only_toggle else st.selectbox(
            "Star Tier Range:", options=star_tiers, index=0
        )

        min_stars_slider = st.slider(
            "Fine Minimum Star Calibrator:",
            min_value=0.0,
            max_value=9.5,
            value=8.0 if elite_only_toggle else 0.0,
            step=0.5,
            format="⭐ %.1f",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Genre & Depth
        st.markdown(
            """
            <div class="cyber-panel">
                <h4 style="margin:0 0 6px 0; color:#fff;">🎭 Genre Matrix & Depth</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_genre_list = movie_engine.get_genres_list() if is_movie_mode else series_engine.get_genres_list()
        selected_genres = st.multiselect(
            "Restrict to Genres:",
            options=active_genre_list,
            default=[],
            placeholder="Select genres...",
        )

        top_k = st.slider("Result Quantum (Top K):", min_value=5, max_value=30, value=12, step=1)

        st.markdown("<br>", unsafe_allow_html=True)

        # Holographic Watchlist
        st.markdown(
            f"""
            <div class="cyber-panel">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#fff;">📌 Holo-Watchlist</h4>
                    <span class="badge-holo-match" style="font-size:0.75rem; padding: 2px 8px;">{len(st.session_state.watchlist)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.watchlist:
            for item in st.session_state.watchlist:
                col_w1, col_w2 = st.columns([4, 1])
                tag_icon = "🎬" if item.get("type") == "Movie" else "📺"
                col_w1.markdown(f"{tag_icon} **{item['title']}** &nbsp;`⭐ {item['vote_average']}`")
                if col_w2.button("✕", key=f"del_wl_{item['id']}", help="Eject"):
                    st.session_state.watchlist = [w for w in st.session_state.watchlist if w["id"] != item["id"]]
                    st.rerun()

            if st.button("🗑️ Purge Watchlist", use_container_width=True):
                st.session_state.watchlist = []
                st.rerun()
        else:
            st.caption("No titles locked in HUD. Click '➕ Watchlist' to store.")

        st.markdown("<br>", unsafe_allow_html=True)

        # HUD Node Status
        total_assets = len(movie_titles) if is_movie_mode else len(series_titles)
        accent_color = "#00f2fe" if is_movie_mode else "#c084fc"
        st.markdown(
            f"""
            <div class="cyber-panel">
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div class="hud-box">
                        <h4 style="color:{accent_color};">{total_assets:,}</h4>
                        <p>{'MOVIES' if is_movie_mode else 'SERIES'}</p>
                    </div>
                    <div class="hud-box">
                        <h4 style="color:{accent_color};">0.003s</h4>
                        <p>LATENCY</p>
                    </div>
                </div>
                <p style="text-align:center; font-size: 0.72rem; color: #4ade80; margin: 8px 0 0 0;">● QUANTUM ENGINE ONLINE</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # MAIN STAGE: NAVIGATION TABS & HERO SECTION
    # -------------------------------------------------------------
    tab_movies, tab_series = st.tabs(["🎬 MOVIES UNIVERSE (4,760)", "📺 TV SERIES REALM (1,926)"])

    # =============================================================
    # TAB 1: MOVIES DIMENSION
    # =============================================================
    with tab_movies:
        st.markdown(
            """
            <div class="cyber-hero">
                <h1 style="margin:0; font-size: 2.2rem; color: #fff; text-shadow: 0 0 15px rgba(0,242,254,0.5);">
                    🎬 MOVIE RECOMMENDATION MATRIX
                </h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8;">
                    TF-IDF Vector Space Analysis across 4,760 Feature Films (Cast, Director, Tagline & Themes)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_m_input, col_m_rnd = st.columns([4, 1.2])
        with col_m_input:
            if "movie_drop_key" not in st.session_state:
                st.session_state.movie_drop_key = st.session_state.current_movie

            current_m_idx = (
                movie_titles.index(st.session_state.movie_drop_key)
                if st.session_state.movie_drop_key in movie_titles
                else 0
            )

            chosen_movie = st.selectbox(
                "🎯 Select a movie you admire:",
                options=movie_titles,
                index=current_m_idx,
                key="movie_drop_key",
            )
            st.session_state.current_movie = chosen_movie

        with col_m_rnd:
            st.write("")
            st.write("")
            st.button("🎲 Surprise Movie", on_click=surprise_movie_callback, use_container_width=True)

        # Compute Movie Recommendations
        with st.spinner(f"Scanning semantic cosmos for '{st.session_state.current_movie}'..."):
            raw_movie_res = movie_engine.recommend(
                st.session_state.current_movie,
                top_k=max(top_k * 4, 40),
                exclude_self=anti_self_toggle,
            )

        m_matched = raw_movie_res["matched_title"] or st.session_state.current_movie
        m_conf = raw_movie_res["match_confidence"]
        m_fallback = raw_movie_res["is_fallback"]
        raw_m_list = raw_movie_res["recommendations"]

        if m_fallback:
            st.warning(f"⚠️ {raw_movie_res['message']}")
        else:
            st.success(f"🎯 Matched: **{m_matched}** &nbsp;•&nbsp; Match Confidence: **{int(m_conf * 100)}%**")

        # Movie Sort Controls Bar
        st.markdown("#### 🔀 Sort Results (High to Low Stars, Ratings, Relevance)")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns([1.3, 1.4, 1.4, 1.4, 1.4])

        with m_col1:
            active = st.session_state.movie_sort == "🎯 Best Match"
            if st.button("🎯 Best Match", key="m_btn_best", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.movie_sort = "🎯 Best Match"
                st.rerun()
        with m_col2:
            active = st.session_state.movie_sort == "⭐ Stars: High → Low"
            if st.button("⭐ Stars: High → Low", key="m_btn_h2l", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.movie_sort = "⭐ Stars: High → Low"
                st.rerun()
        with m_col3:
            active = st.session_state.movie_sort == "⭐ Stars: Low → High"
            if st.button("⭐ Stars: Low → High", key="m_btn_l2h", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.movie_sort = "⭐ Stars: Low → High"
                st.rerun()
        with m_col4:
            active = st.session_state.movie_sort == "🔥 Most Popular"
            if st.button("🔥 Most Popular", key="m_btn_pop", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.movie_sort = "🔥 Most Popular"
                st.rerun()
        with m_col5:
            active = st.session_state.movie_sort == "📅 Newest First"
            if st.button("📅 Newest First", key="m_btn_new", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.movie_sort = "📅 Newest First"
                st.rerun()

        final_movie_list = apply_sorting_and_tier_filter(
            raw_m_list,
            sort_choice=st.session_state.movie_sort,
            star_tier=chosen_star_tier,
            min_stars=min_stars_slider,
            selected_genres=selected_genres,
        )[:top_k]

        st.caption(f"Currently Sorted by: **{st.session_state.movie_sort}** &nbsp;•&nbsp; Showing **{len(final_movie_list)}** Films")
        st.markdown("---")

        if not final_movie_list:
            st.info("No movies satisfied the active Star or Genre restrictions. Lower the filter thresholds in the sidebar.")
        else:
            for m in final_movie_list:
                m_rank = m.get("display_rank", 1)
                m_title = m.get("title", "")
                m_vote = m.get("vote_average", 0.0)
                m_votes_cnt = m.get("vote_count", 0)
                m_pct = m.get("match_percentage", "N/A")
                m_genre = m.get("genre", "")
                m_dir = m.get("director", "Unknown")
                m_cast = m.get("cast", "Unknown")
                m_tag = m.get("tagline", "")
                m_over = m.get("overview", "")
                m_date = m.get("release_date", "")
                m_time = m.get("runtime", 0)
                m_id = f"m_{m.get('movie_id', m_rank)}"
                m_year = m_date.split("-")[-1] if m_date else ""

                with st.container():
                    col_info, col_act = st.columns([3.8, 1.2])
                    with col_info:
                        clean_m_year = str(m_year).strip("() ")
                        yr_txt = f" <span style='color:#64748b; font-size:1.1rem;'>({clean_m_year})</span>" if clean_m_year else ""
                        st.markdown(f"<h3 style='margin:0; color:#fff;'>#{m_rank}. {m_title}{yr_txt}</h3>", unsafe_allow_html=True)
                        if m_tag:
                            st.markdown(f"<p style='color:#00f2fe; font-style:italic; margin:4px 0 8px 0;'>\"{m_tag}\"</p>", unsafe_allow_html=True)

                        if m_genre:
                            pills = "".join([f"<span class='badge-holo-genre'>{g}</span>" for g in m_genre.split() if g])
                            st.markdown(f"<div style='margin-bottom:8px;'>{pills}</div>", unsafe_allow_html=True)

                        st.markdown(f"🎬 <b style='color:#e2e8f0;'>Director:</b> {m_dir} &nbsp;|&nbsp; 👥 <b style='color:#e2e8f0;'>Cast:</b> {m_cast}", unsafe_allow_html=True)
                        if m_over:
                            with st.expander("📖 Storyline & Synopsis"):
                                st.write(m_over)

                    with col_act:
                        if not m_fallback:
                            st.markdown(f"<div class='badge-holo-match'>🎯 {m_pct} Match</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='badge-holo-meta'>🔥 Popular Film</div>", unsafe_allow_html=True)
                        st.write("")
                        st.markdown(f"<div class='badge-holo-star'>⭐ {m_vote:.1f} / 10</div>", unsafe_allow_html=True)
                        st.caption(f"👥 {m_votes_cnt:,} reviews")
                        if m_time > 0:
                            st.markdown(f"<span class='badge-holo-meta'>🕒 {m_time} min</span>", unsafe_allow_html=True)
                        st.write("")

                        in_wl = any(w["id"] == m_id for w in st.session_state.watchlist)
                        if in_wl:
                            st.success("✓ Locked in HUD")
                        else:
                            if st.button("➕ Watchlist", key=f"btn_{m_id}", use_container_width=True):
                                st.session_state.watchlist.append({
                                    "id": m_id,
                                    "title": m_title,
                                    "vote_average": m_vote,
                                    "type": "Movie"
                                })
                                st.rerun()

                    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)

    # =============================================================
    # TAB 2: TV SERIES REALM
    # =============================================================
    with tab_series:
        st.markdown(
            """
            <div class="cyber-hero" style="border-left-color: #a855f7; box-shadow: 0 15px 35px rgba(0,0,0,0.6), 0 0 20px rgba(168,85,247,0.2);">
                <h1 style="margin:0; font-size: 2.2rem; color: #fff; text-shadow: 0 0 15px rgba(168,85,247,0.6);">
                    📺 TV SERIES NEURAL MATRIX
                </h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8;">
                    Deep Semantic Matching across 1,926 Binge-Worthy TV Shows & Streaming Series
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_s_input, col_s_rnd = st.columns([4, 1.2])
        with col_s_input:
            if "series_drop_key" not in st.session_state:
                st.session_state.series_drop_key = st.session_state.current_series

            current_s_idx = (
                series_titles.index(st.session_state.series_drop_key)
                if st.session_state.series_drop_key in series_titles
                else 0
            )

            chosen_series = st.selectbox(
                "🎯 Select a TV show you love:",
                options=series_titles,
                index=current_s_idx,
                key="series_drop_key",
            )
            st.session_state.current_series = chosen_series

        with col_s_rnd:
            st.write("")
            st.write("")
            st.button("🎲 Surprise Series", on_click=surprise_series_callback, use_container_width=True)

        # Compute TV Series Recommendations
        with st.spinner(f"Scanning narrative multiverse for '{st.session_state.current_series}'..."):
            raw_series_res = series_engine.recommend(
                st.session_state.current_series,
                top_k=max(top_k * 4, 40),
                exclude_self=anti_self_toggle,
            )

        s_matched = raw_series_res["matched_title"] or st.session_state.current_series
        s_conf = raw_series_res["match_confidence"]
        s_fallback = raw_series_res["is_fallback"]
        raw_s_list = raw_series_res["recommendations"]

        if s_fallback:
            st.warning(f"⚠️ {raw_series_res['message']}")
        else:
            st.success(f"🎯 Matched: **{s_matched}** &nbsp;•&nbsp; Match Confidence: **{int(s_conf * 100)}%**")

        # Series Sort Controls Bar
        st.markdown("#### 🔀 Sort Results (High to Low Stars, Ratings, Relevance)")
        s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns([1.3, 1.4, 1.4, 1.4, 1.4])

        with s_col1:
            active = st.session_state.series_sort == "🎯 Best Match"
            if st.button("🎯 Best Match", key="s_btn_best", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.series_sort = "🎯 Best Match"
                st.rerun()
        with s_col2:
            active = st.session_state.series_sort == "⭐ Stars: High → Low"
            if st.button("⭐ Stars: High → Low", key="s_btn_h2l", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.series_sort = "⭐ Stars: High → Low"
                st.rerun()
        with s_col3:
            active = st.session_state.series_sort == "⭐ Stars: Low → High"
            if st.button("⭐ Stars: Low → High", key="s_btn_l2h", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.series_sort = "⭐ Stars: Low → High"
                st.rerun()
        with s_col4:
            active = st.session_state.series_sort == "🔥 Most Popular"
            if st.button("🔥 Most Popular", key="s_btn_pop", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.series_sort = "🔥 Most Popular"
                st.rerun()
        with s_col5:
            active = st.session_state.series_sort == "📅 Newest First"
            if st.button("📅 Newest First", key="s_btn_new", use_container_width=True, type="primary" if active else "secondary"):
                st.session_state.series_sort = "📅 Newest First"
                st.rerun()

        final_series_list = apply_sorting_and_tier_filter(
            raw_s_list,
            sort_choice=st.session_state.series_sort,
            star_tier=chosen_star_tier,
            min_stars=min_stars_slider,
            selected_genres=selected_genres,
        )[:top_k]

        st.caption(f"Currently Sorted by: **{st.session_state.series_sort}** &nbsp;•&nbsp; Showing **{len(final_series_list)}** TV Shows")
        st.markdown("---")

        if not final_series_list:
            st.info("No TV shows satisfied the active Star or Genre restrictions. Lower the filter thresholds in the sidebar.")
        else:
            for s in final_series_list:
                s_rank = s.get("display_rank", 1)
                s_title = s.get("title", "")
                s_vote = s.get("vote_average", 0.0)
                s_votes_cnt = s.get("vote_count", 0)
                s_pct = s.get("match_percentage", "N/A")
                s_genre = s.get("genre", "")
                s_over = s.get("overview", "")
                s_year = s.get("year", "")
                s_time = s.get("runtime", "")
                s_id = f"s_{s.get('series_id', s_rank)}"

                with st.container():
                    col_info, col_act = st.columns([3.8, 1.2])
                    with col_info:
                        clean_s_year = re.sub(r"TV Series|TV Mini Series|TV Movie", "", str(s_year), flags=re.I).strip("() ")
                        yr_txt = f" <span style='color:#a855f7; font-size:1.1rem;'>({clean_s_year})</span>" if clean_s_year else ""
                        st.markdown(f"<h3 style='margin:0; color:#fff;'>#{s_rank}. {s_title}{yr_txt}</h3>", unsafe_allow_html=True)

                        if s_genre:
                            pills = "".join([f"<span class='badge-holo-genre badge-genre-purple'>{g}</span>" for g in s_genre.split() if g])
                            st.markdown(f"<div style='margin-bottom:8px;'>{pills}</div>", unsafe_allow_html=True)

                        if s_time:
                            st.markdown(f"📺 <b style='color:#e2e8f0;'>Format:</b> Episodic ({s_time})", unsafe_allow_html=True)

                        if s_over:
                            with st.expander("📖 Show Synopsis & Lore"):
                                st.write(s_over)

                    with col_act:
                        if not s_fallback:
                            st.markdown(f"<div class='badge-holo-purple'>🎯 {s_pct} Match</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='badge-holo-meta'>🔥 Trending Show</div>", unsafe_allow_html=True)
                        st.write("")
                        st.markdown(f"<div class='badge-holo-star'>⭐ {s_vote:.1f} / 10</div>", unsafe_allow_html=True)
                        st.caption(f"👥 {s_votes_cnt:,} IMDb votes")
                        st.write("")

                        in_wl = any(w["id"] == s_id for w in st.session_state.watchlist)
                        if in_wl:
                            st.success("✓ Locked in HUD")
                        else:
                            if st.button("➕ Watchlist", key=f"btn_{s_id}", use_container_width=True):
                                st.session_state.watchlist.append({
                                    "id": s_id,
                                    "title": s_title,
                                    "vote_average": s_vote,
                                    "type": "TV Series"
                                })
                                st.rerun()

                    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
