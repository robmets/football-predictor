"""
Football Predictor — Streamlit Dashboard
Start with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from src.models.poisson_model import PoissonModel
from src.simulation.monte_carlo import MonteCarloSimulator
from src.utils.database import get_session, Match, Team
from config.config import config
from src.collectors.odds_collector import OddsCollector
from src.features.value_bet_detector import ValueBetDetector

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1628;
    border-right: 1px solid #1e2d4a;
}

/* Header */
.dashboard-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0;
    line-height: 1.1;
}

.dashboard-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}

.metric-value.green  { color: #4ade80; }
.metric-value.blue   { color: #38bdf8; }
.metric-value.amber  { color: #fbbf24; }
.metric-value.red    { color: #f87171; }

/* Team vs Team header */
.matchup-header {
    background: linear-gradient(135deg, #111827 0%, #0f1628 100%);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
}

.team-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #f1f5f9;
}

.vs-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #38bdf8;
    background: #0c1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 4px 12px;
    letter-spacing: 0.1em;
}

/* Probability bar */
.prob-bar-container {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
}

.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.prob-team {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: #e2e8f0;
    margin-bottom: 10px;
}

/* Score chip */
.score-chip {
    display: inline-block;
    background: #1e2d4a;
    border: 1px solid #2d4a6e;
    border-radius: 8px;
    padding: 6px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 4px;
}

.score-chip.top {
    background: #0c1e38;
    border-color: #38bdf8;
    color: #38bdf8;
}

/* Warning banner */
.warning-banner {
    background: #1c1204;
    border: 1px solid #78350f;
    border-radius: 10px;
    padding: 14px 18px;
    color: #fbbf24;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}

/* Divider */
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #38bdf8;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* Hide default streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_features(league: str = "BL1") -> pd.DataFrame:
    path = Path(f"data/processed/features_{league}.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_teams_from_db() -> list[str]:
    try:
        session = get_session()
        teams = session.query(Team).all()
        session.close()
        return sorted([t.name for t in teams if t.name])
    except Exception:
        return []


@st.cache_resource
def get_fitted_model(league: str = "BL1") -> PoissonModel | None:
    df = load_features(league)
    if df.empty:
        return None
    model = PoissonModel()
    model.fit(df)
    return model


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="dashboard-header" style="font-size:1.4rem">⚽ FP</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Football Predictor</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🎯 Match Prediction", "📊 Team Ratings", "💰 Value Bets", "📈 Data Explorer", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    league = st.selectbox(
        "Liga",
        options=["BL1", "PL", "PD", "SA", "FL1"],
        format_func=lambda x: config.SUPPORTED_LEAGUES.get(x, x),
    )
    sims = st.select_slider(
        "Simulationen",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
        format_func=lambda x: f"{x:,}",
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#334155;">'
        f'Stand: {datetime.now().strftime("%d.%m.%Y")}<br>'
        'Modell: Poisson + Monte Carlo</p>',
        unsafe_allow_html=True
    )


# ── Helper: Plotly Gauge ──────────────────────────────────────────────────────

def prob_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "family": "Syne", "color": "#f1f5f9"}},
        title={"text": title, "font": {"size": 11, "family": "DM Mono", "color": "#64748b"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#111827",
            "bordercolor": "#1e2d4a",
            "steps": [{"range": [0, 100], "color": "#0a0e1a"}],
            "threshold": {
                "line": {"color": color, "width": 2},
                "thickness": 0.75,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
    )
    return fig


def score_heatmap(score_matrix: dict, home: str, away: str) -> go.Figure:
    max_g = 6
    z = [[0.0] * (max_g + 1) for _ in range(max_g + 1)]
    for (h, a), p in score_matrix.items():
        if h <= max_g and a <= max_g:
            z[h][a] = round(p * 100, 2)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(i) for i in range(max_g + 1)],
        y=[str(i) for i in range(max_g + 1)],
        colorscale=[[0, "#0a0e1a"], [0.5, "#1e3a5f"], [1, "#38bdf8"]],
        showscale=True,
        colorbar=dict(
            title=dict(text="%", font=dict(family="DM Mono", size=10, color="#64748b")),
            tickfont=dict(family="DM Mono", size=10, color="#64748b"),
        ),
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 9, "family": "DM Mono"},
    ))
    fig.update_layout(
        height=320,
        xaxis=dict(title=dict(text=f"Tore {away}", font=dict(family="DM Mono", size=10, color="#64748b")),
                   tickfont=dict(family="DM Mono", size=10, color="#94a3b8")),
        yaxis=dict(title=dict(text=f"Tore {home}", font=dict(family="DM Mono", size=10, color="#64748b")),
                   tickfont=dict(family="DM Mono", size=10, color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0e1a",
        margin=dict(t=10, b=40, l=50, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: MATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎯 Match Prediction":
    st.markdown('<p class="dashboard-header">Match Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Poisson-Modell · Monte Carlo Simulation · Verletzungsgewichtung</p>', unsafe_allow_html=True)
    st.markdown("---")

    df = load_features(league)
    teams = load_teams_from_db()

    if df.empty:
        st.error(f"Keine Daten für {league}. Bitte zuerst `python scripts/build_features.py --league {league}` ausführen.")
        st.stop()

    all_teams = sorted(
        t for t in (set(df["home_team"].dropna()) | set(df["away_team"].dropna()))
        if not str(t).startswith("ID:")
    )
    if teams:
        all_teams = sorted(set(all_teams) | set(teams))

    col1, col2, col3 = st.columns([5, 1, 5])
    with col1:
        home_team = st.selectbox("🏠 Heimmannschaft", all_teams, index=0)
    with col2:
        st.markdown("<br><div style='text-align:center'><span class='vs-badge'>VS</span></div>", unsafe_allow_html=True)
    with col3:
        default_away = 1 if len(all_teams) > 1 else 0
        away_team = st.selectbox("✈️ Auswärtsmannschaft", all_teams, index=default_away)

    if home_team == away_team:
        st.warning("Bitte zwei verschiedene Teams wählen.")
        st.stop()

    run_btn = st.button("⚡ Simulation starten", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Modell wird gefittet & {sims:,} Spiele simuliert..."):
            model = PoissonModel()
            model.fit(df)
            sim = MonteCarloSimulator(model)
            result = sim.simulate(home_team, away_team, n=sims)
            poisson_pred = model.predict(home_team, away_team)
            result["score_matrix"] = poisson_pred["score_matrix"]

        # ── Matchup Header
        st.markdown(f"""
        <div class="matchup-header">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="team-name">{home_team}</div>
                <div><span class="vs-badge">VS</span></div>
                <div class="team-name">{away_team}</div>
            </div>
            <div style="margin-top:12px;font-family:'DM Mono',monospace;font-size:0.72rem;color:#475569;">
                {sims:,} Simulationen · Konfidenz: {result['confidence']} · Favorit: {result['favourite']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 3 Gauges
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(prob_gauge(result["prob_home_win"], f"HEIMSIEG\n{home_team[:18]}", "#4ade80"), use_container_width=True)
        with g2:
            st.plotly_chart(prob_gauge(result["prob_draw"], "UNENTSCHIEDEN", "#94a3b8"), use_container_width=True)
        with g3:
            st.plotly_chart(prob_gauge(result["prob_away_win"], f"AUSWÄRTSSIEG\n{away_team[:18]}", "#f87171"), use_container_width=True)

        # ── Expected Goals & Markets
        st.markdown('<p class="section-title">Erwartete Tore & Märkte</p>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)

        def metric(col, label, value, color=""):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color}">{value}</div>
            </div>""", unsafe_allow_html=True)

        metric(m1, f"xG {home_team[:12]}", f"{result['expected_home_goals']:.2f}", "blue")
        metric(m2, f"xG {away_team[:12]}", f"{result['expected_away_goals']:.2f}", "blue")
        metric(m3, "BTTS", f"{result['prob_btts']:.0%}", "amber")
        metric(m4, "Über 2.5", f"{result['prob_over_2_5']:.0%}", "amber")
        metric(m5, "Über 3.5", f"{result['prob_over_3_5']:.0%}", "amber")

        # ── Score Heatmap + Top Scores
        st.markdown('<p class="section-title">Ergebnis-Wahrscheinlichkeiten</p>', unsafe_allow_html=True)
        hc1, hc2 = st.columns([3, 2])

        with hc1:
            fig_heat = score_heatmap(result.get("score_matrix", {}), home_team, away_team)
            st.plotly_chart(fig_heat, use_container_width=True)

        with hc2:
            st.markdown("**Wahrscheinlichste Ergebnisse**")
            for i, s in enumerate(result["top_scores"][:8]):
                chip_class = "score-chip top" if i == 0 else "score-chip"
                st.markdown(
                    f'<span class="{chip_class}">{s["score"]}</span>'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.8rem;color:#475569;margin-left:8px;">{s["probability"]:.2f}%</span>',
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: TEAM RATINGS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Team Ratings":
    st.markdown('<p class="dashboard-header">Team Ratings</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Angriff · Abwehr · Gesamtstärke aus historischen Daten</p>', unsafe_allow_html=True)
    st.markdown("---")

    df = load_features(league)
    if df.empty:
        st.error("Keine Feature-Daten gefunden.")
        st.stop()

    with st.spinner("Poisson-Modell wird gefittet..."):
        model = PoissonModel()
        model.fit(df)
        ratings = model.team_ratings()

    # Bar chart: overall strength
    fig_bar = go.Figure(go.Bar(
        x=ratings["overall"],
        y=ratings["team"],
        orientation="h",
        marker=dict(
            color=ratings["overall"],
            colorscale=[[0, "#1e3a5f"], [1, "#38bdf8"]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in ratings["overall"]],
        textposition="outside",
        textfont=dict(family="DM Mono", size=10, color="#94a3b8"),
    ))
    fig_bar.update_layout(
        height=max(350, len(ratings) * 28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1e2d4a", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
        yaxis=dict(tickfont=dict(family="Inter", size=11, color="#e2e8f0"), autorange="reversed"),
        margin=dict(l=10, r=80, t=10, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Attack vs Defence scatter
    st.markdown('<p class="section-title">Angriff vs Abwehr</p>', unsafe_allow_html=True)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=ratings["attack"],
        y=ratings["defence"],
        mode="markers+text",
        text=ratings["team"],
        textposition="top center",
        textfont=dict(family="DM Mono", size=9, color="#94a3b8"),
        marker=dict(
            size=12,
            color=ratings["overall"],
            colorscale=[[0, "#1e3a5f"], [1, "#38bdf8"]],
            showscale=False,
            line=dict(width=1, color="#1e2d4a"),
        ),
    ))
    # Quadrant lines
    avg_att = ratings["attack"].mean()
    avg_def = ratings["defence"].mean()
    fig_scatter.add_hline(y=avg_def, line_dash="dot", line_color="#334155", line_width=1)
    fig_scatter.add_vline(x=avg_att, line_dash="dot", line_color="#334155", line_width=1)

    fig_scatter.update_layout(
        height=420,
        xaxis=dict(title=dict(text="Angriffsstärke", font=dict(family="DM Mono", size=10, color="#64748b")),
                   tickfont=dict(family="DM Mono", size=9, color="#64748b"), gridcolor="#111827"),
        yaxis=dict(title=dict(text="Abwehrschwäche (niedrig = besser)", font=dict(family="DM Mono", size=10, color="#64748b")),
                   tickfont=dict(family="DM Mono", size=9, color="#64748b"), gridcolor="#111827"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0e1a",
        margin=dict(t=10, b=50, l=60, r=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("Rohdaten anzeigen"):
        st.dataframe(
            ratings.style.background_gradient(subset=["overall", "attack"], cmap="Blues"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: VALUE BETS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💰 Value Bets":
    st.markdown('<p class="dashboard-header">Value Bets</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Modell vs. Buchmacher · Edge-Analyse · Marktvergleich</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not config.ODDS_API_KEY:
        st.warning("⚠️ Kein ODDS_API_KEY konfiguriert. Bitte in config/.env eintragen.")
        st.code("ODDS_API_KEY=your_key_here", language="bash")
        st.markdown("Kostenlosen Key bekommst du auf [the-odds-api.com](https://the-odds-api.com)")
        st.stop()

    df = load_features(league)
    if df.empty:
        st.error("Keine Feature-Daten. Bitte build_features.py ausführen.")
        st.stop()

    # Manual match input for comparison
    st.markdown('<p class="section-title">Einzelspiel analysieren</p>', unsafe_allow_html=True)

    db_teams = load_teams_from_db()
    all_teams = sorted(
        t for t in (set(df["home_team"].dropna()) | set(df["away_team"].dropna()) | set(db_teams))
        if not str(t).startswith("ID:")
    )

    vc1, vc2, vc3 = st.columns([5, 1, 5])
    with vc1:
        vb_home = st.selectbox("🏠 Heimmannschaft", all_teams, key="vb_home")
    with vc2:
        st.markdown("<br><div style='text-align:center'><span class='vs-badge'>VS</span></div>", unsafe_allow_html=True)
    with vc3:
        vb_away = st.selectbox("✈️ Auswärtsmannschaft", all_teams, index=1, key="vb_away")

    analyze_btn = st.button("🔍 Analysieren", type="primary", use_container_width=True)

    if analyze_btn and vb_home != vb_away:
        with st.spinner("Modell wird gefittet & Odds werden geladen..."):
            # Fit model
            model = PoissonModel()
            model.fit(df)
            sim = MonteCarloSimulator(model)
            model_result = sim.simulate(vb_home, vb_away, n=10_000)

            # Fetch odds
            odds_collector = OddsCollector()
            consensus = odds_collector.get_consensus_odds(league)

        if consensus.empty:
            st.error("Keine Marktdaten verfügbar. Entweder kein Key, keine Verbindung, oder das Spiel ist nicht gelistet.")

            # Show model-only result
            st.markdown('<p class="section-title">Modell-Vorhersage (ohne Marktvergleich)</p>', unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            def model_metric(col, label, val, color):
                col.markdown(f'''<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color}">{val}</div>
                </div>''', unsafe_allow_html=True)
            model_metric(mc1, f"Heimsieg {vb_home[:14]}", f"{model_result['prob_home_win']:.1%}", "green")
            model_metric(mc2, "Unentschieden", f"{model_result['prob_draw']:.1%}", "")
            model_metric(mc3, f"Auswärtssieg {vb_away[:14]}", f"{model_result['prob_away_win']:.1%}", "red")
        else:
            # Find the match in odds
            match_odds = consensus[
                (consensus["home_team"].str.contains(vb_home[:6], case=False, na=False)) &
                (consensus["away_team"].str.contains(vb_away[:6], case=False, na=False))
            ]

            if match_odds.empty:
                st.warning(f"Spiel {vb_home} vs {vb_away} nicht in aktuellen Marktdaten. Spiel evtl. noch nicht gelistet.")
            else:
                mkt = match_odds.iloc[0]
                detector = ValueBetDetector()
                analysis = detector.analyze(
                    home_team=vb_home, away_team=vb_away,
                    model_home=model_result["prob_home_win"],
                    model_draw=model_result["prob_draw"],
                    model_away=model_result["prob_away_win"],
                    market_home=float(mkt["market_home"]),
                    market_draw=float(mkt["market_draw"]),
                    market_away=float(mkt["market_away"]),
                    avg_margin=float(mkt.get("avg_margin", 0)),
                )

                # ── Vergleichs-Tabelle
                st.markdown('<p class="section-title">Modell vs. Markt</p>', unsafe_allow_html=True)

                comp_data = {
                    "Outcome":     ["Heimsieg", "Unentschieden", "Auswärtssieg"],
                    "Modell %":    [analysis["model_home"], analysis["model_draw"], analysis["model_away"]],
                    "Markt %":     [analysis["market_home"], analysis["market_draw"], analysis["market_away"]],
                    "Edge (PP)":   [analysis["edge_home"], analysis["edge_draw"], analysis["edge_away"]],
                }
                comp_df = pd.DataFrame(comp_data)

                # Color-coded bar chart
                fig_comp = go.Figure()
                colors_model  = ["#4ade80", "#94a3b8", "#f87171"]
                colors_market = ["#166534", "#334155", "#7f1d1d"]

                for i, outcome in enumerate(["Heimsieg", "Unentschieden", "Auswärtssieg"]):
                    fig_comp.add_trace(go.Bar(
                        name=f"Modell — {outcome}",
                        x=[outcome], y=[comp_data["Modell %"][i]],
                        marker_color=colors_model[i],
                        text=f"{comp_data['Modell %'][i]:.1f}%",
                        textposition="outside",
                        textfont=dict(family="DM Mono", size=10),
                        offsetgroup=0,
                    ))
                    fig_comp.add_trace(go.Bar(
                        name=f"Markt — {outcome}",
                        x=[outcome], y=[comp_data["Markt %"][i]],
                        marker_color=colors_market[i],
                        text=f"{comp_data['Markt %'][i]:.1f}%",
                        textposition="outside",
                        textfont=dict(family="DM Mono", size=10),
                        offsetgroup=1,
                    ))

                fig_comp.update_layout(
                    barmode="group", height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
                    xaxis=dict(tickfont=dict(family="DM Mono", size=11, color="#e2e8f0"), gridcolor="#111827"),
                    yaxis=dict(title=dict(text="%", font=dict(family="DM Mono", size=10, color="#64748b")),
                               tickfont=dict(family="DM Mono", size=9, color="#64748b"), gridcolor="#111827"),
                    legend=dict(font=dict(family="DM Mono", size=9, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(t=30, b=10, l=40, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # ── Edge-Anzeige
                ec1, ec2, ec3, ec4 = st.columns(4)
                def edge_metric(col, label, edge):
                    color = "green" if edge >= 5 else ("red" if edge <= -5 else "")
                    prefix = "▲" if edge > 0 else ("▼" if edge < 0 else "=")
                    col.markdown(f'''<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value {color}">{prefix}{abs(edge):.1f} PP</div>
                    </div>''', unsafe_allow_html=True)

                edge_metric(ec1, "Edge Heimsieg", analysis["edge_home"])
                edge_metric(ec2, "Edge Unentschieden", analysis["edge_draw"])
                edge_metric(ec3, "Edge Auswärtssieg", analysis["edge_away"])
                ec4.markdown(f'''<div class="metric-card">
                    <div class="metric-label">Buchmacher-Marge</div>
                    <div class="metric-value amber">{analysis["avg_margin_pct"]:.1f}%</div>
                </div>''', unsafe_allow_html=True)

                # ── Value Bets
                st.markdown('<p class="section-title">Value Bets gefunden</p>', unsafe_allow_html=True)
                if analysis["has_value"]:
                    for vb in analysis["value_bets"]:
                        st.markdown(f'''
                        <div style="background:#0c1e0c;border:1px solid #166534;border-radius:10px;padding:16px 20px;margin:8px 0;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div>
                                    <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#4ade80">{vb["rating"]}</span>
                                    <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#94a3b8;margin-left:12px">{vb["outcome"]}</span>
                                </div>
                                <div style="text-align:right;font-family:'DM Mono',monospace;font-size:0.85rem;color:#86efac">
                                    Modell: {vb["model_prob"]:.1f}% | Markt: {vb["market_prob"]:.1f}% | Edge: +{vb["edge_pct"]:.1f} PP
                                </div>
                            </div>
                        </div>''', unsafe_allow_html=True)
                else:
                    st.markdown('''
                    <div style="background:#111827;border:1px solid #1e2d4a;border-radius:10px;padding:16px 20px;color:#64748b;font-family:'DM Mono',monospace;font-size:0.85rem;">
                        Kein Value gefunden — Modell und Markt sind sich einig (Edge < 5 Prozentpunkte)
                    </div>''', unsafe_allow_html=True)

                if analysis["disagreement"]:
                    st.markdown(f'''
                    <div class="warning-banner" style="margin-top:12px">
                        ⚡ Modell und Markt sind sich uneinig über den Favoriten:<br>
                        Modell → <strong>{analysis["model_favourite"]}</strong> &nbsp;|&nbsp; Markt → <strong>{analysis["market_favourite"]}</strong>
                    </div>''', unsafe_allow_html=True)

elif page == "📈 Data Explorer":
    st.markdown('<p class="dashboard-header">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Historische Spieldaten · Feature Matrix · Trendanalyse</p>', unsafe_allow_html=True)
    st.markdown("---")

    df = load_features(league)
    if df.empty:
        st.error("Keine Daten gefunden.")
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    total = len(df)
    h = (df["result"] == "H").sum()
    d = (df["result"] == "D").sum()
    a = (df["result"] == "A").sum()

    def kpi(col, label, val, color=""):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color}">{val}</div></div>', unsafe_allow_html=True)

    kpi(k1, "Spiele gesamt", total)
    kpi(k2, "Heimsiege", f"{h/total:.0%}", "green")
    kpi(k3, "Unentschieden", f"{d/total:.0%}")
    kpi(k4, "Auswärtssiege", f"{a/total:.0%}", "red")

    # Goals over time
    st.markdown('<p class="section-title">Tore pro Spieltag (Saison-Trend)</p>', unsafe_allow_html=True)
    df_goals = df.dropna(subset=["home_goals", "away_goals", "matchday"]).copy()
    df_goals["total_goals"] = df_goals["home_goals"] + df_goals["away_goals"]
    trend = df_goals.groupby(["season", "matchday"])["total_goals"].mean().reset_index()

    fig_trend = go.Figure()
    for season in trend["season"].unique():
        s = trend[trend["season"] == season]
        fig_trend.add_trace(go.Scatter(
            x=s["matchday"], y=s["total_goals"],
            name=str(season), mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=5),
        ))
    fig_trend.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        xaxis=dict(title="Spieltag", gridcolor="#111827", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
        yaxis=dict(title="Ø Tore/Spiel", gridcolor="#111827", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
        legend=dict(font=dict(family="DM Mono", size=9, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Result distribution pie
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown('<p class="section-title">Ergebnisverteilung</p>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Heimsieg", "Unentschieden", "Auswärtssieg"],
            values=[h, d, a],
            hole=0.55,
            marker=dict(colors=["#4ade80", "#94a3b8", "#f87171"]),
            textfont=dict(family="DM Mono", size=10),
        ))
        fig_pie.update_layout(
            height=260, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(family="DM Mono", size=9, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with rc2:
        st.markdown('<p class="section-title">Tore-Verteilung</p>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["home_goals"].dropna(), name="Heimtore",
            marker_color="#38bdf8", opacity=0.7, xbins=dict(size=1),
        ))
        fig_hist.add_trace(go.Histogram(
            x=df["away_goals"].dropna(), name="Auswärtstore",
            marker_color="#f87171", opacity=0.7, xbins=dict(size=1),
        ))
        fig_hist.update_layout(
            barmode="overlay", height=260,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
            xaxis=dict(gridcolor="#111827", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
            yaxis=dict(gridcolor="#111827", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
            legend=dict(font=dict(family="DM Mono", size=9, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=40, l=50, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Feature Matrix (letzte 20 Spiele)"):
        display_cols = ["date", "home_team", "away_team", "result",
                        "home_form_ppg", "away_form_ppg",
                        "home_goals_scored_avg", "away_goals_scored_avg",
                        "position_diff", "h2h_home_win_rate"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].tail(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown('<p class="dashboard-header">Über das Projekt</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Data Analytics · Data Science · Sports Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### Methodik

    Dieses System kombiniert mehrere datengetriebene Ansätze um Spielausgänge vorherzusagen:

    **1. Poisson-Modell (Dixon-Coles)**
    Fußballtore folgen einer Poisson-Verteilung. Das Modell schätzt für jedes Team eine
    Angriffsstärke und Abwehrschwäche via Maximum Likelihood Estimation — plus einen
    Heimvorteil-Koeffizienten. Damit berechnen wir die erwarteten Tore (λ) für beide Teams.

    **2. Monte Carlo Simulation**
    Mit den geschätzten λ-Werten simulieren wir das Spiel 10.000+ mal. Jede Simulation
    zieht unabhängig Tore aus der Poisson-Verteilung. Die aggregierten Ergebnisse ergeben
    saubere Wahrscheinlichkeiten inkl. exakter Ergebnis-Wahrscheinlichkeiten.

    **3. Verletzungsgewichtung (Transfermarkt)**
    Fehlende Schlüsselspieler werden über ihren Marktwert-Anteil am Gesamtkader quantifiziert.
    Teams mit > 15% Kaderausfall erhalten einen Schwächungs-Malus auf ihre Angriffsstärke.

    **4. Feature Engineering**
    Für jedes Spiel werden 34 Features berechnet: Form (gewichtet), xG-Durchschnitte,
    Head-to-Head, Heimvorteil, Restzeit seit letztem Spiel, Tabellenplatz.

    ### Datenquellen
    - **football-data.org** — Spielergebnisse, Spielpläne
    - **Transfermarkt** (lokaler Scraper) — Kaderwerte, Verletzungen
    - **OpenWeatherMap** — Wetterbedingungen
    - **The Odds API** — Buchmacherquoten

    ### These
    > *"Mit ausreichend strukturierten Daten lassen sich zukünftige Ereignisse
    > mit messbarer Wahrscheinlichkeit vorhersagen — nicht deterministisch,
    > aber statistisch signifikant besser als der Zufall."*
    """)