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
from src.models.xgboost_model import XGBoostFeedbackModel
from src.collectors.weather_collector import WeatherCollector
from src.features.injury_impact import calculate_missing_impact
from src.features.live_form import LiveFormCalculator
from src.features.value_bet_detector import ValueBetDetector
from src.features.context_engine import ContextEngine
from src.collectors.sofascore_collector import SofascoreCollector

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
.stApp { background-color: #0a0e1a; color: #e2e8f0; }
[data-testid="stSidebar"] { background-color: #0f1628; border-right: 1px solid #1e2d4a; }
.dashboard-header {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; margin-bottom: 0; line-height: 1.1;
}
.dashboard-sub {
    font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #475569;
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px;
}
.metric-card {
    background: #111827; border: 1px solid #1e2d4a; border-radius: 12px;
    padding: 20px 24px; text-align: center;
}
.metric-label {
    font-family: 'DM Mono', monospace; font-size: 0.68rem; color: #64748b;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;
}
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.metric-value.green  { color: #4ade80; }
.metric-value.blue   { color: #38bdf8; }
.metric-value.amber  { color: #fbbf24; }
.metric-value.red    { color: #f87171; }
.matchup-header {
    background: linear-gradient(135deg, #111827 0%, #0f1628 100%);
    border: 1px solid #1e2d4a; border-radius: 16px; padding: 28px 32px;
    text-align: center; margin: 16px 0;
}
.team-name { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: #f1f5f9; }
.vs-badge {
    font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #38bdf8;
    background: #0c1a2e; border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 4px 12px; letter-spacing: 0.1em;
}
.score-chip {
    display: inline-block; background: #1e2d4a; border: 1px solid #2d4a6e;
    border-radius: 8px; padding: 6px 14px; font-family: 'DM Mono', monospace;
    font-size: 0.9rem; color: #94a3b8; margin: 4px;
}
.score-chip.top { background: #0c1e38; border-color: #38bdf8; color: #38bdf8; }
.warning-banner {
    background: #1c1204; border: 1px solid #78350f; border-radius: 10px;
    padding: 14px 18px; color: #fbbf24; font-family: 'DM Mono', monospace; font-size: 0.8rem;
}
.section-title {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #38bdf8;
    letter-spacing: 0.15em; text-transform: uppercase;
    border-bottom: 1px solid #1e2d4a; padding-bottom: 8px; margin: 24px 0 16px;
}
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
def load_teams_from_db(league: str = "BL1") -> list[str]:
    try:
        session = get_session()
        teams = session.query(Team).filter(Team.league == league).all()
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
        ["🎯 Match Prediction", "📊 Team Ratings", "💰 Value Bets", "📋 Feedback & Training", "📈 Data Explorer", "ℹ️ About"],
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

    from src.utils.database import get_session as _gs, Match as _Match, Team as _Team

    @st.cache_data(ttl=300)
    def _db_status(lg):
        sess = _gs()
        count = sess.query(_Match).filter(_Match.league == lg, _Match.status == "FINISHED").count()
        last  = sess.query(_Match).filter(_Match.league == lg, _Match.status == "FINISHED"
                ).order_by(_Match.date.desc()).first()
        sess.close()
        return count, str(last.date) if last else "—"

    _cnt, _last = _db_status(league)
    st.markdown(
        f'''<p style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#334155;line-height:1.8">
        📊 {_cnt} Spiele in DB &nbsp;·&nbsp; Letztes: {_last}<br>
        Modell: Poisson · Monte Carlo · Sofascore · Verletzungen · Wetter
        </p>''',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Neue Spiele laden", use_container_width=True):
        with st.spinner("Aktualisiere Daten..."):
            try:
                import pandas as _pd
                from src.collectors.football_data_collector import FootballDataCollector as _FDC
                from src.features.feature_builder import FeatureBuilder as _FB
                _FDC().fetch_matches(league=league, seasons=2)
                _FDC().fetch_teams(league=league)
                sess2 = _gs()
                _matches = sess2.query(_Match).filter(_Match.league == league, _Match.status == "FINISHED").all()
                _tmap = {t.api_id: t.name for t in sess2.query(_Team).all()}
                sess2.close()
                def _r(h, a): return "H" if h > a else ("A" if h < a else "D")
                _rows = [{"match_id": m.api_id, "date": str(m.date), "league": m.league,
                          "season": m.season, "matchday": m.matchday,
                          "home_team": _tmap.get(m.home_team_id, f"ID:{m.home_team_id}"),
                          "away_team": _tmap.get(m.away_team_id, f"ID:{m.away_team_id}"),
                          "home_goals": m.home_goals, "away_goals": m.away_goals,
                          "result": _r(m.home_goals, m.away_goals)} for m in _matches]
                _feats = _FB(_pd.DataFrame(_rows)).build_features()
                Path(f"data/processed/features_{league}.csv").write_text(_feats.to_csv(index=False))
                st.cache_data.clear()
                st.success(f"✅ {len(_feats)} Spiele geladen!")
            except Exception as _e:
                st.error(f"Fehler: {_e}")


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
            "bgcolor": "#111827", "bordercolor": "#1e2d4a",
            "steps": [{"range": [0, 100], "color": "#0a0e1a"}],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.75, "value": value * 100},
        },
    ))
    fig.update_layout(height=180, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    return fig


def score_heatmap(score_matrix: dict, home: str, away: str) -> go.Figure:
    max_g = 6
    z = [[0.0] * (max_g + 1) for _ in range(max_g + 1)]
    for (h, a), p in score_matrix.items():
        if h <= max_g and a <= max_g:
            z[h][a] = round(p * 100, 2)
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(i) for i in range(max_g + 1)], y=[str(i) for i in range(max_g + 1)],
        colorscale=[[0, "#0a0e1a"], [0.5, "#1e3a5f"], [1, "#38bdf8"]], showscale=True,
        colorbar=dict(title=dict(text="%", font=dict(family="DM Mono", size=10, color="#64748b")),
                      tickfont=dict(family="DM Mono", size=10, color="#64748b")),
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}", textfont={"size": 9, "family": "DM Mono"},
    ))
    fig.update_layout(
        height=320,
        xaxis=dict(title=dict(text=f"Tore {away}", font=dict(family="DM Mono", size=10, color="#64748b")),
                   tickfont=dict(family="DM Mono", size=10, color="#94a3b8")),
        yaxis=dict(title=dict(text=f"Tore {home}", font=dict(family="DM Mono", size=10, color="#64748b")),
                   tickfont=dict(family="DM Mono", size=10, color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        margin=dict(t=10, b=40, l=50, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: MATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎯 Match Prediction":
    st.markdown('<p class="dashboard-header">Match Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Poisson · Monte Carlo · Sofascore · Verletzungen · Wetter</p>', unsafe_allow_html=True)
    st.markdown("---")

    df = load_features(league)
    teams = load_teams_from_db(league)

    if df.empty:
        st.error(f"Keine Daten für {league}. Bitte zuerst `python scripts/update.py --league {league}` ausführen.")
        st.stop()

    all_teams = sorted(
        t for t in (set(df["home_team"].dropna()) | set(df["away_team"].dropna()) | set(teams))
        if not str(t).startswith("ID:")
    )

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
        # ── Schritt 1: Verletzungen ─────────────────────────────────────────
        with st.spinner("1/4 · Verletzungsanalyse (Transfermarkt)..."):
            model = PoissonModel()
            model.fit(df)

            _h_inj = calculate_missing_impact(home_team)
            _a_inj = calculate_missing_impact(away_team)
            home_inj          = _h_inj[0] if _h_inj else 0.0
            home_missing_names = _h_inj[1] if _h_inj else []
            away_inj          = _a_inj[0] if _a_inj else 0.0
            away_missing_names = _a_inj[1] if _a_inj else []

        # ── Schritt 2: Sofascore Modus A/B/C ───────────────────────────────
        with st.spinner("2/4 · Sofascore Team-Ratings & Verletzungspenalty..."):
            sofascore = SofascoreCollector()
            home_ss_id = sofascore.get_team_id(home_team)
            away_ss_id = sofascore.get_team_id(away_team)

            home_sofascore_penalty = 0.0
            away_sofascore_penalty = 0.0
            home_sofascore_rating  = None
            away_sofascore_rating  = None
            home_starters = []
            away_starters = []
            modus = "C"

            if home_ss_id and away_ss_id:
                match_id = sofascore.get_match_id(home_ss_id, away_team)

                if match_id:
                    # Modus A: bestätigte Live-Aufstellung
                    confirmed = sofascore.get_match_ratings(match_id)
                    if confirmed:
                        h_st = [p for p in confirmed["home_team"]["players"] if p.get("is_starter")]
                        a_st = [p for p in confirmed["away_team"]["players"] if p.get("is_starter")]
                        if len(h_st) == 11 and confirmed.get("confirmed", False):
                            home_starters, away_starters, modus = h_st, a_st, "A"

                    # Modus B: voraussichtliche Aufstellung
                    if modus != "A":
                        predicted = sofascore.get_predicted_match_ratings(match_id)
                        if predicted:
                            h_pl = predicted["home_team"]["players"]
                            a_pl = predicted["away_team"]["players"]
                            h_st = [p for p in h_pl if p.get("is_starter")]
                            a_st = [p for p in a_pl if p.get("is_starter")]
                            if len(h_st) < 11 and len(h_pl) >= 11:
                                h_st, a_st = h_pl[:11], a_pl[:11]
                            if len(h_st) >= 11:
                                home_starters, away_starters = h_st[:11], a_st[:11]
                                modus = "B"

                if home_ss_id:
                    home_ssc = sofascore.get_injured_player_impact(
                        home_ss_id, home_missing_names,
                        match_starters=home_starters if modus in ("A", "B") else None,
                    )
                    home_sofascore_penalty = home_ssc["penalty"]
                    home_sofascore_rating  = home_ssc["team_avg_rating"]

                if away_ss_id:
                    away_ssc = sofascore.get_injured_player_impact(
                        away_ss_id, away_missing_names,
                        match_starters=away_starters if modus in ("A", "B") else None,
                    )
                    away_sofascore_penalty = away_ssc["penalty"]
                    away_sofascore_rating  = away_ssc["team_avg_rating"]

            home_total_impact = home_inj + home_sofascore_penalty
            away_total_impact = away_inj + away_sofascore_penalty

        # ── Schritt 3: Form, Context, Wetter ───────────────────────────────
        with st.spinner("3/4 · Live-Form · Tabelle · Wetter..."):
            form_calc = LiveFormCalculator()
            home_form = form_calc.get_lambda_adjustment(
                home_team, league=league,
                injury_impact=home_total_impact,
                sofascore_team_rating=home_sofascore_rating,
                is_home=True, features_df=df,
            )
            away_form = form_calc.get_lambda_adjustment(
                away_team, league=league,
                injury_impact=away_total_impact,
                sofascore_team_rating=away_sofascore_rating,
                is_home=False, features_df=df,
            )

            try:
                from src.collectors.football_data_collector import FootballDataCollector as _FDC2
                standings = _FDC2().get_live_standings(league)
            except Exception:
                standings = {}

            ctx_engine = ContextEngine()
            matchday = 20
            if standings and home_team in standings:
                matchday = standings[home_team].get("playedGames", 19) + 1
            context = ctx_engine.calculate_context(home_team, away_team, league, matchday, standings)

            home_form["attack_factor"] = round(home_form["attack_factor"] * context["home_motivation"], 3)
            away_form["attack_factor"] = round(away_form["attack_factor"] * context["away_motivation"], 3)

            weather = WeatherCollector().get_match_weather(home_team)

        # ── Schritt 4: Simulation ───────────────────────────────────────────
        with st.spinner(f"4/4 · Simuliere {sims:,} Spiele..."):
            sim = MonteCarloSimulator(model)
            result = sim.simulate(
                home_team, away_team, n=sims,
                home_injury_impact=home_total_impact,
                away_injury_impact=away_total_impact,
                weather_impact=weather["goal_impact_factor"],
                home_form_factor=home_form["attack_factor"],
                away_form_factor=away_form["attack_factor"],
            )
            poisson_pred = model.predict(home_team, away_team)

            result["score_matrix"]       = poisson_pred["score_matrix"]
            result["home_form"]          = home_form
            result["away_form"]          = away_form
            result["home_injury_impact"] = home_total_impact
            result["away_injury_impact"] = away_total_impact
            result["home_form_ppg"]      = home_form["form_ppg"]
            result["away_form_ppg"]      = away_form["form_ppg"]
            result["home_specific_ppg"]  = home_form.get("specific_ppg", 1.5)
            result["away_specific_ppg"]  = away_form.get("specific_ppg", 1.5)
            result["is_derby"]           = context["is_derby"]
            result["match_urgency"]      = context["urgency"]
            result["home_motivation"]    = context["home_motivation"]
            result["away_motivation"]    = context["away_motivation"]

            # XGBoost Ensemble (falls trainiert)
            xgb = XGBoostFeedbackModel()
            if xgb.is_trained:
                result = xgb.get_ensemble(result)

            # ── Vorhersage in DB speichern ──────────────────────────────────
            pred_id = XGBoostFeedbackModel.save_prediction(result, league=league)
            result["prediction_id"] = pred_id

        # ── Matchup Header ──────────────────────────────────────────────────
        derby_badge = " 🔥 DERBY" if result.get("is_derby") else ""
        home_boost = f" ×{result.get('home_motivation', 1.0):.3f}" if result.get("home_motivation", 1.0) > 1.01 else ""
        away_boost = f" ×{result.get('away_motivation', 1.0):.3f}" if result.get("away_motivation", 1.0) > 1.01 else ""
        
        sofascore_badge = ""
        if home_sofascore_rating or away_sofascore_rating:
            modus_label = {"A": "Live-Aufstellung", "B": "Predicted Lineup", "C": "Kader-Schnitt"}.get(modus, modus)
            h_r = f"{home_sofascore_rating:.1f}" if home_sofascore_rating else "—"
            a_r = f"{away_sofascore_rating:.1f}" if away_sofascore_rating else "—"
            sofascore_badge = f" · Sofascore {modus_label}: Ø {h_r} / {a_r}"

        st.markdown(f"""
        <div class="matchup-header">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="team-name">{home_team}<span style="font-size:0.9rem;color:#fbbf24">{home_boost}</span></div>
                <div><span class="vs-badge">VS{derby_badge}</span></div>
                <div class="team-name">{away_team}<span style="font-size:0.9rem;color:#fbbf24">{away_boost}</span></div>
            </div>
            <div style="margin-top:12px;font-family:'DM Mono',monospace;font-size:0.72rem;color:#475569;">
                {sims:,} Simulationen · Konfidenz: {result['confidence']} · Favorit: {result['favourite']}
                {sofascore_badge} · Vorhersage-ID: #{pred_id}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 3 Gauges ────────────────────────────────────────────────────────
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(prob_gauge(result["prob_home_win"], f"HEIMSIEG\n{home_team[:18]}", "#4ade80"), use_container_width=True)
        with g2:
            st.plotly_chart(prob_gauge(result["prob_draw"], "UNENTSCHIEDEN", "#94a3b8"), use_container_width=True)
        with g3:
            st.plotly_chart(prob_gauge(result["prob_away_win"], f"AUSWÄRTSSIEG\n{away_team[:18]}", "#f87171"), use_container_width=True)

        # ── Expected Goals & Markets ────────────────────────────────────────
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

        # ── Score Heatmap + Top Scores ───────────────────────────────────────
        st.markdown('<p class="section-title">Ergebnis-Wahrscheinlichkeiten</p>', unsafe_allow_html=True)
        hc1, hc2 = st.columns([3, 2])
        with hc1:
            st.plotly_chart(score_heatmap(result.get("score_matrix", {}), home_team, away_team), use_container_width=True)
        with hc2:
            st.markdown("**Wahrscheinlichste Ergebnisse**")
            for i, s in enumerate(result["top_scores"][:8]):
                chip_class = "score-chip top" if i == 0 else "score-chip"
                st.markdown(
                    f'<span class="{chip_class}">{s["score"]}</span>'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.8rem;color:#475569;margin-left:8px;">{s["probability"]:.2f}%</span>',
                    unsafe_allow_html=True
                )

        # ── Sofascore Info ───────────────────────────────────────────────────
        if home_sofascore_rating or away_sofascore_rating:
            st.markdown('<p class="section-title">Sofascore Team-Ratings</p>', unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            for col, team, rating, penalty in [
                (sc1, home_team, home_sofascore_rating, home_sofascore_penalty),
                (sc2, away_team, away_sofascore_rating, away_sofascore_penalty),
            ]:
                if rating:
                    rating_color = "#4ade80" if rating >= 55 else ("#fbbf24" if rating >= 45 else "#f87171")
                    penalty_str = f"&nbsp;·&nbsp;Verletzungspenalty: <b style='color:#f87171'>-{penalty:.1f}%</b>" if penalty > 0 else ""
                    col.markdown(
                        f'<div style="background:#0a0e1a;border:1px solid #1e2d4a;border-radius:10px;padding:14px 18px;font-family:\'DM Mono\',monospace;font-size:0.82rem;">'
                        f'<span style="color:#38bdf8">📊 {team[:20]}</span><br>'
                        f'Attribut-Ø: <b style="color:{rating_color}">{rating:.1f}/100</b>{penalty_str}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        st.success(f"✅ Vorhersage gespeichert — ID #{pred_id} (im Feedback-Bereich eintragen nach dem Spiel)")


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

    fig_bar = go.Figure(go.Bar(
        x=ratings["overall"], y=ratings["team"], orientation="h",
        marker=dict(color=ratings["overall"], colorscale=[[0, "#1e3a5f"], [1, "#38bdf8"]], showscale=False),
        text=[f"{v:.3f}" for v in ratings["overall"]], textposition="outside",
        textfont=dict(family="DM Mono", size=10, color="#94a3b8"),
    ))
    fig_bar.update_layout(
        height=max(350, len(ratings) * 28), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1e2d4a", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
        yaxis=dict(tickfont=dict(family="Inter", size=11, color="#e2e8f0"), autorange="reversed"),
        margin=dict(l=10, r=80, t=10, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<p class="section-title">Angriff vs Abwehr</p>', unsafe_allow_html=True)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=ratings["attack"], y=ratings["defence"], mode="markers+text",
        text=ratings["team"], textposition="top center",
        textfont=dict(family="DM Mono", size=9, color="#94a3b8"),
        marker=dict(size=12, color=ratings["overall"], colorscale=[[0, "#1e3a5f"], [1, "#38bdf8"]],
                    showscale=False, line=dict(width=1, color="#1e2d4a")),
    ))
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        margin=dict(t=10, b=50, l=60, r=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("Rohdaten anzeigen"):
        st.dataframe(ratings.style.background_gradient(subset=["overall", "attack"], cmap="Blues"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: VALUE BETS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💰 Value Bets":
    st.markdown('<p class="dashboard-header">Value Bets</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Modell vs. Buchmacher · Edge-Analyse · Marktvergleich</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not config.ODDS_API_KEY:
        st.warning("⚠️ Kein ODDS_API_KEY konfiguriert.")
        st.stop()

    df = load_features(league)
    if df.empty:
        st.error("Keine Feature-Daten.")
        st.stop()

    st.markdown('<p class="section-title">Einzelspiel analysieren</p>', unsafe_allow_html=True)
    db_teams = load_teams_from_db(league)
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

    if st.button("🔍 Analysieren", type="primary", use_container_width=True) and vb_home != vb_away:
        with st.spinner("Modell + Odds laden..."):
            model = PoissonModel()
            model.fit(df)
            sim = MonteCarloSimulator(model)
            model_result = sim.simulate(vb_home, vb_away, n=10_000)
            odds_collector = OddsCollector()
            consensus = odds_collector.get_consensus_odds(league)

        if consensus.empty:
            mc1, mc2, mc3 = st.columns(3)
            def model_metric(col, label, val, color):
                col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color}">{val}</div></div>', unsafe_allow_html=True)
            model_metric(mc1, f"Heimsieg {vb_home[:14]}", f"{model_result['prob_home_win']:.1%}", "green")
            model_metric(mc2, "Unentschieden", f"{model_result['prob_draw']:.1%}", "")
            model_metric(mc3, f"Auswärtssieg {vb_away[:14]}", f"{model_result['prob_away_win']:.1%}", "red")
        else:
            match_odds = consensus[
                (consensus["home_team"].str.contains(vb_home[:6], case=False, na=False)) &
                (consensus["away_team"].str.contains(vb_away[:6], case=False, na=False))
            ]
            if not match_odds.empty:
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
                ec1, ec2, ec3, ec4 = st.columns(4)
                def edge_metric(col, label, edge):
                    color = "green" if edge >= 5 else ("red" if edge <= -5 else "")
                    prefix = "▲" if edge > 0 else ("▼" if edge < 0 else "=")
                    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color}">{prefix}{abs(edge):.1f} PP</div></div>', unsafe_allow_html=True)
                edge_metric(ec1, "Edge Heimsieg", analysis["edge_home"])
                edge_metric(ec2, "Edge Unentschieden", analysis["edge_draw"])
                edge_metric(ec3, "Edge Auswärtssieg", analysis["edge_away"])
                ec4.markdown(f'<div class="metric-card"><div class="metric-label">Buchmacher-Marge</div><div class="metric-value amber">{analysis["avg_margin_pct"]:.1f}%</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEEDBACK & TRAINING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Feedback & Training":
    st.markdown('<p class="dashboard-header">Feedback & Training</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Ergebnisse eintragen · XGBoost trainieren · Testdaten löschen</p>', unsafe_allow_html=True)
    st.markdown("---")

    xgb = XGBoostFeedbackModel()
    from src.utils.database import Prediction as PredModel

    # ── Status ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Modell-Status</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    session = get_session()
    total_preds   = session.query(PredModel).count()
    open_preds    = session.query(PredModel).filter(PredModel.actual_result == None).count()
    closed_preds  = session.query(PredModel).filter(PredModel.actual_result != None).count()
    correct_preds = session.query(PredModel).filter(PredModel.prediction_correct == True).count()
    session.close()

    accuracy_str = f"{correct_preds/closed_preds:.1%}" if closed_preds > 0 else "—"

    def status_card(col, label, value, color=""):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color}">{value}</div></div>', unsafe_allow_html=True)

    status_card(s1, "Vorhersagen gesamt", total_preds)
    status_card(s2, "Offen (kein Ergebnis)", open_preds, "amber")
    status_card(s3, "Trefferquote", accuracy_str, "green" if closed_preds > 0 else "")
    status_card(s4, "XGBoost",
        f"✅ {xgb.accuracy:.1%}" if xgb.is_trained else f"⏳ {closed_preds}/10",
        "green" if xgb.is_trained else "amber")

    # ── Ergebnis eintragen ──────────────────────────────────────────────────
    st.markdown('<p class="section-title">Ergebnis eintragen</p>', unsafe_allow_html=True)

    session = get_session()
    open_predictions = session.query(PredModel).filter(
        PredModel.actual_result == None
    ).order_by(PredModel.created_at.desc()).all()
    session.close()

    if not open_predictions:
        st.markdown('''<div style="background:#111827;border:1px solid #1e2d4a;border-radius:10px;
            padding:16px 20px;color:#64748b;font-family:'DM Mono',monospace;font-size:0.85rem;">
            Keine offenen Vorhersagen — erst eine Simulation auf der Match Prediction Seite starten.
        </div>''', unsafe_allow_html=True)
    else:
        pred_options = {
            f"ID {p.id} | {str(p.created_at)[:10]} | {p.home_team} vs {p.away_team} [{p.predicted_winner}]": p.id
            for p in open_predictions
        }
        selected_label = st.selectbox("Vorhersage auswählen", options=list(pred_options.keys()), key="pred_select")
        selected_id = pred_options[selected_label]
        selected_pred = next(p for p in open_predictions if p.id == selected_id)

        fc1, fc2, fc3 = st.columns([4, 1, 4])
        with fc1:
            home_goals = st.number_input(f"🏠 {selected_pred.home_team[:25]} — Tore", min_value=0, max_value=20, value=0, step=1, key="hg")
        with fc2:
            st.markdown("<br><div style='text-align:center;font-size:1.5rem'>:</div>", unsafe_allow_html=True)
        with fc3:
            away_goals = st.number_input(f"✈️ {selected_pred.away_team[:25]} — Tore", min_value=0, max_value=20, value=0, step=1, key="ag")

        st.markdown(f'''<div style="background:#0a0e1a;border:1px solid #1e2d4a;border-radius:10px;
            padding:12px 20px;font-family:'DM Mono',monospace;font-size:0.8rem;color:#64748b;margin:8px 0;">
            Prognose: Heimsieg {selected_pred.prob_home_win:.1%} |
            Unentschieden {selected_pred.prob_draw:.1%} |
            Auswärtssieg {selected_pred.prob_away_win:.1%}
        </div>''', unsafe_allow_html=True)

        if st.button("✅ Ergebnis speichern", type="primary", use_container_width=True):
            success = XGBoostFeedbackModel.enter_result(selected_id, int(home_goals), int(away_goals))
            if success:
                actual = "H" if home_goals > away_goals else ("A" if home_goals < away_goals else "D")
                correct = actual == selected_pred.predicted_winner
                color = "#4ade80" if correct else "#f87171"
                icon  = "✅" if correct else "❌"
                st.markdown(f'''<div style="background:#0c1e0c;border:1px solid #166534;border-radius:10px;
                    padding:14px 20px;color:{color};font-family:'DM Mono',monospace;font-size:0.9rem;">
                    {icon} Ergebnis gespeichert: {home_goals}:{away_goals}
                    — Prognose war {"RICHTIG" if correct else "FALSCH"}
                </div>''', unsafe_allow_html=True)
                st.rerun()

    # ── Bisherige Ergebnisse ─────────────────────────────────────────────────
    session = get_session()
    past = session.query(PredModel).filter(PredModel.actual_result != None).order_by(PredModel.created_at.desc()).all()
    session.close()

    if past:
        st.markdown('<p class="section-title">Bisherige Ergebnisse</p>', unsafe_allow_html=True)
        rows = []
        for p in past:
            rows.append({
                "Datum":    str(p.created_at)[:10],
                "Spiel":    f"{p.home_team} vs {p.away_team}",
                "Prognose": p.predicted_winner,
                "Konf.":    p.confidence or "?",
                "Ergebnis": f"{p.actual_home_goals}:{p.actual_away_goals}",
                "Korrekt":  "✅" if p.prediction_correct else "❌",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── XGBoost Training ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">XGBoost Training</p>', unsafe_allow_html=True)
    MIN_SAMPLES = 10
    if closed_preds < MIN_SAMPLES:
        st.markdown(f'''<div style="background:#111827;border:1px solid #1e2d4a;border-radius:10px;
            padding:16px 20px;color:#64748b;font-family:'DM Mono',monospace;font-size:0.85rem;">
            ⏳ Noch {MIN_SAMPLES - closed_preds} Ergebnisse bis zum ersten Training
            ({closed_preds} / {MIN_SAMPLES} vorhanden)
        </div>''', unsafe_allow_html=True)
    else:
        tc1, tc2 = st.columns([3, 1])
        with tc1:
            if xgb.is_trained:
                st.markdown(f'''<div style="background:#0c1e0c;border:1px solid #166534;border-radius:10px;
                    padding:14px 20px;font-family:'DM Mono',monospace;font-size:0.82rem;color:#4ade80;">
                    ✅ XGBoost aktiv — Accuracy: {xgb.accuracy:.1%} |
                    Trainiert: {xgb.trained_at.strftime("%d.%m.%Y %H:%M") if xgb.trained_at else "?"} |
                    {xgb.n_training_samples} Samples
                </div>''', unsafe_allow_html=True)
        with tc2:
            if st.button("🧠 Training starten", type="primary", use_container_width=True):
                with st.spinner("XGBoost trainiert..."):
                    result = xgb.train()
                if result["success"]:
                    st.success(f"✅ Accuracy: {result['accuracy']:.1%}")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

    # ── Testdaten / Vorhersagen löschen ──────────────────────────────────────
    st.markdown('<p class="section-title">Vorhersagen verwalten & löschen</p>', unsafe_allow_html=True)

    session = get_session()
    all_preds = session.query(PredModel).order_by(PredModel.created_at.desc()).all()
    session.close()

    if not all_preds:
        st.markdown('''<div style="background:#111827;border:1px solid #1e2d4a;border-radius:10px;
            padding:16px 20px;color:#64748b;font-family:'DM Mono',monospace;font-size:0.85rem;">
            Keine Vorhersagen in der Datenbank.
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#64748b;">'
            f'Wähle Vorhersagen zum Löschen aus. Nützlich um Testläufe zu entfernen.</p>',
            unsafe_allow_html=True
        )

        # Alle Vorhersagen als Tabelle mit Checkboxen
        delete_ids = []
        for p in all_preds:
            status = "⏳ offen" if p.actual_result is None else (
                f"{'✅' if p.prediction_correct else '❌'} {p.actual_home_goals}:{p.actual_away_goals}"
            )
            label = f"ID {p.id} | {str(p.created_at)[:16]} | {p.home_team[:20]} vs {p.away_team[:20]} | {status}"
            if st.checkbox(label, key=f"del_{p.id}"):
                delete_ids.append(p.id)

        if delete_ids:
            col_del, col_info = st.columns([2, 3])
            with col_del:
                if st.button(f"🗑️ {len(delete_ids)} Vorhersage(n) löschen", type="primary", use_container_width=True):
                    session = get_session()
                    for del_id in delete_ids:
                        session.query(PredModel).filter(PredModel.id == del_id).delete()
                    session.commit()
                    session.close()
                    st.success(f"✅ {len(delete_ids)} Vorhersage(n) gelöscht.")
                    st.rerun()
            with col_info:
                st.markdown(
                    f'<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#f87171;padding-top:8px;">'
                    f'⚠️ {len(delete_ids)} ausgewählt — diese Aktion ist nicht rückgängig zu machen</p>',
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Data Explorer":
    st.markdown('<p class="dashboard-header">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Historische Spieldaten · Feature Matrix · Trendanalyse</p>', unsafe_allow_html=True)
    st.markdown("---")

    df = load_features(league)
    if df.empty:
        st.error("Keine Daten gefunden.")
        st.stop()

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

    st.markdown('<p class="section-title">Tore pro Spieltag (Saison-Trend)</p>', unsafe_allow_html=True)
    df_goals = df.dropna(subset=["home_goals", "away_goals", "matchday"]).copy()
    df_goals["total_goals"] = df_goals["home_goals"] + df_goals["away_goals"]
    trend = df_goals.groupby(["season", "matchday"])["total_goals"].mean().reset_index()

    fig_trend = go.Figure()
    for season in trend["season"].unique():
        s = trend[trend["season"] == season]
        fig_trend.add_trace(go.Scatter(x=s["matchday"], y=s["total_goals"], name=str(season),
                                        mode="lines+markers", line=dict(width=2), marker=dict(size=5)))
    fig_trend.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        xaxis=dict(title="Spieltag", gridcolor="#111827", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
        yaxis=dict(title="Ø Tore/Spiel", gridcolor="#111827", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
        legend=dict(font=dict(family="DM Mono", size=9, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown('<p class="section-title">Ergebnisverteilung</p>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Heimsieg", "Unentschieden", "Auswärtssieg"], values=[h, d, a], hole=0.55,
            marker=dict(colors=["#4ade80", "#94a3b8", "#f87171"]), textfont=dict(family="DM Mono", size=10),
        ))
        fig_pie.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                               legend=dict(font=dict(family="DM Mono", size=9, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
                               margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with rc2:
        st.markdown('<p class="section-title">Tore-Verteilung</p>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df["home_goals"].dropna(), name="Heimtore",
                                         marker_color="#38bdf8", opacity=0.7, xbins=dict(size=1)))
        fig_hist.add_trace(go.Histogram(x=df["away_goals"].dropna(), name="Auswärtstore",
                                         marker_color="#f87171", opacity=0.7, xbins=dict(size=1)))
        fig_hist.update_layout(barmode="overlay", height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
                                xaxis=dict(gridcolor="#111827", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
                                yaxis=dict(gridcolor="#111827", tickfont=dict(family="DM Mono", size=9, color="#64748b")),
                                legend=dict(font=dict(family="DM Mono", size=9, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
                                margin=dict(t=10, b=40, l=50, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Feature Matrix (letzte 20 Spiele)"):
        display_cols = ["date", "home_team", "away_team", "result",
                        "home_form_ppg", "away_form_ppg",
                        "home_goals_scored_avg", "away_goals_scored_avg",
                        "position_diff", "h2h_home_win_rate"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].tail(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown('<p class="dashboard-header">Über das Projekt</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-sub">Data Analytics · Data Science · Sports Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### Methodik

    **1. Poisson-Modell (Dixon-Coles)**
    Schätzt Angriffs- & Abwehrstärke per MLE + Heimvorteil. Liefert erwartete Tore (λ).

    **2. Monte Carlo Simulation**
    10.000+ Simulationen mit Poisson-Ziehungen → saubere Wahrscheinlichkeiten inkl. Ergebnis-Matrix.

    **3. Sofascore Team-Ratings**
    Attribut-basiertes Rating (0–100) aus Sofascore. Beeinflusst Lambda-Faktor: ±15% max.

    **4. Verletzungsgewichtung (Transfermarkt)**
    Missing Impact = Marktwert-Ausfälle / Gesamtkaderwert. >15% → Schwächungsmalus.

    **5. Context Engine**
    Tabelle, Derby-Erkennung, Abstiegskampf-, Titel- und Europa-Boosts.

    **6. XGBoost Feedback-Loop**
    Lernt aus Vorhersage-Fehlern. Ab 10 Ergebnissen: 35% XGB + 65% Poisson Ensemble.

    ### Datenquellen
    - **football-data.org** — Spielergebnisse, Spielpläne, Tabellen
    - **Transfermarkt** (lokaler Scraper) — Kaderwerte, Verletzungen, Sperren
    - **Sofascore** (lokale API) — Team-Attribut-Ratings, Verletzungspenalty
    - **OpenWeatherMap** — Wetterbedingungen
    - **The Odds API** — Buchmacherquoten
    """)