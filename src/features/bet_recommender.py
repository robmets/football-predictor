"""
Bet Recommender
Analysiert alle verfügbaren Wettmärkte (H2H, Totals, BTTS) und gibt
konkrete Tipp-Empfehlungen basierend auf Expected Value (Edge) und Kelly-Kriterium.
"""

import pandas as pd
import difflib
from typing import List, Dict
from src.utils.logger import get_logger

log = get_logger(__name__)


class BetRecommender:
    def __init__(self, min_edge: float = 5.0, min_confidence: str = "LOW"):
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.safety_factor = 0.5  # Half-Kelly für sichereres Bankroll-Management
        self.confidence_levels = {"TOSS-UP": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def _fuzzy_match_match(self, df: pd.DataFrame, home: str, away: str, threshold: float = 0.55) -> pd.DataFrame:
        """Findet das Spiel im Odds-DataFrame via Fuzzy Matching."""
        if df.empty:
            return df
            
        best_score = 0.0
        best_home = None
        best_away = None
        
        # Einmaliges Matching auf die ersten Teams, um den API-Namen zu finden
        for idx, row in df.drop_duplicates(subset=['home_team', 'away_team']).iterrows():
            score_h = difflib.SequenceMatcher(None, home.lower(), row["home_team"].lower()).ratio()
            score_a = difflib.SequenceMatcher(None, away.lower(), row["away_team"].lower()).ratio()
            combined = (score_h + score_a) / 2
            
            if combined > best_score:
                best_score = combined
                best_home = row["home_team"]
                best_away = row["away_team"]
                
        if best_score >= threshold:
            return df[(df['home_team'] == best_home) & (df['away_team'] == best_away)]
        return pd.DataFrame()

    def calculate_kelly(self, odds: float, model_prob: float) -> float:
        """
        Berechnet den empfohlenen Einsatz in % der Bankroll nach dem Kelly-Kriterium.
        Formel: f* = (b*p - q) / b  | b = odds - 1, p = prob, q = 1 - prob
        """
        if odds <= 1.0 or model_prob <= 0.0:
            return 0.0
            
        b = odds - 1.0
        p = model_prob
        q = 1.0 - p
        
        kelly_full = (b * p - q) / b
        kelly_safe = kelly_full * self.safety_factor
        
        return max(0.0, kelly_safe * 100) # In Prozent zurückgeben

    def should_recommend(self, edge_pct: float, model_confidence: str) -> bool:
        """Dynamischer Filter: Höherer Edge erlaubt niedrigere Konfidenz."""
        conf_score = self.confidence_levels.get(model_confidence, 0)
        
        # > 15% Edge: Immer zeigen (auch bei TOSS-UP)
        if edge_pct >= 15.0:
            return True
        # > 10% Edge: Ab LOW Confidence
        if edge_pct >= 10.0 and conf_score >= 1:
            return True
        # > 5% Edge: Ab MEDIUM Confidence
        if edge_pct >= self.min_edge and conf_score >= 2:
            return True
            
        return False

    def _evaluate_market(self, market_type: str, recommendation: str, model_prob: float, 
                         odds_df: pd.DataFrame, odds_col: str, implied_col: str, 
                         model_conf: str, explanation: str) -> List[Dict]:
        """Hilfsfunktion zur Auswertung eines spezifischen Marktes über alle Buchmacher."""
        tips = []
        if odds_df.empty or odds_col not in odds_df.columns:
            return tips

        for _, row in odds_df.iterrows():
            odds = row[odds_col]
            implied = row[implied_col]
            bookmaker = row["bookmaker"]
            
            if pd.isna(odds) or odds <= 1.01:
                continue

            edge_pct = (model_prob - implied) * 100

            if self.should_recommend(edge_pct, model_conf):
                kelly_pct = self.calculate_kelly(odds, model_prob)
                
                # Nur Wetten mit positivem Kelly aufnehmen
                if kelly_pct > 0:
                    tips.append({
                        'market': market_type,
                        'recommendation': recommendation,
                        'model_prob': model_prob,
                        'best_odds': odds,
                        'bookmaker': bookmaker,
                        'implied_prob': implied,
                        'edge_pct': edge_pct,
                        'confidence': model_conf,
                        'kelly_stake_pct': kelly_pct,
                        'explanation': explanation
                    })
        return tips

    def analyze_markets(self, simulation_result: dict, odds_data: dict, home_team: str, away_team: str) -> List[Dict]:
        """
        Hauptfunktion: Prüft H2H, Over/Under und BTTS auf Value.
        Gibt die Top-Tipps zurück, sortiert nach Expected Value.
        """
        all_tips = []
        conf = simulation_result.get('confidence', 'LOW')
        
        # 1. Quoten für dieses spezifische Spiel filtern
        h2h_match = self._fuzzy_match_match(odds_data.get('h2h', pd.DataFrame()), home_team, away_team)
        totals_match = self._fuzzy_match_match(odds_data.get('totals', pd.DataFrame()), home_team, away_team)
        btts_match = self._fuzzy_match_match(odds_data.get('btts', pd.DataFrame()), home_team, away_team)

        # ── H2H (1X2) ──
        if not h2h_match.empty:
            all_tips.extend(self._evaluate_market(
                'h2h_home', f'HEIMSIEG {home_team[:10]}', simulation_result.get('prob_home_win', 0),
                h2h_match, 'odds_home', 'implied_home', conf, 'Das Modell sieht einen klaren Heimvorteil gegenüber dem Markt.'
            ))
            all_tips.extend(self._evaluate_market(
                'h2h_draw', 'UNENTSCHIEDEN', simulation_result.get('prob_draw', 0),
                h2h_match, 'odds_draw', 'implied_draw', conf, 'Hohe Wahrscheinlichkeit für ein enges Match ohne klaren Sieger.'
            ))
            all_tips.extend(self._evaluate_market(
                'h2h_away', f'AUSWÄRTSSIEG {away_team[:10]}', simulation_result.get('prob_away_win', 0),
                h2h_match, 'odds_away', 'implied_away', conf, 'Der Markt unterschätzt die Auswärtsstärke deutlich.'
            ))

        # ── Totals (Over/Under) ──
        if not totals_match.empty:
            # Over/Under 2.5
            ou25 = totals_match[totals_match['point'] == 2.5]
            if not ou25.empty:
                prob_over_2_5 = simulation_result.get('prob_over_2_5', 0)
                all_tips.extend(self._evaluate_market(
                    'over_2_5', 'ÜBER 2.5 TORE', prob_over_2_5,
                    ou25, 'odds_over', 'implied_over', conf, 'Beide Teams haben statistisch eine hohe Offensivkraft.'
                ))
                all_tips.extend(self._evaluate_market(
                    'under_2_5', 'UNTER 2.5 TORE', 1.0 - prob_over_2_5,
                    ou25, 'odds_under', 'implied_under', conf, 'Modell erwartet eine defensive, chancenarme Partie.'
                ))

            # Over/Under 3.5
            ou35 = totals_match[totals_match['point'] == 3.5]
            if not ou35.empty:
                prob_over_3_5 = simulation_result.get('prob_over_3_5', 0)
                all_tips.extend(self._evaluate_market(
                    'over_3_5', 'ÜBER 3.5 TORE', prob_over_3_5,
                    ou35, 'odds_over', 'implied_over', conf, 'Extremes Potenzial für ein echtes Torfestival.'
                ))
                all_tips.extend(self._evaluate_market(
                    'under_3_5', 'UNTER 3.5 TORE', 1.0 - prob_over_3_5,
                    ou35, 'odds_under', 'implied_under', conf, 'Ein Schützenfest ist hier laut Simulation sehr unwahrscheinlich.'
                ))

        # ── BTTS (Both Teams To Score) ──
        if not btts_match.empty:
            prob_btts = simulation_result.get('prob_btts', 0)
            all_tips.extend(self._evaluate_market(
                'btts_yes', 'BTTS: JA', prob_btts,
                btts_match, 'odds_yes', 'implied_yes', conf, 'Beide Abwehrreihen sind anfällig, Tore auf beiden Seiten erwartet.'
            ))
            all_tips.extend(self._evaluate_market(
                'btts_no', 'BTTS: NEIN', 1.0 - prob_btts,
                btts_match, 'odds_no', 'implied_no', conf, 'Wahrscheinlich hält mindestens ein Team heute die Null.'
            ))

        # ── Aufräumen und Sortieren ──
        # Wir behalten pro Wett-Typ nur den Buchmacher mit der besten Quote (höchster Edge)
        best_tips_map = {}
        for tip in all_tips:
            key = tip['recommendation']
            if key not in best_tips_map or tip['edge_pct'] > best_tips_map[key]['edge_pct']:
                best_tips_map[key] = tip

        # Sortieren nach Edge (höchster Value zuerst)
        final_tips = list(best_tips_map.values())
        final_tips.sort(key=lambda x: x['edge_pct'], reverse=True)

        return final_tips