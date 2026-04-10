"""
Bet Recommender
Analysiert alle verfügbaren Wettmärkte und gibt simple, verständliche Tipp-Empfehlungen.
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
        self.safety_factor = 0.5
        self.confidence_levels = {"TOSS-UP": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        
        # Übersetzung für absolute Anfänger
        self.risk_translation = {
            "TOSS-UP": "Sehr hoch (50/50 Spiel)",
            "LOW": "Hoch",
            "MEDIUM": "Mittel",
            "HIGH": "Gering"
        }

    def _fuzzy_match_match(self, df: pd.DataFrame, home: str, away: str, threshold: float = 0.55) -> pd.DataFrame:
        if df.empty:
            return df
            
        best_score = 0.0
        best_home = None
        best_away = None
        
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
        if odds <= 1.0 or model_prob <= 0.0:
            return 0.0
            
        b = odds - 1.0
        p = model_prob
        q = 1.0 - p
        
        kelly_full = (b * p - q) / b
        kelly_safe = kelly_full * self.safety_factor
        
        return max(0.0, kelly_safe * 100)

    def should_recommend(self, edge_pct: float, model_confidence: str) -> bool:
        conf_score = self.confidence_levels.get(model_confidence, 0)
        
        if edge_pct >= 15.0:
            return True
        if edge_pct >= 10.0 and conf_score >= 1:
            return True
        if edge_pct >= self.min_edge and conf_score >= 2:
            return True
            
        return False

    def _evaluate_market(self, market_type: str, recommendation: str, model_prob: float, 
                         odds_df: pd.DataFrame, odds_col: str, implied_col: str, 
                         model_conf: str, explanation: str) -> List[Dict]:
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
                
                if kelly_pct > 0:
                    # Einsatz in einen simplen Text übersetzen (auf ganze Euro gerundet für 100 EUR Startkapital)
                    stake_euros = max(1, int(round(kelly_pct)))
                    stake_text = f"{stake_euros} Euro (bei 100 Euro Gesamtbudget)"
                    
                    tips.append({
                        'market': market_type,
                        'recommendation': recommendation,
                        'model_prob': model_prob,
                        'best_odds': odds,
                        'bookmaker': bookmaker,
                        'implied_prob': implied,
                        'edge_pct': edge_pct,
                        'confidence': model_conf,
                        'risk_level': self.risk_translation.get(model_conf, "Unbekannt"),
                        'kelly_stake_pct': kelly_pct,
                        'stake_text': stake_text,
                        'explanation': explanation
                    })
        return tips

    def analyze_markets(self, simulation_result: dict, odds_data: dict, home_team: str, away_team: str) -> List[Dict]:
        all_tips = []
        conf = simulation_result.get('confidence', 'LOW')
        
        h2h_match = self._fuzzy_match_match(odds_data.get('h2h', pd.DataFrame()), home_team, away_team)
        totals_match = self._fuzzy_match_match(odds_data.get('totals', pd.DataFrame()), home_team, away_team)
        btts_match = self._fuzzy_match_match(odds_data.get('btts', pd.DataFrame()), home_team, away_team)

        if not h2h_match.empty:
            all_tips.extend(self._evaluate_market(
                'h2h_home', f'SIEG {home_team[:12]}', simulation_result.get('prob_home_win', 0),
                h2h_match, 'odds_home', 'implied_home', conf, 'Unser System berechnet höhere Siegchancen für das Heimteam als der Buchmacher.'
            ))
            all_tips.extend(self._evaluate_market(
                'h2h_draw', 'UNENTSCHIEDEN', simulation_result.get('prob_draw', 0),
                h2h_match, 'odds_draw', 'implied_draw', conf, 'Sehr hohe Wahrscheinlichkeit für ein enges Spiel ohne klaren Sieger.'
            ))
            all_tips.extend(self._evaluate_market(
                'h2h_away', f'SIEG {away_team[:12]}', simulation_result.get('prob_away_win', 0),
                h2h_match, 'odds_away', 'implied_away', conf, 'Die Auswärtsmannschaft wird vom Buchmacher deutlich unterschätzt.'
            ))

        if not totals_match.empty:
            ou25 = totals_match[totals_match['point'] == 2.5]
            if not ou25.empty:
                prob_over_2_5 = simulation_result.get('prob_over_2_5', 0)
                all_tips.extend(self._evaluate_market(
                    'over_2_5', 'MEHR ALS 2.5 TORE', prob_over_2_5,
                    ou25, 'odds_over', 'implied_over', conf, 'Beide Mannschaften schießen statistisch gesehen sehr viele Tore.'
                ))
                all_tips.extend(self._evaluate_market(
                    'under_2_5', 'WENIGER ALS 2.5 TORE', 1.0 - prob_over_2_5,
                    ou25, 'odds_under', 'implied_under', conf, 'Wir erwarten ein sehr defensives Spiel mit wenigen Torchancen.'
                ))

            ou35 = totals_match[totals_match['point'] == 3.5]
            if not ou35.empty:
                prob_over_3_5 = simulation_result.get('prob_over_3_5', 0)
                all_tips.extend(self._evaluate_market(
                    'over_3_5', 'MEHR ALS 3.5 TORE', prob_over_3_5,
                    ou35, 'odds_over', 'implied_over', conf, 'Es wird ein extrem offensives Spiel mit sehr vielen Toren erwartet.'
                ))
                all_tips.extend(self._evaluate_market(
                    'under_3_5', 'WENIGER ALS 3.5 TORE', 1.0 - prob_over_3_5,
                    ou35, 'odds_under', 'implied_under', conf, 'Es ist fast ausgeschlossen, dass hier 4 oder mehr Tore fallen.'
                ))

        if not btts_match.empty:
            prob_btts = simulation_result.get('prob_btts', 0)
            all_tips.extend(self._evaluate_market(
                'btts_yes', 'BEIDE TEAMS TREFFEN: JA', prob_btts,
                btts_match, 'odds_yes', 'implied_yes', conf, 'Beide Abwehrreihen machen oft Fehler, wir erwarten Tore auf beiden Seiten.'
            ))
            all_tips.extend(self._evaluate_market(
                'btts_no', 'BEIDE TEAMS TREFFEN: NEIN', 1.0 - prob_btts,
                btts_match, 'odds_no', 'implied_no', conf, 'Sehr wahrscheinlich schießt mindestens eine Mannschaft heute kein Tor.'
            ))

        best_tips_map = {}
        for tip in all_tips:
            key = tip['recommendation']
            if key not in best_tips_map or tip['edge_pct'] > best_tips_map[key]['edge_pct']:
                best_tips_map[key] = tip

        final_tips = list(best_tips_map.values())
        final_tips.sort(key=lambda x: x['edge_pct'], reverse=True)

        return final_tips