---
name: project-wc-integration
description: WC (FIFA World Cup) integration into the football predictor — what was built, how it works, key decisions
metadata:
  type: project
---

WC integration built on branch `wc_integration` (June 2026). Tested and working in dashboard.

**Why:** User wants to predict WC 2026 matches using the same pipeline as CL/Bundesliga, treating WC like a tournament (same approach as Champions League).

**What was done:**

1. DB migration: added `stage` (VARCHAR) and `group_name` (VARCHAR) columns to `matches` table
2. Config: added `"WC": "FIFA World Cup"` and `"EC"` to SUPPORTED_LEAGUES
3. CSV importer: `scripts/import_wc_csv.py` — imports 1930-2022 historical data from `data/matches.csv` (964 matches, Men's WC only filter)
4. API collector: `fetch_matches` stores `stage` + `group_name`; `get_live_standings("WC")` returns per-group standings from DB
5. Context engine: WC uses `stage` field (not matchday) for knockout detection; WC_STAGE_CONFIG maps stages to boosts (LAST_16=+20%, QF=+25%, SF=+30%, FINAL=+40%); group stage urgency scales with matchday 1→3
6. Feature builder: `_wc_group_standings()` computes standings within group only; neutral venue → `home_win_rate_at_home=0.33` for WC; `stage` and `group_name` added to feature CSV output; shared `_compute_table()` helper
7. Collector mappings: odds `"WC": "soccer_fifa_world_cup"`, api-football `"WC": 1`, TM national teams via `get_national_team_players()`
8. Team mapping: **dynamic** via existing `map_teams.py --league WC` (NOT hardcoded). 46/48 WC 2026 teams TM-mapped. Cape Verde Islands + Congo DR not found on TM.
9. WC 2026 teams from API: 48 teams fetched with real API IDs (<800000). Dashboard shows only these, not historical synthetic teams (830000+).
10. Dashboard: WC in league selector; WC stage selector (GROUP_STAGE MD1/2/3 + LAST_32/16/QF/SF/3rd/Final); group selector (A-L); group standings table displayed; neutral venue badge; `calculate_missing_impact(league=league)` passes league for WC fallback
11. predict.py: `--stage` and `--group` CLI args for WC
12. 18 new WC tests in `tests/unit/test_wc.py`, 69 total tests pass

**Key architectural decisions:**
- WC uses `stage` field for context (not matchday) since API returns `matchday=None` for knockout rounds
- Synthetic api_ids for historical teams: `830000 + int(T-XX)` from CSV team_id column
- Match api_ids: `700000 + key_id` (1-1248)
- home_advantage = 1.0 for WC (neutral venues), implemented in FeatureBuilder
- Historical data covers 1930-2022 from data/matches.csv; 2026 live data available via API
- Team mapping is DYNAMIC (map_teams.py), not hardcoded — identical to other leagues
- Dashboard load_teams_from_db: WC filters api_id < 800000 to show only current 2026 participants

**Known limitations:**
- France λ very high (6.97 vs Germany) because sparse historical data skews attack ratings. Will improve as 2026 matches come in.
- Group standings table shows historical group data (e.g. Gruppe E = Netherlands/Denmark from old WCs). Will show live data once 2026 group games start (June 11, 2026).
- Cape Verde Islands and Congo DR have no TM mapping (not on Transfermarkt).

**How to apply:** When user asks about WC predictions, WC data pipeline, or tournament mode — refer to this integration. All changes are additive with `if league == "WC"` guards.
