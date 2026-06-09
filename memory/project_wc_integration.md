---
name: project-wc-integration
description: WC (FIFA World Cup) integration into the football predictor — what was built, how it works, key decisions
metadata:
  type: project
---

WC integration built on branch `wc_integration` (June 2026).

**Why:** User wants to predict WC 2026 matches using the same pipeline as CL/Bundesliga, treating WC like a tournament (same approach as Champions League).

**What was done:**

1. DB migration: added `stage` (VARCHAR) and `group_name` (VARCHAR) columns to `matches` table
2. Config: added `"WC": "FIFA World Cup"` and `"EC"` to SUPPORTED_LEAGUES
3. CSV importer: `scripts/import_wc_csv.py` — imports 1930-2014 historical data from `data/WorldCupMatches.csv` (836 matches imported)
4. API collector: `fetch_matches` now stores `stage` + `group_name`; `get_live_standings("WC")` returns per-group standings from DB
5. Context engine: WC uses `stage` field (not matchday) for knockout detection; WC_STAGE_CONFIG maps stages to boosts (LAST_16=+20%, QF=+25%, SF=+30%, FINAL=+40%); group stage urgency scales with matchday 1→3
6. Feature builder: `_wc_group_standings()` computes standings within group only; neutral venue → `home_win_rate_at_home=0.33` for WC; `stage` and `group_name` added to feature CSV output; shared `_compute_table()` helper
7. Collector mappings: odds `"WC": "soccer_fifa_world_cup"`, api-football `"WC": 1`, TM national team support via `get_national_team_players()`
8. Dashboard: WC in league selector; WC stage selector (GROUP_STAGE MD1/2/3 + LAST_32/16/QF/SF/3rd/Final); group selector (A-L); group standings table displayed; neutral venue badge; `calculate_missing_impact(league=league)` passes league for WC fallback
9. predict.py: `--stage` and `--group` CLI args for WC; `calculate_missing_impact` gets `league=league`
10. 18 new WC tests in `tests/unit/test_wc.py`, all 69 tests pass

**Key architectural decisions:**
- WC uses `stage` field for context (not matchday) since API returns `matchday=None` for knockout rounds
- Synthetic api_ids for historical teams: `900000 + hash(name) % 99999` (stable, no collisions with real API IDs)
- home_advantage = 1.0 for WC (neutral venues), implemented in FeatureBuilder
- Historical data only covers 1930-2014; 2026 live data available via API (no historical restriction on paid tier needed)

**How to apply:** When user asks about WC predictions, WC data pipeline, or tournament mode — refer to this integration. All changes are additive with `if league == "WC"` guards.
