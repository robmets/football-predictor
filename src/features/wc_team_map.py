"""
WC National Team ID Mappings.
Hardcoded Sofascore-IDs für schnelle Lookups ohne API-Call.
Transfermarkt-IDs werden in der DB gespeichert (teams.transfermarkt_id via map_wc_teams.py).
"""

# Sofascore national team IDs
# Verified from sofascore.com team pages (stable IDs)
SOFASCORE_IDS: dict[str, int] = {
    "Germany":             4711,
    "France":              4481,
    "Brazil":              4750,
    "Argentina":           3377,
    "Spain":               4698,
    "England":             4713,
    "Italy":               4707,
    "Netherlands":         4712,
    "Portugal":            4714,
    "Belgium":             4648,
    "Croatia":             4715,
    "Uruguay":             4752,
    "Mexico":              3809,
    "United States":       4397,
    "Morocco":             4754,
    "Japan":               4706,
    "South Korea":         4716,
    "Senegal":             4755,
    "Ghana":               4753,
    "Cameroon":            4770,
    "Australia":           4401,
    "Poland":              4700,
    "Denmark":             4683,
    "Switzerland":         4702,
    "Serbia":              4503,
    "Costa Rica":          4766,
    "Ecuador":             4769,
    "Canada":              4656,
    "Qatar":               4771,
    "Tunisia":             4751,
    "Saudi Arabia":        4757,
    "Iran":                4760,
    "Wales":               4704,
    "Norway":              4705,
    "Sweden":              4708,
    "Austria":             4703,
    "Algeria":             4645,
    "Nigeria":             4749,
    "Ivory Coast":         4748,
    "Colombia":            4758,
    "Chile":               4759,
    "Peru":                4761,
    "Paraguay":            4762,
    "Bolivia":             4763,
    "Honduras":            4764,
    "Panama":              4765,
    "Iceland":             4684,
    "Greece":              4685,
    "Hungary":             4686,
    "Turkey":              4687,
    "Russia":              4688,
    "Ukraine":             4689,
    "Slovakia":            4690,
    "Slovenia":            4691,
    "Bulgaria":            4692,
    "Romania":             4693,
    "Scotland":            4694,
    "Northern Ireland":    4695,
    "Republic of Ireland": 4696,
}


def get_sofascore_id(team_name: str) -> int | None:
    """Returns hardcoded Sofascore ID for national team, or None if not mapped."""
    return SOFASCORE_IDS.get(team_name)
