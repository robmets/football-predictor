# src/features/derbies.py

DERBY_PAIRS = [
    # Premier League
    {"Arsenal", "Tottenham Hotspur"},
    {"Manchester United", "Manchester City"},
    {"Liverpool", "Everton"},
    {"Chelsea", "Fulham"},
    {"Chelsea", "Brentford"},
    {"Liverpool", "Manchester United"},
    {"Brighton & Hove Albion", "Crystal Palace"},
    {"Aston Villa", "Wolverhampton Wanderers"},
    {"Newcastle United", "Sunderland"},
    {"Manchester United", "Leeds United"},
    {"Chelsea", "Tottenham Hotspur"},
    
    # La Liga
    {"Real Madrid", "Barcelona"},
    {"Real Madrid", "Atlético Madrid"},
    {"Barcelona", "Espanyol"},
    {"Sevilla", "Real Betis"},
    {"Valencia", "Levante"},
    {"Athletic Club", "Real Sociedad"},
    {"Sporting Gijón", "Real Oviedo"},
    {"Celta Vigo", "Deportivo La Coruña"},
    {"Las Palmas", "Tenerife"},
    
    # Bundesliga
    {"Borussia Dortmund", "Schalke 04"},
    {"Hamburger SV", "Werder Bremen"},
    {"Hamburger SV", "FC St. Pauli"},
    {"1. FC Köln", "Borussia Mönchengladbach"},
    {"1. FC Köln", "Bayer 04 Leverkusen"},
    {"Hertha BSC", "Union Berlin"},
    {"Bayern München", "VfB Stuttgart"},
    {"Bayern München", "1. FC Nürnberg"},
    {"Eintracht Frankfurt", "Kickers Offenbach"},
    {"Hannover 96", "Hertha BSC"},
    {"Bayern München", "Borussia Mönchengladbach"},
    {"Bayern München", "Borussia Dortmund"},
    
    # Serie A
    {"Inter", "AC Milan"},
    {"Roma", "Lazio"},
    {"Juventus", "Torino"},
    {"Genoa", "Sampdoria"},
    {"Bologna", "Parma"},
    {"Bologna", "Fiorentina"},
    {"Roma", "Napoli"},
    {"Napoli", "Palermo"},
    {"Fiorentina", "Juventus"},
    {"Juventus", "Napoli"},
    {"Juventus", "Inter"},
    
    # Ligue 1
    {"Paris Saint-Germain", "Marseille"},
    {"Paris Saint-Germain", "Paris FC"},
    {"Lyon", "Marseille"},
    {"Lille", "Lens"},
    {"Nice", "Monaco"},
    {"Bordeaux", "Toulouse"},
    {"Lyon", "Saint-Étienne"},
    {"Lyon", "Paris Saint-Germain"}
]

def is_derby(team_a: str, team_b: str) -> bool:
    """Prüft blitzschnell, ob zwei Teams ein Derby spielen."""
    # Wir bereinigen "FC", "BSC", etc. für einen robusteren Match
    a_clean = team_a.replace("FC ", "").replace(" CF", "").strip()
    b_clean = team_b.replace("FC ", "").replace(" CF", "").strip()
    
    match_set = {a_clean, b_clean}
    
    # Exakter Match
    for pair in DERBY_PAIRS:
        # Check ob beide Teams im Paar gefunden werden (Teil-Strings erlaubt)
        matches = 0
        for p in pair:
            if p in a_clean or a_clean in p or p in b_clean or b_clean in p:
                matches += 1
        if matches >= 2:
            return True
            
    return False