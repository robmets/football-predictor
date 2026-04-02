from curl_cffi.requests import AsyncSession
import urllib.parse

async def search_team_on_sofascore(team_name: str) -> dict:
    """
    Sucht auf Sofascore nach einem Teamnamen und gibt die ID und den exakten Namen zurück.
    """
    # Den Namen für die URL formatieren (z.B. aus "FC Bayern" wird "FC%20Bayern")
    query = urllib.parse.quote(team_name)
    url = f"https://www.sofascore.com/api/v1/search/all?q={query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com"
    }
    
    async with AsyncSession(impersonate="chrome") as client:
        response = await client.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Fehler bei der Suche: {response.status_code}")
            
        data = response.json()
        
        # Sofascore gibt Ergebnisse in Kategorien zurück. Wir suchen die Kategorie "team"
        results = data.get("results", [])
        for item in results:
            if item.get("type") == "team":
                entity = item.get("entity", {})
                return {
                    "sofascore_id": entity.get("id"),
                    "name": entity.get("name"),
                    "country": entity.get("country", {}).get("name")
                }
                
        # Wenn kein Team gefunden wurde
        return {"error": f"Kein Team für '{team_name}' gefunden."}