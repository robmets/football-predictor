import asyncio
from curl_cffi.requests import AsyncSession

async def get_player_stats_and_attributes(player_id: int) -> dict:
    """
    Holt das Profil und die spezifischen Attribute (Radar-Chart) eines Spielers.
    """
    profile_url = f"https://www.sofascore.com/api/v1/player/{player_id}"
    attributes_url = f"https://www.sofascore.com/api/v1/player/{player_id}/attribute-overviews"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com"
    }
    
    async with AsyncSession(impersonate="chrome") as client:
        # Wir rufen beide URLs gleichzeitig (asynchron) ab, damit die API super schnell ist!
        profile_task = client.get(profile_url, headers=headers)
        attr_task = client.get(attributes_url, headers=headers)
        
        profile_res, attr_res = await asyncio.gather(profile_task, attr_task)
        
        result = {
            "player_id": player_id,
            "profile": {},
            "attributes": {}
        }
        
        # 1. Profil-Daten verarbeiten
        if profile_res.status_code == 200:
            player_data = profile_res.json().get("player", {})
            result["profile"] = {
                "name": player_data.get("name"),
                "position": player_data.get("position"),
                "market_value": player_data.get("proposedMarketValue"), # Sofascores eigener Marktwert
                "country": player_data.get("country", {}).get("name")
            }
            
        # 2. Attribute-Daten (Radar Chart) verarbeiten
        if attr_res.status_code == 200:
            attr_data = attr_res.json()
            # Sofascore speichert die aktuellen Attribute meist im Array unter "averageAttributeOverviews"
            attributes_list = attr_data.get("averageAttributeOverviews", [])
            
            if attributes_list:
                # Wir nehmen den aktuellsten Eintrag (Index 0)
                latest_attributes = attributes_list[0] 
                result["attributes"] = {
                    "attacking": latest_attributes.get("attacking"),
                    "technical": latest_attributes.get("technical"),
                    "tactical": latest_attributes.get("tactical"),
                    "defending": latest_attributes.get("defending"),
                    "creativity": latest_attributes.get("creativity"),
                    "year": latest_attributes.get("year")
                }
                
        return result