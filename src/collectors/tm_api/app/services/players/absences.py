from dataclasses import dataclass
from typing import List, Optional

from app.services.base import TransfermarktBase
from app.utils.utils import trim
from app.utils.xpath import Players

@dataclass
class TransfermarktPlayerAbsences(TransfermarktBase):
    player_id: str = None
    # HIER ist der magische Unterschied: 'ausfaelle' statt 'verletzungen'
    URL: str = "https://www.transfermarkt.com/player/ausfaelle/spieler/{player_id}/plus/1/page/{page_number}"
    page_number: int = 1

    def __post_init__(self):
        self.URL = self.URL.format(player_id=self.player_id, page_number=self.page_number)
        self.page = self.request_url_page()
        self.raise_exception_if_not_found(xpath=Players.Profile.URL)

    def __parse_player_absences(self) -> Optional[List[dict]]:
        absences = self.page.xpath(Players.Injuries.RESULTS)
        player_absences = []

        for absence in absences:
            # Wir steuern die Spalten (td) manuell an, da die Sperren-Tabelle eine Spalte mehr hat!
            season = trim(absence.xpath("./td[1]//text()"))
            absence_type = trim(absence.xpath("./td[3]//text()"))
            date_from = trim(absence.xpath("./td[4]//text()"))
            date_until = trim(absence.xpath("./td[5]//text()"))

            player_absences.append({
                "season": season,
                "injury": absence_type, 
                "fromDate": date_from,
                "untilDate": date_until,
            })

        return player_absences

    def get_player_absences(self) -> dict:
        self.response["id"] = self.player_id
        # Wir geben es als 'injuries' Key zurück, das macht das Kombinieren gleich viel leichter
        self.response["injuries"] = self.__parse_player_absences() 
        return self.response