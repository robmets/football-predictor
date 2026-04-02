from dataclasses import dataclass, field
from typing import Optional
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from lxml import etree
from requests import Response, TooManyRedirects

from app.utils.utils import trim
from app.utils.xpath import Pagination

@dataclass
class TransfermarktBase:
    URL: str
    page: ElementTree = field(default_factory=lambda: None, init=False)
    response: dict = field(default_factory=lambda: {}, init=False)

    def make_request(self, url: Optional[str] = None) -> Response:
        url = self.URL if not url else url
        try:
            response: Response = requests.get(
                url=url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36",
                },
                timeout=10
            )
        except TooManyRedirects:
            raise ValueError(f"Zu viele Weiterleitungen für URL: {url}")
        except Exception as e:
            raise ValueError(f"Fehler bei der Verbindung zu {url}: {e}")

        if 400 <= response.status_code < 500:
            raise ValueError(f"Client Fehler {response.status_code}: {response.reason} für URL: {url}")
        elif 500 <= response.status_code < 600:
            raise ValueError(f"Server Fehler {response.status_code}: {response.reason} für URL: {url}")
            
        return response

    def request_url_bsoup(self) -> BeautifulSoup:
        response: Response = self.make_request()
        return BeautifulSoup(markup=response.content, features="html.parser")

    @staticmethod
    def convert_bsoup_to_page(bsoup: BeautifulSoup) -> ElementTree:
        return etree.HTML(str(bsoup))

    def request_url_page(self) -> ElementTree:
        bsoup: BeautifulSoup = self.request_url_bsoup()
        return self.convert_bsoup_to_page(bsoup=bsoup)

    def raise_exception_if_not_found(self, xpath: str):
        if not self.get_text_by_xpath(xpath):
            raise ValueError(f"Daten auf der Seite nicht gefunden oder ungültige URL: {self.URL}")

    def get_list_by_xpath(self, xpath: str, remove_empty: Optional[bool] = True) -> Optional[list]:
        if self.page is None:
            self.page = self.request_url_page()
            
        elements: list = self.page.xpath(xpath)
        if remove_empty:
            elements_valid: list = [trim(e) for e in elements if trim(e)]
        else:
            elements_valid: list = [trim(e) for e in elements]
        return elements_valid or []

    def get_text_by_xpath(
        self,
        xpath: str,
        pos: int = 0,
        iloc: Optional[int] = None,
        iloc_from: Optional[int] = None,
        iloc_to: Optional[int] = None,
        join_str: Optional[str] = None,
    ) -> Optional[str]:
        if self.page is None:
            self.page = self.request_url_page()
            
        element = self.page.xpath(xpath)
        if not element:
            return None

        if isinstance(element, list):
            element = [trim(e) for e in element if trim(e)]

        if isinstance(iloc, int):
            element = element[iloc]
        if isinstance(iloc_from, int) and isinstance(iloc_to, int):
            element = element[iloc_from:iloc_to]
        if isinstance(iloc_to, int):
            element = element[:iloc_to]
        if isinstance(iloc_from, int):
            element = element[iloc_from:]

        if isinstance(join_str, str):
            return join_str.join([trim(e) for e in element])

        try:
            return trim(element[pos])
        except IndexError:
            return None

    def get_last_page_number(self, xpath_base: str = "") -> int:
        for xpath in [Pagination.PAGE_NUMBER_LAST, Pagination.PAGE_NUMBER_ACTIVE]:
            url_page = self.get_text_by_xpath(xpath_base + xpath)
            if url_page:
                return int(url_page.split("=")[-1].split("/")[-1])
        return 1