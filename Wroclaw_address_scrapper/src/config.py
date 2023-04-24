"""Basic config"""
from dataclasses import dataclass

@dataclass
class Urls:
    """Urls used"""
    main_site = "https://geoportal.wroclaw.pl/poi/ulice"


@dataclass
class Selectors:
    """Selectors used"""
    street_list ='.list-group.list-group-flush > li'
    street = "a"
    address_div = "#bodyId > div.app-layout > main > div > div:nth-child(4) > div \
        > div.row > div > div > div.card.mb-4.box-shadow-lg > div.row.border-top > \
        div.col-12.col-lg-6.border-right > div > div"
    address_list = "#bodyId > div.app-layout > main > div > div:nth-child(4) > div \
        > div.row > div > div > div.card.mb-4.box-shadow-lg > div.row.border-top > \
        div.col-12.col-lg-6.border-right > div > div > a"
    address = ""
    next_page = 'a[title="Idź do następnej strony"]'
