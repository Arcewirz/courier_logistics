"""Basic config"""
from dataclasses import dataclass

@dataclass
class Urls:
    """Urls used"""
    main_site = "https://geoportal.wroclaw.pl/poi/ulice"


@dataclass
class Selectors:
    """Selectors used"""
    street_list ='#id27 > ul > li'
    street = "a"
    address_list = ""
    address = ""
    next_page = ""
