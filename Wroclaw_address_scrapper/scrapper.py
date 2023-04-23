"""Module with Scrapper class."""
import time
import numpy as np
from playwright.async_api import async_playwright


class Scrapper():
    """Scraps data.
    """
    def __init__(self,
                 url: str,
                 list: str) -> None:
        self.url = url

    def exp_sleep() -> None:
        time.sleep(np.random.exponential(1)+2)

    async def scrap(self) -> None:
        """Literally scraps data.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.goto(self.url)
