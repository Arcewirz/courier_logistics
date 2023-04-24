"""Module with Scrapper class."""
import time
import asyncio
import numpy as np
from collections import defaultdict
from playwright.async_api import async_playwright
from src.config import Selectors


class Scrapper():
    """Scraps data.
    """
    def __init__(self,
                 url: str,
                 list: str) -> None:
        self.url = url

    async def exp_sleep(self) -> None:
        """Waits random exponential time
        """
        await asyncio.sleep(np.random.exponential(2)+2)

    async def scrap(self) -> None:
        """Literally scraps data.
        """
        data = defaultdict(list)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False) # True out of debugging
            page = await browser.new_page()
            await page.goto(self.url)
            while True:
                street_list = await page.locator(Selectors.street_list).all()
                for street in street_list:
                    await self.exp_sleep()
                    await street.locator(Selectors.street).click()
                    address_list = await page.locator(Selectors.address_list).all_text_contents()
                    for address in address_list:
                        data[street].append(address)
                    await self.exp_sleep()
                    await page.go_back()
