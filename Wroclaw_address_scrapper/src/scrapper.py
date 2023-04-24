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
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False) # True out of debugging
            page = await browser.new_page()
            await page.goto(self.url)
            for i in range(1000000):
                data = defaultdict(list)
                await page.wait_for_selector(Selectors.street_list)
                street_list = await page.locator(Selectors.street_list).all()
                for street_loc in street_list:
                    # print(street)
                    await page.wait_for_selector(Selectors.street_list)
                    await self.exp_sleep()
                    street = await street_loc.locator(Selectors.street).text_content()
                    street = street.strip()
                    await street_loc.click()
                    await page.wait_for_selector(Selectors.address_div)
                    address_div_loc = page.locator(Selectors.address_div)
                    address_div_text = await address_div_loc.text_content()
                    if "Brak punktów adresowych" in address_div_text:
                        await self.exp_sleep()
                        await page.go_back()
                        continue
                    await page.wait_for_selector(Selectors.address_list)
                    address_list_loc = page.locator(Selectors.address_list)
                    address_list = await address_list_loc.all_text_contents()
                    for address in address_list:
                        data[street].append(address.strip())
                    await self.exp_sleep()
                    await page.go_back()
                with open(str(i), 'x', encoding='utf-8') as f:
                    f.write(str(dict(data)))
                await page.wait_for_selector(Selectors.next_page)
                await page.locator(Selectors.next_page).click()
