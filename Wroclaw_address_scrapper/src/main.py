import asyncio
from src.config import Urls
from src.scrapper import Scrapper

async def main():
    scrp = Scrapper(Urls.main_site, [])
    await scrp.scrap()

if __name__ == '__main__':
    asyncio.run(main())
