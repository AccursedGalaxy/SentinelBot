import asyncio
from datetime import datetime

import aiohttp
from sqlalchemy import update

from config.settings import CG_API_KEY
from data.db import Database
from logger_config import setup_logging

logger = setup_logging()


async def fetch_trending_coins(api_key=CG_API_KEY):
    url = "https://pro-api.coingecko.com/api/v3/search/trending"
    headers = {"x-cg-pro-api-key": api_key}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("coins", [])
            else:
                logger.error(
                    f"Failed to fetch trending coins. Status: {response.status}. Response: {await response.text()}"
                )
                return []


import re


def update_database_sync(data):
    db = Database()
    # Set all coins to not trending
    db.session.query(TrendingCoin).update({TrendingCoin.is_trending: False})
    db.session.commit()

    trending_coin_ids = [item["item"]["coin_id"] for item in data]

    for coin_data in data:
        item = coin_data["item"]
        coin = db.session.query(TrendingCoin).filter_by(coin_id=item["coin_id"]).first()

        # Extract numerical part of the price
        price_str = item["data"]["price"].replace("$", "")
        price_str = re.sub(r"<[^>]*>", "", price_str)  # Remove HTML tags
        price = float(price_str) if price_str else 0.0

        if not coin:
            coin = TrendingCoin(
                coin_id=item["coin_id"],
                name=item["name"],
                symbol=item["symbol"],
                market_cap_rank=item["market_cap_rank"],
                slug=item["slug"],
                price_btc=item["price_btc"],
                score=item["score"],
                last_updated=datetime.now(),
                first_trended_at=datetime.now(),
                is_trending=True,
            )
            db.session.add(coin)
        else:
            coin.is_trending = True
            if coin.coin_id not in trending_coin_ids:
                coin.first_trended_at = datetime.now()

        price_change = PriceChange(
            coin_id=item["coin_id"],
            timestamp=datetime.now(),
            price=price,
            price_btc=item["data"]["price_btc"],
            price_change_percentage_24h=item["data"]["price_change_percentage_24h"][
                "usd"
            ],
            market_cap=item["data"]["market_cap"],
            market_cap_btc=item["data"]["market_cap_btc"],
            total_volume=item["data"]["total_volume"],
            total_volume_btc=item["data"]["total_volume_btc"],
        )
        db.session.add(price_change)

    db.session.commit()


async def update_database(data):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, update_database_sync, data)


async def main():
    Database.create_tables()
    while True:
        data = await fetch_trending_coins()
        if data:
            await update_database(data)
            logger.info("Trending coins updated...sleeping for 15 minutes")
        await asyncio.sleep(900)  # Wait for 15 minutes before fetching again


if __name__ == "__main__":
    asyncio.run(main())
