# trending_analysis.py
from datetime import datetime, timedelta

import disnake

from config.settings import TRENDS_CHANNEL
from data.db import Database
from logger_config import setup_logging

logger = setup_logging()

# database model for reference
# class TrendingCoin(Base):
#     __tablename__ = "trending_coins"

#     id = Column(Integer, primary_key=True)
#     coin_id = Column(Integer, unique=True, nullable=False)
#     name = Column(String, nullable=False)
#     symbol = Column(String, nullable=False)
#     market_cap_rank = Column(Integer)
#     slug = Column(String)
#     price_btc = Column(Float)
#     score = Column(Integer)
#     last_updated = Column(DateTime)
#     is_trending = Column(Boolean, default=True)
#     first_trended_at = Column(DateTime, nullable=True)

#     # Relationship to store historical price changes and other dynamic data
#     price_changes = relationship("PriceChange", back_populates="coin")

#     def __repr__(self):
#         return f"<TrendingCoin(coin_id='{self.coin_id}', name='{self.name}', symbol='{self.symbol}', market_cap_rank='{self.market_cap_rank}')>"


async def analyze_trending_coins(bot):
    session = Database.get_session()
    try:
        logger.info("Analyzing trending coins...")
        # Fetch all currently trending coins
        trending_coins = session.query(TrendingCoin).filter_by(is_trending=True).all()

        # Identify new trending coins and the longest trending coins
        new_trending_coins = []
        longest_trending_coin = None
        longest_duration = timedelta(0)

        for coin in trending_coins:
            # Check if the coin is newly trending (e.g., within the last hour)
            if datetime.utcnow() - coin.first_trended_at < timedelta(hours=1):
                new_trending_coins.append(coin)
                logger.info(f"New trending coin: {coin.name} ({coin.symbol})")

            # Determine the longest trending coin
            trending_duration = datetime.utcnow() - coin.first_trended_at
            if trending_duration > longest_duration:
                longest_duration = trending_duration
                longest_trending_coin = coin
                logger.info(
                    f"Longest trending coin: {coin.name} ({coin.symbol}) - Trending for {trending_duration}"
                )

        # Send updates to the designated Discord channel
        channel = bot.get_channel(TRENDS_CHANNEL)
        if not channel:
            print(f"Channel with ID {TRENDS_CHANNEL} not found.")
            return

        # Notify about new trending coins
        if new_trending_coins:
            message = "📈 **New Trending Coins:**\n" + "\n".join(
                [f"{coin.name} ({coin.symbol})" for coin in new_trending_coins]
            )
            await channel.send(message)

        # Notify about the longest trending coin
        if longest_trending_coin:
            message = f"🏆 **Longest Trending Coin:** {longest_trending_coin.name} ({longest_trending_coin.symbol}) - Trending for {longest_duration}"
            await channel.send(message)

    finally:
        Database.close_session()
