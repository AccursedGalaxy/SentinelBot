import asyncio
import statistics
from datetime import datetime, timedelta

import ccxt.async_support as ccxt
import discord

from config.settings import ALERTS_CHANNEL
from data.db import Database
from data.models import Alert
from logger_config import setup_logging
from utils.crypto_data import fetch_coin_info

logger = setup_logging(name="Alerts Worker", default_color="purple")

RVOL_UP = 2.5
RVOL_DOWN = 0.2

alerts_channel_id = ALERTS_CHANNEL


def format_number(value):
    if value == 0:
        return "0"
    elif abs(value) < 1e-9:
        return f"{value:.10f}"  # For very small numbers, show 10 decimal places
    elif abs(value) < 1e-6:
        return f"{value:.8f}"  # For very small numbers, show 8 decimal places
    elif abs(value) < 1e-3:
        return f"{value:.6f}"  # For small numbers, show 6 decimal places
    else:
        return f"{value:.2f}"  # For larger numbers, show 2 decimal places


def format_currency(value: float) -> str:
    """Format a float as a currency string, converting to a more readable format."""
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.1f}T"
    elif value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.2f}"


# Database model for reference:
# class Alert(Base):
#     __tablename__ = "alerts"

#     id = Column(Integer, primary_key=True)
#     symbol = Column(String, nullable=False)
#     alert_type = Column(String, nullable=False)  # e.g., "RVOL_UP" or "RVOL_DOWN"
#     timestamp = Column(DateTime, nullable=False)
#     last_alerted_at = Column(DateTime, nullable=False)


class CryptoAnalyzer:
    def __init__(self, exchange_id, timeframe, lookback_days, bot, channel_id):
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = getattr(ccxt, exchange_id)()
        self.bot = bot
        self.channel_id = channel_id
        self.db = Database()

    async def send_discord_alert(self, title, description, color):
        channel = self.bot.get_channel(self.channel_id)
        if channel:
            try:
                # Using markdown formatting for the message
                message = f"**{title}**\n{description}"
                await channel.send(message)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        else:
            logger.error(f"Could not find channel with ID {self.channel_id}")

    async def should_send_alert(self, symbol):
        # Check the database for the last alert time for this symbol
        last_alert = self.db.session.query(Alert).filter_by(symbol=symbol).first()
        if (
            last_alert
            and (datetime.utcnow() - last_alert.last_alerted_at).total_seconds() < 86400
        ):
            return False
        return True

    async def update_last_alert_time(self, symbol):
        # Update the last alert time in the database
        alert = self.db.session.query(Alert).filter_by(symbol=symbol).first()
        if alert:
            alert.last_alerted_at = datetime.utcnow()
        else:
            alert = Alert(
                symbol=symbol,
                alert_type="YourAlertType",
                timestamp=datetime.utcnow(),
                last_alerted_at=datetime.utcnow(),
            )
            self.db.session.add(alert)
        self.db.session.commit()

    async def fetch_candles(self, symbol):
        since = self.exchange.parse8601(
            str(datetime.utcnow() - timedelta(days=self.lookback_days))
        )
        candles = await self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
        return candles

    async def analyze_volume(self, symbol, candles):
        if not candles:
            logger.info(f"No candles fetched for {symbol}")
            return

        if not await self.should_send_alert(symbol):
            logger.info(
                f"Skipping alert for {symbol} as it's within the 24-hour threshold."
            )
            return

        volumes = [candle[5] for candle in candles]
        average_volume = statistics.mean(volumes)
        current_volume = candles[-1][5]

        coin_name = symbol.split("/")[0]

        coin_data = await fetch_coin_info(coin_name)

        if coin_data and "market_data" in coin_data:
            # Prepare the message
            title = (
                f"📈 RVOL Alert: {symbol}"
                if current_volume > RVOL_UP * average_volume
                else f"📉 RVOL Alert: {symbol}"
            )
            description = f"Current volume ({current_volume}) is {'significantly higher' if current_volume > RVOL_UP * average_volume else 'significantly lower'} than the 30-day average."
            message = f"**{title}**\n{description}\n\n"

            # Add additional coin data to the message
            message += f"**{coin_data['name']} ({coin_data['symbol'].upper()})**\n"
            message += f"Current Price: ${format_number(coin_data['market_data']['current_price']['usd'])}\n"
            message += f"24h Volume: {format_currency(coin_data['market_data']['total_volume']['usd'])}\n"
            if (
                "usd"
                in coin_data["market_data"]["price_change_percentage_24h_in_currency"]
            ):
                message += f"24h Change: {coin_data['market_data']['price_change_percentage_24h_in_currency']['usd']:.2f}%\n"
            else:
                message += "24h Change: Data not available\n"

            # Send the alert message
            await self.send_discord_alert(
                title,
                message,
                discord.Color.green()
                if current_volume > RVOL_UP * average_volume
                else discord.Color.red(),
            )

            # Update the alert time
            await self.update_last_alert_time(symbol)

            # Generate and send the chart image
            # You need to adapt the chart generation function to return the image as bytes and use it here
            # For example:
            # chart_image_bytes = await generate_chart(coin_data)
            # if chart_image_bytes:
            #     await channel.send(file=discord.File(fp=chart_image_bytes, filename='chart.png'))
        else:
            logger.info(f"Could not fetch additional data for {coin_name}.")

    async def process_symbol(self, symbol):
        logger.info(f"Processing symbol {symbol}")
        candles = await self.fetch_candles(symbol)
        if candles:
            await self.analyze_volume(symbol, candles)

    async def run(self):
        await self.exchange.load_markets()
        symbols = [
            symbol for symbol in self.exchange.symbols if symbol.endswith("/USDT")
        ]
        while True:
            for symbol in symbols:
                if "/" in symbol:
                    await self.process_symbol(symbol)
            logger.info("Completed one loop for all symbols. Starting over...")
            await asyncio.sleep(60)
            await self.exchange.close()
