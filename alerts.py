# TODO: turn moving averages into easily editable variables
import asyncio
import statistics
from datetime import datetime, timedelta
from io import BytesIO

import ccxt.async_support as ccxt
import disnake
import plotly.graph_objects as go

from config.settings import ALERTS_CHANNEL
from data.db import Database
from data.models import Alert
from logger_config import setup_logging
from utils.calcs import format_currency, format_number
from utils.crypto_data import fetch_coin_info

logger = setup_logging(name="Alerts Worker", default_color="purple")

RVOL_UP = 1.5
RVOL_DOWN = 0.3
# timeout duration for alerts in seconds (4 hours)
alert_timeout_duration = 60 * 60 * 4
short_ma_period = 9
long_ma_period = 21

alerts_channel_id = ALERTS_CHANNEL


class CryptoAnalyzer:
    """Class to analyze cryptocurrency data and send alerts."""

    def __init__(self, exchange_id, timeframe, lookback_days, bot, channel_id):
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = getattr(ccxt, exchange_id)()
        self.bot = bot
        self.channel_id = channel_id
        self.db = Database()

    async def calculate_moving_average(self, symbol, period):
        """Calculate the moving average of a given symbol."""
        since = self.exchange.parse8601(
            str(datetime.utcnow() - timedelta(days=period * 2))
        )
        candles = await self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
        closes = [candle[4] for candle in candles]
        return statistics.mean(closes)

    async def plot_ohlcv(self, symbol, candles):
        dates = [datetime.utcfromtimestamp(candle[0] / 1000) for candle in candles]
        closes = [candle[4] for candle in candles]
        lows = [candle[3] for candle in candles]
        highs = [candle[2] for candle in candles]
        volumes = [candle[5] for candle in candles]
        ma_short = await self.calculate_moving_average(symbol, short_ma_period)
        ma_long = await self.calculate_moving_average(symbol, long_ma_period)

        # Create a plot with two y-axes
        fig = go.Figure()

        # Add closing price trace
        fig.add_trace(
            go.Scatter(x=dates, y=closes, name="Close", line=dict(color="blue"))
        )

        # Add low-high range area
        fig.add_trace(
            go.Scatter(
                x=dates, y=highs, name="High", line=dict(width=0), showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=lows,
                name="Low",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.4)",
                showlegend=False,
            )
        )

        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[ma_short] * len(dates),
                name="21-day MA",
                line=dict(color="orange", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[ma_long] * len(dates),
                name="51-day MA",
                line=dict(color="green", dash="dot"),
            )
        )

        # Add volume as bar chart
        fig.add_trace(
            go.Bar(
                x=dates,
                y=volumes,
                name="Volume",
                yaxis="y2",
                marker_color="grey",
                opacity=0.5,
            )
        )

        # Set up the layout
        fig.update_layout(
            title=f"{symbol} - 30 Day Chart",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right"),
            template="plotly_white",
        )

        # Save to BytesIO object
        image_bytes = BytesIO()
        fig.write_image(image_bytes, format="png")
        image_bytes.seek(0)

        return image_bytes

    async def send_discord_alert(self, title, description, color, image_bytes=None):
        channel = self.bot.get_channel(self.channel_id)
        if channel:
            try:
                if image_bytes:
                    disnake_file = disnake.File(image_bytes, filename="chart.png")
                    await channel.send(
                        content=f"**{title}**\n{description}", file=disnake_file
                    )
                else:
                    message = f"**{title}**\n{description}"
                    await channel.send(message)
            except disnake.HTTPException as http_exc:
                logger.error(f"HTTPException while sending alert: {http_exc}")
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        else:
            logger.error(f"Could not find channel with ID {self.channel_id}")

    async def should_send_alert(self, symbol):
        """Check if an alert should be sent for a given symbol."""
        last_alert = self.db.session.query(Alert).filter_by(symbol=symbol).first()
        if (
            last_alert
            and (datetime.utcnow() - last_alert.last_alerted_at).total_seconds()
            < alert_timeout_duration  # 24 hours
        ):
            return False
        return True

    async def update_last_alert_time(self, symbol):
        """Update the last alert time for a given symbol in the database."""
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
        """Fetch Chart data for a given symbol."""
        since = self.exchange.parse8601(
            str(datetime.utcnow() - timedelta(days=self.lookback_days))
        )
        candles = await self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
        return candles

    async def analyze_volume(self, symbol, candles):
        """Analyze the volume of a given symbol and send alerts if necessary."""
        if not candles:
            logger.warning(f"No Chart data available for {symbol}.")
            return

        if not await self.should_send_alert(symbol):
            logger.info(f"Alert for {symbol} is within the cooldown period. Skipping.")
            return

        volumes = [candle[5] for candle in candles]
        average_volume = statistics.mean(volumes)
        current_volume = candles[-1][5]

        if current_volume <= RVOL_UP * average_volume:
            logger.info(f"Volume for {symbol} is not significantly higher. No alert.")
            return

        coin_name = symbol.split("/")[0]
        coin_data = await fetch_coin_info(coin_name)
        if "error" in coin_data:
            logger.error(f"Error fetching data for {coin_name}: {coin_data['error']}")
            return

        # Generate and send the alert with the Chart plot
        title = f"📈 RVOL Alert: {symbol} 📈"
        description = self.generate_alert_description(coin_data, current_volume)
        filename = await self.plot_ohlcv(symbol, candles)
        await self.send_discord_alert(
            title, description, disnake.Color.green(), filename
        )
        await self.update_last_alert_time(symbol)

    def generate_alert_description(self, coin_data, current_volume):
        """Generate the alert description text."""
        description = f"Current volume ({current_volume}) is significantly higher than the 30-day average.\n\n"
        description += f"**{coin_data['name']} ({coin_data['symbol'].upper()})**\n"

        # Check if 'usd' key exists in current_price and total_volume
        current_price = coin_data["market_data"]["current_price"].get(
            "usd", "Data not available"
        )
        total_volume = coin_data["market_data"]["total_volume"].get(
            "usd", "Data not available"
        )

        description += (
            f"Current Price: ${format_number(current_price)}\n"
            if current_price != "Data not available"
            else "Current Price: Data not available\n"
        )
        description += (
            f"24h Volume: {format_currency(total_volume)}\n"
            if total_volume != "Data not available"
            else "24h Volume: Data not available\n"
        )

        change_24h = coin_data["market_data"][
            "price_change_percentage_24h_in_currency"
        ].get("usd")
        if change_24h and isinstance(change_24h, (float, int)):
            description += f"24h Change: {change_24h:.2f}%\n"
        else:
            description += "24h Change: Data not available\n"

        return description

    async def process_symbol(self, symbol):
        """Process a single symbol for analysis."""
        logger.info(f"Processing symbol {symbol}")
        candles = await self.fetch_candles(symbol)
        if candles:
            await self.analyze_volume(symbol, candles)

    async def run(self):
        """Main loop to process all symbols continuously."""
        await self.exchange.load_markets()
        # Filter out only symbols that end with USDT and remove all symbols that include "UP" or "DOWN" or "BULL" or "BEAR"
        symbols = [
            symbol
            for symbol in self.exchange.symbols
            if "USDT" in symbol
            and "UP" not in symbol
            and "DOWN" not in symbol
            and "BULL" not in symbol
            and "BEAR" not in symbol
            and ":USDT" not in symbol
        ]

        max_iterations = 10  # Set your desired number of iterations
        iteration = 0

        while iteration < max_iterations:
            for symbol in symbols:
                if "/" in symbol:
                    await self.process_symbol(symbol)
            logger.info("Completed one loop for all symbols. Starting over...")
            await asyncio.sleep(60)
            iteration += 1

        await self.exchange.close()
