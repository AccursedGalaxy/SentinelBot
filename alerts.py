import asyncio
import statistics
from datetime import datetime, timedelta
from io import BytesIO

import ccxt.async_support as ccxt
import disnake
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import ALERTS_CHANNEL
from data.db import Database
from data.models import Alert
from logger_config import setup_logging
from utils.calcs import format_currency, format_number
from utils.crypto_data import fetch_coin_info

logger = setup_logging(name="Alerts Worker", default_color="purple")

RVOL_UP = 1.6
RVOL_DOWN = 0.3
# timeout duration for alerts in seconds (4 hours)
alert_timeout_duration = 60 * 60 * 4
ma_short = 9
ma_long = 21
sleep_time = 60 * 15  # 15 minutes

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

    async def calculate_rsi(self, symbol, period=14):
        candles = await self.fetch_candles(symbol)
        changes = [candles[i][4] - candles[i - 1][4] for i in range(1, len(candles))]

        gains = [max(change, 0) for change in changes]
        losses = [-min(change, 0) for change in changes]

        average_gain = sum(gains[-period:]) / period
        average_loss = sum(losses[-period:]) / period

        if average_loss == 0:
            return 100  # Prevent division by zero; RSI is considered 100 if there are no losses

        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def calculate_macd(
        self, symbol, short_period=12, long_period=26, signal_period=9
    ):
        candles = await self.fetch_candles(symbol)
        closes = [candle[4] for candle in candles]

        # Convert to pandas Series for EMA calculation
        closes_series = pd.Series(closes)

        # Calculate EMAs
        short_ema = closes_series.ewm(span=short_period, adjust=False).mean()
        long_ema = closes_series.ewm(span=long_period, adjust=False).mean()

        # Calculate MACD and Signal
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()

        return macd, signal

    async def plot_ohlcv(self, symbol, candles, alert_type=None):
        dates = [datetime.utcfromtimestamp(candle[0] / 1000) for candle in candles]
        closes = [candle[4] for candle in candles]
        highs = [candle[2] for candle in candles]
        lows = [candle[3] for candle in candles]
        volumes = [candle[5] for candle in candles]

        # Create a subplot figure with 3 rows to include MACD
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(symbol, "Volume", "MACD"),
            row_width=[0.3, 0.2, 0.5],
        )

        # Main plot with closing prices
        fig.add_trace(
            go.Scatter(x=dates, y=closes, mode="lines", name="Close"), row=1, col=1
        )

        # Volume plot
        fig.add_trace(go.Bar(x=dates, y=volumes, name="Volume"), row=2, col=1)

        # Retrieve MACD and signal line
        macd, signal = await self.calculate_macd(symbol)
        histogram = macd - signal

        # Convert dates for MACD plot alignment
        macd_dates = dates[-len(macd) :]

        # MACD plot
        fig.add_trace(
            go.Scatter(x=macd_dates, y=macd, mode="lines", name="MACD"), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=macd_dates, y=signal, mode="lines", name="Signal Line"),
            row=3,
            col=1,
        )
        fig.add_trace(go.Bar(x=macd_dates, y=histogram, name="Histogram"), row=3, col=1)

        # Update layout
        fig.update_layout(height=800, width=800, title_text=f"{symbol} Analysis")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_layout(template="plotly_white")

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

    async def should_send_alert(self, symbol, alert_type):
        """Check if an alert should be sent for a given symbol and alert type."""
        last_alert = (
            self.db.session.query(Alert)
            .filter_by(symbol=symbol, alert_type=alert_type)
            .first()
        )
        if (
            last_alert
            and (datetime.utcnow() - last_alert.last_alerted_at).total_seconds()
            < alert_timeout_duration
        ):
            return False
        return True

    async def update_last_alert_time(self, symbol, alert_type):
        """Update the last alert time for a given symbol and alert type in the database."""
        alert = (
            self.db.session.query(Alert)
            .filter_by(symbol=symbol, alert_type=alert_type)
            .first()
        )
        if alert:
            alert.last_alerted_at = datetime.utcnow()
        else:
            alert = Alert(
                symbol=symbol,
                alert_type=alert_type,
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

        if not await self.should_send_alert(symbol, alert_type="RVOL"):
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
        title = f"🔔 RVOL Alert: {symbol} 🔔"
        description = self.generate_alert_description(coin_data, current_volume)
        filename = await self.plot_ohlcv(symbol, candles)
        await self.send_discord_alert(
            title, description, disnake.Color.green(), filename
        )
        await self.update_last_alert_time(symbol, alert_type="RVOL")

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
        logger.info(f"Processing symbol {symbol}")
        candles = await self.fetch_candles(symbol)

        if candles:
            await self.analyze_volume(symbol, candles)

            # # RSI Alert
            # rsi = await self.calculate_rsi(symbol)
            # if (rsi > 70 or rsi < 30) and await self.should_send_alert(symbol):
            #     title = f"RSI Alert for {symbol}"
            #     description = (
            #         f"RSI is {'overbought' if rsi > 70 else 'oversold'} at {rsi}."
            #     )
            #     image_bytes = await self.plot_ohlcv(symbol, candles, alert_type="RSI")
            #     await self.send_discord_alert(
            #         title, description, disnake.Color.blue(), image_bytes
            #     )
            #     await self.update_last_alert_time(symbol)

            # MACD Alert
            macd, signal = await self.calculate_macd(symbol)

            # Check for the MACD crossover in the last two bars
            macd_crossover_up = (
                macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]
            )
            macd_crossover_down = (
                macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]
            )

            if (macd_crossover_up) and await self.should_send_alert(
                symbol, alert_type="MACD"
            ):
                direction = "above" if macd_crossover_up else "below"
                title = f"🔔 MACD Alert for {symbol} 🔔"
                description = (
                    f"MACD line has just crossed **{direction}** the signal line."
                )
                image_bytes = await self.plot_ohlcv(symbol, candles, alert_type="MACD")
                await self.send_discord_alert(
                    title, description, disnake.Color.orange(), image_bytes
                )
                await self.update_last_alert_time(symbol, alert_type="MACD")

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

        while True:
            for symbol in symbols:
                if "/" in symbol:
                    await self.process_symbol(symbol)
            logger.info(
                f"Completed one loop for all symbols. Sleeping for {sleep_time} seconds."
            )
            await asyncio.sleep(sleep_time)
