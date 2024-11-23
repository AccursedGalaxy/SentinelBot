"""
Alerts Module.

This module provides functionality to analyze cryptocurrency data and send alerts based on various conditions.
It includes tools for fetching candlestick data, calculating technical indicators, and sending alerts to Discord channels.
"""

import asyncio
import statistics
from datetime import datetime, timedelta
from io import BytesIO

import ccxt.async_support as ccxt
import disnake
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import FALLBACK_CHANNEL
from data.db import Database
from data.models import Alert
from logger_config import setup_logging
from utils.calcs import format_currency, format_number
from utils.crypto_data import fetch_coin_info, get_exchange

logger = setup_logging(name="Alerts Worker", default_color="purple")

RVOL_UP = 1.5
RVOL_UP_EXTREME = 2.4
RVOL_DOWN = 0.3
ABOVE_VWAP = 1.01
BELOW_VWAP = 0.99
# Threshold for large market orders in USD
# $100,000 - 100k USD might change or make it dynamic
LARGE_ORDER_THRESHOLD = 100000
# timeout duration for alerts in seconds (4 hours)
alert_timeout_duration = 60 * 60 * 4
ma_short = 9
ma_long = 21
sleep_time = 60  # 1 minute


class CryptoAnalyzer:
    """Class to analyze cryptocurrency data and send alerts."""

    def __init__(self, exchange_id, timeframe, lookback_days, bot, alert_channels, ping_roles):
        """Initialize the CryptoAnalyzer with the given parameters."""
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = getattr(ccxt, exchange_id)()
        self.bot = bot
        self.alert_channels = alert_channels
        self.ping_roles = ping_roles
        self.db = Database()

    async def calculate_moving_average(self, symbol, period):
        """Calculate the moving average of a given symbol."""
        since = self.exchange.parse8601(str(datetime.now() - timedelta(days=period * 2)))
        candles = await self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
        closes = [candle[4] for candle in candles]
        return statistics.mean(closes)

    async def calculate_rsi(self, symbol, period=14):
        """Calculate the Relative Strength Index (RSI) of a given symbol."""
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

    async def calculate_macd(self, symbol, short_period=12, long_period=26, signal_period=9):
        """Calculate the Moving Average Convergence Divergence (MACD) of a given symbol."""
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

    async def calculate_vwap(self, symbol):
        """Optimized calculation of the Volume Weighted Average Price (VWAP)."""
        candles = await self.fetch_candles(symbol)

        typical_prices = np.array([(candle[2] + candle[3] + candle[4]) / 3 for candle in candles])

        volumes = np.array([candle[5] for candle in candles])

        cumulative_tpv = np.cumsum(typical_prices * volumes)

        cumulative_volume = np.cumsum(volumes)

        vwap = cumulative_tpv / cumulative_volume

        return vwap[-1]

    async def plot_ohlcv(self, symbol, candles, alert_type=None):
        """Optimized plotting of OHLCV data."""
        dates = np.array([datetime.utcfromtimestamp(candle[0] / 1000) for candle in candles])
        closes = np.array([candle[4] for candle in candles])
        highs = np.array([candle[2] for candle in candles])
        lows = np.array([candle[3] for candle in candles])
        volumes = np.array([candle[5] for candle in candles])

        typical_prices = (highs + lows + closes) / 3
        vp = volumes * typical_prices
        cumulative_vp = np.cumsum(vp)
        cumulative_volume = np.cumsum(volumes)
        vwaps = cumulative_vp / cumulative_volume

        # Create a subplot figure with 3 rows to include MACD
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(symbol, "Volume", "MACD"),
            row_width=[0.3, 0.2, 0.5],
        )

        # Main plot with closing prices and VWAP
        fig.add_trace(go.Scatter(x=dates, y=closes, mode="lines", name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=vwaps, mode="lines", name="VWAP"), row=1, col=1)

        # Volume plot
        fig.add_trace(go.Bar(x=dates, y=volumes, name="Volume"), row=2, col=1)

        # Retrieve MACD and signal line
        macd, signal = await self.calculate_macd(symbol)
        histogram = macd - signal
        macd_dates = dates[-len(macd) :]

        # MACD plot
        fig.add_trace(go.Scatter(x=macd_dates, y=macd, mode="lines", name="MACD"), row=3, col=1)
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

    async def get_alert_channel(self, alert_type):
        """Get the alert channel based on the alert type."""
        fallback_channel_id = FALLBACK_CHANNEL
        channel_id = self.alert_channels.get(
            alert_type, self.alert_channels.get("default", fallback_channel_id)
        )
        return self.bot.get_channel(channel_id)

    async def send_discord_alert(
        self, title, description, color, image_bytes=None, alert_type=None
    ):
        """Send an alert to the Discord channel."""
        # Determine the channel based on the alert type
        channel = await self.get_alert_channel(alert_type)

        if channel:
            try:
                content = f"**{title}**\n{description}"
                if alert_type in self.ping_roles:
                    content = f"{self.ping_roles[alert_type]} {content}"
                if image_bytes:
                    disnake_file = disnake.File(image_bytes, filename="chart.png")
                    await channel.send(content=content, file=disnake_file)
                else:
                    await channel.send(content=content)
            except disnake.HTTPException as http_exc:
                logger.error(f"HTTPException while sending alert: {http_exc}")
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        else:
            logger.error(f"Could not find channel with ID {channel}")

    async def should_send_alert(self, symbol, alert_type):
        """Check if an alert should be sent for a given symbol and alert type."""
        last_alert = (
            self.db.session.query(Alert).filter_by(symbol=symbol, alert_type=alert_type).first()
        )
        if (
            last_alert
            and (datetime.now() - last_alert.last_alerted_at).total_seconds()
            < alert_timeout_duration
        ):
            return False
        return True

    async def update_last_alert_time(self, symbol, alert_type):
        """Update the last alert time for a given symbol and alert type in the database."""
        alert = self.db.session.query(Alert).filter_by(symbol=symbol, alert_type=alert_type).first()
        if alert:
            alert.last_alerted_at = datetime.now()
        else:
            alert = Alert(
                symbol=symbol,
                alert_type=alert_type,
                timestamp=datetime.now(),
                last_alerted_at=datetime.now(),
            )
            self.db.session.add(alert)
        self.db.session.commit()

    async def fetch_candles(self, symbol):
        """Fetch Chart data for a given symbol."""
        since = self.exchange.parse8601(str(datetime.now() - timedelta(days=self.lookback_days)))
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
        title = f"\n🔔 RVOL Alert: {symbol} 🔔"
        description = self.generate_alert_description(coin_data, current_volume)
        filename = await self.plot_ohlcv(symbol, candles)
        await self.send_discord_alert(title, description, disnake.Color.green(), filename)
        await self.update_last_alert_time(symbol, alert_type="RVOL")

    def generate_alert_description(self, coin_data, current_volume):
        """Generate the alert description text."""
        description = f"Current volume ({current_volume}) is significantly higher than the 30-day average.\n\n"
        description += f"**{coin_data['name']} ({coin_data['symbol'].upper()})**\n"

        # Check if 'usd' key exists in current_price and total_volume
        current_price = coin_data["market_data"]["current_price"].get("usd", "Data not available")
        total_volume = coin_data["market_data"]["total_volume"].get("usd", "Data not available")

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

        change_24h = coin_data["market_data"]["price_change_percentage_24h_in_currency"].get("usd")
        if change_24h and isinstance(change_24h, (float, int)):
            description += f"24h Change: {change_24h:.2f}%\n"
        else:
            description += "24h Change: Data not available\n"

        return description

    async def check_and_alert_rvol_extreme(self, symbol, candles, current_volume, average_volume):
        """Check and send alerts for extreme RVOL conditions."""
        if current_volume > RVOL_UP_EXTREME * average_volume and await self.should_send_alert(
            symbol, "RVOL_UP_EXTREME"
        ):
            title = f"\n🔔 Extreme RVOL Alert 🔔"
            description = f"{symbol}: Current volume is **significantly** higher than the average."
            image_bytes = await self.plot_ohlcv(symbol, candles, "RVOL_UP_EXTREME")
            await self.send_discord_alert(
                title,
                description,
                disnake.Color.red(),
                image_bytes,
                alert_type="RVOL_UP_EXTREME",
            )
            await self.update_last_alert_time(symbol, "RVOL_UP_EXTREME")

    async def check_and_alert_macd_crossover(self, symbol, candles, macd, signal, histogram):
        """Check and send alerts for MACD crossovers."""
        macd_crossover_up = macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]
        macd_crossover_down = macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]

        if macd_crossover_up and await self.should_send_alert(symbol, "MACD_CROSSOVER_UP"):
            title = f"\n🔔 MACD Crossover Up Alert 🔔"
            description = f"{symbol}: MACD line has just crossed **above** the signal line."
            image_bytes = await self.plot_ohlcv(symbol, candles, "MACD_CROSSOVER_UP")
            await self.send_discord_alert(
                title,
                description,
                disnake.Color.green(),
                image_bytes,
                alert_type="MACD_CROSSOVER_UP",
            )
            await self.update_last_alert_time(symbol, "MACD_CROSSOVER_UP")

        if macd_crossover_down and await self.should_send_alert(symbol, "MACD_CROSSOVER_DOWN"):
            title = f"\n🔔 MACD Crossover Down Alert 🔔"
            description = f"{symbol}: MACD line has just crossed **below** the signal line."
            image_bytes = await self.plot_ohlcv(symbol, candles, "MACD_CROSSOVER_DOWN")
            await self.send_discord_alert(
                title,
                description,
                disnake.Color.orange(),
                image_bytes,
                alert_type="MACD_CROSSOVER_DOWN",
            )
            await self.update_last_alert_time(symbol, "MACD_CROSSOVER_DOWN")

    async def check_and_alert_rvol_macd_cross(
        self, symbol, candles, current_volume, average_volume, macd, signal
    ):
        """Check and send alerts for RVOL and MACD crossovers."""
        macd_crossover_up = macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]
        macd_crossover_down = macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]

        if current_volume > RVOL_UP * average_volume:
            if macd_crossover_up and await self.should_send_alert(symbol, "RVOL_MACD_CROSS_UP"):
                title = f"\n🔔 RVOL Up & MACD Cross Up Alert 🔔"
                description = f"{symbol}: RVOL is up, and MACD line has just crossed **above** the signal line."
                image_bytes = await self.plot_ohlcv(symbol, candles, "RVOL_MACD_CROSS_UP")
                await self.send_discord_alert(
                    title,
                    description,
                    disnake.Color.blue(),
                    image_bytes,
                    alert_type="RVOL_MACD_CROSS_UP",
                )
                await self.update_last_alert_time(symbol, "RVOL_MACD_CROSS_UP")

            if macd_crossover_down and await self.should_send_alert(symbol, "RVOL_MACD_CROSS_DOWN"):
                title = f"\n🔔 RVOL Up & MACD Cross Down Alert 🔔"
                description = f"{symbol}: RVOL is up, and MACD line has just crossed **below** the signal line."
                image_bytes = await self.plot_ohlcv(symbol, candles, "RVOL_MACD_CROSS_DOWN")
                await self.send_discord_alert(
                    title,
                    description,
                    disnake.Color.purple(),
                    image_bytes,
                    alert_type="RVOL_MACD_CROSS_DOWN",
                )
                await self.update_last_alert_time(symbol, "RVOL_MACD_CROSS_DOWN")

    async def check_and_alert_vwap(self, symbol, candles):
        """Check if price is near VWAP and volume is high, then send alerts."""
        vwap = await self.calculate_vwap(symbol)
        current_price = candles[-1][4]

        if BELOW_VWAP * vwap < current_price < ABOVE_VWAP * vwap and await self.should_send_alert(
            symbol, "VWAP_ALERT"
        ):
            title = f"\n🔔 VWAP Alert 🔔"
            description = f"{symbol}: Price is above VWAP and volume is significantly higher than the average."
            image_bytes = await self.plot_ohlcv(symbol, candles, "VWAP_ALERT")
            await self.send_discord_alert(
                title,
                description,
                disnake.Color.blue(),
                image_bytes,
                alert_type="VWAP_ALERT",
            )
            await self.update_last_alert_time(symbol, "VWAP_ALERT")

    async def fetch_and_alert_large_orders(self, symbol):
        """Fetch large market orders and send alerts if necessary."""
        try:
            exchange = await get_exchange("binance")
            if not exchange:
                raise Exception("Failed to initialize exchange")

            orders = await exchange.fetch_trades(symbol)
            large_orders = [
                order
                for order in orders
                if order["amount"] * order["price"] > LARGE_ORDER_THRESHOLD
            ]
            for order in large_orders:
                if await self.should_send_alert(symbol, "LARGE_ORDER"):
                    await self.send_large_order_alert(symbol, order)
                    await self.update_last_alert_time(symbol, "LARGE_ORDER")
        except Exception as e:
            logger.error(f"Error processing large market orders for {symbol}: {e}")
        finally:
            if exchange:
                await exchange.close()

    async def send_large_order_alert(self, symbol, order):
        """Send an alert for a large market order."""
        try:
            if not isinstance(order, dict) or "amount" not in order or "price" not in order:
                logger.error(f"Invalid order format for {symbol}: {order}")
                return

            # Extract amount, price, and side safely
            amount = order["amount"]
            price = order["price"]
            # Normalize and capitalize the side
            side = order.get("side", "unknown").capitalize()

            # Format the alert message to include side of the trade
            message = (
                f"🚨 Large **{side}** Alert for {symbol} 🚨\n"
                f"Amount: {amount} {symbol.split('/')[0]}\n"
                f"Price: ${price:.2f}\n"
                f"Total: ${amount * price:.2f}\n"
            )

            channel = await self.get_alert_channel("LARGE_ORDER")
            if channel:
                await channel.send(message)
        except Exception as e:
            logger.error(f"Failed to send large order alert for {symbol}: {e}")

    async def process_symbol(self, symbol):
        """Process a given symbol and send alerts based on various conditions."""
        logger.info(f"Processing symbol {symbol}")
        try:
            candles = await self.fetch_candles(symbol)
            if candles:
                # Calculate MACD and histogram once
                macd, signal = await self.calculate_macd(symbol)

                # Calculate volume metrics once
                volumes = [candle[5] for candle in candles]
                average_volume = statistics.mean(volumes)
                current_volume = candles[-1][5]

                await self.check_and_alert_rvol_macd_cross(
                    symbol, candles, current_volume, average_volume, macd, signal
                )
                await self.fetch_and_alert_large_orders(symbol)

        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange not available when processing {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error when processing {symbol}: {str(e)}")

    async def run(self):
        """Main loop to process all symbols continuously."""
        await self.exchange.load_markets()
        symbols = [
            symbol
            for symbol in self.exchange.symbols
            if symbol.endswith("/USDT")
            and all(keyword not in symbol for keyword in ["UP", "DOWN", "BULL", "BEAR", ":USDT"])
        ]

        while True:
            for symbol in symbols:
                if "/" in symbol:
                    await self.process_symbol(symbol)
            logger.info(f"Completed one loop for all symbols. Sleeping for {sleep_time} seconds.")
            await asyncio.sleep(sleep_time)
