"""
Alerts Module.

This module provides functionality to analyze cryptocurrency data and send alerts based on various conditions.
It includes tools for fetching candlestick data, calculating technical indicators, and sending alerts to Discord channels.
"""

import asyncio
import os
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List

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
from utils.chart import PlotChart
from utils.crypto_data import fetch_coin_info, get_exchange

logger = setup_logging(name="Alerts Worker", default_color="purple")

RVOL_UP = 1.5
RVOL_UP_EXTREME = 2.4
RVOL_DOWN = 0.3
ABOVE_VWAP = 1.01
BELOW_VWAP = 0.99
# Threshold for large market orders in USD
# $100,000 - 100k USD might change or make it dynamic
LARGE_ORDER_THRESHOLD = 1000000
# timeout duration for alerts in seconds (4 hours)
alert_timeout_duration = 60 * 60 * 4
ma_short = 9
ma_long = 21
sleep_time = 60  # 1 minute
ANALYSIS_WINDOW_HOURS = 4  # Hours to look back for order analysis
TOP_SYMBOLS_COUNT = 5  # Number of "hot" symbols to track
MIN_ORDER_SIZE_USD = 100000  # Minimum order size to track ($100k)
VOLUME_UPDATE_INTERVAL = 300  # Seconds between volume analysis updates (5 min)
MIN_VOLUME_THRESHOLD = 1000000  # Minimum volume to consider a symbol ($1M)
SIGNIFICANT_PRICE_MOVE = 0.02  # 2% price move threshold
VOLUME_SPIKE_THRESHOLD = 2.0  # 2x normal volume
BUY_SELL_RATIO_THRESHOLD = 1.5  # Significant imbalance threshold


@dataclass
class TradeData:
    timestamp: float
    price: float
    amount: float
    side: str
    total_value: float


class MarketData:
    def __init__(self, max_age_hours: int = 24):
        self.trades: deque[TradeData] = deque(maxlen=1000)  # Limit maximum trades stored
        self.max_age_seconds = max_age_hours * 3600
        self.last_update = 0  # Track last update time

    def add_trade(self, trade: dict):
        """Add a new trade and remove old ones."""
        current_time = datetime.now().timestamp()

        # Only process if enough time has passed since last update
        if current_time - self.last_update < 1:  # 1 second minimum between updates
            return

        trade_data = TradeData(
            timestamp=trade["timestamp"] / 1000,
            price=float(trade["price"]),
            amount=float(trade["amount"]),
            side=trade["side"],
            total_value=float(trade["amount"]) * float(trade["price"]),
        )

        # Remove old trades
        while self.trades and (current_time - self.trades[0].timestamp) > self.max_age_seconds:
            self.trades.popleft()

        self.trades.append(trade_data)
        self.last_update = current_time

    def calculate_metrics(self) -> dict:
        """Calculate current metrics from stored trades."""
        if not self.trades:
            return {}

        current_time = datetime.now().timestamp()
        active_trades = [
            t for t in self.trades if (current_time - t.timestamp) <= self.max_age_seconds
        ]

        if not active_trades:
            return {}

        # Basic metrics
        total_volume = sum(t.total_value for t in active_trades)
        buy_trades = [t for t in active_trades if t.side == "buy"]
        sell_trades = [t for t in active_trades if t.side == "sell"]

        # Time-based analysis
        hourly_volumes = self._calculate_hourly_volumes(active_trades)
        volume_trend = self._calculate_volume_trend(hourly_volumes)

        # Price analysis
        price_changes = self._calculate_price_changes(active_trades)

        return {
            "total_volume": total_volume,
            "buy_volume": sum(t.total_value for t in buy_trades),
            "sell_volume": sum(t.total_value for t in sell_trades),
            "trade_count": len(active_trades),
            "buy_count": len(buy_trades),
            "sell_count": len(sell_trades),
            "avg_trade_size": total_volume / len(active_trades),
            "volume_trend": volume_trend,
            "price_volatility": price_changes["volatility"],
            "price_trend": price_changes["trend"],
            "large_trades": self._identify_large_trades(active_trades),
            "hourly_volumes": hourly_volumes,
        }

    def _calculate_hourly_volumes(self, trades: List[TradeData]) -> List[float]:
        """Calculate volume for each hour in the last 24 hours."""
        hours = [0] * 24
        current_hour = datetime.now().hour

        for trade in trades:
            trade_hour = datetime.fromtimestamp(trade.timestamp).hour
            hour_index = (trade_hour - current_hour) % 24
            hours[hour_index] += trade.total_value

        return hours

    def _calculate_volume_trend(self, hourly_volumes: List[float]) -> float:
        """Calculate volume trend using linear regression."""
        if not any(hourly_volumes):
            return 0

        x = np.arange(len(hourly_volumes))
        y = np.array(hourly_volumes)
        slope, _ = np.polyfit(x, y, 1)

        return slope

    def _calculate_price_changes(self, trades: List[TradeData]) -> dict:
        """Analyze price changes and volatility."""
        prices = [t.price for t in trades]
        changes = np.diff(prices) / prices[:-1]

        return {
            "volatility": np.std(changes) if len(changes) > 0 else 0,
            "trend": (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0,
        }

    def _identify_large_trades(self, trades: List[TradeData]) -> List[TradeData]:
        """Identify significant trades using dynamic thresholds."""
        if not trades:
            return []

        volumes = [t.total_value for t in trades]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        # Consider trades > 2 standard deviations above mean as large
        threshold = mean_volume + (2 * std_volume)

        return [t for t in trades if t.total_value > threshold]


class CryptoAnalyzer:
    """Class to analyze cryptocurrency data and send alerts."""

    def __init__(self, exchange_id, timeframe, lookback_days, bot, alert_channels, ping_roles):
        """Initialize the CryptoAnalyzer with the given parameters."""
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = None  # Will be initialized in run()
        self.bot = bot
        self.alert_channels = alert_channels
        self.ping_roles = ping_roles
        self.db = Database()
        self.market_data: Dict[str, MarketData] = {}
        self.hot_symbols = set()

    async def initialize_exchange(self):
        """Initialize exchange connection."""
        if self.exchange is None:
            self.exchange = getattr(ccxt, self.exchange_id)()
            await self.exchange.load_markets()

            # Initialize market_data for USDT pairs
            markets = await self.exchange.fetch_markets()
            usdt_symbols = [
                market["symbol"] for market in markets if market["symbol"].endswith("/USDT")
            ]

            for symbol in usdt_symbols:
                self.market_data[symbol] = MarketData()

            logger.info(f"Initialized {len(usdt_symbols)} USDT trading pairs")

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
            title = "\n🔔 Extreme RVOL Alert 🔔"
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
            title = "\n🔔 MACD Crossover Up Alert 🔔"
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
            title = "\n🔔 MACD Crossover Down Alert 🔔"
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
                title = "\n🔔 RVOL Up & MACD Cross Up Alert 🔔"
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
                title = "\n🔔 RVOL Up & MACD Cross Down Alert 🔔"
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
            title = "\n🔔 VWAP Alert 🔔"
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
        """Fetches and processes large market orders"""
        try:
            exchange = await get_exchange("binance")
            if not exchange:
                raise Exception("Failed to initialize exchange")

            # Fetch recent trades
            orders = await exchange.fetch_trades(symbol)

            # Filter for large orders exceeding threshold
            large_orders = [
                order
                for order in orders
                if order["amount"] * order["price"] > LARGE_ORDER_THRESHOLD
            ]

            # Process each large order
            for order in large_orders:
                if await self.should_send_alert(symbol, "LARGE_ORDER"):
                    await self.send_large_order_alert(symbol, order)
                    await self.update_last_alert_time(symbol, "LARGE_ORDER")

        finally:
            if exchange:
                await exchange.close()

    async def send_large_order_alert(self, symbol: str, order: dict) -> None:
        """Sends alert with chart for large market orders"""
        try:
            # Validate order data
            if not isinstance(order, dict) or "amount" not in order or "price" not in order:
                logger.error(f"Invalid order format for {symbol}: {order}")
                return

            # Extract order details
            amount = float(order["amount"])
            price = float(order["price"])
            timestamp = order.get("timestamp", datetime.now().timestamp() * 1000)
            side = order.get("side", "unknown").capitalize()

            # Generate chart showing the order
            chart_file = await self.generate_large_order_chart(
                symbol=symbol,
                timeframe="15m",
                order_price=price,
                order_timestamp=timestamp,
                order_side=side,
            )

            # Format alert message
            message = (
                f"🚨 Large **{side}** Alert for {symbol} 🚨\n"
                f"Amount: {amount} {symbol.split('/')[0]}\n"
                f"Price: ${price:.2f}\n"
                f"Total: ${amount * price:.2f}\n"
            )

            # Send to Discord with chart
            channel = await self.get_alert_channel("LARGE_ORDER")
            if channel and chart_file:
                with open(chart_file, "rb") as f:
                    discord_file = disnake.File(f, filename="large_order_chart.png")
                    await channel.send(content=message, file=discord_file)

                # Cleanup
                try:
                    os.remove(chart_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary chart file: {e}")

        except Exception as e:
            logger.error(f"Failed to send large order alert for {symbol}: {e}")

    async def generate_large_order_chart(
        self,
        symbol: str,
        timeframe: str,
        order_price: float,
        order_timestamp: float,
        order_side: str,
    ) -> str | None:
        """Generates a chart with the large order indicator"""
        try:
            # Fetch OHLCV data
            ohlcv = await PlotChart.get_ohlcv_data(symbol.split("/")[0], timeframe)
            if not ohlcv:
                logger.error(f"Failed to fetch OHLCV data for {symbol}")
                return None

            # Create DataFrame and calculate EMAs
            df = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], unit="ms")
            df.set_index("Date", inplace=True)

            df["20ema"] = df["Close"].rolling(window=20).mean()
            df["50ema"] = df["Close"].rolling(window=50).mean()

            # Create plot with candlesticks, EMAs and order marker
            fig = go.Figure()

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Price",
                )
            )

            # Add EMAs
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["20ema"], name="20 EMA", line=dict(color="green", width=1)
                )
            )

            # Add order indicator
            order_time = datetime.fromtimestamp(order_timestamp / 1000)
            marker_color = "green" if order_side.lower() == "buy" else "red"
            marker_symbol = "triangle-up" if order_side.lower() == "buy" else "triangle-down"

            fig.add_trace(
                go.Scatter(
                    x=[order_time],
                    y=[order_price],
                    mode="markers",
                    name=f"Large {order_side}",
                    marker=dict(
                        symbol=marker_symbol,
                        size=15,
                        color=marker_color,
                        line=dict(width=2, color="white"),
                    ),
                )
            )

            # Save and return chart
            chart_file = f"charts/large_order_{symbol.replace('/', '_')}_{int(order_timestamp)}.png"
            fig.write_image(chart_file, scale=1.5, width=1000, height=600)
            return chart_file

        except Exception as e:
            logger.error(f"Failed to generate chart for large order alert: {e}")
            return None

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
            logger.error(f"Exchange not available when processing {symbol}: {e!s}")
        except Exception as e:
            logger.error(f"Unexpected error when processing {symbol}: {e!s}")

    async def analyze_market_volume(self):
        """Modified to use historical data for analysis."""
        exchange = None
        try:
            logger.info("Starting market volume analysis...")
            exchange = await get_exchange("binance")
            if not exchange:
                raise Exception("Failed to initialize exchange")

            # Process symbols in batches to avoid overloading
            batch_size = 50
            total_processed = 0
            symbols = list(self.market_data.keys())

            logger.info(f"Processing {len(symbols)} symbols in batches of {batch_size}")

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}"
                )

                # Process batch with timeout protection
                try:
                    async with asyncio.timeout(60):  # 60 second timeout per batch
                        for symbol in batch:
                            try:
                                trades = await exchange.fetch_trades(
                                    symbol, limit=100  # Reduced limit for faster processing
                                )

                                if trades:
                                    for trade in trades:
                                        self.market_data[symbol].add_trade(trade)
                                    total_processed += 1

                                    if total_processed % 10 == 0:  # Log progress every 10 symbols
                                        logger.info(
                                            f"Processed {total_processed}/{len(symbols)} symbols"
                                        )

                            except Exception as e:
                                logger.error(f"Error fetching trades for {symbol}: {e}")
                                continue

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout processing batch starting at symbol {i}")
                    continue

                # Add small delay between batches to avoid rate limits
                await asyncio.sleep(1)

            logger.info(f"Completed initial data collection. Processing {total_processed} symbols")

            # Calculate scores only for symbols with data
            scores = []
            scored_count = 0

            for symbol, data in self.market_data.items():
                try:
                    metrics = data.calculate_metrics()
                    if not metrics:
                        continue

                    # Only consider symbols with significant volume
                    if metrics["total_volume"] < MIN_VOLUME_THRESHOLD:
                        continue

                    score = self._calculate_symbol_score(metrics)
                    scores.append((symbol, score, metrics))
                    scored_count += 1

                    if scored_count % 10 == 0:
                        logger.info(f"Calculated scores for {scored_count} symbols")

                except Exception as e:
                    logger.error(f"Error calculating metrics for {symbol}: {e}")
                    continue

            logger.info(f"Found {len(scores)} symbols with significant activity")

            if not scores:
                logger.warning("No symbols met the activity criteria")
                return

            # Sort and identify hot symbols
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            new_hot_symbols = set(symbol for symbol, score, _ in sorted_scores[:TOP_SYMBOLS_COUNT])

            if new_hot_symbols:
                logger.info(f"Hot symbols identified: {', '.join(new_hot_symbols)}")
                logger.info("Top 5 scores:")
                for symbol, score, _ in sorted_scores[:5]:
                    logger.info(f"{symbol}: {score:.2f}")

            # Alert on newly hot symbols
            for symbol in new_hot_symbols - self.hot_symbols:
                metrics = next(m for s, _, m in sorted_scores if s == symbol)
                await self.send_hot_symbol_alert(symbol, metrics)

            self.hot_symbols = new_hot_symbols

        except Exception as e:
            logger.error(f"Error in market analysis: {e}", exc_info=True)
        finally:
            if exchange:
                await exchange.close()

    def _calculate_symbol_score(self, metrics: dict) -> float:
        """Calculate a comprehensive score for symbol activity."""
        if not metrics:
            return 0.0

        # Volume score
        volume_score = metrics["total_volume"] / MIN_VOLUME_THRESHOLD

        # Trend scores
        volume_trend_score = max(0, metrics["volume_trend"]) * 2
        price_trend_score = abs(metrics["price_trend"]) * 3

        # Activity scores
        trade_frequency = metrics["trade_count"] / (24 * 3600)  # trades per second
        activity_score = min(1, trade_frequency * 10)

        # Large trade impact
        large_trade_score = len(metrics["large_trades"]) / max(1, metrics["trade_count"])

        # Combine scores with weights
        return (
            volume_score * 0.3
            + volume_trend_score * 0.2
            + price_trend_score * 0.2
            + activity_score * 0.15
            + large_trade_score * 0.15
        )

    async def send_hot_symbol_alert(self, symbol: str, volume_data: dict):
        """Sends enhanced alert for a hot symbol with detailed market metrics."""
        try:
            chart_file = await self.generate_volume_analysis_chart(
                symbol, volume_data["large_trades"]
            )

            # Format detailed message with market insights
            buy_sell_status = (
                "🟢 Buying Pressure"
                if volume_data["buy_sell_ratio"] > BUY_SELL_RATIO_THRESHOLD
                else "🔴 Selling Pressure"
                if volume_data["buy_sell_ratio"] < 1 / BUY_SELL_RATIO_THRESHOLD
                else "⚪ Neutral"
            )

            message = (
                f"🔥 **Market Activity Alert**: {symbol} 🔥\n\n"
                f"**Price Action ({ANALYSIS_WINDOW_HOURS}h)**\n"
                f"Current: ${format_number(volume_data['current_price'])}\n"
                f"Change: {format_number(volume_data['price_change_pct'])}%\n"
                f"Volatility: {format_number(volume_data['volatility'] * 100)}%\n\n"
                f"**Volume Analysis**\n"
                f"Total Volume: ${format_currency(volume_data['total_volume_usd'])}\n"
                f"Market Status: {buy_sell_status}\n"
                f"Buy/Sell Ratio: {format_number(volume_data['buy_sell_ratio'])}\n"
                f"Buy Orders: {volume_data['buy_count']:,}\n"
                f"Sell Orders: {volume_data['sell_count']:,}\n\n"
                f"**Large Trade Analysis**\n"
                f"Large Trades: {len(volume_data['large_trades'])}\n"
                f"Market Impact: {format_number(volume_data['large_trade_impact'] * 100)}%\n"
                f"Avg Trade Size: ${format_currency(volume_data['avg_trade_size'])}\n\n"
                f"**Trading Activity**\n"
                f"Total Trades: {volume_data['trade_count']:,}\n"
                f"Avg Time Between Trades: {format_number(volume_data['avg_time_between_trades'])}s\n"
            )

            # Add market context or warnings
            if volume_data["buy_sell_ratio"] > BUY_SELL_RATIO_THRESHOLD:
                message += "\n⚠️ **Strong buying pressure detected**\n"
            elif volume_data["buy_sell_ratio"] < 1 / BUY_SELL_RATIO_THRESHOLD:
                message += "\n⚠️ **Heavy selling pressure detected**\n"

            if volume_data["volatility"] > 0.05:  # 5% volatility
                message += "⚠️ **High volatility alert**\n"

            if volume_data["large_trade_impact"] > 0.3:  # 30% of volume from large trades
                message += "⚠️ **Significant whale activity**\n"

            # Send to Discord
            channel = await self.get_alert_channel("HOT_MARKET")
            if channel and chart_file:
                with open(chart_file, "rb") as f:
                    discord_file = disnake.File(f, filename="market_analysis.png")
                    await channel.send(content=message, file=discord_file)

                try:
                    os.remove(chart_file)
                except Exception as e:
                    logger.warning(f"Failed to remove chart file: {e}")

        except Exception as e:
            logger.error(f"Failed to send hot symbol alert for {symbol}: {e}")

    async def generate_volume_analysis_chart(
        self,
        symbol: str,
        large_trades: list,
    ) -> str | None:
        """Generates a chart showing price action and large trades."""
        try:
            # Fetch OHLCV data
            ohlcv = await PlotChart.get_ohlcv_data(symbol.split("/")[0], "15m")
            if not ohlcv:
                logger.error(f"Failed to fetch OHLCV data for {symbol}")
                return None

            # Create DataFrame
            df = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], unit="ms")
            df.set_index("Date", inplace=True)

            # Create subplots for price and volume
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(f"{symbol} Price", "Volume"),
                row_heights=[0.7, 0.3],
            )

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Price",
                ),
                row=1,
                col=1,
            )

            # Add volume bars
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)

            # Add markers for large trades
            for trade in large_trades:
                trade_time = datetime.fromtimestamp(trade["timestamp"] / 1000)
                marker_color = "green" if trade["side"] == "buy" else "red"

                # Add marker on price chart
                fig.add_trace(
                    go.Scatter(
                        x=[trade_time],
                        y=[trade["price"]],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-up" if trade["side"] == "buy" else "triangle-down",
                            size=12,
                            color=marker_color,
                            line=dict(width=1, color="white"),
                        ),
                        name=f"{trade['side'].title()} ${format_currency(trade['amount'] * trade['price'])}",
                    ),
                    row=1,
                    col=1,
                )

            # Update layout
            fig.update_layout(
                title=f"{symbol} Volume Analysis",
                xaxis_title="Time",
                yaxis_title="Price (USDT)",
                template="plotly_dark",
                showlegend=True,
                height=800,
            )

            # Save chart
            if not os.path.exists("charts"):
                os.makedirs("charts")

            chart_file = f"charts/volume_analysis_{symbol.replace('/', '_')}.png"
            fig.write_image(chart_file, scale=1.5, width=1000, height=800)
            return chart_file

        except Exception as e:
            logger.error(f"Failed to generate volume analysis chart: {e}")
            return None

    async def run(self):
        """Modified run method to include volume analysis"""
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                logger.info("Starting market analysis system...")
                await self.initialize_exchange()

                while True:
                    try:
                        logger.info("Starting analysis cycle...")

                        # Run volume analysis with timeout protection
                        async with asyncio.timeout(300):  # 5 minute timeout
                            await self.analyze_market_volume()

                        # Process hot symbols in detail
                        if self.hot_symbols:
                            logger.info(f"Processing {len(self.hot_symbols)} hot symbols...")
                            for symbol in self.hot_symbols:
                                await self.process_symbol(symbol)

                        logger.info(
                            f"Completed analysis cycle. Hot symbols: {', '.join(self.hot_symbols)}"
                        )

                        next_update = datetime.now() + timedelta(seconds=VOLUME_UPDATE_INTERVAL)
                        logger.info(f"Next update at {next_update.strftime('%H:%M:%S')}")

                        await asyncio.sleep(VOLUME_UPDATE_INTERVAL)

                    except asyncio.TimeoutError:
                        logger.error("Analysis cycle timed out")
                        await asyncio.sleep(60)
                    except Exception as e:
                        logger.error(f"Error in analysis cycle: {e}", exc_info=True)
                        await asyncio.sleep(60)

            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Fatal error in run loop (attempt {retry_count}/{max_retries}): {e}",
                    exc_info=True,
                )
                if retry_count < max_retries:
                    await asyncio.sleep(60)
                else:
                    logger.critical("Max retries reached, shutting down analysis system")
                    break
            finally:
                if self.exchange:
                    await self.exchange.close()
