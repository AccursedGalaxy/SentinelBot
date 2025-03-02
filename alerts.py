"""
Alerts Module.

This module provides functionality to analyze cryptocurrency data and send alerts based on various conditions.
It includes tools for fetching candlestick data, calculating technical indicators, and sending alerts to Discord channels.
"""

import asyncio
import os
import statistics
from datetime import datetime, timedelta
from io import BytesIO

import ccxt.async_support as ccxt
import disnake
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.db import Database
from data.models import Alert
from logger_config import setup_logging
from utils.calcs import format_currency, format_number
from utils.chart import PlotChart
from utils.context_gatherer import ContextGatherer
from utils.crypto_data import fetch_coin_info, fetch_ohlcv, get_exchange
from utils.exchange_manager import exchange_manager
from utils.llm_service import LLMService

logger = setup_logging(name="Alerts Worker", default_color="purple")

RVOL_UP = 1.5
RVOL_UP_EXTREME = 2.4
RVOL_DOWN = 0.3
ABOVE_VWAP = 1.01
BELOW_VWAP = 0.99
# timeout duration for alerts in seconds (4 hours)
alert_timeout_duration = 60 * 60 * 4
ma_short = 9
ma_long = 21
sleep_time = 60  # 1 minute
RVOL_MACD_THRESHOLD = 1.8  # Threshold for RVOL MACD cross alerts
MIN_USD_VOLUME = 1000000  # $1 million USD minimum 24h volume for alerts
ORDER_MIN_USD_VOLUME = 500000


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
        self.llm_service = LLMService()
        self.context_gatherer = ContextGatherer()
        self.chart_generator = PlotChart()  # Initialize the chart generator

        # Provide a reference to this analyzer in the context gatherer
        self.context_gatherer.analyzer = self

    async def calculate_moving_average(self, symbol, period):
        """Calculate the moving average of a given symbol."""
        since = self.exchange.parse8601(str(datetime.now() - timedelta(days=period * 2)))
        candles = await self.exchange.fetch_ohlcv(symbol, self.timeframe, since)
        closes = [candle[4] for candle in candles]
        return statistics.mean(closes)

    async def calculate_rsi(self, candles, period=14):
        """Calculate the Relative Strength Index (RSI)."""
        closes = pd.Series([candle[4] for candle in candles])
        delta = closes.diff().dropna()

        # Get gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean().fillna(0)
        avg_loss = loss.rolling(window=period).mean().fillna(0)

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50  # Default to neutral if not enough data

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
        """Generate an enhanced OHLCV chart with better visualization."""
        # Make sure the charts directory exists
        os.makedirs("charts", exist_ok=True)

        # File path for the chart
        chart_file = f"charts/{symbol.replace('/', '_')}_analysis.png"

        # Extract data from candles
        dates = np.array([datetime.utcfromtimestamp(candle[0] / 1000) for candle in candles])
        opens = np.array([candle[1] for candle in candles])
        highs = np.array([candle[2] for candle in candles])
        lows = np.array([candle[3] for candle in candles])
        closes = np.array([candle[4] for candle in candles])
        volumes = np.array([candle[5] for candle in candles])

        # Calculate price changes for coloring volume bars
        price_changes = np.diff(closes, prepend=closes[0])
        volume_colors = [
            "rgba(0, 180, 0, 0.7)" if change >= 0 else "rgba(180, 0, 0, 0.7)"
            for change in price_changes
        ]

        # Calculate indicators
        typical_prices = (highs + lows + closes) / 3
        vp = volumes * typical_prices
        cumulative_vp = np.cumsum(vp)
        cumulative_volume = np.cumsum(volumes)
        vwaps = cumulative_vp / cumulative_volume

        # Create a subplot figure with adjusted heights
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(f"{symbol}", "Volume", "MACD"),
            row_heights=[0.5, 0.2, 0.3],  # Adjusted heights for better proportion
        )

        # Main price chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name="Price",
                increasing_line_color="rgba(0, 180, 0, 0.7)",
                decreasing_line_color="rgba(180, 0, 0, 0.7)",
            ),
            row=1,
            col=1,
        )

        # Add VWAP line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=vwaps,
                mode="lines",
                name="VWAP",
                line=dict(color="rgba(255, 165, 0, 0.8)", width=2),
            ),
            row=1,
            col=1,
        )

        # Volume chart with color coding
        fig.add_trace(
            go.Bar(x=dates, y=volumes, name="Volume", marker_color=volume_colors, opacity=0.8),
            row=2,
            col=1,
        )

        # Calculate MACD indicators
        macd, signal = await self.calculate_macd(symbol)
        histogram = macd - signal
        macd_dates = dates[-len(macd) :]

        # MACD histogram colors
        histogram_colors = [
            "rgba(0, 180, 0, 0.7)" if h >= 0 else "rgba(180, 0, 0, 0.7)" for h in histogram
        ]

        # MACD plot with enhanced styling
        fig.add_trace(
            go.Scatter(
                x=macd_dates,
                y=macd,
                mode="lines",
                name="MACD",
                line=dict(color="rgba(100, 100, 255, 0.9)", width=2),
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=macd_dates,
                y=signal,
                mode="lines",
                name="Signal Line",
                line=dict(color="rgba(255, 165, 0, 0.9)", width=2),
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Bar(x=macd_dates, y=histogram, name="Histogram", marker_color=histogram_colors),
            row=3,
            col=1,
        )

        # Update layout for better aesthetics
        fig.update_layout(
            height=1000,  # Taller chart
            width=1200,  # Wider chart
            title_text=f"{symbol} Analysis",
            title_font=dict(size=24),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white",  # Clean white template
            plot_bgcolor="rgba(245, 245, 245, 0.9)",
            margin=dict(b=40, t=100, r=40, l=40),
        )

        # Add grid lines for better readability
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(211, 211, 211, 0.6)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(211, 211, 211, 0.8)",
            title_font=dict(size=14),
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(211, 211, 211, 0.6)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(211, 211, 211, 0.8)",
            title_font=dict(size=14),
        )

        # Customize specific axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)

        # Remove rangeslider from candlestick
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Save the figure to an image file
        fig.write_image(chart_file, scale=1.5, width=1200, height=1000)

        return chart_file

    async def get_alert_channel(self, alert_type):
        """Get the appropriate Discord channel for the alert type."""
        try:
            if alert_type == "LARGE_ORDER" and "LARGE_ORDER" in self.alert_channels:
                channel_id = self.alert_channels["LARGE_ORDER"]
            else:
                # All other alerts go to the default channel
                channel_id = self.alert_channels.get("DEFAULT", self.alert_channels.get(alert_type))

            if not channel_id:
                logger.warning(f"No channel configured for alert type: {alert_type}")
                return None

            channel = self.bot.get_channel(channel_id)
            if not channel:
                logger.warning(f"Could not find channel with ID: {channel_id}")
                return None

            return channel
        except Exception as e:
            logger.error(f"Error getting alert channel: {e}")
            return None

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

    async def should_send_alert(self, symbol, alert_type, cooldown_minutes=60):
        """
        Check if an alert should be sent based on cooldown period.
        Returns True if no recent alert exists or the cooldown has expired.
        """
        try:
            # Calculate the cutoff time for the cooldown
            cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)

            # Check if we've sent this alert type for this symbol recently
            recent_alert = (
                self.db.session.query(Alert)
                .filter(Alert.symbol == symbol)
                .filter(Alert.alert_type == alert_type)
                .filter(Alert.llm_sent == True)  # Only consider alerts we actually sent
                .filter(Alert.last_alerted_at >= cutoff_time)
                .order_by(Alert.last_alerted_at.desc())
                .first()
            )

            # Return True if no recent alert or if cooldown has expired
            return recent_alert is None

        except Exception as e:
            logger.error(f"Error checking alert cooldown: {e}")
            return True  # If there's an error, default to allowing the alert

    async def update_last_alert_time(self, symbol, alert_type):
        """Update the last alert time for a given symbol and alert type in the database."""
        alert = self.db.session.query(Alert).filter_by(symbol=symbol, alert_type=alert_type).first()
        if alert:
            # Convert to Python datetime before assignment
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
        """Fetch candlestick data for a symbol."""
        try:
            # Calculate the timestamp for lookback days ago
            since = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)

            # Use the exchange manager to fetch OHLCV data
            candles = await fetch_ohlcv(symbol, self.timeframe, limit=500)

            if not candles or len(candles) < 2:
                return None

            return candles

        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return None

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

    async def send_large_order_alert(self, symbol: str, order: dict) -> None:
        """Send an alert for a large market order with a chart showing the order point.

        Args:
            symbol: Trading pair symbol (e.g. 'BTC/USDT')
            order: Dictionary containing order details
        """
        try:
            if not isinstance(order, dict) or "amount" not in order or "price" not in order:
                logger.error(f"Invalid order format for {symbol}: {order}")
                return

            # Extract order details
            amount = float(order["amount"])
            price = float(order["price"])
            timestamp = order.get("timestamp", datetime.now().timestamp() * 1000)  # Convert to ms
            side = order.get("side", "unknown").capitalize()

            # Generate chart with order indicator
            timeframe = "15m"  # Use 15m timeframe for better detail around the order
            chart_file = await self.generate_large_order_chart(
                symbol=symbol,
                timeframe=timeframe,
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

            # Send alert with chart
            channel = await self.get_alert_channel("LARGE_ORDER")
            if channel and chart_file:
                with open(chart_file, "rb") as f:
                    discord_file = disnake.File(f, filename="large_order_chart.png")
                    await channel.send(content=message, file=discord_file)

                # Clean up the chart file
                try:
                    os.remove(chart_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary chart file: {e}")

        except Exception as e:
            logger.error(f"Failed to send large order alert for {symbol}: {e}")

    async def generate_large_order_chart(
        self, symbol, timeframe="15m", order_price=None, order_timestamp=None, order_side=None
    ):
        """Generate a chart with order indicator for large orders."""
        try:
            # Create charts directory if it doesn't exist
            os.makedirs("charts", exist_ok=True)

            # Use the PlotChart class to generate the basic chart
            chart_file = await PlotChart.plot_ohlcv_chart(symbol.split("/")[0], timeframe)

            # For now, we'll use the basic chart - in the future you could enhance
            # this to draw the order marker on the chart
            return chart_file
        except Exception as e:
            logger.error(f"Error generating large order chart for {symbol}: {e}")
            return None

    async def process_symbol(self, symbol):
        """Process a given symbol and send grouped alerts based on various conditions."""
        logger.info(f"Processing symbol {symbol}")
        try:
            # Reset session at the beginning to ensure a clean state
            self.db.session.rollback()

            candles = await self.fetch_candles(symbol)
            if candles:
                # Calculate MACD and histogram once
                macd, signal = await self.calculate_macd(symbol)

                # Calculate volume metrics once
                volumes = [candle[5] for candle in candles]
                average_volume = statistics.mean(volumes)
                current_volume = candles[-1][5]
                current_price = candles[-1][4]

                # Calculate USD volume to filter low-volume tokens
                usd_volume_24h = current_volume * current_price

                # Skip processing if 24h USD volume is below threshold
                if usd_volume_24h < MIN_USD_VOLUME:
                    logger.debug(
                        f"Skipping {symbol} - insufficient 24h volume (${format_currency(usd_volume_24h)})"
                    )
                    return

                # Fetch recent large orders to include in alert context
                # The context gatherer will use this data when preparing alert context
                await self.get_recent_large_orders(symbol)

                # Initialize list to collect all triggered alerts
                triggered_alerts = []

                # Check for MACD crossovers
                macd_crossover_up = (
                    macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]
                )
                macd_crossover_down = (
                    macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]
                )

                # Calculate additional technical indicators
                rsi = await self.calculate_rsi(candles)
                bb = await self.calculate_bollinger_bands(candles)

                # Get price movement
                price_change_pct = ((candles[-1][4] - candles[-2][4]) / candles[-2][4]) * 100

                # Base indicator data that will be used in all alerts
                base_indicators = {
                    "macd": float(macd.iloc[-1]),
                    "signal": float(signal.iloc[-1]),
                    "histogram": float(macd.iloc[-1] - signal.iloc[-1]),
                    "current_volume": current_volume,
                    "average_volume": average_volume,
                    "volume_ratio": current_volume / average_volume,
                    "rsi": rsi,
                    "bollinger_bands": bb,
                    "price_change_24h": price_change_pct,
                }

                # Define the RVOL_MACD_THRESHOLD constant if it doesn't exist
                RVOL_MACD_THRESHOLD = 1.8  # You can adjust this value

                # Check each alert condition and add to triggered alerts if not in cooldown

                # MACD crossover up
                if macd_crossover_up and await self.should_send_alert(symbol, "MACD_CROSSOVER_UP"):
                    triggered_alerts.append(
                        {
                            "type": "MACD_CROSSOVER_UP",
                            "data": {
                                "price": current_price,
                                "volume": current_volume,
                                "indicators": base_indicators,
                            },
                        }
                    )

                # MACD crossover down
                if macd_crossover_down and await self.should_send_alert(
                    symbol, "MACD_CROSSOVER_DOWN"
                ):
                    triggered_alerts.append(
                        {
                            "type": "MACD_CROSSOVER_DOWN",
                            "data": {
                                "price": current_price,
                                "volume": current_volume,
                                "indicators": base_indicators,
                            },
                        }
                    )

                # RVOL extreme
                if (
                    current_volume > RVOL_UP_EXTREME * average_volume
                    and await self.should_send_alert(symbol, "RVOL_UP_EXTREME")
                ):
                    triggered_alerts.append(
                        {
                            "type": "RVOL_UP_EXTREME",
                            "data": {
                                "price": current_price,
                                "volume": current_volume,
                                "indicators": base_indicators,
                            },
                        }
                    )

                # RVOL MACD cross up
                if (
                    macd_crossover_up
                    and current_volume > RVOL_MACD_THRESHOLD * average_volume
                    and await self.should_send_alert(symbol, "RVOL_MACD_CROSS_UP")
                ):
                    triggered_alerts.append(
                        {
                            "type": "RVOL_MACD_CROSS_UP",
                            "data": {
                                "price": current_price,
                                "volume": current_volume,
                                "indicators": base_indicators,
                            },
                        }
                    )

                # RVOL MACD cross down
                if (
                    macd_crossover_down
                    and current_volume > RVOL_MACD_THRESHOLD * average_volume
                    and await self.should_send_alert(symbol, "RVOL_MACD_CROSS_DOWN")
                ):
                    triggered_alerts.append(
                        {
                            "type": "RVOL_MACD_CROSS_DOWN",
                            "data": {
                                "price": current_price,
                                "volume": current_volume,
                                "indicators": base_indicators,
                            },
                        }
                    )

                # If we have multiple alerts, process them as a group
                if len(triggered_alerts) > 1 and await self.should_send_alert(
                    symbol, "GROUPED_ALERT"
                ):
                    await self.process_grouped_alerts(symbol, triggered_alerts, candles)
                # If we have just one alert, process it normally
                elif len(triggered_alerts) == 1:
                    alert_info = triggered_alerts[0]
                    await self.process_alert_with_llm(
                        symbol, alert_info["type"], alert_info["data"]
                    )

        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange not available when processing {symbol}: {e!s}")
            self.db.session.rollback()
        except Exception as e:
            logger.error(f"Unexpected error when processing {symbol}: {e!s}")
            self.db.session.rollback()

    async def process_large_orders_with_llm(self, symbol):
        """Process large orders with LLM analysis before alerting."""
        # Check for cooldown on large order alerts for this symbol
        if not await self.should_send_alert(symbol, "LARGE_ORDER", cooldown_minutes=30):
            return  # Skip if in cooldown

        try:
            # First, check if the symbol meets the minimum volume requirement
            candles = await self.fetch_candles(symbol)
            if not candles or len(candles) < 1:
                return

            current_volume = candles[-1][5]
            current_price = candles[-1][4]
            usd_volume_24h = current_volume * current_price

            # Skip if volume is too low
            if usd_volume_24h < ORDER_MIN_USD_VOLUME:
                logger.debug(
                    f"Skipping large order check for {symbol} - insufficient 24h volume (${format_currency(usd_volume_24h)})"
                )
                return

            # Get trades using the exchange manager
            trades = await exchange_manager.execute("binance", "fetch_trades", symbol, limit=100)
            if not trades:
                return

            # Define minimum order size (restore the filtering logic)
            large_order_threshold = ORDER_MIN_USD_VOLUME * 0.05  # 5% of min volume threshold

            # Filter to only process actual large orders
            large_orders = [
                order
                for order in trades
                if order["amount"] * order["price"] > large_order_threshold
            ]

            # Skip if no large orders found
            if not large_orders:
                return

            # Sort orders by size (largest first) and take the top 3
            large_orders.sort(key=lambda x: x["amount"] * x["price"], reverse=True)
            largest_orders = large_orders[:3]  # Limit to 3 largest orders

            # Process each large order
            for order in largest_orders:
                # Extract order details
                amount = float(order["amount"])
                price = float(order["price"])
                side = order["side"]
                total_value = amount * price
                timestamp = order.get("timestamp", datetime.now().timestamp() * 1000)

                # Only process truly significant orders (adjust threshold if needed)
                if total_value < large_order_threshold:
                    continue

                # Prepare alert data
                alert_data = {
                    "price": price,
                    "amount": amount,
                    "total_value": total_value,
                    "side": side.capitalize(),
                    "timestamp": timestamp,
                }

                # Process through LLM
                alert, llm_result = await self.process_alert_with_llm(
                    symbol, "LARGE_ORDER", alert_data
                )

                # Check if the alert should be sent based on LLM analysis
                if alert is not None and llm_result.get("should_send", False):
                    logger.info(
                        f"Large order alert for {symbol} approved by LLM, sending to Discord"
                    )

                    # Generate chart with order marker
                    chart_file = await self.generate_large_order_chart(
                        symbol,
                        timeframe="15m",
                        order_price=price,
                        order_timestamp=timestamp,
                        order_side=side,
                    )

                    # Convert file to BytesIO for Discord sending
                    if chart_file:
                        try:
                            with open(chart_file, "rb") as f:
                                image_bytes = BytesIO(f.read())
                                image_bytes.seek(0)

                            # Send alert to Discord with enhanced formatting
                            channel = await self.get_alert_channel("LARGE_ORDER")
                            if channel:
                                await self.send_enhanced_discord_alert(
                                    alert, llm_result, image_bytes
                                )
                                logger.info(f"Sent large order alert for {symbol} to Discord")
                            else:
                                logger.error(f"No channel found for LARGE_ORDER alerts")

                            # Clean up the chart file
                            try:
                                os.remove(chart_file)
                            except Exception as e:
                                logger.warning(f"Failed to remove temporary chart file: {e}")
                        except Exception as e:
                            logger.error(f"Error sending large order alert to Discord: {e}")
                else:
                    logger.info(
                        f"Large order alert for {symbol} rejected by LLM or processing failed"
                    )

        except Exception as e:
            logger.error(f"Error processing large market orders for {symbol}: {e}")

    async def run(self):
        """Main loop to process all symbols continuously."""
        symbol_stats = {"total": 0, "filtered": 0, "processed": 0}

        while True:
            try:
                # Reset stats for this run
                symbol_stats = {"total": 0, "filtered": 0, "processed": 0}

                # Get a managed exchange instance
                exchange = await exchange_manager.get_exchange(self.exchange_id)
                if not exchange:
                    logger.error(
                        f"Failed to initialize {self.exchange_id} exchange, retrying in 30s"
                    )
                    await asyncio.sleep(30)
                    continue

                try:
                    # Get symbols from the market cache
                    symbols = [
                        symbol
                        for symbol in exchange.symbols
                        if symbol.endswith("/USDT")
                        and all(
                            keyword not in symbol
                            for keyword in ["UP", "DOWN", "BULL", "BEAR", ":USDT"]
                        )
                    ]

                    symbol_stats["total"] = len(symbols)

                    # Process each symbol
                    for symbol in symbols:
                        if "/" in symbol:
                            # Fetch candles to check volume
                            candles = await self.fetch_candles(symbol)
                            if candles and len(candles) > 0:
                                current_volume = candles[-1][5]
                                current_price = candles[-1][4]
                                usd_volume_24h = current_volume * current_price

                                if usd_volume_24h < MIN_USD_VOLUME:
                                    symbol_stats["filtered"] += 1
                                    continue

                                symbol_stats["processed"] += 1
                                await self.process_symbol(symbol)

                    # Log stats every run
                    logger.info(
                        f"Processed {symbol_stats['processed']} symbols "
                        f"(filtered {symbol_stats['filtered']} low-volume symbols out of {symbol_stats['total']} total)"
                    )

                    # Close idle connections to free up resources
                    await exchange_manager.close_idle()

                except Exception as e:
                    logger.error(f"Error during symbol processing: {e}")

                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                await asyncio.sleep(30)  # Longer sleep on error before retrying

    async def calculate_bollinger_bands(self, candles, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        closes = pd.Series([candle[4] for candle in candles])

        # Calculate middle band (SMA)
        middle_band = closes.rolling(window=period).mean().iloc[-1]

        # Calculate standard deviation
        std = closes.rolling(window=period).std().iloc[-1]

        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)

        # Calculate current price position within bands (%)
        current_price = closes.iloc[-1]
        band_width = upper_band - lower_band
        if band_width > 0:
            position_percent = ((current_price - lower_band) / band_width) * 100
        else:
            position_percent = 50

        return {
            "upper": upper_band,
            "middle": middle_band,
            "lower": lower_band,
            "width": band_width,
            "position_percent": position_percent,
        }

    async def process_grouped_alerts(self, symbol, triggered_alerts, candles):
        """Process multiple alerts for the same symbol as a group."""
        try:
            logger.info(f"Processing {len(triggered_alerts)} grouped alerts for {symbol}")

            # Create a batch ID to associate these alerts
            batch_id = f"{symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Collect alert types and data for a combined context
            alert_types = [alert["type"] for alert in triggered_alerts]

            # Use the data from the first alert as they all share the same indicators
            combined_data = triggered_alerts[0]["data"].copy()
            combined_data["triggered_alerts"] = alert_types

            # Process through LLM with special handling for grouped alerts
            alert, llm_result = await self.process_alert_with_llm(
                symbol, "GROUPED_ALERT", combined_data, batch_id=batch_id
            )

            if alert is not None and llm_result.get("should_send", False):
                # Generate a combined chart
                image_bytes = await self.plot_ohlcv(symbol, candles, "GROUPED_ALERT")

                # Send the combined alert
                await self.send_enhanced_discord_alert(alert, llm_result, image_bytes)

                # Mark all individual alerts as processed
                self.db.session.rollback()  # Ensure clean state
                try:
                    for alert_info in triggered_alerts:
                        individual_alert = Alert(
                            symbol=symbol,
                            alert_type=alert_info["type"],
                            timestamp=datetime.now(),
                            last_alerted_at=datetime.now(),
                            data=alert_info["data"],
                            llm_processed=True,
                            llm_sent=False,  # Individual alerts aren't sent
                            batch_id=batch_id,  # Link to the batch
                        )
                        self.db.session.add(individual_alert)
                    self.db.session.commit()
                except Exception as e:
                    logger.error(f"Error saving individual alerts for batch {batch_id}: {e}")
                    self.db.session.rollback()

        except Exception as e:
            logger.error(f"Error processing grouped alerts for {symbol}: {e}")
            self.db.session.rollback()

    async def get_recent_large_orders(self, symbol, hours_lookback=4):
        """
        Fetch recent large orders for a symbol to provide context for alerts.

        Args:
            symbol: The trading pair to check (e.g., "BTC/USDT")
            hours_lookback: How many hours back to check for large orders

        Returns:
            Dictionary with large order statistics
        """
        try:
            # First, check if the symbol meets the minimum volume requirement
            candles = await self.fetch_candles(symbol)
            if not candles or len(candles) < 1:
                return {"count": 0, "orders": []}

            # Get trades using the exchange manager
            trades = await exchange_manager.execute("binance", "fetch_trades", symbol, limit=200)
            if not trades:
                return {"count": 0, "orders": []}

            # Define minimum order size
            large_order_threshold = ORDER_MIN_USD_VOLUME * 0.05  # 5% of min volume threshold

            # Calculate the cutoff time (hours ago)
            cutoff_time = (datetime.now() - timedelta(hours=hours_lookback)).timestamp() * 1000

            # Filter to only include recent large orders
            large_orders = [
                {
                    "amount": float(order["amount"]),
                    "price": float(order["price"]),
                    "side": order["side"],
                    "value_usd": float(order["amount"]) * float(order["price"]),
                    "timestamp": order.get("timestamp", 0),
                }
                for order in trades
                if (
                    order["amount"] * order["price"] > large_order_threshold
                    and order.get("timestamp", 0) > cutoff_time
                )
            ]

            # Skip if no large orders found
            if not large_orders:
                return {"count": 0, "orders": []}

            # Sort by value (largest first)
            large_orders.sort(key=lambda x: x["value_usd"], reverse=True)

            # Calculate statistics
            buy_orders = [order for order in large_orders if order["side"] == "buy"]
            sell_orders = [order for order in large_orders if order["side"] == "sell"]

            # Total value by side
            buy_value = sum(order["value_usd"] for order in buy_orders)
            sell_value = sum(order["value_usd"] for order in sell_orders)

            # Net value (positive = more buys, negative = more sells)
            net_value = buy_value - sell_value

            return {
                "count": len(large_orders),
                "buy_count": len(buy_orders),
                "sell_count": len(sell_orders),
                "total_value_usd": buy_value + sell_value,
                "net_value_usd": net_value,
                "largest_orders": large_orders[:3],  # Top 3 largest orders
            }

        except Exception as e:
            logger.error(f"Error fetching recent large orders for {symbol}: {e}")
            return {"count": 0, "orders": []}

    async def process_alert_with_llm(self, symbol, alert_type, alert_data, batch_id=None):
        """Process an alert through the LLM to determine if it should be sent."""
        alert = None
        try:
            logger.info(f"Processing {alert_type} alert for {symbol} with LLM")

            # Get alert context
            context = await self.context_gatherer.get_alert_context(symbol, alert_type)

            # Debug output for large orders
            if alert_type == "LARGE_ORDER":
                logger.debug(f"Large order data for {symbol}: {alert_data}")

            # Ensure clean session state
            self.db.session.rollback()

            # Create a database record for the alert
            alert = Alert(
                symbol=symbol,
                alert_type=alert_type,
                timestamp=datetime.now(),
                last_alerted_at=datetime.now(),
                data=alert_data,
                batch_id=batch_id,
            )
            self.db.session.add(alert)
            self.db.session.commit()

            # Format alert for LLM
            alert_for_llm = {
                "symbol": symbol,
                "alert_type": alert_type,
                "timestamp": datetime.now().isoformat(),
                "price": alert_data.get("price"),
                "volume": alert_data.get("volume"),
                "indicators": alert_data,
                "data": alert_data,  # Include all data
            }

            # Send to LLM for evaluation
            llm_result = await self.llm_service.evaluate_alert(
                alert_for_llm,
                context["historical_alerts"],
                context["market_context"],
                context["symbol_details"],
                context.get("large_order_context", {}),
            )

            # Update the alert with LLM results
            alert.llm_processed = True
            alert.llm_sent = llm_result.get("should_send", False)
            alert.llm_analysis = llm_result.get("analysis", "")
            alert.llm_reasoning = llm_result.get("reasoning", "")
            alert.llm_title = llm_result.get("title", "")
            self.db.session.commit()

            # If LLM approves the alert, send it to Discord
            if llm_result.get("should_send", False):
                logger.info(f"LLM approved alert for {symbol}, sending to Discord...")

                # Fetch candles first
                candles = await self.fetch_candles(symbol)
                if candles:
                    chart_file = await self.plot_ohlcv(symbol, candles, alert_type)
                else:
                    logger.warning(f"No candles available for {symbol}, skipping chart generation")
                    chart_file = None

                image_bytes = None

                if chart_file:
                    try:
                        with open(chart_file, "rb") as f:
                            image_bytes = BytesIO(f.read())
                            image_bytes.seek(0)

                        # Clean up the chart file
                        try:
                            os.remove(chart_file)
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary chart file: {e}")
                    except Exception as e:
                        logger.error(f"Error reading chart file: {e}")

                # Send the alert to Discord
                await self.send_enhanced_discord_alert(alert, llm_result, image_bytes)
            else:
                logger.info(f"LLM rejected alert for {symbol}, not sending to Discord")

            return alert, llm_result

        except Exception as e:
            logger.error(f"Error processing alert with LLM: {e}")
            if alert:
                alert.llm_processed = True
                alert.llm_sent = True  # Default to sending on error
                alert.llm_analysis = f"Error in LLM processing: {str(e)}"
                alert.llm_reasoning = (
                    "Error occurred during LLM processing, defaulting to sending alert"
                )
                self.db.session.commit()
            return alert, {
                "should_send": True,
                "title": f"Alert for {symbol}",
                "analysis": f"Error in LLM processing: {str(e)}",
                "key_levels": "No key levels identified",
            }

    async def send_enhanced_discord_alert(self, alert, llm_result, image_bytes=None):
        """Send an enhanced alert with LLM analysis to Discord."""
        try:
            # Check if alert is None
            if alert is None:
                logger.error("Cannot send alert: alert object is None")
                return

            logger.debug(f"Attempting to send alert for {alert.symbol} of type {alert.alert_type}")

            channel = await self.get_alert_channel(alert.alert_type)
            if not channel:
                logger.error(f"No channel found for alert type {alert.alert_type}")
                return

            logger.info(f"Found Discord channel for {alert.alert_type}: {channel.name}")

            # Get symbol name without the /USDT part
            symbol_base = alert.symbol.split("/")[0] if "/" in alert.symbol else alert.symbol

            # Extract price data from alert
            price = "Unknown"
            if alert.data and "price" in alert.data:
                price = format_currency(alert.data["price"])

            logger.debug(f"Creating embed for {symbol_base} at price {price}")

            # Create embedded message with rich formatting
            embed = disnake.Embed(
                title=f"💹 {llm_result.get('title', f'Alert: {symbol_base}')}",
                description=llm_result.get("analysis", "Alert analysis not available"),
                color=disnake.Color.gold(),
            )

            # Add market context field only if it's relevant/different
            market_context = llm_result.get("market_context", "No market context available")
            if market_context and not market_context.startswith("No market context"):
                embed.add_field(name="Market Context", value=market_context, inline=False)

            # Add key levels if available
            if "key_levels" in llm_result and llm_result["key_levels"]:
                embed.add_field(
                    name="Key Levels to Watch", value=llm_result["key_levels"], inline=False
                )

            # Add current price field
            embed.add_field(name="Current Price", value=price, inline=True)

            # Add volume field if available
            if alert.data and "volume" in alert.data:
                volume = alert.data["volume"]
                if "price" in alert.data:
                    formatted_volume = f"{format_number(volume)} (≈${format_currency(volume * alert.data['price'])})"
                else:
                    formatted_volume = format_number(volume)
                embed.add_field(name="Volume (24h)", value=formatted_volume, inline=True)

            logger.debug(f"Embed created, attempting to send to channel")

            # If there's an image, attach it to the message
            if image_bytes:
                file = disnake.File(fp=image_bytes, filename=f"{symbol_base}_chart.png")
                embed.set_image(url=f"attachment://{symbol_base}_chart.png")
                await channel.send(embed=embed, file=file)
                logger.info(f"Sent enhanced alert with chart for {symbol_base} to Discord")
            else:
                await channel.send(embed=embed)
                logger.info(f"Sent enhanced alert without chart for {symbol_base} to Discord")

        except Exception as e:
            logger.error(f"Error sending enhanced alert to Discord: {str(e)}")

    async def generate_chart(self, symbol, alert_type=None):
        """Generate a chart for the given symbol."""
        try:
            # Create charts directory if it doesn't exist
            os.makedirs("charts", exist_ok=True)

            # Fetch candles for the symbol
            candles = await self.fetch_candles(symbol)
            if not candles or len(candles) < 10:
                logger.warning(f"Not enough candle data to generate chart for {symbol}")
                return None

            # Use our custom plotting function
            chart_file = f"charts/{symbol.replace('/', '_')}_analysis.png"
            await self.plot_ohlcv(symbol, candles, alert_type)

            return chart_file
        except Exception as e:
            logger.error(f"Error generating chart for {symbol}: {e}")
            return None
