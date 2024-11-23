import asyncio
import os
from datetime import datetime, timedelta

import ccxt.async_support as ccxt
import colorlog
import pandas as pd
import plotly.graph_objects as go

from logger_config import setup_logging

logger = setup_logging()


class PlotChart:
    @staticmethod
    async def fetch_ohlcv(exchange, symbol, timeframe):
        try:
            ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe)
            return ohlcv
        except Exception as e:
            logger.info(f"Failed to fetch data from {exchange.id}: {e}")
            return None
        finally:
            await exchange.close()

    @staticmethod
    async def get_ohlcv_data(symbol, timeframe):
        # Define the list of exchanges to query
        exchanges = [
            ccxt.binance(),
            ccxt.kucoin(),
            ccxt.bybit(),
        ]

        tasks = [PlotChart.fetch_ohlcv(exchange, symbol, timeframe) for exchange in exchanges]
        results = await asyncio.gather(*tasks)

        # Return the first successful result
        for result in results:
            if result:
                return result

        return None

    @staticmethod
    async def plot_ohlcv_chart(symbol, time_frame):
        # Create charts folder if it doesn't exist
        if not os.path.exists("charts"):
            os.makedirs("charts")

        # Fetch OHLCV data from exchanges
        ohlcv = await PlotChart.get_ohlcv_data(symbol, time_frame)

        if ohlcv is None:
            logger.error(f"No exchange supports the market for {symbol} or data fetch failed.")
            return None

        # Define the time horizon for each time frame
        time_horizon = {
            "1m": timedelta(hours=12),
            "5m": timedelta(days=1),
            "15m": timedelta(days=3),
            "1h": timedelta(days=7),
            "4h": timedelta(weeks=2),
            "1d": timedelta(weeks=12),
            "1w": timedelta(weeks=80),
            "1M": timedelta(weeks=324),
        }

        # Filter data based on the selected time frame
        start_time = datetime.now() - time_horizon.get(time_frame, timedelta(weeks=4))
        ohlcv = [entry for entry in ohlcv if datetime.fromtimestamp(entry[0] // 1000) >= start_time]

        if not ohlcv:
            logger.error(f"No data available for {symbol} in the specified time frame.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df.set_index("Date", inplace=True)

        # Calculate the moving averages
        df["200ema"] = df["Close"].rolling(window=200).mean()
        df["50ema"] = df["Close"].rolling(window=50).mean()
        df["20ema"] = df["Close"].rolling(window=20).mean()

        # Create a Plotly figure
        fig = go.Figure()

        # Add OHLCV data
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
        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["200ema"],
                name="200 EMA",
                line=dict(color="orange", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["50ema"],
                name="50 EMA",
                line=dict(color="yellow", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["20ema"],
                name="20 EMA",
                line=dict(color="green", width=1),
            )
        )

        # Customize the layout
        fig.update_layout(
            title=f"{symbol} OHLCV Chart ({time_frame})",
            xaxis=dict(
                type="date",
                tickformat="%H:%M %b-%d",
                tickmode="auto",
                nticks=10,
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(title="Price (USDT)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            margin=dict(b=40, t=40, r=40, l=40),
        )

        # Save the chart as a PNG image
        chart_file = f"charts/{symbol}_chart_{time_frame}.png"
        try:
            fig.write_image(chart_file, scale=1.5, width=1000, height=600)
            # logger.info(f"Chart saved successfully: {chart_file}")
            return chart_file
        except Exception as e:
            logger.error(f"Failed to save chart for {symbol}: {e}")
            return None


class PlotAllTimeChart:
    @staticmethod
    async def fetch_ohlcv(exchange, symbol, timeframe):
        try:
            ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe)
            return ohlcv
        except Exception as e:
            logger.info(f"Failed to fetch data from {exchange.id}: {e}")
            return None
        finally:
            await exchange.close()

    @staticmethod
    async def get_ohlcv_data(symbol, timeframe):
        # Define the list of exchanges to query
        exchanges = [
            ccxt.binance(),
            ccxt.kucoin(),
            ccxt.bybit(),
        ]

        tasks = [PlotChart.fetch_ohlcv(exchange, symbol, timeframe) for exchange in exchanges]
        results = await asyncio.gather(*tasks)

        # Return the first successful result
        for result in results:
            if result:
                return result

        return None

    @staticmethod
    async def plot_ohlcv_chart(symbol):
        # Create charts folder if it doesn't exist
        time_frame = "1w"
        if not os.path.exists("charts"):
            os.makedirs("charts")

        # Fetch OHLCV data from exchanges
        ohlcv = await PlotChart.get_ohlcv_data(symbol, time_frame)

        if ohlcv is None:
            logger.error(f"No exchange supports the market for {symbol} or data fetch failed.")
            return None

        # Define the time horizon for each time frame
        time_horizon = {
            "1w": timedelta(weeks=150),
        }

        # Filter data based on the selected time frame
        start_time = datetime.now() - time_horizon.get(time_frame, timedelta(weeks=4))
        ohlcv = [entry for entry in ohlcv if datetime.fromtimestamp(entry[0] // 1000) >= start_time]

        if not ohlcv:
            logger.error(f"No data available for {symbol} in the specified time frame.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df.set_index("Date", inplace=True)

        # Calculate the moving averages
        df["200ema"] = df["Close"].rolling(window=200).mean()
        df["50ema"] = df["Close"].rolling(window=50).mean()
        df["20ema"] = df["Close"].rolling(window=20).mean()

        # Create a Plotly figure
        fig = go.Figure()

        # Add OHLCV data
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
        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["200ema"],
                name="200 EMA",
                line=dict(color="orange", width=1),
            )
        )

        # Customize the layout
        fig.update_layout(
            title=f"{symbol} OHLCV Chart ({time_frame})",
            xaxis=dict(
                type="date",
                tickformat="%H:%M %b-%d",
                tickmode="auto",
                nticks=10,
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(title="Price (USDT)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            margin=dict(b=40, t=40, r=40, l=40),
        )

        # Save the chart as a PNG image
        chart_file = f"charts/{symbol}_chart_{time_frame}.png"

        try:
            fig.write_image(chart_file, scale=1.5, width=1000, height=600)
            # logger.info(f"Chart saved successfully: {chart_file}")
            return chart_file
        except Exception as e:
            logger.error(f"Failed to save chart for {symbol}: {e}")
            return None
