import asyncio
import logging

import ccxt.async_support as ccxt
import colorlog
import requests

from config.settings import CMC_API_KEY

# Set up colorful logging
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(levelname)s - %(message)s")
)

logger = colorlog.getLogger("CryptoBettingBot")
logger.setLevel(logging.INFO)


async def get_exchange(exchange_id):
    """Get an exchange object."""
    try:
        if not isinstance(exchange_id, str):
            raise ValueError("Exchange ID must be a string")

        exchange_class = getattr(ccxt, exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Exchange '{exchange_id}' not found in ccxt")

        exchange = exchange_class()
        await exchange.load_markets()
        return exchange
    except Exception as e:
        logger.error(f"Error initializing exchange {exchange_id}: {e}")
        return None


async def get_markets(exchange):
    """Get a list of markets from an exchange object."""
    try:
        all_markets = await exchange.fetch_markets()
        # Filter out only markets with USDT as quote currency
        markets = [market for market in all_markets if market["quote"] == "USDT"]
        return markets
    except Exception as e:
        logger.error(f"Error fetching markets: {e}")
        return None


async def fetch_current_price(symbol):
    """Get current price from an exchange object."""
    try:
        exchange = await get_exchange("binance")
        if not exchange:
            raise Exception("Failed to initialize exchange")

        ticker = await exchange.fetch_ticker(symbol)
        return ticker["last"]
    except Exception as e:
        logger.error(f"Error fetching current price for {symbol}: {e}")
        return None
    finally:
        if exchange:
            await exchange.close()


async def validate_ticker(ticker):
    """
    Validate a ticker.
    - check if the ticker is found in the exchange's markets
    """
    try:
        exchange = await get_exchange("binance")
        if not exchange:
            raise Exception("Failed to initialize exchange")

        markets = await get_markets(exchange)
        if not markets:
            raise Exception("Failed to fetch markets")

        # Debug: Print out a few market symbols to check format
        logger.info(
            f"Sample market symbols: {[market['symbol'] for market in markets[:5]]}"
        )

        # Modify this line if market symbols are in a different format
        ticker_found = any(
            market["symbol"].replace("/", "") == ticker for market in markets
        )
        if not ticker_found:
            raise Exception(f"Ticker '{ticker}' not found in exchange markets")

        return True
    except Exception as e:
        logger.error(f"Error validating ticker {ticker}: {e}")
        return False
    finally:
        if exchange:
            await exchange.close()


async def fetch_prices(tickers):
    """Fetch the prices for multiple tickers using ccxt asyncio support."""
    exchange = None
    try:
        # Initialize the exchange
        exchange = await get_exchange("bybit")
        if not exchange:
            raise Exception("Failed to initialize exchange")

        # Fetch the prices for all tickers
        prices = {}
        for ticker in tickers:
            ticker = ticker.upper()
            ticker_data = await exchange.fetch_ticker(ticker)
            prices[ticker] = ticker_data["last"]

        return prices
    except Exception as e:
        logger.error(f"Error fetching prices for {tickers}: {e}")
    finally:
        # Close the exchange connection properly
        if exchange:
            await exchange.close()


async def fetch_price(symbol):
    """Fetch the price for a single ticker using ccxt asyncio support."""
    exchange = None
    try:
        # Initialize the exchange
        exchange = await get_exchange("bybit")
        if not exchange:
            raise Exception("Failed to initialize exchange")

        # Fetch the prices for all tickers
        ticker = symbol.upper()
        ticker_data = await exchange.fetch_ticker(ticker)
        return ticker_data["last"]
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
    finally:
        # Close the exchange connection properly
        if exchange:
            await exchange.close()


async def fetch_ohlcv(symbol, timeframe, since=None, limit=None):
    """Fetch OHLCV data for a symbol using ccxt asyncio support."""
    exchange = None
    try:
        # Initialize the exchange
        exchange = await get_exchange("binance")
        if not exchange:
            raise Exception("Failed to initialize exchange")

        # Fetch the OHLCV data
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
    finally:
        # Close the exchange connection properly
        if exchange:
            await exchange.close()
