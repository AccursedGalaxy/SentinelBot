import asyncio
import json
import os
import time

import aiohttp
import ccxt.async_support as ccxt
import requests

from config.settings import CG_API_KEY, CMC_API_KEY
from logger_config import setup_logging

logger = setup_logging()
request_semaphore = asyncio.Semaphore(5)
COIN_LIST_CACHE_FILE = "coin_list_cache.json"


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


async def fetch_coin_info(symbol, api_key=CG_API_KEY):
    # Check if the cache file exists and load it
    if os.path.exists(COIN_LIST_CACHE_FILE):
        with open(COIN_LIST_CACHE_FILE, "r") as file:
            coins_list = json.load(file)
    else:
        # If the cache file doesn't exist, fetch the data and create the cache
        coins_url = "https://api.coingecko.com/api/v3/coins/list"
        async with aiohttp.ClientSession() as session:
            async with session.get(coins_url) as response:
                if response.status != 200:
                    return {"error": f"Failed to fetch coins list, status code: {response.status}"}
                coins_list = await response.json()
                with open(COIN_LIST_CACHE_FILE, "w") as file:
                    json.dump(coins_list, file)

    symbol_to_id = {coin["symbol"].upper(): coin["id"] for coin in coins_list}

    coin_id = symbol_to_id.get(symbol.upper())
    if not coin_id:
        return {"error": f"CoinGecko does not support the symbol: {symbol}"}

    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "true",
    }
    headers = {"x-cg-pro-api-key": api_key}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status != 200:
                return {
                    "error": f"Failed to fetch data for {coin_id}, status code: {response.status}"
                }

            return await response.json()


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
        logger.info(f"Sample market symbols: {[market['symbol'] for market in markets[:5]]}")

        # Modify this line if market symbols are in a different format
        ticker_found = any(market["symbol"].replace("/", "") == ticker for market in markets)
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


async def fetch_top_gainers_losers(api_key, category="gainers", time_period="24h", top_coins=300):
    url = "https://pro-api.coingecko.com/api/v3/coins/top_gainers_losers"
    headers = {
        "x-cg-pro-api-key": api_key,
    }
    params = {
        "vs_currency": "usd",
        "duration": time_period,
        "top_coins": top_coins,  # Include the top_coins parameter
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Top {category} fetched successfully")
                if category == "gainers":
                    return data.get("top_gainers", [])[:5]  # Get top 5 gainers
                elif category == "losers":
                    return data.get("top_losers", [])[:5]  # Get top 5 losers
                else:
                    logger.error(f"Invalid category: {category}")
                    return []
            else:
                error_message = await response.text()
                logger.error(
                    f"Error fetching data from CoinGecko API: {response.status} - {error_message}"
                )
                return []


historical_data_cache = {}


async def fetch_historical_data(coin_id, days=1):
    cache_key = f"{coin_id}_{days}"
    if cache_key in historical_data_cache:
        # Return cached data if available
        return historical_data_cache[cache_key]

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    if not 2 <= days <= 90:
        params["interval"] = "daily"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                # Cache the data
                historical_data_cache[cache_key] = data["prices"]
                return data["prices"]
            elif response.status == 429:
                logger.error(
                    f"Rate limit exceeded. Retrying after 60 seconds. Response: {await response.text()}"
                )
                time.sleep(60)  # Wait for 60 seconds before retrying
                return await fetch_historical_data(coin_id, days)
            else:
                logger.error(
                    f"Failed to fetch historical data for {coin_id}. Status: {response.status}. Response: {await response.text()}"
                )
                return None


async def fetch_new_coins(api_key=CG_API_KEY):
    url = "https://pro-api.coingecko.com/api/v3/coins/list/new"
    headers = {
        "x-cg-pro-api-key": api_key,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                new_coins = await response.json()
                # Convert activation time to a readable format
                for coin in new_coins:
                    coin["activated_at"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(coin["activated_at"])
                    )
                return new_coins
            else:
                logger.error(
                    f"Failed to fetch new coins. Status: {response.status}. Response: {await response.text()}"
                )
                return None


async def fetch_coin_data(coin_id, api_key=CG_API_KEY):
    """
    Fetches important and valuable information for a specified coin asynchronously.

    Args:
    coin_id (str): The ID of the coin to fetch data for.

    Returns:
    dict: A dictionary containing the fetched coin data.
    """
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=true&market_data=false&community_data=true&developer_data=true&sparkline=true"
    headers = {
        "x-cg-pro-api-key": api_key,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                important_data = {
                    "id": data["id"],
                    "symbol": data["symbol"],
                    "name": data["name"],
                    "hashing_algorithm": data["hashing_algorithm"],
                    "description": data["description"]["en"],
                    "genesis_date": data["genesis_date"],
                    "sentiment_votes_up_percentage": data["sentiment_votes_up_percentage"],
                    "sentiment_votes_down_percentage": data["sentiment_votes_down_percentage"],
                    "market_cap_rank": data.get("market_cap_rank"),  # Use get to avoid KeyError
                    "community_data": data["community_data"],
                    "developer_data": data["developer_data"],
                    "links": data["links"],
                    "image": data["image"],
                    "last_updated": data["last_updated"],
                }
                return important_data
            else:
                return {"error": "Failed to fetch data"}


async def fetch_trending_coins(api_key=CG_API_KEY):
    url = "https://pro-api.coingecko.com/api/v3/search/trending"
    headers = {
        "x-cg-pro-api-key": api_key,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                # Ensure we return the list of coins correctly
                return data.get("coins", [])
            else:
                logger.error(
                    f"Failed to fetch trending coins. Status: {response.status}. Response: {await response.text()}"
                )
                return []


async def fetch_coins_by_category(category, api_key=CG_API_KEY):
    """
    Fetches coins data by category asynchronously.

    Args:
    category (str): The category of the coins to fetch data for.

    Returns:
    list: A list of dictionaries containing the fetched coins data.
    """
    url = f"https://pro-api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "category": category,
        "per_page": 25,
        "page": 1,
        "sparkline": "true",  # Convert boolean to string
        "price_change_percentage": "1h",
    }
    headers = {
        "x-cg-pro-api-key": api_key,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                coins_data = await response.json()
                return coins_data
            else:
                logger.error(
                    f"Failed to fetch coins by category '{category}'. Status: {response.status}. Response: {await response.text()}"
                )
                return []


async def fetch_category_info(api_key=CG_API_KEY):
    url = "https://pro-api.coingecko.com/api/v3/coins/categories"
    headers = {"x-cg-pro-api-key": "CG-6sKW9pMTnWQ7oWN5bME8edSX"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                logger.error(
                    f"Failed to fetch category info. Status: {response.status}. Response: {await response.text()}"
                )
                return None
