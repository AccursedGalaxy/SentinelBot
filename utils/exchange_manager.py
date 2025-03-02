import asyncio
import time

import ccxt.async_support as ccxt

from logger_config import setup_logging

logger = setup_logging("Exchange Manager", "yellow")

# Store last request times for each endpoint to respect rate limits
last_request_times = {}
# Default rate limits (requests per second)
DEFAULT_RATE_LIMIT = 5  # 5 requests per second
RATE_LIMITS = {
    "exchangeInfo": 0.2,  # Allow 1 request per 5 seconds
    "fetch_trades": 1,  # Allow 1 request per second
    "fetch_ohlcv": 2,  # Allow 2 requests per second
}

# Global semaphore to limit concurrent requests
request_semaphore = asyncio.Semaphore(10)


async def rate_limited(endpoint_name):
    """Decorator to apply rate limiting to exchange API calls"""
    rate_limit = RATE_LIMITS.get(endpoint_name, DEFAULT_RATE_LIMIT)

    # Calculate delay needed based on last request time
    last_time = last_request_times.get(endpoint_name, 0)
    current_time = time.time()
    delay = max(0, (1.0 / rate_limit) - (current_time - last_time))

    # Sleep if we need to respect rate limits
    if delay > 0:
        logger.debug(f"Rate limiting {endpoint_name}: sleeping for {delay:.2f}s")
        await asyncio.sleep(delay)

    # Update the last request time
    last_request_times[endpoint_name] = time.time()


class ExchangeManager:
    """Manager for exchange connections with proper rate limiting and error handling"""

    def __init__(self):
        self._exchanges = {}
        self._init_locks = {}

    async def get_exchange(self, exchange_id="binance"):
        """Get an exchange instance with automatic connection management and rate limiting"""
        if exchange_id not in self._init_locks:
            self._init_locks[exchange_id] = asyncio.Lock()

        # Use a lock to prevent multiple initialization of the same exchange
        async with self._init_locks[exchange_id]:
            # If we already have an initialized exchange, return it
            if exchange_id in self._exchanges and self._exchanges[exchange_id]["initialized"]:
                return self._exchanges[exchange_id]["instance"]

            # Initialize a new exchange
            try:
                # Apply rate limiting to exchangeInfo endpoint
                await rate_limited("exchangeInfo")

                # Create the exchange instance
                exchange = getattr(ccxt, exchange_id)()

                # Configure the exchange
                exchange.enableRateLimit = True
                exchange.options["warnOnFetchOpenOrdersWithoutSymbol"] = False

                # Load markets with a retry mechanism
                retry_count = 3
                for attempt in range(retry_count):
                    try:
                        await exchange.load_markets()
                        break
                    except Exception as e:
                        if attempt < retry_count - 1:
                            wait_time = 2**attempt  # Exponential backoff
                            logger.warning(
                                f"Retrying load_markets() for {exchange_id} after {wait_time}s: {e}"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            # Last attempt failed
                            await exchange.close()
                            raise

                # Store the exchange instance
                self._exchanges[exchange_id] = {
                    "instance": exchange,
                    "initialized": True,
                    "last_used": time.time(),
                }

                logger.info(f"Successfully initialized {exchange_id} exchange")
                return exchange

            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id} exchange: {e}")
                self._exchanges[exchange_id] = {
                    "instance": None,
                    "initialized": False,
                    "last_used": time.time(),
                }
                return None

    async def execute(self, exchange_id, method_name, *args, **kwargs):
        """Execute an exchange method with rate limiting and error handling"""
        exchange = await self.get_exchange(exchange_id)
        if not exchange:
            raise Exception(f"Could not initialize {exchange_id} exchange")

        # Apply rate limiting
        await rate_limited(method_name)

        # Use a semaphore to limit concurrent requests
        async with request_semaphore:
            try:
                # Update last used time
                self._exchanges[exchange_id]["last_used"] = time.time()

                # Call the method
                method = getattr(exchange, method_name)
                return await method(*args, **kwargs)

            except Exception as e:
                logger.error(f"Error in {exchange_id}.{method_name}(): {e}")
                raise

    async def close_all(self):
        """Close all exchange connections"""
        for exchange_id, exchange_data in self._exchanges.items():
            if exchange_data["initialized"] and exchange_data["instance"]:
                try:
                    await exchange_data["instance"].close()
                    logger.debug(f"Closed {exchange_id} exchange connection")
                except Exception as e:
                    logger.error(f"Error closing {exchange_id} exchange: {e}")

        # Clear the exchanges dictionary
        self._exchanges = {}

    async def close_idle(self, idle_threshold=300):  # 5 minutes
        """Close exchange connections that haven't been used for a while"""
        current_time = time.time()
        for exchange_id, exchange_data in list(self._exchanges.items()):
            if (
                exchange_data["initialized"]
                and exchange_data["instance"]
                and current_time - exchange_data["last_used"] > idle_threshold
            ):
                try:
                    await exchange_data["instance"].close()
                    del self._exchanges[exchange_id]
                    logger.debug(f"Closed idle {exchange_id} exchange connection")
                except Exception as e:
                    logger.error(f"Error closing idle {exchange_id} exchange: {e}")


# Singleton instance
exchange_manager = ExchangeManager()
