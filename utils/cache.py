import functools
import time


def cache_response(timeout):
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            result, expiry = cache.get(key, (None, 0))

            if time.time() < expiry:
                return result

            result = await func(*args, **kwargs)
            cache[key] = (result, time.time() + timeout)
            return result

        return wrapper

    return decorator
