import logging
import os

import colorlog


def setup_logging():
    logger = colorlog.getLogger("Sentinel")
    if not logger.handlers:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
        )
        logger.addHandler(handler)

        # Set log level based on environment variable, default to INFO
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level))

    return logger
