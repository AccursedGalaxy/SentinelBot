import logging
import os

import colorlog


def setup_logging(name="Sentinel", default_color="green"):
    logger = colorlog.getLogger(name)
    if not logger.handlers:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)s - %(name)s - %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": default_color,
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
        logger.addHandler(handler)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level))
    return logger
