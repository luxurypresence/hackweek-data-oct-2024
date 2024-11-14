"""A custom log configutation"""
import logging
import os
import time

def config(level) -> logging.Logger:
    """Config custom logging

    Args:
    - level (int): Logging level. Can take one of the following values:
        * logging.DEBUG
        * logging.INFO
        * logging.WARNING
        * logging.ERROR
        * logging.CRITICAL

    Returns:
    - logger (Logger object): Custom logger object with specified logging level and settings.
    """
    formatter = logging.Formatter(
        "[%(levelname)-8s][%(asctime)s][%(filename)-15s][%(lineno)4d][%(threadName)10s] - %(message)s"
    )
    formatter.converter = time.gmtime

    channel = logging.StreamHandler()
    channel.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.handlers = []
    logger.addHandler(channel)

    logger.propagate = False

    return logger


# Let's only log INFO and above in production by default, and DEBUG and above in any other ones, such as dev or stage
LEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()

# Object to be used by other modules
logger = config(LEVEL)
