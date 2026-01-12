import logging


def get_logger() -> logging.Logger:
    """Initialize the logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
