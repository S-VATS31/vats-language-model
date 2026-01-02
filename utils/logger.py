import logging
import os
import sys

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up logging module.
    
    Args:
        name (str): Name of logger.
        log_file (str): File to be logged to.
        level (int): Level to be logged to. Defaults to `logging.INFO`.
    
    Returns:
        logging.Logger: Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
