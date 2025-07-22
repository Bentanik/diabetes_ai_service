import logging
import sys
from pathlib import Path


def setup_logging(debug: bool = False):
    Path("logs").mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]
    if not debug:
        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        handlers=handlers,
        force=True,
    )

    silence_loggers = [
        "httpx",
        "httpcore",
        "openai",
        "urllib3",
        "asyncio",
        "multipart",
        "uvicorn.access",
        "pymongo",
        "pdfminer",
    ]

    for logger_name in silence_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


setup_logging(debug=True)
