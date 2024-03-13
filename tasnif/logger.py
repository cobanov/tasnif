import logging

from rich.logging import RichHandler

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level="INFO",
    format=log_format,
    datefmt="[%X]",
    handlers=[RichHandler(show_time=False, show_level=False)],
)


def info(msg):
    logging.info(msg)


def error(msg):
    logging.error(msg)


if __name__ == "__main__":
    info("info message")
    error("error message")
