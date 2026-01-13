import logging
from rich.logging import RichHandler

def get_logger(name: str) -> logging.Logger:
    # Safe to call multiple times; RichHandler remains the primary handler.
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)
