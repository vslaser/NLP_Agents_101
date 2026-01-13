from __future__ import annotations
import time
from functools import wraps
from utils.logger import get_logger

log = get_logger(__name__)

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            ms = (time.perf_counter() - start) * 1000
            log.info(f"{fn.__name__} took {ms:.1f}ms")
    return wrapper
