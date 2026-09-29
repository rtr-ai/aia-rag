import contextlib
from typing import Iterator

_in_flight = 0


def in_flight() -> int:
    return _in_flight


@contextlib.contextmanager
def track() -> Iterator[None]:
    global _in_flight
    _in_flight += 1
    try:
        yield
    finally:
        _in_flight -= 1
