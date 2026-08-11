import logging
import os
import threading
from utils import path_utils
from logging.handlers import WatchedFileHandler


_HANDLER_LOCK = threading.RLock()
_HANDLER_KEY = "_aia_rag_handler_key"


def get_logger(name: str, filename="log.txt"):
    """
    Return a configured logger without adding duplicate handlers.

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logs_directory = os.path.join(path_utils.get_project_root(), "logs")
    os.makedirs(logs_directory, exist_ok=True)
    file_path = os.path.abspath(os.path.join(logs_directory, filename))

    formatter = logging.Formatter(
        fmt="{asctime}  {levelname: <7} ---  {name: <25}   : {message}", style="{"
    )

    with _HANDLER_LOCK:
        handler_keys = {
            getattr(handler, _HANDLER_KEY, None) for handler in logger.handlers
        }

        console_key = ("console", None)
        if console_key not in handler_keys:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            setattr(console_handler, _HANDLER_KEY, console_key)
            logger.addHandler(console_handler)

        file_key = ("file", file_path)
        if file_key not in handler_keys:
            file_handler = WatchedFileHandler(
                file_path, mode="a", encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            setattr(file_handler, _HANDLER_KEY, file_key)
            logger.addHandler(file_handler)

    return logger
