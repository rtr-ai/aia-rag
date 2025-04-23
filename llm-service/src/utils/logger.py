import logging
import os
from utils import path_utils
from logging.handlers import WatchedFileHandler


def get_logger(name: str, filename="log.txt"):
    """
    Creates a new logger.

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logs_directory = os.path.join(path_utils.get_project_root(), "logs")
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)
    file_path = os.path.join(logs_directory, filename)

    formatter = logging.Formatter(
        fmt="{asctime}  {levelname: <7} ---  {name: <25}   : {message}", style="{"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = WatchedFileHandler(file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
