import logging
import sys
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from utils import logger as logger_module


class LoggerTest(unittest.TestCase):
    def setUp(self):
        self.logger_name = f"test.logger.{uuid.uuid4()}"
        self.logger = logging.getLogger(self.logger_name)
        self.logger.handlers.clear()

    def close_handlers(self):
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    def tearDown(self):
        self.close_handlers()
        logging.Logger.manager.loggerDict.pop(self.logger_name, None)

    def test_repeated_calls_reuse_handlers(self):
        with tempfile.TemporaryDirectory() as project_root, patch.object(
            logger_module.path_utils, "get_project_root", return_value=project_root
        ):
            try:
                first = logger_module.get_logger(self.logger_name)
                second = logger_module.get_logger(self.logger_name)

                self.assertIs(first, second)
                self.assertEqual(len(first.handlers), 2)

                logger_module.get_logger(self.logger_name, "chat.log")
                self.assertEqual(len(first.handlers), 3)
            finally:
                self.close_handlers()

    def test_concurrent_calls_create_one_handler_pair(self):
        with tempfile.TemporaryDirectory() as project_root, patch.object(
            logger_module.path_utils, "get_project_root", return_value=project_root
        ):
            try:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(
                        executor.map(
                            lambda _: logger_module.get_logger(self.logger_name),
                            range(50),
                        )
                    )

                self.assertEqual(len(self.logger.handlers), 2)
            finally:
                self.close_handlers()


if __name__ == "__main__":
    unittest.main()
