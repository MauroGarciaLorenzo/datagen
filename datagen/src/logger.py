import os
from datetime import datetime

class Logger:
    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    def __init__(self, level="INFO", dst_dir=None):
        self.level = self.LEVELS.get(level.upper(), 20)
        self.dst_dir = dst_dir
        self.log_file = None

        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
            self.log_file = open(os.path.join(dst_dir, f"log.txt"), "a")

    def _log(self, level_name, message):
        if self.LEVELS[level_name] >= self.level:
            msg = f"[{datetime.now().isoformat()}] [{level_name}] {message}"
            print(msg, flush=True)
            if self.log_file:
                print(msg, file=self.log_file, flush=True)

    def debug(self, msg):    self._log("DEBUG", msg)
    def info(self, msg):     self._log("INFO", msg)
    def warning(self, msg):  self._log("WARNING", msg)
    def error(self, msg):    self._log("ERROR", msg)
    def critical(self, msg): self._log("CRITICAL", msg)

    def get_logging_level(self):
        for name, value in self.LEVELS.items():
            if value == self.level:
                return name
        return "UNKNOWN"

def setup_logger(logging_level, dst_dir):
    global logger
    os.makedirs(dst_dir, exist_ok=True)

    logger = Logger(level=logging_level, dst_dir=dst_dir)

logger = Logger()
