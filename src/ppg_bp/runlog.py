"""Progress messages printed to the console and appended to a log file."""

import threading
import time
from pathlib import Path


class RunLog:
    """Callable logger: log("message") prints "HH:MM:SS message" and appends it to the file.

    A lock keeps lines from different threads whole.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        with self._lock:
            print(line, flush=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
