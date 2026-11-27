"""Local run directories and console logs for the single-Episode pilot."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import sys


def create_run_directory(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for name in ("config", "logs", "metrics", "plots", "cache", "checkpoints", "diagnostics"):
        (root / name).mkdir()


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


@contextmanager
def capture_console(path):
    """Keep stdout and stderr beside the run without an external tee file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            yield
