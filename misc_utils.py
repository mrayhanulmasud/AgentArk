"""Small utilities the training and eval scripts expect to import.

This module was referenced in the original research code but not shipped.
It provides a minimal ``print_colored`` helper so imports succeed on any
platform, including macOS.
"""
from __future__ import annotations

import sys


_ANSI = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "reset": "\033[0m",
}


def print_colored(message: str, color: str = "white", *, file=None) -> None:
    stream = file if file is not None else sys.stdout
    code = _ANSI.get(color.lower(), "")
    reset = _ANSI["reset"] if code else ""
    stream.write(f"{code}{message}{reset}\n")
    stream.flush()
