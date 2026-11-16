"""Logging setup for partC_rev1 commands."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .partC_rev1_paths import partC_rev1_ensure_directory, partC_rev1_resolve_path


def partC_rev1_setup_logging(output_dir: str | Path = "partC_rev1_outputs", *, level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging for partC_rev1."""

    directory = partC_rev1_ensure_directory(partC_rev1_resolve_path(output_dir))
    log_path = directory / "partC_rev1_run.log"

    logger = logging.getLogger("partC_rev1")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("partC_rev1 logging initialized at %s", log_path)
    return logger
