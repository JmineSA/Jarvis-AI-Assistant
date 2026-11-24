"""Console + rotating-file logging."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_HANDLER_MARK = "_jarvis_handler"


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    max_bytes: int = 1_048_576,
    backup_count: int = 3,
    console_level: str | None = None,
) -> None:
    """Configure the root logger. Safe to call more than once.

    `console_level` quiets the terminal (for interactive modes) while the log
    file still records everything at `level`.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove handlers added by a previous call so logs are not duplicated.
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    if console_level is not None:
        console_numeric = logging.getLevelName(console_level.upper())
        if isinstance(console_numeric, int):
            console.setLevel(console_numeric)
    handlers: list[logging.Handler] = [console]

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(
                logging.handlers.RotatingFileHandler(
                    log_dir / "jarvis.log",
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
            )
        except OSError as exc:
            logging.getLogger(__name__).warning("File logging disabled: %s", exc)

    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARK, True)
        root.addHandler(handler)
