"""Centralized logging configuration.

`configure_logging` sets up the `bictool` logger tree with a console
handler plus any extra handlers passed in (e.g. `QtLogHandler`, so the GUI's
live trace panel receives every log record from every layer). Every module
should log via `logging.getLogger(__name__)` - records propagate up to the
`bictool` logger automatically, so nothing else needs registering per module.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class QtLogHandler(QObject, logging.Handler):
    """Bridges the standard `logging` module to a Qt signal, so any log call
    anywhere in the app - including from a background `QThreadPool` worker
    thread (`Worker`, `DicomLoadWorker`) - shows up in the GUI. Qt queues
    cross-thread signal delivery to the receiving (GUI-thread) slot
    automatically, so this is safe to connect directly to a widget's update
    method without any extra locking on the caller's part.
    """

    message_logged = Signal(str)

    def __init__(self) -> None:
        QObject.__init__(self)
        logging.Handler.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self.message_logged.emit(message)


def configure_logging(level: int = logging.DEBUG, extra_handlers: list[logging.Handler] | None = None) -> None:
    """Configures the `bictool` logger tree: a console handler plus any
    `extra_handlers` (e.g. a `QtLogHandler` feeding the GUI's trace panel),
    all at `level`.

    Defaults to DEBUG rather than INFO: this app is under active
    development, and the entire point of this configuration existing right
    now is maximum visibility into what's happening step by step while
    diagnosing failures - not production-quiet defaults.

    Safe to call more than once (e.g. once per `MainWindowView()` in tests)
    - clears previously attached handlers first instead of accumulating
    duplicates.
    """
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    logger = logging.getLogger("bictool")
    logger.setLevel(level)
    logger.propagate = False  # keep output self-contained, no duplicate root-logger handlers
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    for handler in extra_handlers or []:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
