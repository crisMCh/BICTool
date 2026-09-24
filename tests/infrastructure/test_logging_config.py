import logging

from bictool.infrastructure.logging_config import QtLogHandler, configure_logging


def test_qt_log_handler_emits_formatted_message():
    handler = QtLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    received = []
    handler.message_logged.connect(received.append)

    record = logging.LogRecord("test.logger", logging.INFO, __file__, 1, "hello", None, None)
    handler.emit(record)

    assert received == ["INFO:hello"]


def test_configure_logging_routes_records_to_extra_handlers():
    handler = QtLogHandler()
    received = []
    handler.message_logged.connect(received.append)

    configure_logging(level=logging.DEBUG, extra_handlers=[handler])
    logger = logging.getLogger("bictool.some.module")
    logger.debug("a debug message")
    logger.error("an error message")

    assert any("a debug message" in msg for msg in received)
    assert any("an error message" in msg for msg in received)


def test_configure_logging_is_idempotent_and_does_not_duplicate_handlers():
    handler_a = QtLogHandler()
    received_a = []
    handler_a.message_logged.connect(received_a.append)
    configure_logging(extra_handlers=[handler_a])

    handler_b = QtLogHandler()
    received_b = []
    handler_b.message_logged.connect(received_b.append)
    configure_logging(extra_handlers=[handler_b])

    logging.getLogger("bictool.idempotency_check").info("one message")

    # The first configure_logging call's handlers should have been cleared,
    # so only handler_b (from the second call) should receive this.
    assert received_a == []
    assert len(received_b) == 1


def test_configure_logging_does_not_propagate_to_root_logger():
    handler = QtLogHandler()
    configure_logging(extra_handlers=[handler])

    bictool_logger = logging.getLogger("bictool")
    assert bictool_logger.propagate is False
