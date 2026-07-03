import logging

import symtorch


def test_enable_logging_attaches_stream_handler():
    logger = logging.getLogger("symtorch")
    symtorch.enable_logging(logging.DEBUG)
    assert logger.level == logging.DEBUG
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_enable_logging_idempotent():
    logger = logging.getLogger("symtorch")
    symtorch.enable_logging()
    n = len(logger.handlers)
    symtorch.enable_logging()
    assert len(logger.handlers) == n


def test_package_logger_has_null_handler_by_default():
    logger = logging.getLogger("symtorch")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
