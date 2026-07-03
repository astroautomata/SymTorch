"""Package-level logging setup for symtorch (library-friendly: silent by default)."""

import logging

_package_logger = logging.getLogger("symtorch")
_package_logger.addHandler(logging.NullHandler())


def enable_logging(level: int = logging.INFO) -> None:
    """Show symtorch progress messages (distill progress, cache hits, etc.).

    Libraries stay silent by default; call this once in notebooks/scripts to
    restore the chatty behavior.
    """
    _package_logger.setLevel(level)
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in _package_logger.handlers
    )
    if not has_stream:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        _package_logger.addHandler(handler)
