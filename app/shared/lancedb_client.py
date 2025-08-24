"""
Singleton for managing the LanceDB connection with diagnostics.

Adds logging, ensures the database directory exists, and wraps the
connection in a short timeout to surface potential hangs quickly.
"""
import os
import logging
import threading
import time

import lancedb
from ..shared.config import config

_db_connection = None
_logger = logging.getLogger(__name__)


def _connect_lancedb(db_path: str, result_holder, error_holder) -> None:
    """Worker function to connect to LanceDB and capture result or error."""
    try:
        result_holder.append(lancedb.connect(db_path))
    except Exception as exc:  # pragma: no cover - diagnostic path
        error_holder.append(exc)


def get_db_connection():
    """
    Returns a singleton instance of the LanceDB connection.

    If connection hangs beyond LANCEDB_CONNECT_TIMEOUT_SECONDS (default 15s),
    raises a TimeoutError so callers can fail fast instead of hitting Celery
    soft time limits.
    """
    global _db_connection
    if _db_connection is None:
        db_path = os.path.abspath(config.LANCEDB_PATH)
        # Ensure directory exists (LanceDB uses a directory path)
        os.makedirs(db_path, exist_ok=True)

        timeout_seconds = int(os.getenv("LANCEDB_CONNECT_TIMEOUT_SECONDS", "15"))
        _logger.info(
            f"Connecting to LanceDB at '{db_path}' with timeout {timeout_seconds}s..."
        )

        result_holder = []  # type: ignore[var-annotated]
        error_holder = []  # type: ignore[var-annotated]
        start = time.time()
        thread = threading.Thread(
            target=_connect_lancedb, args=(db_path, result_holder, error_holder), daemon=True
        )
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            _logger.error(
                "LanceDB connect did not complete within %ss (path=%s). "
                "Consider checking file locks, filesystem latency, or try a path without spaces.",
                timeout_seconds,
                db_path,
            )
            raise TimeoutError(
                f"LanceDB connect timeout after {timeout_seconds}s at path: {db_path}"
            )

        if error_holder:
            _logger.error("LanceDB connect failed: %s", error_holder[0])
            raise error_holder[0]

        _db_connection = result_holder[0]
        _logger.info("LanceDB connection established in %.3fs", time.time() - start)

    return _db_connection
