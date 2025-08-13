"""
Singleton for managing the LanceDB connection.
"""
import lancedb
from ..shared.config import config

_db_connection = None

def get_db_connection():
    """
    Returns a singleton instance of the LanceDB connection.
    """
    global _db_connection
    if _db_connection is None:
        _db_connection = lancedb.connect(config.LANCEDB_PATH)
    return _db_connection
