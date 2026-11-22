import os

import psycopg
from pgvector.psycopg import register_vector


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "supply_chain_ai"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "connect_timeout": 5
}


def get_db_connection():
    """
    Create and return a PostgreSQL connection.

    Registers pgvector so Python can work with vector columns.
    """

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "DB_PASSWORD environment variable is not configured."
        )

    connection = psycopg.connect(**DB_CONFIG)

    try:
        register_vector(connection)
        return connection
    except Exception:
        connection.close()
        raise