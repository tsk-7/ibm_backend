import os
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse

import mysql.connector


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def database_path() -> Path:
    configured = os.getenv("DATAINSIGHT_DB_PATH")
    path = Path(configured) if configured else PROJECT_ROOT / "database" / "datainsight.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class MySQLConnection:
    dialect = "mysql"

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query: str, parameters: tuple = ()):
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query.replace("?", "%s"), parameters)
        return cursor

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()


def _mysql_config() -> dict | None:
    if os.getenv("DATAINSIGHT_DB_PATH"):
        return None
    database_url = os.getenv("DATAINSIGHT_DATABASE_URL")
    if not database_url:
        return None
    parsed = urlparse(database_url)
    if parsed.scheme not in {"mysql", "mysql+mysqlconnector"}:
        raise ValueError("DATAINSIGHT_DATABASE_URL must use the mysql scheme")
    database = parsed.path.lstrip("/")
    if not database or not database.replace("_", "").isalnum():
        raise ValueError("MySQL database name must contain only letters, numbers, or underscores")
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": database,
    }


def get_connection() -> sqlite3.Connection | MySQLConnection:
    mysql_config = _mysql_config()
    if mysql_config:
        return MySQLConnection(mysql.connector.connect(**mysql_config))
    connection = sqlite3.connect(database_path(), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    from app.database.tables import create_tables

    mysql_config = _mysql_config()
    if mysql_config:
        database = mysql_config.pop("database")
        connection = mysql.connector.connect(**mysql_config)
        try:
            cursor = connection.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        finally:
            connection.close()
        mysql_config["database"] = database

    with get_connection() as connection:
        create_tables(connection)
