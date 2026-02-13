import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

from config import MD_DIR_PATH

DB_FILE_NAME = "mathbot.sqlite3"
DB_PATH = os.path.join(MD_DIR_PATH, DB_FILE_NAME)
DB_LOCK = threading.RLock()


def _ensure_db_parent_dir():
    parent = os.path.dirname(DB_PATH)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def get_connection():
    _ensure_db_parent_dir()
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


@contextmanager
def transaction():
    with DB_LOCK:
        conn = get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db():
    with transaction() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                fingerprint TEXT,
                notion_page_id TEXT,
                created_at TEXT NOT NULL,
                last_updated TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_title ON concepts(title);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_fingerprint ON concepts(fingerprint);")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def write_log(level, message):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with transaction() as conn:
        conn.execute(
            "INSERT INTO logs(level, message, created_at) VALUES (?, ?, ?)",
            (str(level), str(message), created_at),
        )


def fetch_all_concepts():
    init_db()
    with DB_LOCK:
        conn = get_connection()
        try:
            cur = conn.execute(
                """
                SELECT id, title, content, fingerprint, notion_page_id, created_at, last_updated
                FROM concepts
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def replace_all_concepts(data):
    init_db()
    with transaction() as conn:
        conn.execute("DELETE FROM concepts")
        for item in data:
            conn.execute(
                """
                INSERT INTO concepts(title, content, fingerprint, notion_page_id, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.get("title", ""),
                    item.get("content", ""),
                    item.get("fingerprint", ""),
                    item.get("notion_page_id"),
                    item.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    item.get("last_updated"),
                ),
            )
    return True
