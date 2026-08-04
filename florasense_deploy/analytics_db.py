"""
analytics_db.py — lightweight local analytics for FloraSense AI.

No external DB needed — SQLite is a single file, zero setup, plenty for a
project at this scale. Logs every query (species identified, confidence,
latency, whether it used each tool) plus optional thumbs-up/down feedback,
and exposes simple aggregate stats for the dashboard.
"""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.getenv("FLORASENSE_ANALYTICS_DB", "florasense_analytics.db")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                had_image INTEGER NOT NULL,
                predicted_species TEXT,
                confidence REAL,
                is_confident INTEGER,
                tools_used TEXT,
                latency_seconds REAL NOT NULL,
                answer_preview TEXT,
                feedback INTEGER
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_species ON interactions(predicted_species)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)")


def log_interaction(
    question: str,
    had_image: bool,
    predicted_species: str = None,
    confidence: float = None,
    is_confident: bool = None,
    tools_used: list = None,
    latency_seconds: float = 0.0,
    answer_preview: str = "",
) -> int:
    """Logs one interaction, returns its row id (used later to attach feedback)."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO interactions
                (timestamp, question, had_image, predicted_species, confidence,
                 is_confident, tools_used, latency_seconds, answer_preview, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                datetime.utcnow().isoformat(),
                question,
                int(had_image),
                predicted_species,
                confidence,
                int(is_confident) if is_confident is not None else None,
                json.dumps(tools_used or []),
                latency_seconds,
                answer_preview[:200],
            ),
        )
        return cur.lastrowid


def log_feedback(interaction_id: int, positive: bool):
    with _connect() as conn:
        conn.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (1 if positive else 0, interaction_id),
        )


def get_summary_stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM interactions").fetchone()["c"]
        if total == 0:
            return {
                "total_queries": 0,
                "avg_latency": 0.0,
                "avg_confidence": 0.0,
                "confident_rate": 0.0,
                "positive_feedback_rate": None,
                "top_species": [],
                "tool_usage": {},
            }

        avg_latency = conn.execute("SELECT AVG(latency_seconds) AS a FROM interactions").fetchone()["a"]

        conf_rows = conn.execute(
            "SELECT confidence FROM interactions WHERE confidence IS NOT NULL"
        ).fetchall()
        avg_confidence = (sum(r["confidence"] for r in conf_rows) / len(conf_rows)) if conf_rows else 0.0

        confident_count = conn.execute(
            "SELECT COUNT(*) AS c FROM interactions WHERE is_confident = 1"
        ).fetchone()["c"]
        classified_count = conn.execute(
            "SELECT COUNT(*) AS c FROM interactions WHERE is_confident IS NOT NULL"
        ).fetchone()["c"]
        confident_rate = (confident_count / classified_count) if classified_count else 0.0

        fb_rows = conn.execute(
            "SELECT feedback FROM interactions WHERE feedback IS NOT NULL"
        ).fetchall()
        positive_feedback_rate = (
            sum(r["feedback"] for r in fb_rows) / len(fb_rows) if fb_rows else None
        )

        top_species = conn.execute(
            """
            SELECT predicted_species, COUNT(*) AS n
            FROM interactions
            WHERE predicted_species IS NOT NULL
            GROUP BY predicted_species
            ORDER BY n DESC
            LIMIT 10
            """
        ).fetchall()

        tool_usage = {}
        for row in conn.execute("SELECT tools_used FROM interactions").fetchall():
            for tool_name in json.loads(row["tools_used"] or "[]"):
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        return {
            "total_queries": total,
            "avg_latency": round(avg_latency or 0.0, 2),
            "avg_confidence": round(avg_confidence, 3),
            "confident_rate": round(confident_rate, 3),
            "positive_feedback_rate": (
                round(positive_feedback_rate, 3) if positive_feedback_rate is not None else None
            ),
            "top_species": [(r["predicted_species"], r["n"]) for r in top_species],
            "tool_usage": tool_usage,
        }


def get_recent_interactions(limit: int = 20) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


class Timer:
    """Small context manager for measuring latency around the agent call."""

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
