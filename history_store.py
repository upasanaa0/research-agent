"""
Clean history store — saves only user question + final answer.
Separate from LangGraph checkpoints which save full pipeline state.
"""
import psycopg2
import psycopg2.extras
from datetime import datetime
from config import DB_URI


def setup_history_table():
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id SERIAL PRIMARY KEY,
            thread_id TEXT NOT NULL,
            asked_at TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            answered_from TEXT DEFAULT 'research'
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[history] Table ready.")


def save_session(thread_id: str, question: str, answer: str, answered_from: str = "research"):
    """
    answered_from: 'research' | 'memory'
    """
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO session_history (thread_id, asked_at, question, answer, answered_from)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        thread_id,
        datetime.now().isoformat(),
        question,
        answer,
        answered_from
    ))
    conn.commit()
    cur.close()
    conn.close()
    print(f"[history] Saved session for thread={thread_id}")


def fetch_history_for_thread(thread_id: str) -> list[dict]:
    """Fetch all Q&A pairs for a specific thread, oldest first."""
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, thread_id, asked_at, question, answer, answered_from
        FROM session_history
        WHERE thread_id = %s
        ORDER BY asked_at ASC
    """, (thread_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def fetch_all_history(limit: int = 50) -> list[dict]:
    """Fetch most recent sessions across all threads."""
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, thread_id, asked_at, question, answer, answered_from
        FROM session_history
        ORDER BY asked_at DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def fetch_all_threads() -> list[dict]:
    """Fetch distinct thread IDs with their question count and last activity."""
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            thread_id,
            COUNT(*) as question_count,
            MIN(asked_at) as first_asked,
            MAX(asked_at) as last_asked
        FROM session_history
        GROUP BY thread_id
        ORDER BY last_asked DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]