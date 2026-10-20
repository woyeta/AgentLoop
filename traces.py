"""
Stores structured traces of agent runs in SQLite.
"""

import json
import sqlite3
import threading
import time
import uuid

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    tool        TEXT NOT NULL,
    query       TEXT NOT NULL,
    model       TEXT,
    status      TEXT,
    n_steps     INTEGER,
    started_at  REAL NOT NULL,
    ended_at    REAL
);

CREATE TABLE IF NOT EXISTS steps (
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    step              INTEGER NOT NULL,
    type              TEXT NOT NULL,
    thought           TEXT,
    action            TEXT,
    args_json         TEXT,
    raw_output        TEXT,
    parse_ok          INTEGER NOT NULL,
    validation_error  TEXT,
    error             TEXT,
    observation       TEXT,
    model_ms          REAL,
    tool_ms           REAL,
    n_tokens          INTEGER,
    PRIMARY KEY (run_id, step)
);
"""

_STEP_COLUMNS = (
    "type", "thought", "action", "args_json", "raw_output", "parse_ok", "validation_error",
    "error", "observation", "model_ms", "tool_ms", "n_tokens",
)


class TraceStore:
    """Thread-safe writer and reader for run traces."""

    def __init__(self, path: str = "traces.db"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def start_run(self, tool: str, query: str, model: str = "") -> str:
        run_id = uuid.uuid4().hex
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, tool, query, model, started_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, tool, query, model, time.time()),
            )
        return run_id

    def log_step(self, run_id: str, step: int, **values) -> None:
        if "args_json" in values and not isinstance(values["args_json"], (str, type(None))):
            values["args_json"] = json.dumps(values["args_json"])
        row = [values.get(col) for col in _STEP_COLUMNS]
        row[_STEP_COLUMNS.index("parse_ok")] = int(bool(values.get("parse_ok", True)))
        placeholders = ", ".join("?" for _ in range(len(_STEP_COLUMNS) + 2))
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO steps (run_id, step, {', '.join(_STEP_COLUMNS)}) VALUES ({placeholders})",
                (run_id, step, *row),
            )

    def end_run(self, run_id: str, status: str, n_steps: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET status = ?, n_steps = ?, ended_at = ? WHERE run_id = ?",
                (status, n_steps, time.time(), run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            run = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            steps = self._conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY step", (run_id,)
            ).fetchall()
        return {**dict(run), "steps": [dict(s) for s in steps]}

    def query(self, sql: str, params: tuple = ()) -> list:
        """Runs a read-only query, e.g. for latency or recovery metrics."""
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def close(self) -> None:
        self._conn.close()


RECOVERY_RATE_SQL = """
SELECT COUNT(*) AS invalid_steps,
       SUM(nxt.parse_ok = 1 AND nxt.validation_error IS NULL AND nxt.type != 'error') AS recovered
FROM steps cur
JOIN steps nxt ON nxt.run_id = cur.run_id AND nxt.step = cur.step + 1
WHERE cur.parse_ok = 0 OR cur.validation_error IS NOT NULL
"""

LATENCY_SQL = """
SELECT AVG(model_ms) AS avg_model_ms, AVG(tool_ms) AS avg_tool_ms, AVG(n_tokens) AS avg_tokens
FROM steps
"""
