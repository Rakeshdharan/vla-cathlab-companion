"""SQLite state for the control plane: runs (frozen selections) and jobs (stage executions)."""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from pydantic import BaseModel, Field

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL, selection TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, kind TEXT, name TEXT, run_id TEXT, worker TEXT, state TEXT,
    params TEXT, result TEXT, created REAL, updated REAL
);
"""


class Run(BaseModel):
    id: str
    created: float
    selection: dict
    note: str = ""


class Job(BaseModel):
    id: str
    kind: str
    name: str
    run_id: str | None = None
    worker: str | None = None
    state: str
    params: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
    created: float
    updated: float


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    # runs ---------------------------------------------------------------
    def create_run(self, selection: dict, note: str = "") -> Run:
        run = Run(id=new_id("run"), created=time.time(), selection=selection, note=note)
        with self._lock, self._db:
            self._db.execute("INSERT INTO runs VALUES (?,?,?,?)",
                             (run.id, run.created, json.dumps(selection), note))
        return run

    def get_run(self, run_id: str) -> Run | None:
        row = self._db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return None if row is None else Run(id=row["id"], created=row["created"],
                                            selection=json.loads(row["selection"]), note=row["note"])

    def list_runs(self, limit: int = 50) -> list[Run]:
        rows = self._db.execute("SELECT id FROM runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self.get_run(r["id"]) for r in rows]

    # jobs ---------------------------------------------------------------
    def create_job(self, kind: str, name: str, params: dict | None = None, run_id: str | None = None,
                   worker: str | None = None, state: str = "queued") -> Job:
        now = time.time()
        job = Job(id=new_id(kind[:3]), kind=kind, name=name, run_id=run_id, worker=worker, state=state,
                  params=params or {}, result={}, created=now, updated=now)
        with self._lock, self._db:
            self._db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (job.id, kind, name, run_id, worker, state, json.dumps(job.params), "{}", now, now))
        return job

    def update_job(self, job_id: str, state: str | None = None, result: dict | None = None,
                   worker: str | None = None) -> Job:
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                raise KeyError(job_id)
            merged = {**job.result, **(result or {})}
            with self._db:
                self._db.execute("UPDATE jobs SET state=?, result=?, worker=?, updated=? WHERE id=?",
                                 (state or job.state, json.dumps(merged), worker or job.worker, time.time(), job_id))
            return self.get_job(job_id)

    def get_job(self, job_id: str) -> Job | None:
        row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return Job(id=row["id"], kind=row["kind"], name=row["name"], run_id=row["run_id"], worker=row["worker"],
                   state=row["state"], params=json.loads(row["params"]), result=json.loads(row["result"]),
                   created=row["created"], updated=row["updated"])

    def list_jobs(self, kind: str | None = None, run_id: str | None = None, limit: int = 100) -> list[Job]:
        sql, args = "SELECT id FROM jobs WHERE 1=1", []
        if kind:
            sql, args = sql + " AND kind=?", [*args, kind]
        if run_id:
            sql, args = sql + " AND run_id=?", [*args, run_id]
        rows = self._db.execute(sql + " ORDER BY created DESC LIMIT ?", (*args, limit)).fetchall()
        return [self.get_job(r["id"]) for r in rows]
