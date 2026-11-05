"""Small SQLite store. Every operation owns its connection (worker-thread safe)."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid
from .cloud_sync import FirebaseSync


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "evidence").mkdir(exist_ok=True)
        self.path = directory / "examwatch.sqlite3"
        self.cloud = FirebaseSync(directory / "evidence")
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS students (
                  id TEXT PRIMARY KEY, name TEXT NOT NULL, embeddings TEXT NOT NULL,
                  created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                  id TEXT PRIMARY KEY, title TEXT NOT NULL, started_at TEXT NOT NULL,
                  ended_at TEXT);
                CREATE TABLE IF NOT EXISTS alerts (
                  id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                  student_id TEXT, student_name TEXT NOT NULL, track_id INTEGER NOT NULL,
                  kind TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL,
                  offset_seconds REAL NOT NULL, image TEXT NOT NULL, acknowledged INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS transcripts (
                  id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                  start_seconds REAL NOT NULL, end_seconds REAL NOT NULL, text TEXT NOT NULL,
                  created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS observations (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
                  track_id INTEGER NOT NULL, student_id TEXT, student_name TEXT NOT NULL,
                  emotion TEXT NOT NULL, confidence REAL NOT NULL, posture TEXT NOT NULL,
                  offset_seconds REAL NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS observations_session ON observations(session_id, id);
                CREATE INDEX IF NOT EXISTS alerts_session ON alerts(session_id, created_at);
                CREATE INDEX IF NOT EXISTS transcripts_session ON transcripts(session_id, start_seconds);
            """)
            # A crashed application cannot resume capture without a new explicit session.
            db.execute("UPDATE sessions SET ended_at=? WHERE ended_at IS NULL", (utc_now(),))
        self._sync_existing()

    def _sync_existing(self):
        if self.cloud.state != "connected":
            return
        with self.connection() as db:
            students = [dict(r) for r in db.execute("SELECT * FROM students")]
            sessions = [dict(r) for r in db.execute("SELECT * FROM sessions")]
            alerts = [dict(r) for r in db.execute("SELECT * FROM alerts")]
            transcripts = [dict(r) for r in db.execute("SELECT * FROM transcripts")]
            observations = [dict(r) for r in db.execute("SELECT * FROM observations")]
            settings = [dict(r) for r in db.execute("SELECT * FROM app_settings")]
        for row in students:
            row["embeddings"] = json.loads(row["embeddings"])
            self.cloud.enqueue("set", "students", row["id"], row)
        for collection, rows in (("sessions", sessions), ("alerts", alerts),
                                 ("transcripts", transcripts), ("observations", observations)):
            for row in rows:
                self.cloud.enqueue("set", collection, row.get("id", row.get("id")), row)
        for row in alerts:
            self.cloud.upload_evidence(row["session_id"], row["image"], row["id"])
        for row in settings:
            self.cloud.enqueue("set", "settings", row["key"],
                               {"key": row["key"], "value": json.loads(row["value"])})

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def students(self, with_embeddings=False):
        with self.connection() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM students ORDER BY name")]
        for row in rows:
            vectors = json.loads(row.pop("embeddings"))
            row["samples"] = len(vectors)
            if with_embeddings:
                row["embeddings"] = vectors
        return rows

    def setting(self, key, default=None):
        with self.connection() as db:
            row=db.execute('SELECT value FROM app_settings WHERE key=?',(key,)).fetchone()
        return json.loads(row['value']) if row else default

    def set_setting(self, key, value):
        with self.connection() as db:
            db.execute('INSERT INTO app_settings(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                       (key,json.dumps(value)))
        self.cloud.enqueue("set", "settings", key, {"key": key, "value": value})

    def add_student(self, student_id, name, embeddings):
        row = {"id": student_id, "name": name, "embeddings": embeddings, "created_at": utc_now()}
        with self.connection() as db:
            db.execute("INSERT INTO students VALUES (?,?,?,?)",
                       (student_id, name, json.dumps(embeddings), row["created_at"]))
        self.cloud.enqueue("set", "students", student_id, row)

    def delete_student(self, student_id):
        with self.connection() as db:
            deleted = db.execute("DELETE FROM students WHERE id=?", (student_id,)).rowcount > 0
        if deleted:
            self.cloud.enqueue("delete", "students", student_id)
        return deleted

    def new_session(self, title):
        row = dict(id=uuid.uuid4().hex, title=title, started_at=utc_now(), ended_at=None)
        with self.connection() as db:
            db.execute("INSERT INTO sessions VALUES (:id,:title,:started_at,:ended_at)", row)
        self.cloud.enqueue("set", "sessions", row["id"], row)
        return row

    def end_session(self, session_id):
        ended_at = utc_now()
        with self.connection() as db:
            db.execute("UPDATE sessions SET ended_at=? WHERE id=?", (ended_at, session_id))
        self.cloud.enqueue("set", "sessions", session_id, {"ended_at": ended_at})

    def sessions(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM sessions ORDER BY started_at DESC")]

    def session(self, session_id):
        with self.connection() as db:
            r = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            return dict(r) if r else None

    def add_alert(self, session_id, track, kind, detail, offset, image):
        row = dict(id=uuid.uuid4().hex, session_id=session_id, student_id=track.student_id,
                   student_name=track.name, track_id=track.id, kind=kind, detail=detail,
                   created_at=utc_now(), offset_seconds=offset, image=image, acknowledged=0)
        with self.connection() as db:
            db.execute("""INSERT INTO alerts VALUES (:id,:session_id,:student_id,:student_name,
                :track_id,:kind,:detail,:created_at,:offset_seconds,:image,:acknowledged)""", row)
        self.cloud.enqueue("set", "alerts", row["id"], row)
        self.cloud.upload_evidence(session_id, image, row["id"])
        return row

    def acknowledge(self, alert_id):
        with self.connection() as db:
            changed = db.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,)).rowcount > 0
            row = db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone() if changed else None
        if row:
            self.cloud.enqueue("set", "alerts", alert_id, dict(row))
        return changed

    def add_transcript(self, session_id, start, end, text):
        row = {"id": uuid.uuid4().hex, "session_id": session_id, "start_seconds": start,
               "end_seconds": end, "text": text, "created_at": utc_now()}
        with self.connection() as db:
            db.execute("INSERT INTO transcripts VALUES (?,?,?,?,?,?)",
                       (row["id"], session_id, start, end, text, row["created_at"]))
        self.cloud.enqueue("set", "transcripts", row["id"], row)

    def observe(self, session_id, track, offset):
        row = {"session_id": session_id, "track_id": track.id, "student_id": track.student_id,
               "student_name": track.name, "emotion": track.emotion,
               "confidence": track.emotion_score, "posture": track.posture,
               "offset_seconds": offset, "created_at": utc_now()}
        with self.connection() as db:
            cursor = db.execute("""INSERT INTO observations (session_id,track_id,student_id,student_name,
                emotion,confidence,posture,offset_seconds,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                       (session_id, track.id, track.student_id, track.name, track.emotion,
                        track.emotion_score, track.posture, offset, row["created_at"]))
            row["id"] = cursor.lastrowid
        self.cloud.enqueue("set", "observations", row["id"], row)

    def records(self, session_id, table, limit=300):
        if table not in {"alerts", "transcripts", "observations"}:
            raise ValueError("Invalid record table")
        order = "start_seconds" if table == "transcripts" else "rowid"
        with self.connection() as db:
            rows = [dict(r) for r in db.execute(
                f"SELECT * FROM {table} WHERE session_id=? ORDER BY {order} DESC LIMIT ?",
                (session_id, limit))]
        if table == 'alerts':
            for row in rows:
                path = self.directory / 'evidence' / row['image']
                row['evidence_type'] = 'video' if path.suffix == '.webm' else 'image'
                row['evidence_status'] = ('ready' if path.is_file() else
                    'failed' if path.with_suffix('.failed').is_file() else 'pending')
        return rows

    def evidence(self, alert_id):
        with self.connection() as db:
            r = db.execute("SELECT image FROM alerts WHERE id=?", (alert_id,)).fetchone()
            return self.directory / "evidence" / r[0] if r else None
