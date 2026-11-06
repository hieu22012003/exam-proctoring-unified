"""Small SQLite store. Every operation owns its connection (worker-thread safe)."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import unicodedata
import uuid
from .cloud_sync import FirebaseSync

# Columns added after the first release; created on startup when missing.
MIGRATIONS = {
    "students": [("role", "TEXT NOT NULL DEFAULT 'student'")],
    "sessions": [("roster", "TEXT")],
    "alerts": [("category", "TEXT"), ("decision", "TEXT"), ("note", "TEXT"), ("reviewer", "TEXT"),
               ("reviewed_at", "TEXT"), ("signature", "TEXT"), ("thumbnail", "TEXT")],
    "transcripts": [("audio", "TEXT"), ("speaker_id", "TEXT"), ("speaker_name", "TEXT"), ("speaker_score", "REAL")],
    "observations": [("mood", "TEXT")],
}
DECISIONS = {"violation": "Xác nhận vi phạm", "warning": "Nhắc nhở", "dismissed": "Không vi phạm"}


def fold(text):
    """Lower-case and strip Vietnamese diacritics so 'nhac bai' finds 'Nhắc bài'."""
    text = unicodedata.normalize("NFD", str(text or "").lower().replace("đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def mood_of(emotion, tired=False):
    """Group the 7 model emotions into the 4 support states shown to proctors."""
    if tired:
        return "tired"
    return {"neutral": "normal", "happy": "normal", "fear": "anxious", "surprise": "anxious",
            "angry": "anxious", "disgust": "anxious", "sad": "sad"}.get(emotion, "unknown")


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
            for table, columns in MIGRATIONS.items():
                existing = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
                for name, kind in columns:
                    if name not in existing:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
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

    def add_student(self, student_id, name, embeddings, role="student"):
        row = {"id": student_id, "name": name, "embeddings": embeddings, "created_at": utc_now(), "role": role}
        with self.connection() as db:
            db.execute("INSERT INTO students (id,name,embeddings,created_at,role) VALUES (?,?,?,?,?)",
                       (student_id, name, json.dumps(embeddings), row["created_at"], role))
        self.cloud.enqueue("set", "students", student_id, row)

    def delete_student(self, student_id):
        with self.connection() as db:
            deleted = db.execute("DELETE FROM students WHERE id=?", (student_id,)).rowcount > 0
        if deleted:
            self.cloud.enqueue("delete", "students", student_id)
        return deleted

    def new_session(self, title, roster=None):
        row = dict(id=uuid.uuid4().hex, title=title, started_at=utc_now(), ended_at=None,
                   roster=json.dumps(roster or [], ensure_ascii=False))
        with self.connection() as db:
            db.execute("INSERT INTO sessions (id,title,started_at,ended_at,roster) VALUES (:id,:title,:started_at,:ended_at,:roster)", row)
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

    def add_alert(self, session_id, track, kind, detail, offset, image, thumbnail=None):
        row = dict(id=uuid.uuid4().hex, session_id=session_id, student_id=track.student_id,
                   student_name=track.name, track_id=track.id, kind=kind, detail=detail,
                   created_at=utc_now(), offset_seconds=offset, image=image, acknowledged=0,
                   thumbnail=thumbnail)
        with self.connection() as db:
            db.execute("""INSERT INTO alerts (id,session_id,student_id,student_name,track_id,kind,detail,
                created_at,offset_seconds,image,acknowledged,thumbnail) VALUES (:id,:session_id,:student_id,
                :student_name,:track_id,:kind,:detail,:created_at,:offset_seconds,:image,:acknowledged,:thumbnail)""", row)
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

    def review(self, alert_id, category, decision, note, reviewer, signature=None):
        if decision not in DECISIONS:
            raise ValueError("Quyết định không hợp lệ.")
        reviewed_at = utc_now()
        with self.connection() as db:
            changed = db.execute("""UPDATE alerts SET acknowledged=1, category=?, decision=?, note=?,
                reviewer=?, reviewed_at=?, signature=COALESCE(?, signature) WHERE id=?""",
                (category, decision, note, reviewer, reviewed_at, signature, alert_id)).rowcount > 0
            row = db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone() if changed else None
        if row:
            self.cloud.enqueue("set", "alerts", alert_id, dict(row))
        return dict(row) if row else None

    def _evidence_fields(self, row):
        if not row.get('image'):
            row.update(evidence_type='none', evidence_status='none', has_thumbnail=False,
                       decision_label=DECISIONS.get(row.get('decision') or '', ''))
            row['has_signature'] = bool(row.pop('signature', None))
            return row
        path = self.directory / 'evidence' / row['image']
        row['evidence_type'] = 'video' if path.suffix == '.webm' else 'image'
        row['evidence_status'] = ('ready' if path.is_file() else
            'failed' if path.with_suffix('.failed').is_file() else 'pending')
        row['has_thumbnail'] = bool(row.get('thumbnail') and (self.directory / 'evidence' / row['thumbnail']).is_file()) \
            or (row['evidence_type'] == 'image' and row['evidence_status'] == 'ready')
        row['decision_label'] = DECISIONS.get(row.get('decision') or '', '')
        row['has_signature'] = bool(row.pop('signature', None))  # served separately
        return row

    def search_alerts(self, q="", kind="", status="", session_id="", date_from="", date_to="", limit=500):
        sql = "SELECT a.*, s.title AS session_title FROM alerts a JOIN sessions s ON s.id=a.session_id WHERE 1=1"
        args = []
        if kind: sql += " AND a.kind=?"; args.append(kind)
        if session_id: sql += " AND a.session_id=?"; args.append(session_id)
        if status == "open": sql += " AND a.acknowledged=0"
        elif status == "reviewed": sql += " AND a.acknowledged=1"
        elif status in DECISIONS: sql += " AND a.decision=?"; args.append(status)
        if date_from: sql += " AND a.created_at>=?"; args.append(date_from)
        if date_to: sql += " AND a.created_at<=?"; args.append(date_to)
        sql += " ORDER BY a.created_at DESC LIMIT ?"; args.append(max(1, min(int(limit), 5000)))
        with self.connection() as db:
            rows = [dict(r) for r in db.execute(sql, args)]
        if q:
            needle = fold(q)
            rows = [r for r in rows if needle in fold(" ".join(str(r.get(k) or "") for k in
                    ("student_name", "detail", "note", "session_title", "category", "reviewer", "student_id")))]
        return [self._evidence_fields(r) for r in rows]

    def evidence_file(self, name):
        return self.directory / "evidence" / name if name else None

    def alert(self, alert_id):
        with self.connection() as db:
            r = db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(r) if r else None

    def add_transcript(self, session_id, start, end, text, audio=None, speaker=None):
        speaker = speaker or {}
        row = {"id": uuid.uuid4().hex, "session_id": session_id, "start_seconds": start,
               "end_seconds": end, "text": text, "created_at": utc_now(), "audio": audio,
               "speaker_id": speaker.get("id"), "speaker_name": speaker.get("name"), "speaker_score": speaker.get("score")}
        with self.connection() as db:
            db.execute("""INSERT INTO transcripts (id,session_id,start_seconds,end_seconds,text,created_at,audio,
                speaker_id,speaker_name,speaker_score) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                       (row["id"], session_id, start, end, text, row["created_at"], audio,
                        row["speaker_id"], row["speaker_name"], row["speaker_score"]))
        self.cloud.enqueue("set", "transcripts", row["id"], row)
        return row

    def search_transcripts(self, q="", session_id="", limit=500, speaker=""):
        sql = "SELECT t.*, s.title AS session_title, s.started_at AS session_started FROM transcripts t JOIN sessions s ON s.id=t.session_id"
        args = []
        clauses = []
        if session_id: clauses.append("t.session_id=?"); args.append(session_id)
        if speaker == "unknown": clauses.append("t.speaker_name IS NULL")
        elif speaker: clauses.append("t.speaker_id=?"); args.append(speaker)
        if clauses: sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY s.started_at DESC, t.start_seconds DESC LIMIT ?"; args.append(5000)
        with self.connection() as db:
            rows = [dict(r) for r in db.execute(sql, args)]
        if q:
            needle = fold(q)
            rows = [r for r in rows if needle in fold(r["text"] + " " + (r.get("speaker_name") or ""))]
        for r in rows:
            r["has_audio"] = bool(r.get("audio") and (self.directory / "evidence" / r["audio"]).is_file())
        return rows[:max(1, min(int(limit), 5000))]

    def speakers(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("""SELECT speaker_id AS id, MAX(speaker_name) AS name, COUNT(*) AS lines
                FROM transcripts WHERE speaker_id IS NOT NULL GROUP BY speaker_id ORDER BY name""")]

    def transcript_audio(self, transcript_id):
        with self.connection() as db:
            r = db.execute("SELECT audio FROM transcripts WHERE id=?", (transcript_id,)).fetchone()
        return self.directory / "evidence" / r[0] if r and r[0] else None

    def mood_series(self, session_id, student_id="", bucket=10):
        sql = "SELECT offset_seconds, student_id, student_name, mood, emotion FROM observations WHERE session_id=?"
        args = [session_id]
        if student_id: sql += " AND student_id=?"; args.append(student_id)
        with self.connection() as db:
            rows = [dict(r) for r in db.execute(sql + " ORDER BY offset_seconds", args)]
        buckets, people = {}, {}
        for r in rows:
            mood = r["mood"] or mood_of(r["emotion"], False)
            if mood == "unknown":
                continue
            b = int(r["offset_seconds"] // bucket) * bucket
            counts = buckets.setdefault(b, {"normal": 0, "anxious": 0, "tired": 0, "sad": 0})
            counts[mood] = counts.get(mood, 0) + 1
            if r["student_id"]:
                person = people.setdefault(r["student_id"], {"id": r["student_id"], "name": r["student_name"],
                                                             "normal": 0, "anxious": 0, "tired": 0, "sad": 0})
                person[mood] = person.get(mood, 0) + 1
        series = []
        for b in sorted(buckets):
            total = sum(buckets[b].values()) or 1
            series.append({"offset": b, **{k: round(v / total, 3) for k, v in buckets[b].items()}})
        return {"bucket_seconds": bucket, "series": series, "students": sorted(people.values(), key=lambda p: p["name"])}

    def attendance(self, session_id):
        row = self.session(session_id)
        if not row:
            return None
        roster = json.loads(row.get("roster") or "[]")
        with self.connection() as db:
            seen = {r["student_id"]: dict(r) for r in db.execute(
                """SELECT student_id, MIN(student_name) AS name, MIN(offset_seconds) AS first_seen,
                   MAX(offset_seconds) AS last_seen, COUNT(*) AS samples FROM observations
                   WHERE session_id=? AND student_id IS NOT NULL GROUP BY student_id""", (session_id,))}
        people = []
        for student in roster:
            s = seen.get(student["id"])
            people.append({"id": student["id"], "name": student["name"], "present": bool(s),
                           "first_seen": s["first_seen"] if s else None, "last_seen": s["last_seen"] if s else None})
        known = {p["id"] for p in roster}
        extra = [{"id": sid, "name": s["name"], "present": True, "first_seen": s["first_seen"],
                  "last_seen": s["last_seen"], "unlisted": True} for sid, s in seen.items() if sid not in known]
        people += extra
        return {"total": len(roster), "present": sum(p["present"] for p in people if not p.get("unlisted")),
                "absent": sum(not p["present"] for p in people), "people": people}

    def observe(self, session_id, track, offset):
        mood = mood_of(track.emotion, getattr(track, "tired", False))
        row = {"session_id": session_id, "track_id": track.id, "student_id": track.student_id,
               "student_name": track.name, "emotion": track.emotion,
               "confidence": track.emotion_score, "posture": track.posture,
               "offset_seconds": offset, "created_at": utc_now(), "mood": mood}
        with self.connection() as db:
            cursor = db.execute("""INSERT INTO observations (session_id,track_id,student_id,student_name,
                emotion,confidence,posture,offset_seconds,created_at,mood) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                       (session_id, track.id, track.student_id, track.name, track.emotion,
                        track.emotion_score, track.posture, offset, row["created_at"], mood))
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
            rows = [self._evidence_fields(row) for row in rows]
        if table == 'transcripts':
            for r in rows:
                r["has_audio"] = bool(r.get("audio") and (self.directory / "evidence" / r["audio"]).is_file())
        return rows

    def evidence(self, alert_id):
        with self.connection() as db:
            r = db.execute("SELECT image FROM alerts WHERE id=?", (alert_id,)).fetchone()
            return self.directory / "evidence" / r[0] if r else None
