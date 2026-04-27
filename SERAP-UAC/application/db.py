from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Session:
    id: int
    created_at: str
    top_k: int
    resume_systeme: str
    profil: dict[str, Any]


@dataclass(frozen=True)
class RecommandationRow:
    session_id: int
    rang: int
    code_filiere: str
    filiere: str
    domaine: str
    cycle: str
    score_final: float
    score_regles: float
    score_graphe: float
    score_cnn: float
    explication: str


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).astimezone().replace(microsecond=0).isoformat()


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    resume_systeme TEXT NOT NULL,
                    profil_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recommandations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    rang INTEGER NOT NULL,
                    code_filiere TEXT NOT NULL,
                    filiere TEXT NOT NULL,
                    domaine TEXT NOT NULL,
                    cycle TEXT NOT NULL,
                    score_final REAL NOT NULL,
                    score_regles REAL NOT NULL,
                    score_graphe REAL NOT NULL,
                    score_cnn REAL NOT NULL,
                    explication TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                """
            )

    def save_session(
        self,
        profil: dict[str, Any],
        top_k: int,
        resume_systeme: str,
        recommandations: list[dict[str, Any]],
    ) -> int:
        created_at = _utc_now_iso()
        profil_json = json.dumps(profil, ensure_ascii=False)
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO sessions(created_at, top_k, resume_systeme, profil_json) VALUES(?,?,?,?)",
                (created_at, int(top_k), resume_systeme, profil_json),
            )
            session_id = int(cur.lastrowid)
            for reco in recommandations:
                con.execute(
                    """
                    INSERT INTO recommandations(
                        session_id, rang, code_filiere, filiere, domaine, cycle,
                        score_final, score_regles, score_graphe, score_cnn, explication
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        session_id,
                        int(reco["rang"]),
                        str(reco["code_filiere"]),
                        str(reco["filiere"]),
                        str(reco["domaine"]),
                        str(reco["cycle"]),
                        float(reco["score_final"]),
                        float(reco["score_regles"]),
                        float(reco["score_graphe"]),
                        float(reco["score_cnn"]),
                        str(reco["explication"]),
                    ),
                )
        return session_id

    def list_sessions(self, limit: int = 30) -> list[Session]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, created_at, top_k, resume_systeme, profil_json FROM sessions ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out: list[Session] = []
        for r in rows:
            out.append(
                Session(
                    id=int(r["id"]),
                    created_at=str(r["created_at"]),
                    top_k=int(r["top_k"]),
                    resume_systeme=str(r["resume_systeme"]),
                    profil=json.loads(str(r["profil_json"])),
                )
            )
        return out

    def list_recommandations_for_sessions(
        self, session_ids: list[int]
    ) -> dict[int, list[RecommandationRow]]:
        if not session_ids:
            return {}
        placeholders = ",".join(["?"] * len(session_ids))
        query = f"""
            SELECT session_id, rang, code_filiere, filiere, domaine, cycle,
                   score_final, score_regles, score_graphe, score_cnn, explication
            FROM recommandations
            WHERE session_id IN ({placeholders})
            ORDER BY session_id DESC, rang ASC
        """
        with self._connect() as con:
            rows = con.execute(query, [int(sid) for sid in session_ids]).fetchall()
        out: dict[int, list[RecommandationRow]] = {}
        for r in rows:
            rec = RecommandationRow(
                session_id=int(r["session_id"]),
                rang=int(r["rang"]),
                code_filiere=str(r["code_filiere"]),
                filiere=str(r["filiere"]),
                domaine=str(r["domaine"]),
                cycle=str(r["cycle"]),
                score_final=float(r["score_final"]),
                score_regles=float(r["score_regles"]),
                score_graphe=float(r["score_graphe"]),
                score_cnn=float(r["score_cnn"]),
                explication=str(r["explication"]),
            )
            out.setdefault(rec.session_id, []).append(rec)
        return out

    def delete_session(self, session_id: int) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM recommandations WHERE session_id = ?", (int(session_id),))
            con.execute("DELETE FROM sessions WHERE id = ?", (int(session_id),))

    def delete_all_sessions(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM recommandations")
            con.execute("DELETE FROM sessions")
