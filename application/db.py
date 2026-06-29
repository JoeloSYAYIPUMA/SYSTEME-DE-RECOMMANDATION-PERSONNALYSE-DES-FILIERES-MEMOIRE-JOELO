from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
DOSSIER_SORTIES = BASE_DIR / "sorties"
CHEMIN_BD = DOSSIER_SORTIES / "serap.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recommandation_session (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  resume_systeme TEXT NOT NULL,
  top_k INTEGER NOT NULL,
  profil_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommandation_item (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  rang INTEGER NOT NULL,
  code_filiere TEXT NOT NULL,
  filiere TEXT NOT NULL,
  domaine TEXT NOT NULL,
  score_final REAL NOT NULL,
  score_regles REAL NOT NULL,
  score_graphe REAL NOT NULL,
  score_cnn REAL NOT NULL,
  explication TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES recommandation_session(id)
);

CREATE INDEX IF NOT EXISTS idx_item_session ON recommandation_item(session_id);
"""


def _connect(db_path: Path = CHEMIN_BD) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path = CHEMIN_BD) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def enregistrer_recommandation(
    profil: Dict[str, Optional[float]],
    resultat: Dict[str, Any],
    top_k: int,
    db_path: Path = CHEMIN_BD,
) -> int:
    resume_systeme = str(resultat.get("resume_systeme", ""))
    items: List[Dict[str, Any]] = list(resultat.get("top_k", []))
    profil_json = json.dumps(profil, ensure_ascii=False)

    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO recommandation_session (resume_systeme, top_k, profil_json) VALUES (?, ?, ?)",
            (resume_systeme, int(top_k), profil_json),
        )
        session_id = int(cur.lastrowid)

        lignes: List[Tuple[Any, ...]] = []
        for idx, reco in enumerate(items, start=1):
            lignes.append(
                (
                    session_id,
                    int(idx),
                    str(reco.get("code_filiere", "")),
                    str(reco.get("filiere", "")),
                    str(reco.get("domaine", "")),
                    float(reco.get("score_final", 0.0)),
                    float(reco.get("score_regles", 0.0)),
                    float(reco.get("score_graphe", 0.0)),
                    float(reco.get("score_cnn", 0.0)),
                    str(reco.get("explication", "")),
                )
            )

        conn.executemany(
            """
            INSERT INTO recommandation_item (
              session_id, rang, code_filiere, filiere, domaine,
              score_final, score_regles, score_graphe, score_cnn, explication
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            lignes,
        )
        return session_id


def lister_sessions(limit: int = 50, db_path: Path = CHEMIN_BD) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        lignes = conn.execute(
            """
            SELECT id, created_at, resume_systeme, top_k, profil_json
            FROM recommandation_session
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    sessions: List[Dict[str, Any]] = []
    for lig in lignes:
        sessions.append(
            {
                "id": int(lig["id"]),
                "created_at": str(lig["created_at"]),
                "resume_systeme": str(lig["resume_systeme"]),
                "top_k": int(lig["top_k"]),
                "profil": json.loads(str(lig["profil_json"])) if lig["profil_json"] else {},
            }
        )
    return sessions


def lister_items(session_id: int, db_path: Path = CHEMIN_BD) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        lignes = conn.execute(
            """
            SELECT rang, code_filiere, filiere, domaine,
                   score_final, score_regles, score_graphe, score_cnn, explication
            FROM recommandation_item
            WHERE session_id = ?
            ORDER BY rang ASC
            """,
            (int(session_id),),
        ).fetchall()
    items: List[Dict[str, Any]] = []
    for lig in lignes:
        items.append(
            {
                "rang": int(lig["rang"]),
                "code_filiere": str(lig["code_filiere"]),
                "filiere": str(lig["filiere"]),
                "domaine": str(lig["domaine"]),
                "score_final": float(lig["score_final"]),
                "score_regles": float(lig["score_regles"]),
                "score_graphe": float(lig["score_graphe"]),
                "score_cnn": float(lig["score_cnn"]),
                "explication": str(lig["explication"]),
            }
        )
    return items
