from __future__ import annotations

import hashlib
import json
import secrets
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

CREATE TABLE IF NOT EXISTS utilisateur (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  nom TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  mot_de_passe_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'etudiant',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS session_utilisateur (
  token TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (user_id) REFERENCES utilisateur(id)
);
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
        _creer_admin_defaut(conn)


def _hash_mot_de_passe(mot_de_passe: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", mot_de_passe.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"{salt}${digest.hex()}"


def _verifier_mot_de_passe(mot_de_passe: str, valeur_hash: str) -> bool:
    try:
        salt, _ = valeur_hash.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(_hash_mot_de_passe(mot_de_passe, salt), valeur_hash)


def _creer_admin_defaut(conn: sqlite3.Connection) -> None:
    existe = conn.execute("SELECT id FROM utilisateur WHERE role = 'admin' LIMIT 1").fetchone()
    if existe:
        return
    conn.execute(
        "INSERT INTO utilisateur (nom, email, mot_de_passe_hash, role) VALUES (?, ?, ?, ?)",
        ("Administrateur", "admin@serap-uac.local", _hash_mot_de_passe("admin123"), "admin"),
    )


def creer_utilisateur(nom: str, email: str, mot_de_passe: str, role: str = "etudiant", db_path: Path = CHEMIN_BD) -> Tuple[bool, str]:
    role = role if role in {"etudiant", "admin"} else "etudiant"
    with _connect(db_path) as conn:
        existe = conn.execute("SELECT id FROM utilisateur WHERE lower(email) = lower(?)", (email,)).fetchone()
        if existe:
            return False, "Un compte existe deja avec cette adresse email."
        conn.execute(
            "INSERT INTO utilisateur (nom, email, mot_de_passe_hash, role) VALUES (?, ?, ?, ?)",
            (nom.strip(), email.strip().lower(), _hash_mot_de_passe(mot_de_passe), role),
        )
    return True, "Compte cree avec succes."


def authentifier(email: str, mot_de_passe: str, db_path: Path = CHEMIN_BD) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as conn:
        ligne = conn.execute(
            "SELECT id, nom, email, mot_de_passe_hash, role FROM utilisateur WHERE lower(email) = lower(?)",
            (email.strip(),),
        ).fetchone()
        if not ligne or not _verifier_mot_de_passe(mot_de_passe, str(ligne["mot_de_passe_hash"])):
            return None
        return {"id": int(ligne["id"]), "nom": str(ligne["nom"]), "email": str(ligne["email"]), "role": str(ligne["role"])}


def creer_session(user_id: int, db_path: Path = CHEMIN_BD) -> str:
    token = secrets.token_urlsafe(32)
    with _connect(db_path) as conn:
        conn.execute("INSERT INTO session_utilisateur (token, user_id) VALUES (?, ?)", (token, int(user_id)))
    return token


def utilisateur_par_token(token: str, db_path: Path = CHEMIN_BD) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with _connect(db_path) as conn:
        ligne = conn.execute(
            """
            SELECT u.id, u.nom, u.email, u.role
            FROM session_utilisateur s
            JOIN utilisateur u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if not ligne:
            return None
        return {"id": int(ligne["id"]), "nom": str(ligne["nom"]), "email": str(ligne["email"]), "role": str(ligne["role"])}


def supprimer_session(token: str, db_path: Path = CHEMIN_BD) -> None:
    if not token:
        return
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM session_utilisateur WHERE token = ?", (token,))


def lister_utilisateurs(db_path: Path = CHEMIN_BD) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        lignes = conn.execute(
            """
            SELECT id, nom, email, role, created_at
            FROM utilisateur
            ORDER BY id DESC
            """
        ).fetchall()
    return [
        {
            "id": int(ligne["id"]),
            "nom": str(ligne["nom"]),
            "email": str(ligne["email"]),
            "role": str(ligne["role"]),
            "created_at": str(ligne["created_at"]),
        }
        for ligne in lignes
    ]


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
