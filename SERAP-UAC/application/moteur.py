from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class Recommandation:
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


@dataclass(frozen=True)
class Resultat:
    resume_systeme: str
    top_k: list[Recommandation]


_COND_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*(>=|<=|==|!=|>|<)\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _score_from_20(v: float) -> float:
    return _clamp01(float(v) / 20.0)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _parse_condition(expr: str) -> tuple[str, str, float] | None:
    m = _COND_RE.match(expr or "")
    if not m:
        return None
    return m.group(1), m.group(2), float(m.group(3))


def _eval_condition(var: str, op: str, rhs: float, profil: dict[str, Any]) -> bool:
    lhs = _safe_float(profil.get(var))
    if lhs is None:
        return False
    if op == ">=":
        return lhs >= rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">":
        return lhs > rhs
    if op == "<":
        return lhs < rhs
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    return False


class MoteurSERAP:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.data_dir = base_dir / "donnees"
        self.model_dir = base_dir / "modeles"

        self.exigences = pd.read_csv(self.data_dir / "exigences_filieres.csv")
        self.regles = pd.read_csv(self.data_dir / "regles_expertes.csv")
        with open(self.data_dir / "programmes_uac.json", "r", encoding="utf-8") as f:
            self.programmes = json.load(f)

        # Normalisation types
        self.exigences["obligatoire"] = self.exigences["obligatoire"].astype(int)
        self.exigences["poids"] = self.exigences["poids"].astype(float)
        self.exigences["seuil_minimal"] = self.exigences["seuil_minimal"].astype(float)

    def competences_disponibles(self) -> list[str]:
        return sorted(set(self.exigences["competence"].astype(str).tolist()))

    def filieres_par_cycle(self, cycle: str) -> pd.DataFrame:
        df = self.exigences[self.exigences["cycle"] == cycle].copy()
        return df

    def recommander(self, profil: dict[str, Any], top_k: int = 5) -> Resultat:
        cycle = str(profil.get("cycle") or "Licence")
        df_cycle = self.filieres_par_cycle(cycle)
        if df_cycle.empty:
            return Resultat(
                resume_systeme=f"Aucune filière configurée pour le cycle '{cycle}'.",
                top_k=[],
            )

        regroup = (
            df_cycle.groupby(["code_filiere", "filiere", "domaine", "cycle"], dropna=False)
            .apply(lambda x: x.to_dict(orient="records"))
            .to_dict()
        )

        recos: list[Recommandation] = []
        for (code, filiere, domaine, cyc), reqs in regroup.items():
            score_exigences, exp_exig = self._score_exigences(reqs, profil)
            score_regles, exp_reg = self._score_regles(code, profil)
            score_graphe = self._score_graphe(reqs, profil)
            score_cnn = self._score_cnn(reqs, profil)

            # Pondération simple, somme = 1
            w_ex, w_reg, w_graph, w_cnn = 0.6, 0.2, 0.15, 0.05
            score_final = _clamp01(
                w_ex * score_exigences
                + w_reg * score_regles
                + w_graph * score_graphe
                + w_cnn * score_cnn
            )

            explication = " ".join([s for s in [exp_exig, exp_reg] if s]).strip()
            if not explication:
                explication = "Recommandation basée sur l'adéquation globale du profil."

            recos.append(
                Recommandation(
                    rang=0,
                    code_filiere=str(code),
                    filiere=str(filiere),
                    domaine=str(domaine),
                    cycle=str(cyc),
                    score_final=float(score_final),
                    score_regles=float(score_regles),
                    score_graphe=float(score_graphe),
                    score_cnn=float(score_cnn),
                    explication=explication,
                )
            )

        recos.sort(key=lambda r: r.score_final, reverse=True)
        recos = recos[: int(top_k)]
        ranked: list[Recommandation] = []
        for i, r in enumerate(recos, start=1):
            ranked.append(
                Recommandation(
                    rang=i,
                    code_filiere=r.code_filiere,
                    filiere=r.filiere,
                    domaine=r.domaine,
                    cycle=r.cycle,
                    score_final=r.score_final,
                    score_regles=r.score_regles,
                    score_graphe=r.score_graphe,
                    score_cnn=r.score_cnn,
                    explication=r.explication,
                )
            )

        resume = (
            "Fusion exigences (pondérées), règles expertes et similarité de compétences "
            f"— cycle: {cycle}, top_k: {top_k}."
        )
        return Resultat(resume_systeme=resume, top_k=ranked)

    def _score_exigences(
        self, reqs: list[dict[str, Any]], profil: dict[str, Any]
    ) -> tuple[float, str]:
        total_w = 0.0
        acc = 0.0
        obligatoires_manquants = 0
        obligatoires_sous_seuil = 0

        for r in reqs:
            comp = str(r["competence"])
            poids = float(r["poids"])
            obligatoire = int(r["obligatoire"]) == 1
            seuil = float(r["seuil_minimal"])
            v = _safe_float(profil.get(comp))

            if v is None:
                if obligatoire:
                    obligatoires_manquants += 1
                continue

            if obligatoire and v < seuil:
                obligatoires_sous_seuil += 1

            s = _score_from_20(v)
            total_w += poids
            acc += poids * s

        if total_w <= 0:
            base = 0.0
        else:
            base = acc / total_w

        # pénalité douce sur obligations manquantes/sous seuil
        penalty = 0.0
        penalty += 0.08 * obligatoires_manquants
        penalty += 0.10 * obligatoires_sous_seuil
        score = _clamp01(base - penalty)

        exp = []
        if obligatoires_manquants:
            exp.append(f"{obligatoires_manquants} compétence(s) obligatoire(s) non renseignée(s).")
        if obligatoires_sous_seuil:
            exp.append(f"{obligatoires_sous_seuil} compétence(s) obligatoire(s) sous le seuil minimal.")
        return score, " ".join(exp)

    def _score_regles(self, code_filiere: str, profil: dict[str, Any]) -> tuple[float, str]:
        df = self.regles[self.regles["code_filiere"] == code_filiere].copy()
        if df.empty:
            return 0.0, ""

        df["priorite"] = df["priorite"].astype(int)
        df = df.sort_values(["priorite", "code_regle"])

        score = 0.0
        explications: list[str] = []

        for _, r in df.iterrows():
            cond = _parse_condition(str(r["si"]))
            if not cond:
                continue
            var, op, rhs = cond
            if not _eval_condition(var, op, rhs, profil):
                continue
            action = str(r["alors"])
            typ = str(r["type_regle"])
            explications.append(str(r["explication"]))
            if "augmenter_score_symbolique" in action:
                score += 0.15 if typ == "admissibilite" else 0.10
            elif "reduire_score_symbolique" in action:
                score -= 0.15 if typ == "exclusion_relative" else 0.10

        return _clamp01(0.5 + score), " ".join(explications[:2])

    def _score_graphe(self, reqs: list[dict[str, Any]], profil: dict[str, Any]) -> float:
        # Similarité simple: proportion des compétences requises renseignées
        comps = [str(r["competence"]) for r in reqs]
        if not comps:
            return 0.0
        known = sum(1 for c in comps if _safe_float(profil.get(c)) is not None)
        return _clamp01(known / float(len(comps)))

    def _score_cnn(self, reqs: list[dict[str, Any]], profil: dict[str, Any]) -> float:
        # Placeholder: si aucun modèle disponible, retourner un score neutre.
        # (Les fichiers du modèle CNN sont attendus dans `modeles/`.)
        _ = reqs, profil
        return 0.5


def libelles_par_defaut(competences: Iterable[str]) -> dict[str, str]:
    mapping = {
        "francais": "Français",
        "communication": "Communication",
        "interet_social": "Intérêt social",
        "empathie": "Empathie",
        "logique": "Logique",
        "informatique": "Informatique",
        "economie": "Économie",
        "anglais": "Anglais",
        "interet_affaires": "Intérêt pour les affaires",
        "leadership": "Leadership",
        "terrain": "Aptitude terrain",
        "biologie": "Biologie",
        "interet_environnement": "Intérêt environnement",
        "dessin": "Dessin",
        "mathematiques": "Mathématiques",
        "physique": "Physique",
        "interet_technologie": "Intérêt technologie",
        "gestion_projet": "Gestion de projet",
        "reseau": "Réseau",
        "entrepreneuriat": "Entrepreneuriat",
        "education": "Éducation",
        "journalisme": "Journalisme",
        "entreprise_agricole": "Entreprise agricole",
        "environnement": "Environnement",
        "architecture": "Architecture",
        "genie_civil": "Génie civil",
        "electronique": "Électronique",
        "programmation": "Programmation",
        "intelligence_artificielle": "Intelligence artificielle",
        "psychologie_clinique": "Psychologie clinique",
        "recherche": "Recherche",
        "innovation": "Innovation",
        "econometrie": "Économétrie",
        "psychologie": "Psychologie",
    }
    return {c: mapping.get(c, c.replace("_", " ").capitalize()) for c in competences}


def groupes_par_defaut(competences: list[str]) -> dict[str, list[str]]:
    acad = []
    transv = []
    interets = []

    for c in competences:
        if c in {"francais", "anglais", "mathematiques", "physique", "biologie", "economie", "informatique", "logique"}:
            acad.append(c)
        elif c.startswith("interet_") or c in {"terrain", "environnement", "architecture", "electronique", "reseau", "programmation", "intelligence_artificielle"}:
            interets.append(c)
        else:
            transv.append(c)

    return {
        "Compétences académiques": sorted(acad),
        "Compétences transversales": sorted(transv),
        "Centres d'intérêt / Spécialités": sorted(interets),
    }

