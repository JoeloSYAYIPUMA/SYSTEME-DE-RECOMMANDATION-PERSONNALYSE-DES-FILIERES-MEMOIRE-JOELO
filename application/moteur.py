from __future__ import annotations  # Active les annotations différées pour les types.

from pathlib import Path  # Importe Path pour manipuler les chemins de fichiers.
from typing import Dict, List, Optional  # Importe les types utiles pour annoter le code.

import json  # Importe json pour lire les fichiers JSON.
import joblib  # Importe joblib pour charger les objets sérialisés.
import numpy as np  # Importe NumPy pour les tableaux numériques.
import pandas as pd  # Importe pandas pour lire les fichiers CSV.
# PyTorch est optionnel : en environnement Windows, le chargement des DLL peut échouer.
# Le système continue alors en "mode dégradé" (règles + graphe) sans CNN.
try:  # Tente d'importer PyTorch.
    import torch  # type: ignore  # Importe PyTorch pour charger le réseau neuronal.
    from torch import nn  # type: ignore  # Importe le module neural network de PyTorch.
    TORCH_DISPONIBLE = True  # Indique que PyTorch est utilisable.
except Exception:  # Couvre ImportError et erreurs de DLL (OSError/WinError).
    torch = None  # type: ignore[assignment]  # Rend la variable accessible sans PyTorch.
    nn = None  # type: ignore[assignment]  # Rend la variable accessible sans PyTorch.
    TORCH_DISPONIBLE = False  # Indique que PyTorch est indisponible.

BASE_DIR = Path(__file__).resolve().parents[1]  # Définit le dossier racine du projet.
DOSSIER_DONNEES = BASE_DIR / "donnees"  # Définit le dossier qui contient les données.
DOSSIER_MODELES = BASE_DIR / "modeles"  # Définit le dossier qui contient les modèles entraînés.

COLONNES_COMPETENCES = [  # Liste toutes les colonnes utilisées comme variables de profil.
    "francais",  # Représente le niveau ou la note en français.
    "anglais",  # Représente le niveau ou la note en anglais.
    "mathematiques",  # Représente le niveau ou la note en mathématiques.
    "physique",  # Représente le niveau ou la note en physique.
    "chimie",  # Représente le niveau ou la note en chimie.
    "biologie",  # Représente le niveau ou la note en biologie.
    "economie",  # Représente le niveau ou la note en économie.
    "informatique",  # Représente le niveau ou la note en informatique.
    "logique",  # Représente le niveau ou la note en logique.
    "dessin",  # Représente le niveau ou la note en dessin.
    "communication",  # Représente l’aptitude de communication.
    "leadership",  # Représente l’aptitude de leadership.
    "empathie",  # Représente l’aptitude relationnelle et empathique.
    "terrain",  # Représente l’aptitude de terrain ou d’intervention pratique.
    "interet_social",  # Représente l’intérêt pour le social.
    "interet_affaires",  # Représente l’intérêt pour les affaires et la gestion.
    "interet_environnement",  # Représente l’intérêt pour l’environnement.
    "interet_technologie",  # Représente l’intérêt pour la technologie.
]  # Ferme la liste des colonnes de compétences.


if TORCH_DISPONIBLE:  # Définit le réseau uniquement si PyTorch est disponible.

    class ReseauCNN1D(nn.Module):  # Définit le réseau neuronal convolutif 1D.
        def __init__(self, nb_variables: int, nb_classes: int) -> None:  # Initialise l’architecture du modèle.
            super().__init__()  # Appelle l’initialisation de la classe parente.
            self.extracteur = nn.Sequential(  # Définit la partie convolutionnelle du réseau.
                nn.Conv1d(1, 16, kernel_size=3, padding=1),  # Applique une première convolution 1D.
                nn.ReLU(),  # Applique une activation ReLU.
                nn.Conv1d(16, 32, kernel_size=3, padding=1),  # Applique une deuxième convolution 1D.
                nn.ReLU(),  # Applique une deuxième activation ReLU.
                nn.AdaptiveAvgPool1d(16),  # Réduit la dimension à une longueur fixe.
            )  # Ferme le bloc extracteur.
            self.classifieur = nn.Sequential(  # Définit la tête de classification.
                nn.Flatten(),  # Aplati les cartes de caractéristiques.
                nn.Linear(32 * 16, 128),  # Projette vers une couche dense de 128 neurones.
                nn.ReLU(),  # Applique une activation ReLU.
                nn.Dropout(0.20),  # Applique un dropout pour limiter le surapprentissage.
                nn.Linear(128, 64),  # Projette vers une couche dense de 64 neurones.
                nn.ReLU(),  # Applique une activation ReLU.
                nn.Linear(64, nb_classes),  # Produit les logits pour toutes les classes.
            )  # Ferme le bloc classifieur.

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # Définit la propagation avant.
            x = x.unsqueeze(1)  # Ajoute la dimension canal attendue par Conv1d.
            x = self.extracteur(x)  # Passe le tenseur dans l’extracteur convolutionnel.
            return self.classifieur(x)  # Retourne les logits produits par le classifieur.
else:

    class ReseauCNN1D:  # Type de secours quand PyTorch est indisponible.
        pass


class MoteurSERAP:  # Définit le moteur principal du système hybride.
    def __init__(self) -> None:  # Initialise le moteur complet.
        self.exigences = pd.read_csv(DOSSIER_DONNEES / "exigences_filieres.csv")  # Charge les exigences des filières.
        self.regles_expertes = pd.read_csv(DOSSIER_DONNEES / "regles_expertes.csv")  # Charge le catalogue des règles expertes.
        self.exigences_par_code = {code: sous.copy() for code, sous in self.exigences.groupby("code_filiere")}  # Pré-indexe les exigences par filière pour accélérer les calculs.
        self.programmes = self._charger_programmes()  # Charge la structure publique des programmes UAC.
        self.colonnes_scores = COLONNES_COMPETENCES.copy()  # Copie la liste des colonnes de compétences.
        self.normaliseur = None  # Initialise le normaliseur à None.
        self.classes_modele: List[str] = []  # Initialise la liste des classes du modèle.
        self.modele_cnn: Optional[ReseauCNN1D] = None  # Initialise le modèle CNN à None.
        self.modele_charge = False  # Indique que le modèle n’est pas encore chargé.
        self.alpha = 0.40  # Initialise le poids des règles.
        self.beta = 0.25  # Initialise le poids du graphe.
        self.gamma = 0.35  # Initialise le poids du CNN.
        self._charger_modele_si_disponible()  # Tente de charger le modèle entraîné.

    def _charger_programmes(self) -> Dict:  # Charge le fichier JSON des programmes UAC.
        chemin = DOSSIER_DONNEES / "programmes_uac_publics.json"  # Définit le chemin du JSON.
        if not chemin.exists():  # Vérifie si le fichier existe.
            return {}  # Retourne un dictionnaire vide s’il est absent.
        return json.loads(chemin.read_text(encoding="utf-8"))  # Lit et décode le fichier JSON.

    def _charger_modele_si_disponible(self) -> None:  # Charge le modèle neuronal et les métadonnées si présents.
        if not TORCH_DISPONIBLE:  # Vérifie si PyTorch est utilisable dans cet environnement.
            return  # Ne tente pas de charger le modèle si PyTorch ne se charge pas.
        chemin_modele = DOSSIER_MODELES / "modele_cnn_serap.pt"  # Définit le chemin du modèle PyTorch.
        chemin_meta = DOSSIER_MODELES / "pretraitement_serap.joblib"  # Définit le chemin des métadonnées.
        if not chemin_modele.exists() or not chemin_meta.exists():  # Vérifie l’existence des deux fichiers.
            return  # Arrête la fonction si les fichiers n’existent pas.
        meta = joblib.load(chemin_meta)  # Charge les métadonnées du modèle.
        paquet = torch.load(chemin_modele, map_location="cpu")  # Charge l’état du réseau sur le CPU.
        self.normaliseur = meta.get("normaliseur")  # Récupère le normaliseur.
        self.classes_modele = list(meta.get("classes", []))  # Récupère la liste des classes.
        self.colonnes_scores = list(meta.get("colonnes_scores", COLONNES_COMPETENCES))  # Récupère les colonnes de scores.
        self.alpha = float(meta.get("alpha", self.alpha))  # Récupère le poids alpha.
        self.beta = float(meta.get("beta", self.beta))  # Récupère le poids beta.
        self.gamma = float(meta.get("gamma", self.gamma))  # Récupère le poids gamma.
        self.modele_cnn = ReseauCNN1D(nb_variables=int(paquet["nb_variables"]), nb_classes=int(paquet["nb_classes"]))  # Reconstruit l’architecture du réseau.
        self.modele_cnn.load_state_dict(paquet["etat_modele"])  # Charge les poids du réseau.
        self.modele_cnn.eval()  # Place le réseau en mode évaluation.
        self.modele_charge = True  # Active le drapeau de modèle chargé.

    def score_regles(self, profil: Dict[str, Optional[float]], code_filiere: str) -> float:  # Calcule le score symbolique d’une filière.
        sous = self.exigences_par_code.get(code_filiere)  # Récupère directement les exigences de la filière.
        if sous is None or sous.empty:  # Vérifie si aucune exigence n’existe.
            return 0.0  # Retourne zéro si la filière est introuvable.
        total = 0.0  # Initialise la somme pondérée.
        poids_total = 0.0  # Initialise la somme des poids.
        for _, lig in sous.iterrows():  # Parcourt chaque exigence de la filière.
            competence = str(lig["competence"])  # Récupère le nom de la compétence.
            seuil = float(lig["seuil_minimal"])  # Récupère le seuil minimal attendu.
            poids = float(lig["poids"])  # Récupère le poids de la compétence.
            obligatoire = int(lig["obligatoire"])  # Récupère le caractère obligatoire.
            valeur = profil.get(competence, None)  # Récupère la valeur de la compétence dans le profil.
            if valeur in (None, "") or pd.isna(valeur):  # Vérifie si la valeur n’est pas observée.
                score = 0.45 if obligatoire == 0 else 0.25  # Affecte une pénalité douce pour le non observé.
            elif float(valeur) < max(seuil - 4.0, 0.0):  # Vérifie si la valeur est très inférieure au seuil.
                score = max(0.05, float(valeur) / max(seuil, 1.0))  # Attribue un score faible mais non nul.
            else:  # Couvre le cas d’une valeur acceptable ou bonne.
                score = min(1.20, float(valeur) / max(seuil, 1.0))  # Transforme la valeur en score relatif plafonné.
            total += score * poids  # Ajoute la contribution pondérée.
            poids_total += poids  # Ajoute le poids au total.
        return float(total / poids_total) if poids_total else 0.0  # Retourne le score moyen pondéré.

    def score_graphe(self, profil: Dict[str, Optional[float]], code_filiere: str) -> float:  # Calcule le score relationnel de graphe.
        sous = self.exigences_par_code.get(code_filiere)  # Récupère directement les exigences de la filière.
        numerateur = 0.0  # Initialise le numérateur du score.
        denominateur = 0.0  # Initialise le dénominateur du score.
        for _, lig in sous.iterrows():  # Parcourt chaque exigence.
            competence = str(lig["competence"])  # Récupère la compétence.
            poids = float(lig["poids"])  # Récupère le poids de la compétence.
            valeur = profil.get(competence, None)  # Récupère la valeur observée dans le profil.
            if valeur in (None, "") or pd.isna(valeur):  # Ignore les compétences non observées.
                continue  # Passe à la compétence suivante.
            numerateur += (float(valeur) / 20.0) * poids  # Ajoute la contribution normalisée.
            denominateur += poids  # Ajoute le poids au dénominateur.
        return float(numerateur / denominateur) if denominateur else 0.0  # Retourne la moyenne pondérée.

    def _vectoriser_profil(self, profil: Dict[str, Optional[float]]) -> np.ndarray:  # Transforme le profil en vecteur pour le CNN.
        if self.normaliseur is None:  # Vérifie que le normaliseur est chargé.
            raise RuntimeError("Le normaliseur est indisponible. Veuillez entraîner le modèle.")  # Lève une erreur explicite.
        scores = []  # Initialise la liste des scores bruts.
        masques = []  # Initialise la liste des indicateurs de disponibilité.
        for colonne in self.colonnes_scores:  # Parcourt toutes les colonnes de score.
            valeur = profil.get(colonne, None)  # Récupère la valeur fournie.
            if valeur in (None, "") or pd.isna(valeur):  # Vérifie si la valeur est absente.
                scores.append(0.0)  # Insère zéro comme valeur de remplacement.
                masques.append(0.0)  # Indique que la valeur est absente.
            else:  # Couvre le cas où la valeur est disponible.
                scores.append(float(valeur))  # Ajoute la valeur brute.
                masques.append(1.0)  # Indique que la valeur est disponible.
        scores = np.array(scores, dtype=np.float32).reshape(1, -1)  # Convertit la liste en tableau 2D.
        scores_df = pd.DataFrame(scores, columns=self.colonnes_scores)  # Reconstruit un DataFrame avec les noms de colonnes attendus.
        scores_std = self.normaliseur.transform(scores_df)  # Normalise les scores selon le prétraitement appris.
        masque_np = np.array(masques, dtype=np.float32).reshape(1, -1)  # Convertit le masque en tableau 2D.
        return np.concatenate([scores_std, masque_np], axis=1).astype(np.float32)  # Concatène scores normalisés et masque.

    def probabilites_cnn(self, profil: Dict[str, Optional[float]]) -> Dict[str, float]:  # Retourne les probabilités du CNN par filière.
        if not TORCH_DISPONIBLE:  # Vérifie si PyTorch est utilisable.
            return {}  # Retourne vide : pas de CNN.
        if not self.modele_charge or self.modele_cnn is None:  # Vérifie que le modèle est chargé.
            return {}  # Retourne un dictionnaire vide si le modèle est absent.
        x = self._vectoriser_profil(profil)  # Vectorise le profil pour le CNN.
        with torch.no_grad():  # Désactive le calcul de gradient pour l’inférence.
            logits = self.modele_cnn(torch.tensor(x, dtype=torch.float32))  # Calcule les logits du réseau.
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Convertit les logits en probabilités.
        return {code: float(score) for code, score in zip(self.classes_modele, proba)}  # Associe chaque probabilité à sa filière.

    def _competences_cles(self, code_filiere: str) -> List[str]:  # Retourne les compétences les plus importantes d’une filière.
        sous = self.exigences_par_code.get(code_filiere)  # Récupère directement les exigences de la filière.
        if sous.empty:  # Vérifie que la filière existe.
            return []  # Retourne une liste vide en cas d’absence.
        return list(sous.sort_values(["obligatoire", "poids"], ascending=[False, False])["competence"].head(3))  # Retourne les trois compétences majeures.

    def _expliquer_recommandation(self, profil: Dict[str, Optional[float]], ligne: Dict[str, float]) -> str:  # Construit une explication textuelle.
        code = ligne["code_filiere"]  # Récupère le code de la filière.
        filiere = ligne["filiere"]  # Récupère le nom de la filière.
        competences_cles = self._competences_cles(code)  # Récupère les compétences les plus importantes.
        fortes = []  # Initialise la liste des compétences fortes.
        fragiles = []  # Initialise la liste des compétences fragiles.
        manquantes = []  # Initialise la liste des compétences manquantes.
        sous = self.exigences_par_code.get(code)  # Récupère directement les exigences pour la filière.
        for _, lig in sous.iterrows():  # Parcourt les exigences de la filière.
            competence = str(lig["competence"])  # Récupère la compétence concernée.
            seuil = float(lig["seuil_minimal"])  # Récupère le seuil attendu.
            obligatoire = int(lig["obligatoire"])  # Récupère le caractère obligatoire.
            valeur = profil.get(competence, None)  # Récupère la valeur observée.
            if valeur in (None, "") or pd.isna(valeur):  # Vérifie si la donnée manque.
                if obligatoire == 1:  # Vérifie si la compétence est obligatoire.
                    manquantes.append(competence)  # Ajoute la compétence aux manquantes.
                continue  # Passe à l’exigence suivante.
            if float(valeur) >= seuil:  # Vérifie si la compétence atteint le seuil.
                fortes.append(f"{competence}={float(valeur):.1f}")  # Ajoute la compétence à la liste forte.
            elif obligatoire == 1:  # Couvre le cas d’une compétence obligatoire sous le seuil.
                fragiles.append(f"{competence}={float(valeur):.1f} < {seuil:.0f}")  # Ajoute la fragilité repérée.
        parties = []  # Initialise les segments d’explication.
        parties.append(f"{filiere} est proposée car les scores règles, graphe et CNN convergent vers cette filière.")  # Explique la convergence des trois modules.
        if competences_cles:  # Vérifie que les compétences clés existent.
            parties.append(f"Compétences clés visées : {', '.join(competences_cles)}.")  # Ajoute les compétences clés.
        if fortes:  # Vérifie s’il existe des points forts.
            parties.append(f"Atouts observés : {', '.join(fortes[:4])}.")  # Ajoute jusqu’à quatre atouts.
        if fragiles:  # Vérifie s’il existe des fragilités.
            parties.append(f"Points de vigilance : {', '.join(fragiles[:3])}.")  # Ajoute jusqu’à trois fragilités.
        if manquantes:  # Vérifie s’il existe des données manquantes importantes.
            parties.append(f"Données à compléter pour une meilleure précision : {', '.join(manquantes[:3])}.")  # Ajoute les données manquantes.
        parties.append(f"Contributions numériques : règles={ligne['score_regles']:.3f}, graphe={ligne['score_graphe']:.3f}, cnn={ligne['score_cnn']:.3f}, score final={ligne['score_final']:.3f}.")  # Ajoute les détails numériques.
        return " ".join(parties)  # Retourne l’explication complète.

    def recommander(self, profil: Dict[str, Optional[float]], top_k: int = 3) -> Dict[str, object]:  # Retourne les meilleures recommandations pour un profil.
        proba_cnn = self.probabilites_cnn(profil)  # Calcule les probabilités du CNN pour toutes les filières.
        lignes = []  # Initialise la liste des recommandations candidates.
        for code in self.exigences["code_filiere"].drop_duplicates():  # Parcourt toutes les filières connues.
            sous = self.exigences_par_code.get(code)  # Récupère directement les lignes de la filière.
            sr = self.score_regles(profil, code)  # Calcule le score des règles.
            sg = self.score_graphe(profil, code)  # Calcule le score du graphe.
            scnn = float(proba_cnn.get(code, 0.0))  # Récupère la probabilité du CNN.
            score_final = self.alpha * sr + self.beta * sg + self.gamma * scnn if self.modele_charge else 0.60 * sr + 0.40 * sg  # Réalise l’hybridation complète ou le mode dégradé.
            ligne = {  # Construit l’objet recommandation brut.
                "code_filiere": code,  # Sauvegarde le code de la filière.
                "filiere": str(sous["filiere"].iloc[0]),  # Sauvegarde le nom de la filière.
                "domaine": str(sous["domaine"].iloc[0]),  # Sauvegarde le domaine de la filière.
                "score_final": float(score_final),  # Sauvegarde le score final.
                "score_regles": float(sr),  # Sauvegarde le score des règles.
                "score_graphe": float(sg),  # Sauvegarde le score du graphe.
                "score_cnn": float(scnn),  # Sauvegarde le score du CNN.
            }  # Ferme le dictionnaire temporaire.
            ligne["explication"] = self._expliquer_recommandation(profil, ligne)  # Ajoute l’explication complète.
            lignes.append(ligne)  # Ajoute la ligne à la liste globale.
        lignes = sorted(lignes, key=lambda x: x["score_final"], reverse=True)[:top_k]  # Trie et garde les top-k meilleures filières.
        resume = "Hybridation active avec règles + graphe + CNN." if self.modele_charge else "Mode dégradé : seuls les modules règles et graphe sont actifs."  # Prépare le résumé du mode utilisé.
        resume += f" Pondérations actuelles : alpha={self.alpha:.2f}, beta={self.beta:.2f}, gamma={self.gamma:.2f}."  # Ajoute les pondérations.
        return {"top_k": lignes, "resume_systeme": resume}  # Retourne les recommandations et le résumé système.
