from __future__ import annotations  # Active les annotations différées.

import json  # Importe json pour écrire le résumé final.
import os  # Importe os pour contrôler certains paramètres d’exécution.
import random  # Importe random pour fixer la reproductibilité.
import sys  # Importe sys pour ajuster le chemin d’import.
from pathlib import Path  # Importe Path pour manipuler les chemins.
from typing import Dict, List, Tuple  # Importe les types utiles pour l’annotation.

os.environ.setdefault("OMP_NUM_THREADS", "1")  # Limite les threads OpenMP pour éviter les blocages CPU.
os.environ.setdefault("MKL_NUM_THREADS", "1")  # Limite aussi les threads MKL.

import joblib  # Importe joblib pour sauvegarder les objets Python.
import matplotlib.pyplot as plt  # Importe matplotlib pour générer les figures.
import numpy as np  # Importe NumPy pour les calculs numériques.
import pandas as pd  # Importe pandas pour manipuler les tableaux.
import torch  # Importe PyTorch pour le CNN.
from sklearn.ensemble import RandomForestClassifier  # Importe la baseline Random Forest.
from sklearn.linear_model import LogisticRegression  # Importe la baseline Régression Logistique.
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # Importe les métriques d’évaluation.
from sklearn.model_selection import train_test_split  # Importe la fonction de découpage train/val/test.
from sklearn.neural_network import MLPClassifier  # Importe la baseline MLP.
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Importe l’encodeur d’étiquettes et le normaliseur.
from torch import nn  # Importe le module neural network.
from torch.utils.data import DataLoader, TensorDataset  # Importe les utilitaires de chargement de données.

torch.set_num_threads(1)  # Fixe explicitement le nombre de threads PyTorch à 1.

BASE_DIR = Path(__file__).resolve().parents[1]  # Définit le dossier racine du projet.
sys.path.insert(0, str(BASE_DIR))  # Ajoute la racine au chemin d’import pour charger l’application.

from application.moteur import COLONNES_COMPETENCES, MoteurSERAP, ReseauCNN1D  # Importe les briques métier partagées.

DOSSIER_DONNEES = BASE_DIR / "donnees"  # Définit le dossier de données.
DOSSIER_MODELES = BASE_DIR / "modeles"  # Définit le dossier des modèles.
DOSSIER_SORTIES = BASE_DIR / "sorties"  # Définit le dossier des sorties.
DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)  # Crée le dossier des modèles si besoin.
DOSSIER_SORTIES.mkdir(parents=True, exist_ok=True)  # Crée le dossier des sorties si besoin.

SEED = 42  # Définit la graine aléatoire globale.

def fixer_graine(seed: int = SEED) -> None:  # Définit les graines aléatoires.
    random.seed(seed)  # Fixe la graine Python.
    np.random.seed(seed)  # Fixe la graine NumPy.
    torch.manual_seed(seed)  # Fixe la graine PyTorch CPU.
    if torch.cuda.is_available():  # Vérifie si CUDA est disponible.
        torch.cuda.manual_seed_all(seed)  # Fixe aussi les graines GPU.

def charger_profils() -> pd.DataFrame:  # Charge le jeu de données générique.
    return pd.read_csv(DOSSIER_DONNEES / "profils_etudiants_generiques.csv")  # Retourne le DataFrame des profils.

def preparer_jeux(df: pd.DataFrame) -> Dict[str, object]:  # Prépare train, validation et test.
    colonnes_scores = [c for c in COLONNES_COMPETENCES if c in df.columns]  # Sélectionne les colonnes présentes.
    X_scores = df[colonnes_scores].astype(float)  # Extrait les scores numériques bruts.
    X_masques = (~X_scores.isna()).astype(np.float32)  # Crée un masque binaire de disponibilité.
    y_textes = df["code_filiere_reelle"].astype(str).values  # Extrait les labels textuels.
    encodeur = LabelEncoder()  # Initialise l’encodeur de classes.
    y = encodeur.fit_transform(y_textes)  # Encode les classes en entiers.
    X_train_scores, X_temp_scores, X_train_mask, X_temp_mask, y_train, y_temp, idx_train, idx_temp = train_test_split(  # Découpe train et temporaire.
        X_scores,  # Passe les scores bruts.
        X_masques,  # Passe les masques.
        y,  # Passe les labels encodés.
        np.arange(len(df)),  # Passe les indices de lignes.
        test_size=0.40,  # Réserve 40 % pour validation + test.
        random_state=SEED,  # Fixe la reproductibilité du split.
        stratify=y,  # Maintient la distribution des classes.
    )  # Ferme le premier split.
    X_val_scores, X_test_scores, X_val_mask, X_test_mask, y_val, y_test, idx_val, idx_test = train_test_split(  # Découpe validation et test.
        X_temp_scores,  # Passe les scores temporaires.
        X_temp_mask,  # Passe les masques temporaires.
        y_temp,  # Passe les labels temporaires.
        idx_temp,  # Passe les indices temporaires.
        test_size=0.50,  # Coupe moitié validation et moitié test.
        random_state=SEED,  # Fixe la reproductibilité.
        stratify=y_temp,  # Maintient la distribution des classes.
    )  # Ferme le deuxième split.
    normaliseur = StandardScaler()  # Initialise le standardiseur.
    X_train_std = normaliseur.fit_transform(X_train_scores.fillna(0.0))  # Ajuste le standardiseur sur train.
    X_val_std = normaliseur.transform(X_val_scores.fillna(0.0))  # Transforme la validation.
    X_test_std = normaliseur.transform(X_test_scores.fillna(0.0))  # Transforme le test.
    X_train = np.concatenate([X_train_std, X_train_mask.values], axis=1).astype(np.float32)  # Concatène scores et masques pour train.
    X_val = np.concatenate([X_val_std, X_val_mask.values], axis=1).astype(np.float32)  # Concatène scores et masques pour validation.
    X_test = np.concatenate([X_test_std, X_test_mask.values], axis=1).astype(np.float32)  # Concatène scores et masques pour test.
    return {  # Retourne tous les objets préparés.
        "X_train": X_train,  # Sauvegarde X_train.
        "X_val": X_val,  # Sauvegarde X_val.
        "X_test": X_test,  # Sauvegarde X_test.
        "y_train": y_train.astype(np.int64),  # Sauvegarde y_train.
        "y_val": y_val.astype(np.int64),  # Sauvegarde y_val.
        "y_test": y_test.astype(np.int64),  # Sauvegarde y_test.
        "profils_val": df.iloc[idx_val].reset_index(drop=True),  # Sauvegarde les profils de validation.
        "profils_test": df.iloc[idx_test].reset_index(drop=True),  # Sauvegarde les profils de test.
        "encodeur": encodeur,  # Sauvegarde l’encodeur.
        "normaliseur": normaliseur,  # Sauvegarde le normaliseur.
        "colonnes_scores": colonnes_scores,  # Sauvegarde la liste des colonnes de score.
    }  # Ferme le dictionnaire.

def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:  # Calcule les métriques principales.
    return {  # Retourne un dictionnaire de métriques.
        "accuracy": float(accuracy_score(y_vrai, y_pred)),  # Calcule l’accuracy.
        "precision_macro": float(precision_score(y_vrai, y_pred, average="macro", zero_division=0)),  # Calcule la précision macro.
        "recall_macro": float(recall_score(y_vrai, y_pred, average="macro", zero_division=0)),  # Calcule le rappel macro.
        "f1_macro": float(f1_score(y_vrai, y_pred, average="macro", zero_division=0)),  # Calcule la F1 macro.
    }  # Ferme le dictionnaire de métriques.

def entrainer_baselines(paquets: Dict[str, object]) -> Dict[str, Dict[str, float]]:  # Entraîne les baselines classiques.
    resultats: Dict[str, Dict[str, float]] = {}  # Initialise le dictionnaire de résultats.
    logreg = LogisticRegression(max_iter=1500, random_state=SEED)  # Crée la régression logistique.
    logreg.fit(paquets["X_train"], paquets["y_train"])  # Entraîne la régression logistique.
    pred_logreg = logreg.predict(paquets["X_test"])  # Prédit sur le jeu de test.
    resultats["regression_logistique"] = calculer_metriques(paquets["y_test"], pred_logreg)  # Enregistre les métriques.
    joblib.dump(logreg, DOSSIER_MODELES / "modele_logistique.joblib")  # Sauvegarde le modèle logistique.
    rf = RandomForestClassifier(n_estimators=250, random_state=SEED)  # Crée la forêt aléatoire.
    rf.fit(paquets["X_train"], paquets["y_train"])  # Entraîne la forêt.
    pred_rf = rf.predict(paquets["X_test"])  # Prédit sur le test.
    resultats["random_forest"] = calculer_metriques(paquets["y_test"], pred_rf)  # Enregistre les métriques.
    joblib.dump(rf, DOSSIER_MODELES / "modele_random_forest.joblib")  # Sauvegarde la forêt.
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)  # Crée le MLP classique.
    mlp.fit(paquets["X_train"], paquets["y_train"])  # Entraîne le MLP.
    pred_mlp = mlp.predict(paquets["X_test"])  # Prédit sur le test.
    resultats["mlp"] = calculer_metriques(paquets["y_test"], pred_mlp)  # Enregistre les métriques.
    joblib.dump(mlp, DOSSIER_MODELES / "modele_mlp.joblib")  # Sauvegarde le MLP.
    return resultats  # Retourne les résultats des baselines.

def entrainer_cnn(paquets: Dict[str, object], nb_epoques: int = 25, patience: int = 6) -> Tuple[ReseauCNN1D, Dict[str, List[float]]]:  # Entraîne le CNN 1D.
    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sélectionne CPU ou GPU.
    modele = ReseauCNN1D(nb_variables=int(paquets["X_train"].shape[1]), nb_classes=int(len(paquets["encodeur"].classes_))).to(appareil)  # Instancie le CNN.
    train_loader = DataLoader(TensorDataset(torch.tensor(paquets["X_train"], dtype=torch.float32), torch.tensor(paquets["y_train"], dtype=torch.long)), batch_size=32, shuffle=True)  # Crée le DataLoader train.
    val_loader = DataLoader(TensorDataset(torch.tensor(paquets["X_val"], dtype=torch.float32), torch.tensor(paquets["y_val"], dtype=torch.long)), batch_size=64, shuffle=False)  # Crée le DataLoader validation.
    perte_fn = nn.CrossEntropyLoss()  # Définit la fonction de perte.
    optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)  # Définit l’optimiseur Adam.
    historique = {"perte_train": [], "perte_val": [], "accuracy_val": []}  # Initialise l’historique d’entraînement.
    meilleur_etat = None  # Initialise le meilleur état du modèle.
    meilleure_perte = float("inf")  # Initialise la meilleure perte observée.
    attente = 0  # Initialise le compteur d’arrêt anticipé.
    for _ in range(nb_epoques):  # Parcourt les époques d’entraînement.
        modele.train()  # Passe le modèle en mode entraînement.
        pertes_train = []  # Initialise la liste des pertes train de l’époque.
        for xb, yb in train_loader:  # Parcourt les mini-lots de train.
            xb = xb.to(appareil)  # Envoie X sur le bon appareil.
            yb = yb.to(appareil)  # Envoie y sur le bon appareil.
            optimiseur.zero_grad()  # Réinitialise les gradients.
            logits = modele(xb)  # Calcule les logits du batch.
            perte = perte_fn(logits, yb)  # Calcule la perte du batch.
            perte.backward()  # Propage les gradients.
            optimiseur.step()  # Met à jour les poids du modèle.
            pertes_train.append(float(perte.item()))  # Enregistre la perte du batch.
        modele.eval()  # Passe le modèle en mode évaluation.
        pertes_val = []  # Initialise la liste des pertes validation.
        y_pred = []  # Initialise la liste des prédictions validation.
        y_reel = []  # Initialise la liste des labels validation.
        with torch.no_grad():  # Désactive le calcul de gradient en validation.
            for xb, yb in val_loader:  # Parcourt les mini-lots de validation.
                xb = xb.to(appareil)  # Envoie X validation sur l’appareil.
                yb = yb.to(appareil)  # Envoie y validation sur l’appareil.
                logits = modele(xb)  # Calcule les logits validation.
                pertes_val.append(float(perte_fn(logits, yb).item()))  # Enregistre la perte validation.
                y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())  # Enregistre les prédictions validation.
                y_reel.extend(yb.cpu().numpy().tolist())  # Enregistre les labels validation.
        perte_train = float(np.mean(pertes_train)) if pertes_train else 0.0  # Calcule la perte moyenne train.
        perte_val = float(np.mean(pertes_val)) if pertes_val else 0.0  # Calcule la perte moyenne validation.
        acc_val = float(accuracy_score(y_reel, y_pred))  # Calcule l’accuracy de validation.
        historique["perte_train"].append(perte_train)  # Ajoute la perte train à l’historique.
        historique["perte_val"].append(perte_val)  # Ajoute la perte validation à l’historique.
        historique["accuracy_val"].append(acc_val)  # Ajoute l’accuracy validation à l’historique.
        if perte_val < meilleure_perte:  # Vérifie si la perte validation s’améliore.
            meilleure_perte = perte_val  # Met à jour la meilleure perte.
            meilleur_etat = {k: v.cpu().clone() for k, v in modele.state_dict().items()}  # Copie l’état du meilleur modèle.
            attente = 0  # Réinitialise le compteur d’attente.
        else:  # Couvre le cas sans amélioration.
            attente += 1  # Incrémente le compteur d’attente.
            if attente >= patience:  # Vérifie le critère d’arrêt anticipé.
                break  # Interrompt l’entraînement.
    if meilleur_etat is not None:  # Vérifie qu’un meilleur état a été capturé.
        modele.load_state_dict(meilleur_etat)  # Recharge le meilleur état observé.
    return modele, historique  # Retourne le modèle entraîné et l’historique.

def probabilites_cnn(modele: ReseauCNN1D, X: np.ndarray) -> np.ndarray:  # Produit les probabilités du CNN.
    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sélectionne l’appareil.
    modele.eval()  # Passe le modèle en mode évaluation.
    with torch.no_grad():  # Désactive les gradients pour l’inférence.
        logits = modele(torch.tensor(X, dtype=torch.float32).to(appareil))  # Calcule les logits.
        return torch.softmax(logits, dim=1).cpu().numpy()  # Retourne les probabilités.

def top3_accuracy(probabilites: np.ndarray, y_vrai: np.ndarray) -> float:  # Calcule la Top-3 accuracy.
    top3 = np.argsort(probabilites, axis=1)[:, -3:]  # Extrait les trois classes les plus probables.
    succes = [(y_vrai[i] in top3[i]) for i in range(len(y_vrai))]  # Vérifie si la classe vraie est dans le top 3.
    return float(np.mean(succes))  # Retourne la moyenne de succès.

def coherence_logique(moteur: MoteurSERAP, profils: pd.DataFrame, classes: List[str], pred_indices: np.ndarray) -> float:  # Mesure la cohérence logique.
    ratios = []  # Initialise la liste des ratios de cohérence.
    for i, (_, ligne) in enumerate(profils.iterrows()):  # Parcourt les profils.
        profil = {c: (None if pd.isna(ligne[c]) else float(ligne[c])) for c in COLONNES_COMPETENCES}  # Construit le profil Python.
        code = classes[int(pred_indices[i])]  # Traduit l’indice prédit en code filière.
        ratios.append(moteur.score_regles(profil, code))  # Ajoute le score de règles de la prédiction.
    return float(np.mean(ratios))  # Retourne la moyenne des scores de règles.

def choisir_poids_hybrides(moteur: MoteurSERAP, profils_val: pd.DataFrame, classes: List[str], proba_cnn_val: np.ndarray, y_val: np.ndarray) -> Tuple[Tuple[float, float, float], float]:  # Cherche les meilleurs poids alpha, beta, gamma.
    meilleurs_poids = (0.35, 0.20, 0.45)  # Initialise un triplet par défaut.
    meilleur_f1 = -1.0  # Initialise le meilleur score F1.
    for alpha in [0.20, 0.25, 0.30, 0.35, 0.40]:  # Parcourt quelques valeurs pour alpha.
        for beta in [0.10, 0.15, 0.20, 0.25, 0.30]:  # Parcourt quelques valeurs pour beta.
            gamma = round(1.0 - alpha - beta, 2)  # Déduit gamma pour conserver une somme égale à 1.
            if gamma < 0.20:  # Exclut les cas où gamma devient trop faible.
                continue  # Passe à la combinaison suivante.
            predictions = []  # Initialise la liste des prédictions validation.
            for i, (_, ligne) in enumerate(profils_val.iterrows()):  # Parcourt les profils de validation.
                profil = {c: (None if pd.isna(ligne[c]) else float(ligne[c])) for c in COLONNES_COMPETENCES}  # Reconstruit le profil.
                scores = []  # Initialise les scores pour toutes les filières.
                for j, code in enumerate(classes):  # Parcourt les classes.
                    sr = moteur.score_regles(profil, code)  # Calcule le score règles.
                    sg = moteur.score_graphe(profil, code)  # Calcule le score graphe.
                    scnn = float(proba_cnn_val[i, j])  # Récupère la probabilité CNN.
                    scores.append(alpha * sr + beta * sg + gamma * scnn)  # Ajoute le score hybride.
                predictions.append(int(np.argmax(scores)))  # Garde la meilleure classe.
            f1 = float(f1_score(y_val, np.array(predictions), average="macro", zero_division=0))  # Calcule la F1 macro.
            if f1 > meilleur_f1:  # Vérifie si la combinaison est meilleure.
                meilleur_f1 = f1  # Met à jour la meilleure F1.
                meilleurs_poids = (alpha, beta, gamma)  # Met à jour les meilleurs poids.
    return meilleurs_poids, meilleur_f1  # Retourne les meilleurs poids et la meilleure F1.

def predire_hybride(moteur: MoteurSERAP, profils: pd.DataFrame, classes: List[str], proba_cnn: np.ndarray, poids: Tuple[float, float, float]) -> Tuple[np.ndarray, List[Dict[str, float]]]:  # Produit les prédictions hybrides.
    alpha, beta, gamma = poids  # Décompacte les poids.
    predictions = []  # Initialise la liste des indices prédits.
    details = []  # Initialise la liste des détails explicatifs.
    for i, (_, ligne) in enumerate(profils.iterrows()):  # Parcourt les profils.
        profil = {c: (None if pd.isna(ligne[c]) else float(ligne[c])) for c in COLONNES_COMPETENCES}  # Construit le profil.
        scores = []  # Initialise les scores pour toutes les filières.
        for j, code in enumerate(classes):  # Parcourt les filières candidates.
            sr = moteur.score_regles(profil, code)  # Calcule le score règles.
            sg = moteur.score_graphe(profil, code)  # Calcule le score graphe.
            scnn = float(proba_cnn[i, j])  # Récupère le score CNN.
            score = alpha * sr + beta * sg + gamma * scnn  # Réalise l’hybridation.
            scores.append((code, score, sr, sg, scnn))  # Ajoute le tuple de score.
        scores = sorted(scores, key=lambda x: x[1], reverse=True)  # Trie les filières du meilleur au moins bon.
        meilleur = scores[0]  # Sélectionne la meilleure filière.
        predictions.append(classes.index(meilleur[0]))  # Ajoute l’indice de la meilleure classe.
        details.append({  # Ajoute un dictionnaire de détails.
            "id_etudiant": ligne["id_etudiant"],  # Sauvegarde l’identifiant étudiant.
            "code_reel": ligne["code_filiere_reelle"],  # Sauvegarde le code réel.
            "code_predit": meilleur[0],  # Sauvegarde le code prédit.
            "score_hybride": float(meilleur[1]),  # Sauvegarde le score hybride.
            "score_regles": float(meilleur[2]),  # Sauvegarde le score règles.
            "score_graphe": float(meilleur[3]),  # Sauvegarde le score graphe.
            "score_cnn": float(meilleur[4]),  # Sauvegarde le score CNN.
        })  # Ferme le dictionnaire de détail.
    return np.array(predictions, dtype=int), details  # Retourne les indices prédits et les détails.

def sauvegarder_modeles(modele_cnn: ReseauCNN1D, paquets: Dict[str, object], poids: Tuple[float, float, float], historique: Dict[str, List[float]]) -> None:  # Sauvegarde le CNN et les métadonnées.
    torch.save({  # Sauvegarde l’état du réseau au format PyTorch.
        "etat_modele": modele_cnn.state_dict(),  # Sauvegarde les poids du réseau.
        "nb_variables": int(paquets["X_train"].shape[1]),  # Sauvegarde la dimension d’entrée.
        "nb_classes": int(len(paquets["encodeur"].classes_)),  # Sauvegarde le nombre de classes.
    }, DOSSIER_MODELES / "modele_cnn_serap.pt")  # Définit le fichier de sortie.
    joblib.dump({  # Sauvegarde les métadonnées utiles à l’inférence.
        "normaliseur": paquets["normaliseur"],  # Sauvegarde le standardiseur.
        "classes": list(paquets["encodeur"].classes_),  # Sauvegarde la liste des classes.
        "colonnes_scores": list(paquets["colonnes_scores"]),  # Sauvegarde les colonnes utilisées.
        "alpha": float(poids[0]),  # Sauvegarde alpha.
        "beta": float(poids[1]),  # Sauvegarde beta.
        "gamma": float(poids[2]),  # Sauvegarde gamma.
    }, DOSSIER_MODELES / "pretraitement_serap.joblib")  # Définit le nom du fichier de métadonnées.
    pd.DataFrame(historique).to_csv(DOSSIER_SORTIES / "historique_cnn.csv", index=False)  # Exporte l’historique de l’entraînement.

def generer_figures(df_metriques: pd.DataFrame, df_ablation: pd.DataFrame) -> None:  # Génère les figures PNG.
    plt.figure(figsize=(10, 5))  # Crée la figure des métriques.
    plt.bar(df_metriques["modele"], df_metriques["accuracy"])  # Trace les accuracies par modèle.
    plt.xticks(rotation=25, ha="right")  # Oriente les labels de l’axe x.
    plt.ylim(0, 1.0)  # Fixe l’échelle verticale.
    plt.title("Comparaison des accuracies des modèles")  # Définit le titre de la figure.
    plt.tight_layout()  # Ajuste l’agencement.
    plt.savefig(DOSSIER_SORTIES / "comparaison_accuracy.png", dpi=200)  # Sauvegarde la figure.
    plt.close()  # Ferme la figure.
    plt.figure(figsize=(8, 5))  # Crée la figure d’ablation.
    plt.bar(df_ablation["variante"], df_ablation["f1_macro"])  # Trace la F1 macro des variantes.
    plt.xticks(rotation=20, ha="right")  # Oriente les labels de l’axe x.
    plt.ylim(0, 1.0)  # Fixe l’échelle verticale.
    plt.title("Ablation de SERAP (F1 macro)")  # Définit le titre de la figure.
    plt.tight_layout()  # Ajuste l’agencement.
    plt.savefig(DOSSIER_SORTIES / "ablation_f1.png", dpi=200)  # Sauvegarde la figure.
    plt.close()  # Ferme la figure.

def generer_predictions_explicatives(profils_test: pd.DataFrame) -> pd.DataFrame:  # Génère les sorties explicatives avec le moteur final.
    moteur = MoteurSERAP()  # Recharge le moteur avec le modèle désormais sauvegardé.
    lignes = []  # Initialise la liste des lignes de sortie.
    for _, ligne in profils_test.iterrows():  # Parcourt tous les profils de test.
        profil = {c: (None if pd.isna(ligne[c]) else float(ligne[c])) for c in COLONNES_COMPETENCES}  # Reconstruit le profil.
        resultat = moteur.recommander(profil, top_k=3)  # Calcule les recommandations top-3.
        top1 = resultat["top_k"][0]  # Sélectionne la première recommandation.
        top3_resume = " | ".join([f"{r['code_filiere']}:{r['score_final']:.3f}" for r in resultat["top_k"]])  # Construit un résumé top-3.
        lignes.append({  # Ajoute une ligne de sortie explicative.
            "id_etudiant": ligne["id_etudiant"],  # Sauvegarde l’identifiant étudiant.
            "code_reel": ligne["code_filiere_reelle"],  # Sauvegarde la filière réelle.
            "code_predit": top1["code_filiere"],  # Sauvegarde la filière prédite.
            "filiere_predite": top1["filiere"],  # Sauvegarde le nom de la filière prédite.
            "score_final": round(float(top1["score_final"]), 4),  # Sauvegarde le score final.
            "score_regles": round(float(top1["score_regles"]), 4),  # Sauvegarde le score règles.
            "score_graphe": round(float(top1["score_graphe"]), 4),  # Sauvegarde le score graphe.
            "score_cnn": round(float(top1["score_cnn"]), 4),  # Sauvegarde le score CNN.
            "top3_resume": top3_resume,  # Sauvegarde le résumé top-3.
            "explication": top1["explication"],  # Sauvegarde l’explication textuelle.
        })  # Ferme le dictionnaire de sortie.
    return pd.DataFrame(lignes)  # Retourne le DataFrame final.

def main() -> None:  # Définit le point d’entrée du script.
    fixer_graine(SEED)  # Fixe les graines pour la reproductibilité.
    df = charger_profils()  # Charge les profils étudiants génériques.
    paquets = preparer_jeux(df)  # Prépare les jeux train, validation et test.
    moteur = MoteurSERAP()  # Instancie le moteur métier.
    moteur.normaliseur = paquets["normaliseur"]  # Alimente le moteur avec le normaliseur courant.
    moteur.classes_modele = list(paquets["encodeur"].classes_)  # Alimente le moteur avec la liste des classes.
    resultats = entrainer_baselines(paquets)  # Entraîne les modèles de référence.
    modele_cnn, historique = entrainer_cnn(paquets)  # Entraîne le CNN 1D.
    proba_val = probabilites_cnn(modele_cnn, paquets["X_val"])  # Calcule les probabilités validation du CNN.
    proba_test = probabilites_cnn(modele_cnn, paquets["X_test"])  # Calcule les probabilités test du CNN.
    pred_cnn = np.argmax(proba_test, axis=1)  # Extrait les classes CNN prédites.
    resultats["cnn_1d"] = calculer_metriques(paquets["y_test"], pred_cnn)  # Sauvegarde les métriques du CNN.
    classes = list(paquets["encodeur"].classes_)  # Récupère les codes filières dans l’ordre du modèle.
    poids_hybrides, f1_val = choisir_poids_hybrides(moteur, paquets["profils_val"], classes, proba_val, paquets["y_val"])  # Recherche les meilleurs poids.
    pred_h, details_h = predire_hybride(moteur, paquets["profils_test"], classes, proba_test, poids_hybrides)  # Calcule les prédictions hybrides.
    resultats["hybride_serap"] = calculer_metriques(paquets["y_test"], pred_h)  # Sauvegarde les métriques de l’hybride.
    metriques_rows = []  # Initialise les lignes du tableau métriques.
    for nom, mesures in resultats.items():  # Parcourt tous les résultats enregistrés.
        top3 = top3_accuracy(proba_test, paquets["y_test"]) if nom in ("cnn_1d", "hybride_serap") else np.nan  # Calcule la top-3 accuracy si pertinent.
        pred_indices = pred_h if nom == "hybride_serap" else pred_cnn if nom == "cnn_1d" else None  # Définit les prédictions à utiliser pour la cohérence logique.
        coherence = coherence_logique(moteur, paquets["profils_test"], classes, pred_indices) if pred_indices is not None else np.nan  # Calcule la cohérence logique si possible.
        metriques_rows.append({  # Ajoute une ligne de métriques.
            "modele": nom,  # Sauvegarde le nom du modèle.
            "accuracy": round(mesures["accuracy"], 4),  # Sauvegarde l’accuracy.
            "precision_macro": round(mesures["precision_macro"], 4),  # Sauvegarde la précision macro.
            "recall_macro": round(mesures["recall_macro"], 4),  # Sauvegarde le rappel macro.
            "f1_macro": round(mesures["f1_macro"], 4),  # Sauvegarde la F1 macro.
            "top3_accuracy": round(float(top3), 4) if not np.isnan(top3) else "",  # Sauvegarde la top-3 accuracy.
            "coherence_logique": round(float(coherence), 4) if not np.isnan(coherence) else "",  # Sauvegarde la cohérence logique.
        })  # Ferme la ligne des métriques.
    df_metriques = pd.DataFrame(metriques_rows)  # Convertit les métriques en DataFrame.
    df_metriques.to_csv(DOSSIER_SORTIES / "metriques_modeles.csv", index=False)  # Exporte le tableau des métriques.
    alpha, beta, gamma = poids_hybrides  # Décompacte les poids hybrides.
    variantes = {  # Prépare les variantes d’ablation.
        "sans_regles": (0.0, beta / (beta + gamma), gamma / (beta + gamma)),  # Définit la variante sans règles.
        "sans_graphe": (alpha / (alpha + gamma), 0.0, gamma / (alpha + gamma)),  # Définit la variante sans graphe.
        "sans_cnn": (alpha / (alpha + beta), beta / (alpha + beta), 0.0),  # Définit la variante sans CNN.
        "complet": poids_hybrides,  # Définit la variante complète.
    }  # Ferme le dictionnaire des variantes.
    rows_ablation = []  # Initialise la liste des lignes d’ablation.
    for nom, poids in variantes.items():  # Parcourt les variantes.
        pred_tmp, _ = predire_hybride(moteur, paquets["profils_test"], classes, proba_test, poids)  # Recalcule les prédictions de la variante.
        mesures = calculer_metriques(paquets["y_test"], pred_tmp)  # Calcule les métriques de la variante.
        rows_ablation.append({  # Ajoute la ligne de la variante.
            "variante": nom,  # Sauvegarde le nom de la variante.
            "accuracy": round(mesures["accuracy"], 4),  # Sauvegarde l’accuracy.
            "precision_macro": round(mesures["precision_macro"], 4),  # Sauvegarde la précision macro.
            "recall_macro": round(mesures["recall_macro"], 4),  # Sauvegarde le rappel macro.
            "f1_macro": round(mesures["f1_macro"], 4),  # Sauvegarde la F1 macro.
        })  # Ferme la ligne d’ablation.
    df_ablation = pd.DataFrame(rows_ablation)  # Convertit l’ablation en DataFrame.
    df_ablation.to_csv(DOSSIER_SORTIES / "ablation_modeles.csv", index=False)  # Exporte le tableau d’ablation.
    sauvegarder_modeles(modele_cnn, paquets, poids_hybrides, historique)  # Sauvegarde le CNN et les métadonnées.
    predictions_explicatives = generer_predictions_explicatives(paquets["profils_test"])  # Génère les sorties explicatives finales.
    predictions_explicatives.to_csv(DOSSIER_SORTIES / "predictions_explicatives_test.csv", index=False)  # Exporte les prédictions explicatives.
    pd.DataFrame(details_h).to_csv(DOSSIER_SORTIES / "details_scores_hybrides_test.csv", index=False)  # Exporte les détails numériques hybrides.
    generer_figures(df_metriques, df_ablation)  # Génère les figures PNG.
    resume = {  # Construit le résumé final de l’entraînement.
        "classes": classes,  # Sauvegarde la liste des classes.
        "poids_hybrides": {"alpha_regles": poids_hybrides[0], "beta_graphe": poids_hybrides[1], "gamma_cnn": poids_hybrides[2]},  # Sauvegarde les poids hybrides.
        "f1_validation_meilleure": f1_val,  # Sauvegarde la meilleure F1 en validation.
        "nombre_profils_total": int(len(df)),  # Sauvegarde la taille totale du dataset.
        "taille_train": int(len(paquets["y_train"])),  # Sauvegarde la taille du train.
        "taille_validation": int(len(paquets["y_val"])),  # Sauvegarde la taille de la validation.
        "taille_test": int(len(paquets["y_test"])),  # Sauvegarde la taille du test.
    }  # Ferme le résumé.
    with open(DOSSIER_SORTIES / "resume_entrainement.json", "w", encoding="utf-8") as fichier:  # Ouvre le fichier JSON en écriture.
        json.dump(resume, fichier, ensure_ascii=False, indent=2)  # Écrit le résumé JSON lisible.
    print("Entraînement terminé avec succès.")  # Affiche le message de fin.
    print("Poids hybrides retenus :", poids_hybrides)  # Affiche les poids d’hybridation.
    print("Sorties enregistrées dans :", DOSSIER_SORTIES)  # Affiche le dossier des sorties.

if __name__ == "__main__":  # Vérifie que le script est exécuté directement.
    main()  # Lance le processus complet.
