# SERAP-UAC — Projet complet en français

Ce projet fournit un prototype complet d’un système hybride de recommandation académique pour l’Université de l’Assomption au Congo (UAC).

## Contenu
- données publiques UAC structurées ;
- exigences des filières au format long ;
- règles expertes ;
- profils étudiants génériques ;
- modèle de template pour données réelles anonymisées ;
- moteur hybride règles + graphe + CNN ;
- entraînement, sauvegarde du modèle et sorties ;
- interface web FastAPI ;
- recommandations explicatives.

## Lancer l’entraînement
```bash
pip install -r requirements.txt
python scripts/entrainer_modele.py
```

## Lancer l’interface
```bash
uvicorn application.main:app --reload
```

## Enregistrement des recommandations (SQLite)
Chaque requête de recommandation (HTML et API) est automatiquement enregistrée dans `sorties/serap.db`.

## Données
- une cellule vide dans `profils_etudiants_generiques.csv` signifie « non observé » ;
- une valeur faible signifie « observé mais faible » ;
- une compétence absente des exigences d’une filière n’est pas pénalisée pour cette filière.
