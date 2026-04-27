# PROJETMEMOIREOR

Petit projet Python pour générer une présentation PowerPoint (`.pptx`) à partir d’un fichier Markdown.

Il contient aussi une application web FastAPI dans `SERAP-UAC/`.

## Prérequis

- Python 3.10+ (testé avec Python 3.12)
- Dépendance : `python-pptx`

Installation :

```bash
python -m pip install -U python-pptx
```

## Lancer

### Générateur PPTX (Markdown → PowerPoint)

Depuis la racine du projet :

```bash
python tools/make_pptx_from_md.py
```

Avec un autre fichier d’entrée / sortie :

```bash
python tools/make_pptx_from_md.py --src presentation_SNMP_nouvelle.md --out presentation.pptx
```

## Format Markdown attendu

- `# Titre` : titre global (slide de couverture)
- `## Titre de slide` : une slide
- `### ...` ou `- ...` : puces dans le contenu

### Application web SERAP-UAC (FastAPI)

Depuis le dossier `SERAP-UAC/` :

```bash
python -m uvicorn application.main:app --host 127.0.0.1 --port 8000
```

Puis ouvre `http://127.0.0.1:8000/` dans ton navigateur. Pour arrêter : `Ctrl+C`.

#### Login (historique protégé)

- Page : `/login`
- Identifiants par défaut : `admin` / `admin`
- Variables d’environnement :
  - `SERAP_ADMIN_USER`
  - `SERAP_ADMIN_PASS`
  - `SERAP_SECRET_KEY` (secret pour signer le cookie de session)

## Mettre en ligne (Render)

Le repo contient un blueprint Render prêt à l’emploi : `render.yaml`.

1) Pousse le projet sur GitHub
2) Sur Render : **New** → **Blueprint** → sélectionne le repo
3) Render crée le service `serap-uac` automatiquement

Notes :
- L’app écoute sur `$PORT` (Render) via la commande `uvicorn ... --port $PORT`.
- Sur le plan gratuit, le disque peut être éphémère : l’historique SQLite peut être perdu après redémarrage.
