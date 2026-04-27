from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .db import Database
from .moteur import MoteurSERAP, groupes_par_defaut, libelles_par_defaut


BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="SERAP-UAC")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SERAP_SECRET_KEY", "dev-secret-change-me"),
    same_site="lax",
    https_only=False,
)

moteur = MoteurSERAP(BASE_DIR)
competences = moteur.competences_disponibles()
libelles = libelles_par_defaut(competences)
groupes = groupes_par_defaut(competences)

db = Database(BASE_DIR / "sorties" / "serap_uac.sqlite3")

def render_template(request: Request, name: str, context: dict[str, Any]) -> HTMLResponse:
    # Work around an incompatibility between some Starlette/Jinja2 versions where
    # `Jinja2Templates.TemplateResponse()` can call `env.get_template(name, globals)`
    # positionally and treat the globals dict as a "parent" key (unhashable).
    template = templates.env.get_template(name)
    html = template.render(**context, request=request)
    return HTMLResponse(html)

def is_authenticated(request: Request) -> bool:
    return bool(getattr(request, "session", {}).get("user"))


def require_auth(request: Request, next_path: str = "/") -> RedirectResponse | None:
    if is_authenticated(request):
        return None
    return RedirectResponse(url=f"/login?next={next_path}", status_code=303)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    resume = (
        "Exigences filières + règles expertes + (CNN neutre) "
        f"— {len(competences)} compétences, {len(set(moteur.exigences['code_filiere']))} filières."
    )
    return render_template(
        request,
        "index.html",
        {
            "resume_systeme": resume,
            "groupes": groupes,
            "libelles": libelles,
        },
    )

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/") -> Any:
    if is_authenticated(request):
        return RedirectResponse(url=next or "/", status_code=303)
    return render_template(request, "login.html", {"next": next or "/", "error": None})


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form(default="/")) -> Any:
    expected_user = os.environ.get("SERAP_ADMIN_USER", "admin")
    expected_pass = os.environ.get("SERAP_ADMIN_PASS", "admin")
    if username == expected_user and password == expected_pass:
        request.session["user"] = username
        return RedirectResponse(url=next or "/", status_code=303)
    return render_template(
        request,
        "login.html",
        {"next": next or "/", "error": "Identifiants invalides."},
    )


@app.post("/logout")
def logout(request: Request, next: str = Form(default="/")) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url=next or "/", status_code=303)


@app.post("/recommander", response_class=HTMLResponse)
async def recommander(request: Request, cycle: str = Form(default="Licence")) -> Any:
    profil: dict[str, Any] = {"cycle": cycle}
    form = await request.form()
    data = dict(form)

    for c in competences:
        val = data.get(c)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            profil[c] = None
        else:
            try:
                profil[c] = float(val)
            except Exception:
                profil[c] = None

    resultat = moteur.recommander(profil, top_k=5)
    champs_renseignes = sum(1 for k, v in profil.items() if k != "cycle" and v is not None)

    # Persist
    recommandations = [
        {
            "rang": r.rang,
            "code_filiere": r.code_filiere,
            "filiere": r.filiere,
            "domaine": r.domaine,
            "cycle": r.cycle,
            "score_final": r.score_final,
            "score_regles": r.score_regles,
            "score_graphe": r.score_graphe,
            "score_cnn": r.score_cnn,
            "explication": r.explication,
        }
        for r in resultat.top_k
    ]
    db.save_session(profil=profil, top_k=5, resume_systeme=resultat.resume_systeme, recommandations=recommandations)

    return render_template(
        request,
        "resultat.html",
        {
            "profil": profil,
            "resultat": resultat,
            "libelles": libelles,
            "champs_renseignes": champs_renseignes,
        },
    )


@app.get("/historique", response_class=HTMLResponse)
def historique(request: Request) -> Any:
    auth_redirect = require_auth(request, next_path="/historique")
    if auth_redirect is not None:
        return auth_redirect
    sessions = db.list_sessions(limit=30)
    ids = [s.id for s in sessions]
    items_par_session = db.list_recommandations_for_sessions(ids)
    return render_template(
        request,
        "historique.html",
        {
            "sessions": sessions,
            "items_par_session": items_par_session,
            "libelles": libelles,
        },
    )


@app.post("/historique/{session_id}/delete")
def delete_historique_session(request: Request, session_id: int) -> RedirectResponse:
    auth_redirect = require_auth(request, next_path="/historique")
    if auth_redirect is not None:
        return auth_redirect
    db.delete_session(session_id)
    return RedirectResponse(url="/historique", status_code=303)


@app.post("/historique/delete_all")
def delete_historique_all(request: Request) -> RedirectResponse:
    auth_redirect = require_auth(request, next_path="/historique")
    if auth_redirect is not None:
        return auth_redirect
    db.delete_all_sessions()
    return RedirectResponse(url="/historique", status_code=303)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
