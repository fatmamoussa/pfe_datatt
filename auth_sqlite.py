"""
=============================================================
auth_sqlite.py — Authentification SQLite
Chatbot Tunisie Telecom
=============================================================

CORRECTIONS v2 :
  [1] get_db() corrigé — @contextmanager avec conn.close() garanti
  [2] Compte admin fixe créé automatiquement au démarrage
      Email    : admin@telecom.tn
      Password : 123456
      Le compte admin est inséré dans la DB si absent.
      Pas de token hardcodé — il passe par le login normal.
  [3] Hachage mot de passe renforcé — PBKDF2-HMAC-SHA256 avec sel
=============================================================
"""

import os
import re
import uuid
import random
import hashlib
import sqlite3
import logging
import smtplib
import threading
from contextlib import contextmanager
from email.message import EmailMessage
from datetime import datetime, timedelta
from typing import Optional

from fastapi import HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────
AUTH_DB_PATH = os.environ.get("AUTH_DB_PATH", "/app/data/auth.db")
TOKEN_TTL_HOURS  = 24
CODE_TTL_MINUTES = 15

ABSTRACT_API_KEY = os.getenv("ABSTRACT_API_KEY", "")

# ─── COMPTE ADMIN FIXE ────────────────────────────────────
ADMIN_EMAIL    = "admin@telecom.tn"
ADMIN_PASSWORD = "123456"
ADMIN_NAME     = "Admin Tunisie Telecom"

# ─── BLACKLIST DOMAINES JETABLES ──────────────────────────
DISPOSABLE_DOMAINS = {
    "yopmail.com","guerrillamail.com","guerrillamail.net","guerrillamail.info",
    "mailinator.com","trashmail.com","trashmail.net","trashmail.io",
    "tempmail.com","temp-mail.org","tempail.com","10minutemail.com",
    "throwam.com","sharklasers.com","guerrillamailblock.com","grr.la",
    "spam4.me","dispostable.com","fakeinbox.com","maildrop.cc",
    "spamgourmet.com","mytemp.email","emailondeck.com","discard.email",
    "spamhereplease.com","crapmail.org","filzmail.com","nomail.pw",
    "rcpt.at","ruu.be","say.email","sneakemail.com","spambob.net",
    "spambob.org","spamcannot.com","spamcannot.org","spamcannon.com",
    "objectmail.com","obobbo.com","mt2015.com","hulapla.de",
    "inoutmail.de","mail2rss.org","mega.zik.dj","courriel.fr.nf",
    "hide.biz.st","moncourrier.fr.nf","monemail.fr.nf","monmail.fr.nf",
}

# ─── SCHEMAS PYDANTIC ─────────────────────────────────────
class RegisterRequest(BaseModel):
    name:     str
    email:    str
    password: str

class VerifyRequest(BaseModel):
    email: str
    code:  str

class LoginRequest(BaseModel):
    email:    str
    password: str

class ResendRequest(BaseModel):
    email: str

class CheckEmailRequest(BaseModel):
    email: str


# ═══════════════ DB — FIX [1] ═══════════════

_db_lock = threading.Lock()

@contextmanager
def get_db():
    """
    FIX [1] : context manager correct avec fermeture explicite.
    Garantit conn.close() même en cas d'exception.
    """
    os.makedirs(os.path.dirname(AUTH_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_auth_db():
    """
    Initialise la base et crée le compte admin fixe s'il n'existe pas.
    FIX [2] : l'admin passe par le login normal (email + password).
    """
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                email       TEXT NOT NULL UNIQUE,
                pwd_hash    TEXT NOT NULL,
                pwd_salt    TEXT NOT NULL DEFAULT '',
                role        TEXT NOT NULL DEFAULT 'user',
                is_verified INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

            CREATE TABLE IF NOT EXISTS verif (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                email      TEXT NOT NULL,
                code       TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS tokens (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                token      TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_tokens_token ON tokens(token);
        """)
        _migrate_auth_db(db)

        # FIX [2] : créer le compte admin automatiquement si absent
        existing = db.execute(
            "SELECT id FROM users WHERE email = ?", (ADMIN_EMAIL,)
        ).fetchone()

        if not existing:
            salt     = _gen_salt()
            pwd_hash = hash_pwd(ADMIN_PASSWORD, salt)
            db.execute(
                "INSERT INTO users (name, email, pwd_hash, pwd_salt, role, is_verified) "
                "VALUES (?, ?, ?, ?, 'admin', 1)",
                (ADMIN_NAME, ADMIN_EMAIL, pwd_hash, salt)
            )
            logger.info("[AUTH] Compte admin créé : %s", ADMIN_EMAIL)
        else:
            logger.info("[AUTH] Compte admin existant : %s", ADMIN_EMAIL)

    logger.info("[AUTH] DB prête : %s", AUTH_DB_PATH)


def _migrate_auth_db(conn):
    """Migrations pour compatibilité avec les bases existantes."""
    for sql in [
        "ALTER TABLE users ADD COLUMN pwd_salt TEXT NOT NULL DEFAULT ''",
    ]:
        try:
            conn.execute(sql)
        except Exception:
            pass


# ═══════════════ HELPERS ═══════════════

def _gen_salt() -> str:
    return uuid.uuid4().hex

def hash_pwd(pwd: str, salt: str = "") -> str:
    """
    FIX [3] : PBKDF2-HMAC-SHA256 avec sel (100 000 itérations).
    Compatibilité : si salt vide (anciens comptes), utilise SHA-256 simple.
    """
    if not salt:
        return hashlib.sha256(pwd.encode()).hexdigest()
    dk = hashlib.pbkdf2_hmac('sha256', pwd.encode('utf-8'), salt.encode('utf-8'), 100_000)
    return dk.hex()

def verify_pwd(pwd: str, pwd_hash: str, salt: str) -> bool:
    return hash_pwd(pwd, salt) == pwd_hash

def gen_code() -> str:
    return str(random.randint(100000, 999999))

def gen_token() -> str:
    return str(uuid.uuid4())

def now_str() -> str:
    return datetime.now().isoformat()

def expiry_str(minutes: int = 0, hours: int = 0) -> str:
    return (datetime.now() + timedelta(minutes=minutes, hours=hours)).isoformat()


# ═══════════════ ENVOI EMAIL SMTP ═══════════════

def send_verification_email_async(email: str, code: str, name: str = ""):
    def _send():
        try:
            smtp_host      = os.getenv("SMTP_HOST")
            smtp_port      = int(os.getenv("SMTP_PORT", 587))
            smtp_user      = os.getenv("SMTP_USER")
            smtp_password  = os.getenv("SMTP_PASSWORD")
            smtp_from      = os.getenv("SMTP_FROM", smtp_user)
            smtp_from_name = os.getenv("SMTP_FROM_NAME", "Tunisie Telecom")

            if not all([smtp_host, smtp_user, smtp_password]):
                logger.warning("[AUTH] SMTP non configuré — email non envoyé (mode démo)")
                return

            prenom = name or email.split('@')[0]
            msg = EmailMessage()
            msg["Subject"] = "Vérification de votre compte Tunisie Telecom"
            msg["From"]    = f"{smtp_from_name} <{smtp_from}>"
            msg["To"]      = email

            msg.set_content(
                f"Bonjour {prenom},\n\n"
                f"Votre code de vérification : {code}\n\n"
                f"Valable {CODE_TTL_MINUTES} minutes.\n\n"
                f"— Assistant IA Tunisie Telecom"
            )
            msg.add_alternative(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;background:#f4f6fa;padding:20px;">
<div style="max-width:500px;margin:0 auto;background:#fff;border-radius:16px;padding:28px;">
  <h2 style="color:#003A8C;">Vérification de compte</h2>
  <p>Bonjour {prenom},</p>
  <p>Votre code pour <strong>Tunisie Telecom</strong> :</p>
  <div style="background:#E8F0FA;font-size:32px;font-weight:700;text-align:center;
              padding:20px;border-radius:12px;letter-spacing:8px;">{code}</div>
  <p>Valable <strong>{CODE_TTL_MINUTES} minutes</strong>.</p>
  <hr style="border:none;border-top:1px solid #e0e0e0;margin:20px 0;">
  <p style="font-size:12px;color:#777;">Ne répondez pas à cet email.</p>
</div></body></html>""", subtype="html")

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            logger.info("[AUTH] Email envoyé à %s", email)
        except Exception as e:
            logger.error("[AUTH] Erreur email à %s : %s", email, e)

    threading.Thread(target=_send, daemon=True).start()


# ═══════════════ VALIDATION EMAIL ═══════════════

def _check_format(email: str):
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Format d'e-mail invalide"
    parts = email.split("@")
    if len(parts[0]) < 1 or len(parts[1]) < 4:
        return False, "Format d'e-mail invalide"
    return True, ""

def _check_disposable(email: str):
    domain = email.split("@")[1].lower()
    if domain in DISPOSABLE_DOMAINS:
        return False, f"Les adresses @{domain} ne sont pas acceptées (domaine temporaire)"
    return True, ""

def _check_mx(email: str):
    domain = email.split("@")[1].lower()
    try:
        import dns.resolver
        try:
            answers = dns.resolver.resolve(domain, 'MX', lifetime=5)
            if not answers:
                return False, f"Le domaine @{domain} n'accepte pas d'e-mails"
            return True, ""
        except dns.resolver.NXDOMAIN:
            return False, f"Le domaine @{domain} n'existe pas"
        except dns.resolver.NoAnswer:
            try:
                dns.resolver.resolve(domain, 'A', lifetime=5)
                return True, ""
            except Exception:
                return False, f"Le domaine @{domain} est inaccessible"
        except dns.exception.Timeout:
            return True, ""
        except Exception as e:
            logger.warning("[AUTH] DNS erreur %s : %s", domain, e)
            return True, ""
    except ImportError:
        return True, ""

async def _check_abstract_api(email: str):
    if not ABSTRACT_API_KEY:
        return True, "", {}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(
                "https://emailvalidation.abstractapi.com/v1/",
                params={"api_key": ABSTRACT_API_KEY, "email": email}
            )
        if r.status_code != 200:
            return True, "", {}
        data = r.json()
        if not data.get("is_valid_format", {}).get("value", True):
            return False, "Format non reconnu par l'API", data
        if data.get("is_disposable_email", {}).get("value", False):
            return False, "Adresse e-mail temporaire non acceptée", data
        if not data.get("is_mx_found", {}).get("value", True):
            return False, "Le domaine n'accepte pas de messages", data
        if data.get("deliverability") == "UNDELIVERABLE":
            return False, "Cette adresse e-mail n'est pas joignable", data
        return True, "", data
    except ImportError:
        return True, "", {}
    except Exception as e:
        logger.warning("[AUTH] Abstract API erreur : %s", e)
        return True, "", {}

async def _verify_email_full(email: str) -> dict:
    email = email.strip().lower()
    checks = {}

    ok, reason = _check_format(email)
    checks["format"] = ok
    if not ok:
        return {"valid": False, "reason": reason, "checks": checks, "score": 0}

    ok, reason = _check_disposable(email)
    checks["not_disposable"] = ok
    if not ok:
        return {"valid": False, "reason": reason, "checks": checks, "score": 25}

    ok, reason = _check_mx(email)
    checks["mx_record"] = ok
    if not ok:
        return {"valid": False, "reason": reason, "checks": checks, "score": 50}

    ok, reason, _ = await _check_abstract_api(email)
    checks["api_check"] = ok
    if not ok:
        return {"valid": False, "reason": reason, "checks": checks, "score": 75}

    score = 25 * sum([
        checks.get("format", False),
        checks.get("not_disposable", False),
        checks.get("mx_record", False),
        checks.get("api_check", False),
    ])
    return {"valid": True, "reason": "", "checks": checks, "score": score}


# ═══════════════ ROUTE CHECK EMAIL ═══════════════

async def check_email_route(req: CheckEmailRequest) -> dict:
    email = req.email.strip().lower()
    if not email:
        return {"valid": False, "reason": "E-mail requis", "score": 0, "checks": {}}

    result = await _verify_email_full(email)

    if result["valid"]:
        with get_db() as db:
            existing = db.execute(
                "SELECT is_verified FROM users WHERE email = ?", (email,)
            ).fetchone()
            if existing and existing["is_verified"]:
                return {
                    "valid":  False,
                    "reason": "Cette adresse e-mail est déjà utilisée",
                    "score":  result["score"],
                    "checks": result["checks"],
                }
            if existing and not existing["is_verified"]:
                result["warning"] = "Un compte non vérifié existe pour cet email"
    return result


# ═══════════════ ROUTES AUTH ═══════════════

def register_route(req: RegisterRequest):
    email = req.email.strip().lower()
    name  = req.name.strip()
    pwd   = req.password

    # Bloquer la réinscription sur le compte admin
    if email == ADMIN_EMAIL:
        raise HTTPException(409, "Cette adresse e-mail est déjà utilisée")

    if not name or len(name) < 2:
        raise HTTPException(400, "Nom trop court (minimum 2 caractères)")
    ok, reason = _check_format(email)
    if not ok:
        raise HTTPException(400, reason)
    ok, reason = _check_disposable(email)
    if not ok:
        raise HTTPException(400, reason)
    if len(pwd) < 6:
        raise HTTPException(400, "Mot de passe : minimum 6 caractères")

    salt     = _gen_salt()
    pwd_hash = hash_pwd(pwd, salt)

    with get_db() as db:
        existing = db.execute(
            "SELECT is_verified FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing and existing["is_verified"]:
            raise HTTPException(409, "Cette adresse e-mail est déjà utilisée")
        if existing and not existing["is_verified"]:
            db.execute("DELETE FROM users WHERE email = ?", (email,))
            db.execute("DELETE FROM verif  WHERE email = ?", (email,))

        db.execute(
            "INSERT INTO users (name, email, pwd_hash, pwd_salt, role) VALUES (?,?,?,?,'user')",
            (name, email, pwd_hash, salt)
        )
        code    = gen_code()
        expires = expiry_str(minutes=CODE_TTL_MINUTES)
        db.execute(
            "INSERT INTO verif (email, code, expires_at) VALUES (?,?,?)",
            (email, code, expires)
        )

    logger.info("[AUTH] Register: %s", email)
    send_verification_email_async(email, code, name)

    return {
        "status":     "verification_required",
        "message":    f"Code envoyé à {email}",
        "demo_code":  code,       # Supprimer en production réelle
        "email_sent": True,
    }


def verify_route(req: VerifyRequest):
    email = req.email.strip().lower()
    code  = req.code.strip()

    with get_db() as db:
        row = db.execute("""
            SELECT id, expires_at, used FROM verif
            WHERE email = ? AND code = ?
            ORDER BY id DESC LIMIT 1
        """, (email, code)).fetchone()

        if not row:
            raise HTTPException(400, "Code incorrect")
        if row["used"]:
            raise HTTPException(400, "Code déjà utilisé")
        if datetime.fromisoformat(row["expires_at"]) < datetime.now():
            raise HTTPException(400, "Code expiré — demandez un nouveau code")

        db.execute("UPDATE verif SET used=1 WHERE id=?", (row["id"],))
        db.execute("UPDATE users SET is_verified=1 WHERE email=?", (email,))

        user = db.execute(
            "SELECT id, name, email, role FROM users WHERE email=?", (email,)
        ).fetchone()

        token   = gen_token()
        expires = expiry_str(hours=TOKEN_TTL_HOURS)
        db.execute(
            "INSERT INTO tokens (user_id, token, expires_at) VALUES (?,?,?)",
            (user["id"], token, expires)
        )

    logger.info("[AUTH] Verified: %s", email)
    return {
        "status": "ok",
        "token":  token,
        "user":   {"name": user["name"], "email": user["email"], "role": user["role"]}
    }


def login_route(req: LoginRequest):
    """
    FIX [2] : l'admin se connecte normalement avec email + password.
    Pas de token hardcodé — le compte admin est dans la DB.
    """
    email = req.email.strip().lower()
    pwd   = req.password

    with get_db() as db:
        user = db.execute(
            "SELECT id, name, email, role, is_verified, pwd_hash, pwd_salt "
            "FROM users WHERE email=?",
            (email,)
        ).fetchone()

    if not user or not verify_pwd(pwd, user["pwd_hash"], user["pwd_salt"]):
        raise HTTPException(401, "E-mail ou mot de passe incorrect")

    # L'admin est toujours vérifié (créé avec is_verified=1)
    if not user["is_verified"]:
        code    = gen_code()
        expires = expiry_str(minutes=CODE_TTL_MINUTES)
        with get_db() as db:
            db.execute(
                "INSERT INTO verif (email, code, expires_at) VALUES (?,?,?)",
                (email, code, expires)
            )
        send_verification_email_async(email, code, user["name"])
        return {
            "status":    "needs_verification",
            "demo_code": code,
            "email_sent": True,
        }

    with get_db() as db:
        # Nettoyer les tokens expirés
        db.execute(
            "DELETE FROM tokens WHERE user_id=? AND expires_at < ?",
            (user["id"], now_str())
        )
        token   = gen_token()
        expires = expiry_str(hours=TOKEN_TTL_HOURS)
        db.execute(
            "INSERT INTO tokens (user_id, token, expires_at) VALUES (?,?,?)",
            (user["id"], token, expires)
        )

    logger.info("[AUTH] Login: %s (role=%s)", email, user["role"])
    return {
        "status": "ok",
        "token":  token,
        "user":   {"name": user["name"], "email": user["email"], "role": user["role"]}
    }


def resend_route(req: ResendRequest):
    email = req.email.strip().lower()
    with get_db() as db:
        user = db.execute(
            "SELECT id, name FROM users WHERE email=?", (email,)
        ).fetchone()
        if not user:
            raise HTTPException(404, "E-mail non trouvé")
        code    = gen_code()
        expires = expiry_str(minutes=CODE_TTL_MINUTES)
        db.execute(
            "INSERT INTO verif (email, code, expires_at) VALUES (?,?,?)",
            (email, code, expires)
        )
    send_verification_email_async(email, code, user["name"] if user else "")
    return {
        "status":    "ok",
        "demo_code": code,
        "email_sent": True,
    }


def require_auth(authorization: str = Header(None)):
    """Dependency FastAPI — vérifie le token Bearer depuis la DB."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token manquant")

    token = authorization.split(" ", 1)[1].strip()

    with get_db() as db:
        row = db.execute("""
            SELECT u.id, u.name, u.email, u.role, t.expires_at
            FROM tokens t JOIN users u ON t.user_id = u.id
            WHERE t.token = ?
        """, (token,)).fetchone()

    if not row:
        raise HTTPException(401, "Token invalide")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now():
        raise HTTPException(401, "Session expirée — reconnectez-vous")
    return dict(row)


def me_route(user: dict):
    return {"name": user["name"], "email": user["email"], "role": user["role"]}


def logout_route(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return {"status": "ok"}
    token = authorization.split(" ", 1)[1].strip()
    with get_db() as db:
        db.execute("DELETE FROM tokens WHERE token=?", (token,))
    return {"status": "ok"}
