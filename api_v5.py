#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
API v1.6 — Chatbot Tunisie Telecom — Phi-3.5-mini-instruct
CORRECTIONS v1.6 :
  - MODEL_DIR = microsoft/Phi-3.5-mini-instruct (modele de base)
  - MAX_INPUT_LENGTH 2048 (etait 1024)
  - RAG_CONTEXT_CHARS reduit a 400/600 (evite troncature)
  - PROMPT_LEAKS_PHI allege (moins de faux rejets)
  - Prompts systeme en anglais (meilleure instruction-following)
  - _clean_phi_output supprime "Response:" et fragments question
  - temperature=None quand do_sample=False (evite le warning)
=============================================================
"""

import os, re, time, uuid, sqlite3, logging, threading, statistics, gc, asyncio
import traceback
import unicodedata
import hashlib
from collections import Counter, deque, OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager
import unicodedata as _ud
import re as _re
import torch
from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rapidfuzz import fuzz, process

from auth_sqlite import (
    init_auth_db, require_auth,
    RegisterRequest, VerifyRequest, LoginRequest, ResendRequest,
    CheckEmailRequest,
    register_route, verify_route, login_route,
    resend_route, me_route, logout_route,
    check_email_route,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
# [FIX-v1.6] Modele de base par defaut
MODEL_DIR  = os.environ.get("MODEL_DIR", "microsoft/Phi-3.5-mini-instruct")

CONF_LOW             = float(os.environ.get("CONF_LOW", "0.55"))
CONFIDENCE_THRESHOLD = CONF_LOW

MAX_NEW_TOKENS     = int(os.environ.get("MAX_NEW_TOKENS", "180"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.3"))
GEN_TIMEOUT_S      = float(os.environ.get("GEN_TIMEOUT_S", "60.0"))

SHORT_QUERY_MAX_WORDS = 4

MIN_CONTEXT_COVERAGE = float(os.environ.get("MIN_CONTEXT_COVERAGE", "0.04"))

SEMANTIC_CACHE_ENABLED = os.environ.get("SEMANTIC_CACHE_ENABLED", "1") == "1"
SEMANTIC_CACHE_SIZE    = int(os.environ.get("SEMANTIC_CACHE_SIZE", "256"))

CHROMA_DB_DIR   = os.environ.get("CHROMA_DB_DIR",   "chroma_tt_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "tt_train")
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K           = int(os.environ.get("TOP_K", "3"))

DB_PATH    = os.environ.get("DB_PATH",    "data/chatbot.db")
STATIC_DIR = os.environ.get("STATIC_DIR", "static")
API_HOST   = "0.0.0.0"
API_PORT   = 8000
MAX_HISTORY        = 10
RAG_HISTORY_WINDOW = int(os.environ.get("RAG_HISTORY_WINDOW", "0"))

# [FIX-v1.6] MAX_INPUT_LENGTH 2048
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "2048"))

# [FIX-v1.6] Contexte RAG reduit pour eviter troncature du prompt
RAG_CONTEXT_CHARS_SIMPLE    = int(os.environ.get("RAG_CONTEXT_CHARS_SIMPLE",    "400"))
RAG_CONTEXT_CHARS_MULTILINE = int(os.environ.get("RAG_CONTEXT_CHARS_MULTILINE", "600"))

TORCH_COMPILE_PHI = os.environ.get("TORCH_COMPILE_PHI", "false").lower() == "true"
MIN_FREE_VRAM_GB  = float(os.environ.get("MIN_FREE_VRAM_GB", "0.2"))
USE_4BIT          = os.environ.get("USE_4BIT", "true").lower() == "true"

GEN_MAX_RETRIES = int(os.environ.get("GEN_MAX_RETRIES", "2"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_cuda_generation_lock = threading.Lock()

_metrics_lock = threading.Lock()
_global_metrics = {
    "total_requests": 0, "rag_used_count": 0, "hors_sujet_count": 0,
    "greeting_count": 0, "hallucination_count": 0, "extractive_count": 0,
    "generative_count": 0, "generative_timeout_count": 0, "cache_hit_count": 0,
    "validation_rejected_count": 0,
    "total_latency_ms": 0, "confidence_scores": [], "response_lengths": [],
    "positive_feedback": 0, "negative_feedback": 0,
    "rag_irrelevant_count": 0, "no_info_count": 0,
    "lang_rejected_count": 0,
    "arabic_query_count": 0,
    "validation_rejection_reasons": Counter(),
    "generative_error_oom": 0,
    "generative_error_runtime": 0,
    "generative_error_other": 0,
    "generative_vram_skip": 0,
    "generative_retry_success": 0,
    "generative_retry_fail": 0,
}

_live_metrics_lock    = threading.Lock()
_live_metrics_history = deque(maxlen=50)

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
GREETINGS_FR = [
    "bonjour","bonsoir","salut","hello","hi","bjr","bsr",
    "coucou","hey","allo","bj","good morning",
]
GREETINGS_AR = [
    "\u0633\u0644\u0627\u0645","\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645","\u0645\u0631\u062d\u0628\u0627","\u0627\u0647\u0644\u0627","\u0635\u0628\u0627\u062d \u0627\u0644\u062e\u064a\u0631","\u0645\u0633\u0627\u0621 \u0627\u0644\u062e\u064a\u0631",
    "\u0623\u0647\u0644\u0627","\u0623\u0647\u0644\u0627\u064b \u0648\u0633\u0647\u0644\u0627\u064b","\u064a\u0633\u0639\u062f \u0635\u0628\u0627\u062d\u0643","\u064a\u0633\u0639\u062f \u0645\u0633\u0627\u0643","\u0647\u0644\u0627","salam",
]
GREETINGS = GREETINGS_FR + GREETINGS_AR

GREETING_RESPONSE_FR = (
    "Bonjour ! Je suis l'assistant virtuel de Tunisie Telecom.\n"
    "Je peux vous aider sur :\n"
    "- Les offres mobiles (Hayya, forfaits 4G/5G)\n"
    "- Internet fixe (ADSL, Fibre, NetBox)\n"
    "- La recharge et le solde\n"
    "- Le roaming international\n\n"
    "Comment puis-je vous aider ?"
)

GREETING_RESPONSE_AR = (
    "\u0645\u0631\u062d\u0628\u0627\u064b! \u0623\u0646\u0627 \u0627\u0644\u0645\u0633\u0627\u0639\u062f \u0627\u0644\u0627\u0641\u062a\u0631\u0627\u0636\u064a \u0644\u0640 Tunisie Telecom.\n"
    "\u064a\u0645\u0643\u0646\u0646\u064a \u0645\u0633\u0627\u0639\u062f\u062a\u0643 \u0641\u064a:\n"
    "- \u0639\u0631\u0648\u0636 \u0627\u0644\u0647\u0627\u062a\u0641 \u0627\u0644\u0645\u062d\u0645\u0648\u0644 (Hayya\u060c \u0628\u0627\u0642\u0627\u062a 4G/5G)\n"
    "- \u0627\u0644\u0625\u0646\u062a\u0631\u0646\u062a \u0627\u0644\u062b\u0627\u0628\u062a (ADSL\u060c Fibre\u060c NetBox)\n"
    "- \u0627\u0644\u0634\u062d\u0646 \u0648\u0627\u0644\u0631\u0635\u064a\u062f\n"
    "- \u0627\u0644\u062a\u062c\u0648\u0627\u0644 \u0627\u0644\u062f\u0648\u0644\u064a\n\n"
    "\u0643\u064a\u0641 \u064a\u0645\u0643\u0646\u0646\u064a \u0645\u0633\u0627\u0639\u062f\u062a\u0643\u061f"
)

LANG_NOT_SUPPORTED_MSG = (
    "Je reponds uniquement en francais et en arabe. / "
    "\u0623\u0646\u0627 \u0623\u062c\u064a\u0628 \u0641\u0642\u0637 \u0628\u0627\u0644\u0644\u063a\u0629 \u0627\u0644\u0641\u0631\u0646\u0633\u064a\u0629 \u0648\u0627\u0644\u0639\u0631\u0628\u064a\u0629."
)

HORS_SUJET_FR = (
    "Je suis l'assistant Tunisie Telecom et je reponds uniquement "
    "aux questions sur nos offres et services."
)
HORS_SUJET_AR = (
    "\u0623\u0646\u0627 \u0645\u0633\u0627\u0639\u062f Tunisie Telecom \u0648\u0644\u0627 \u0623\u062c\u064a\u0628 \u0625\u0644\u0627 \u0639\u0644\u0649 \u0627\u0644\u0623\u0633\u0626\u0644\u0629 \u0627\u0644\u0645\u062a\u0639\u0644\u0642\u0629 "
    "\u0628\u0639\u0631\u0648\u0636\u0646\u0627 \u0648\u062e\u062f\u0645\u0627\u062a\u0646\u0627."
)

FALLBACK_NO_INFO_FR = (
    "Je n'ai pas trouve d'information precise sur ce sujet. "
    "Contactez le service client au 1298."
)
FALLBACK_NO_INFO_AR = (
    "\u0644\u0645 \u0623\u062c\u062f \u0645\u0639\u0644\u0648\u0645\u0627\u062a \u062f\u0642\u064a\u0642\u0629 \u062d\u0648\u0644 \u0647\u0630\u0627 \u0627\u0644\u0645\u0648\u0636\u0648\u0639. "
    "\u064a\u0631\u062c\u0649 \u0627\u0644\u062a\u0648\u0627\u0635\u0644 \u0645\u0639 \u062e\u062f\u0645\u0629 \u0627\u0644\u0639\u0645\u0644\u0627\u0621 \u0639\u0644\u0649 \u0627\u0644\u0631\u0642\u0645 1298."
)
FALLBACK_NO_INFO = FALLBACK_NO_INFO_FR

# ─────────────────────────────────────────────────────────────
# [FIX-v1.6] PROMPTS SYSTEME EN ANGLAIS — meilleure instruction-following
# ─────────────────────────────────────────────────────────────
GENERATIVE_SYSTEM_PROMPT = (
    "You are the official assistant of Tunisie Telecom. "
    "Answer ONLY using the CONTEXT provided below. "
    "Respond in French in 2-3 short sentences. "
    "Do NOT use your general knowledge. "
    "Do NOT repeat the question. "
    "If the context does not contain the answer, respond exactly: "
    "'Je n'ai pas trouve d'information precise. Contactez le 1298.' "
    "Do NOT mention Ooredoo, Orange or Tunisiana."
)

GENERATIVE_SYSTEM_PROMPT_MULTILINE = (
    "You are the official assistant of Tunisie Telecom. "
    "Explain the steps from the CONTEXT below in 3-5 French sentences. "
    "Use ONLY what is in the context. "
    "Do NOT invent steps or codes. "
    "If not found, respond: 'Contactez le 1298.'"
)

GENERATIVE_SYSTEM_PROMPT_AR = (
    "You are the official assistant of Tunisie Telecom. "
    "Answer ONLY using the CONTEXT provided below. "
    "Respond in Arabic in 2-3 sentences. "
    "Do NOT use general knowledge. "
    "If not found, respond: '\u064a\u0631\u062c\u0649 \u0627\u0644\u0627\u062a\u0635\u0627\u0644 \u0628\u0627\u0644\u0631\u0642\u0645 1298.'"
)

GENERATIVE_SYSTEM_PROMPT_AR_MULTILINE = (
    "You are the official assistant of Tunisie Telecom. "
    "Explain the steps from the CONTEXT below in 3-5 Arabic sentences. "
    "Use ONLY what is in the context. "
    "If not found, respond: '\u064a\u0631\u062c\u0649 \u0627\u0644\u0627\u062a\u0635\u0627\u0644 \u0628\u0627\u0644\u0631\u0642\u0645 1298.'"
)

TELECOM_KEYWORDS = [
    "365","3echra","5gtt","activation","advanced","anti","appel",
    "audiotex","avantages","big","bip","bleu","bonus","box","by",
    "cession","cloud","code","codes","comparaison","connect","conso",
    "corporate","couverture","data","ddos","desactivation","dim","double",
    "duo","duree","easy","ehdia","el","eleve","eligibilite","energy",
    "entreprise","esim","esports","fancy","fast","fixe","forfait",
    "forfaits","freeze","general","hadranet","hajj","housing","hybride",
    "iaas","inscription","international","internet","jaweknet",
    "joignabilite","kallemni","lights","ligne","link","manque","marhaba",
    "messagerie","microsoft","mms","mobile","mobiles","mobiracid",
    "mobirif","musique","my","national","net","numero","office","offre",
    "offres","one","optimum","options","pack","partage","partages",
    "pass","paye","pbx","platine","plus","portabilite","post","postpaid",
    "prepaid","prepayee","prepayees","presse","privilege","prix","pro",
    "probleme","profix","prolongation","rapides","rapido","recharge",
    "reseau","resiliation","roaming","saff","sajalni","select","services",
    "smart","sms","solde","sos","souscription","suivi","support",
    "tabba3ni","tarif","telecom","tfadhal","trankil","transfert","tt",
    "tunisie","ussd","validite","vas","vdc","vert","vocale","vod",
    "vpn","waffi",
    "4g","5g","3g","wifi","signal","debit","fibre","adsl","vdsl",
    "sim","esim","credit","facture","recharge","solde","activer",
    "souscrire","abonnement","abonner","inscrire","paiement","payer",
    "1298","mytt","agence","prepaye","postpaye","etranger","itinerance",
    "minutes","go","gigaoctet","mb","gb","panne","depannage","technique",
    "assistance","carte","fonctionne","disponible","cout","coute",
    "taraji","mouzikti","sport","jeu","game","tv","streaming","tourist",
    "samifehri","sami","fehri","svod","lorawan","m2m","iot","elissa",
    "\u0627\u0646\u062a\u0631\u0646\u062a","\u0634\u0628\u0643\u0629","\u0631\u0635\u064a\u062f","\u0634\u062d\u0646","\u0639\u0631\u0636","\u0639\u0631\u0648\u0636","\u0627\u0634\u062a\u0631\u0627\u0643","\u062a\u0641\u0639\u064a\u0644",
    "\u0628\u0627\u0642\u0629","\u0628\u0627\u0642\u0627\u062a","\u0645\u0643\u0627\u0644\u0645\u0629","\u0631\u0633\u0627\u0644\u0629","\u0628\u064a\u0627\u0646\u0627\u062a","\u062a\u062c\u0648\u0627\u0644","\u0641\u0627\u062a\u0648\u0631\u0629",
    "\u062e\u062f\u0645\u0629","\u062e\u062f\u0645\u0627\u062a","\u0647\u0627\u062a\u0641","\u0645\u0648\u0628\u0627\u064a\u0644","\u0633\u064a\u0645","\u0631\u0642\u0645","\u062a\u062d\u0648\u064a\u0644",
]

TELECOM_PRODUCT_NAMES = [
    "activation","advanced","anti ddos","appel","audiotex",
    "avantages","big bonus","bleu","bonus","box 5gtt pro","cession",
    "cloud","cloud pbx","cloud vdc","connect","conso","corporate",
    "data","dim connect","dim net corporate","double appel","duo",
    "easy saff","ehdia net","esim","esports by tt","fancy","fast link",
    "forfait","forfait partage","forfaits el 3echra","forfaits fancy",
    "forfaits internet","hadranet","hajj","hybride","iaas","inscription",
    "internet fixe","internet mobile","jaweknet","kallemni","marhaba",
    "messagerie vocale","microsoft office 365","mobile postpaid",
    "mobile prepaid","mobiracid","mobirif post paye","musique vod",
    "my tt","numero audiotex","numero bleu","numero platine","numero vert",
    "offre hajj","offres prepayees","one connect","optimum plus",
    "options tt","pack pro","pass marhaba","pass roaming data","platine",
    "portabilite","profix","prolongation de validite","rapido pro",
    "recharge","roaming","saff","sajalni","select plus","smart energy",
    "smart freeze","smart lights","smart roaming","sms appel manque",
    "sms joignabilite","sms plus","sos bip","sos solde","souscription",
    "suivi conso","tabba3ni","tfadhal","trankil","transfert d appel",
    "transfert internet","tt presse","tunisie telecom","ussd",
    "validite","vas","vert","vocale","vpn international","vpn national",
    "waffi",
]

# [FIX-v1.6] PROMPT_LEAKS allege — seulement les vrais leaks
PROMPT_LEAKS_PHI = [
    "as an ai","as a language model",
    "je suis un assistant ia","en tant qu'ia",
    "\u0648\u0641\u0642\u0627\u064b \u0644\u0645\u0639\u0631\u0641\u062a\u064a","\u0628\u062d\u0633\u0628 \u0645\u0639\u0644\u0648\u0645\u0627\u062a\u064a",
]

HALLUCINATION_SIGNALS = ["ooredoo","myoredoo","orange tunisie","tunisiana","vodafone"]
GENERATION_NOISE      = ["casino","karting"]

_MULTILINE_KEYWORDS = [
    "inscrire","inscription","souscrire","souscription","activer","activation",
    "acceder","acces","telecharger","telechargement","comment","etapes",
    "possibilites","plusieurs","manieres","facons",
    "\u0643\u064a\u0641\u064a\u0629","\u062e\u0637\u0648\u0627\u062a","\u0637\u0631\u064a\u0642\u0629","\u0643\u064a\u0641","\u062a\u0641\u0639\u064a\u0644","\u0627\u0634\u062a\u0631\u0627\u0643","\u062a\u0633\u062c\u064a\u0644",
]

BRUIT_DEBUT_PHRASE = [
    "Marketing","Contexte","Description","Concept","Source",
    "Flash","DCCM","Cible","Segment","Rappelons","DOCUMENT","Direction",
]

_NOISE_PATTERNS = [
    r"Flash [Cc]ommercial \d{1,2}/\d{1,2}/\d{4}",
    r"Flash [Ii]nfo \d{1,2}/\d{1,2}/\d{4}",
    r"Source d.information\s+Direction\s+\S+(\s+\S+){0,3}",
    r"DCCM\s+Page\s+\d+/\d+",
    r"\(cid:\d+\)",
    r"Cible\s*:?\s*Toute la clientele\s*(de\s*)?Tunisie Telecom",
    r"Direction (?:Marketing|VAS|Commerciale|Reseau).*",
    r"Strictement confidentiel.*",
    r"DOCUMENT DE TRAVAIL.*",
    r"Date de lancement\s*:?\s*\d{2}/\d{2}/\d{4}",
    r"This information is confidential.*",
    r"/\d{4}\s+(?:DCACM|DMFI|DM|DC)\s*/.*",
]

_JSONL_ARTIFACTS = [
    (r'\bavec\s+est\b',                     'est'),
    (r'\best\s+de\s+de\b',                  'est de'),
    (r'\bde\s+de\b',                        'de'),
    (r'voici\s+la\s+r[ee]ponse\s+de\s+Tunisie\s+Telecom\s*:\s*', ''),
    (r'\s{2,}',                             ' '),
]

TELECOM_THEME_KEYWORDS = {
    "Hayya":     ["hayya"],
    "Roaming":   ["roaming","itinerance","international","etranger","\u062a\u062c\u0648\u0627\u0644"],
    "Internet":  ["internet","data","go","mo","4g","5g","debit","connexion","\u0627\u0646\u062a\u0631\u0646\u062a","\u0628\u064a\u0627\u0646\u0627\u062a"],
    "NetBox":    ["netbox","net box"],
    "ADSL/Fixe": ["adsl","vdsl","fixe","fibre","elissa","box"],
    "Forfait":   ["forfait","pack","abonnement","souscrire","offre","\u0628\u0627\u0642\u0629","\u0627\u0634\u062a\u0631\u0627\u0643","\u0639\u0631\u0636"],
    "Recharge":  ["recharge","solde","credit","sos","122","balance","\u0634\u062d\u0646","\u0631\u0635\u064a\u062f"],
    "Mobile":    ["mobile","sim","esim","numero","telephone","\u0647\u0627\u062a\u0641","\u0645\u0648\u0628\u0627\u064a\u0644","\u0631\u0642\u0645"],
    "Activation":["activer","activation","desactiver","ussd","code","\u062a\u0641\u0639\u064a\u0644"],
    "Facture":   ["facture","payer","paiement","frais","\u0641\u0627\u062a\u0648\u0631\u0629"],
    "Corporate": ["corporate","entreprise","b2b","professionnel"],
}

VOCAB_TELECOM = [
    "hayya","forfait","internet","recharge","abonnement","facture",
    "roaming","netbox","adsl","fibre","activation","solde","credit",
    "tarif","offre","mobile","sim","reseau","couverture","debit",
    "illimite","gratuit","appel","sms","4g","5g","wifi","ussd",
    "activer","desactiver","souscrire","telecom","tunisie","prix",
    "cout","minute","mois","go","mo","data","forfaits","pack",
    "esim","vdsl","elissa","fixe","prepaye","postpaye","numero",
    "signal","connexion","hotspot","telechargement","streaming",
    "international","etranger","appels","messages","gigaoctet",
]

CORRECTIONS_DIRECTES = {
    "haya":"hayya","haiya":"hayya","hayia":"hayya",
    "forfet":"forfait","internt":"internet",
    "aboneman":"abonnement","abonement":"abonnement",
    "recharg":"recharge","recharje":"recharge",
    "netboc":"netbox","netbok":"netbox",
}

_TELECOM_VALID_WORDS = {
    "tunisie","telecom","clients","client","offre","service","services",
    "forfait","internet","mobile","reseau","disponible","activer",
    "souscrire","abonnement","recharge","solde","appel","sms","data",
    "go","mo","dt","dinar","mois","jours","gratuit","illimite",
    "tarif","prix","cout","numero","code","ussd","fibre","adsl",
    "netbox","roaming","international","pack","option","bonus",
}


# ─────────────────────────────────────────────────────────────
# CACHE SEMANTIQUE LRU
# ─────────────────────────────────────────────────────────────
class SemanticCache:
    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def _key(self, query: str) -> str:
        normalized = unicodedata.normalize("NFKC", query.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[dict]:
        if not SEMANTIC_CACHE_ENABLED:
            return None
        k = self._key(query)
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)
                return self._cache[k]
        return None

    def set(self, query: str, value: dict):
        if not SEMANTIC_CACHE_ENABLED:
            return
        k = self._key(query)
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)
            self._cache[k] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

_semantic_cache = SemanticCache(maxsize=SEMANTIC_CACHE_SIZE)


# ─────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────
def normalize_query(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def correct_telecom_keywords(query: str) -> str:
    words = query.lower().split()
    corrected = []
    for word in words:
        if len(word) <= 3:
            corrected.append(word); continue
        if word in CORRECTIONS_DIRECTES:
            corrected.append(CORRECTIONS_DIRECTES[word]); continue
        match = process.extractOne(word, VOCAB_TELECOM, scorer=fuzz.ratio, score_cutoff=70)
        corrected.append(match[0] if match and match[0] != word else word)
    return " ".join(corrected)

def preprocess_query(query: str) -> str:
    return normalize_query(correct_telecom_keywords(query))

def detect_theme(question: str) -> str:
    q = question.lower()
    for theme, kws in TELECOM_THEME_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return theme
    return ""

def detect_language(text: str) -> str:
    arabic_chars  = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text))
    latin_chars   = len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', text))
    other_scripts = len(re.findall(r'[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', text))
    total = arabic_chars + latin_chars + other_scripts

    if total == 0:
        return 'fr'
    if other_scripts / total > 0.30:
        return 'other'
    if arabic_chars / total >= 0.40:
        return 'ar'

    if latin_chars > 0:
        words_lower = set(re.findall(r'[a-z]{3,}', text.lower()))
        fr_indicators = {
            "bonjour","bonsoir","salut","merci","oui","non","je","tu","il","elle",
            "nous","vous","ils","elles","le","la","les","un","une","des","du","de",
            "et","est","que","qui","comment","pourquoi","quand","quel","quelle",
            "sur","avec","pour","dans","par","mais","aussi","plus","tres","bien",
            "avez","avoir","faire","aller","pouvoir","vouloir","savoir","voir",
            "offre","forfait","recharge","solde","abonnement","reseau","internet",
            "mobile","telecom","tunisie","service","activation","roaming","facture",
        }
        en_indicators = {
            "the","and","for","are","with","this","that","have","from","they",
            "will","your","what","when","how","can","please","thank","hello",
            "good","morning","evening","need","want","help","call","plan","data",
        }
        fr_score = len(words_lower & fr_indicators)
        en_score = len(words_lower & en_indicators)
        if en_score > fr_score and en_score >= 2:
            return 'other'

    return 'fr'

def is_greeting(query: str) -> bool:
    q = query.strip()
    if len(q) <= 60 and any(g in q for g in GREETINGS_AR):
        return True
    q_lower = q.lower().strip()
    return len(q_lower) <= 40 and any(g in q_lower for g in GREETINGS_FR)

def _normalize_for_kw(text: str) -> str:
    text = _ud.normalize('NFD', text.lower())
    text = ''.join(c for c in text if _ud.category(c) != 'Mn')
    return _re.sub(r'\s+', ' ', _re.sub(r'[^\w\s]', ' ', text)).strip()

def is_telecom_related(query: str) -> bool:
    q_norm  = _normalize_for_kw(query)
    q_words = set(q_norm.split())
    for kw in TELECOM_KEYWORDS:
        if kw in q_words: return True
    for product in TELECOM_PRODUCT_NAMES:
        if product in q_norm: return True
    for kw in TELECOM_KEYWORDS:
        if '\u0600' <= kw[0] <= '\u06FF' and kw in query:
            return True
    return False

def _is_multiline_query(query: str) -> bool:
    return any(kw in normalize_query(query.lower()) for kw in _MULTILINE_KEYWORDS) or \
           any(kw in query for kw in _MULTILINE_KEYWORDS if '\u0600' <= kw[0] <= '\u06FF')

def is_rag_relevant(chunks: List[dict], query: str) -> bool:
    if not chunks:
        return False
    stop = {"le","la","de","du","un","une","est","que","quel","comment","pour","dans","au","et","en"}
    q_norm  = normalize_query(query.lower())
    q_words = {w for w in q_norm.split() if len(w) > 3 and w not in stop}
    if not q_words:
        return True
    required = max(1, int(len(q_words) * 0.30))
    for chunk in chunks:
        combined = normalize_query((chunk.get("text","") + " " + chunk.get("filename","")).lower())
        if sum(1 for w in q_words if w in combined) >= required:
            return True
    return False

def clean_chunk(text: str) -> str:
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()

def clean_jsonl_artifacts(text: str) -> str:
    for pattern, replacement in _JSONL_ARTIFACTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = text.strip()
    if text and text[-1] not in '.!?\u061f':
        text = text.rstrip(',;:') + '.'
    return text

def get_fallback(lang: str) -> str:
    return FALLBACK_NO_INFO_AR if lang == 'ar' else FALLBACK_NO_INFO_FR

def get_hors_sujet(lang: str) -> str:
    return HORS_SUJET_AR if lang == 'ar' else HORS_SUJET_FR

def get_greeting_response(lang: str) -> str:
    return GREETING_RESPONSE_AR if lang == 'ar' else GREETING_RESPONSE_FR

def _get_free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        used  = torch.cuda.memory_allocated(0)
        return (total - used) / 1e9
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# GESTIONNAIRE MODELE
# ─────────────────────────────────────────────────────────────
class Phi35ModelManager:
    def __init__(self):
        self._model      = None
        self._tokenizer  = None
        self._lock       = threading.RLock()
        self._loaded     = False
        self._quant_mode = "unknown"

    def load(self):
        if self._loaded:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise SystemExit("CUDA non disponible.")

        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("[GPU] %s | VRAM: %.1f GB", name, vram)

        model_path = MODEL_DIR
        logger.info("[MODEL] Chargement Phi-3.5 : %s", model_path)
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if USE_4BIT:
            logger.info("[MODEL] Mode 4-bit NF4 active")
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map={"": "cuda:0"},
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
                self._quant_mode = "4bit_nf4"
                logger.info("[MODEL] Charge en 4-bit NF4")
            except ImportError:
                logger.warning("[MODEL] bitsandbytes non installe -> fallback fp16")
                self._load_fp16(model_path)
            except Exception as e:
                logger.warning("[MODEL] Erreur 4-bit (%s) -> fallback fp16", e)
                self._load_fp16(model_path)
        else:
            logger.info("[MODEL] Mode fp16")
            self._load_fp16(model_path)

        self._model.eval()
        self._model.config.use_cache = True

        if TORCH_COMPILE_PHI and hasattr(torch, "compile"):
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead", fullgraph=False)
                logger.info("[MODEL] torch.compile active")
            except Exception as e:
                logger.warning("[MODEL] torch.compile non disponible : %s", e)

        self._warmup()

        elapsed   = time.time() - t0
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        vram_free = _get_free_vram_gb()
        logger.info(
            "[MODEL] Pret en %.1fs | quant=%s | VRAM utilisee: %.2f GB | libre: %.2f GB",
            elapsed, self._quant_mode, vram_used, vram_free
        )
        self._loaded = True

    def _load_fp16(self, model_path: str):
        from transformers import AutoModelForCausalLM
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        self._quant_mode = "fp16"

    def _warmup(self):
        logger.info("[MODEL] Warmup...")
        try:
            dummy = self._tokenizer("Test.", return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                for _ in range(2):
                    _ = self._model.generate(
                        **dummy, max_new_tokens=5, do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[MODEL] Warmup OK")
        except Exception as e:
            logger.warning("[MODEL] Warmup echoue : %s", e)

    def get(self):
        if not self._loaded:
            raise RuntimeError("Modele non charge.")
        return self._model, self._tokenizer

    def is_loaded(self) -> bool:
        return self._loaded

    def quant_mode(self) -> str:
        return self._quant_mode

_gen_manager = Phi35ModelManager()


# ─────────────────────────────────────────────────────────────
# CONTEXTE RAG
# ─────────────────────────────────────────────────────────────
def _clean_chunk_for_context(text: str) -> str:
    text = clean_chunk(text)
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 15:
            continue
        if any(line.startswith(b) for b in BRUIT_DEBUT_PHRASE):
            continue
        if re.match(r'^[\d\s.,|DT%GoMo-]+$', line):
            continue
        clean_lines.append(line)
    return ' '.join(clean_lines).strip()


def _build_rag_context(chunks: List[dict], is_multiline: bool = False) -> str:
    max_chars = RAG_CONTEXT_CHARS_MULTILINE if is_multiline else RAG_CONTEXT_CHARS_SIMPLE
    parts = []
    for chunk in chunks[:3]:
        cleaned = _clean_chunk_for_context(chunk["text"])
        if cleaned and len(cleaned) > 20:
            parts.append(cleaned[:max_chars])
    context = " | ".join(parts)
    logger.info("[CTX] %d chars | chunks=%d | '%s'...",
                len(context), len(chunks), context[:120])
    return context


def _extract_answer_from_chunk(text: str) -> str:
    idx    = text.find('?')
    idx_ar = text.find('\u061f')
    candidates = [i for i in [idx, idx_ar] if i != -1]
    if candidates:
        best = min(candidates)
        if best < len(text) * 0.6:
            after = text[best + 1:].strip()
            if len(after) > 10:
                return clean_jsonl_artifacts(after)
    return clean_jsonl_artifacts(text)


# ─────────────────────────────────────────────────────────────
# VALIDATION POST-GENERATION
# ─────────────────────────────────────────────────────────────
def _token_overlap(text1: str, text2: str) -> float:
    w1 = set(normalize_query(text1.lower()).split())
    w2 = set(normalize_query(text2.lower()).split())
    if not w1 or not w2: return 0.0
    return len(w1 & w2) / max(len(w1), len(w2))


def _validate_phi_response(
    response: str,
    query: str,
    context: str,
    chunks: List[dict] = None,
    attempt: int = 1,
) -> tuple:
    if len(response.strip()) < 10:
        return False, "trop_court"

    if any(sig in response.lower() for sig in HALLUCINATION_SIGNALS):
        return False, "hallucination"

    resp_norm = normalize_query(response.lower())
    for leak in PROMPT_LEAKS_PHI:
        leak_norm = (normalize_query(leak.lower())
                     if not any('\u0600' <= c <= '\u06FF' for c in leak)
                     else leak)
        if leak_norm in resp_norm or leak in response:
            return False, "connaissance_generale"

    if any(n in response.lower() for n in GENERATION_NOISE):
        return False, "bruit"

    overlap = _token_overlap(response, context)
    if overlap > 0.97 and len(response) > 150:
        return False, f"copie_contexte_{overlap:.2f}"

    extended_context = context
    if chunks:
        for c in chunks[:3]:
            extended_context += " " + c.get("text", "")

    context_words  = set(normalize_query(extended_context.lower()).split())
    response_words = [
        w for w in normalize_query(response.lower()).split()
        if len(w) > 4 and w not in _TELECOM_VALID_WORDS
    ]

    if response_words:
        coverage = sum(1 for w in response_words if w in context_words) / len(response_words)
        if coverage < MIN_CONTEXT_COVERAGE:
            logger.debug(
                "[VALID-DEBUG] couverture=%.3f | reponse='%s'...",
                coverage, response[:80]
            )
            return False, f"couverture_faible_{coverage:.2f}"

    return True, "ok"


# ─────────────────────────────────────────────────────────────
# REPONSE EXTRACTIVE (fallback)
# ─────────────────────────────────────────────────────────────
def build_extractive_answer(chunks: List[dict], query: str, lang: str = 'fr') -> str:
    if not chunks:
        return get_fallback(lang)
    stop = {"le","la","de","du","un","une","est","que","pour","dans","au","et","en"}
    q_norm  = normalize_query(query.lower())
    q_words = {w for w in q_norm.split() if len(w) > 3 and w not in stop}
    all_sentences = []
    for chunk in chunks:
        raw       = clean_chunk(chunk["text"])
        ans_part  = _extract_answer_from_chunk(raw)
        sentences = [s.strip() for s in re.split(r'[.!?\u061f;]\s+|\n', ans_part) if len(s.strip()) > 25]
        bonus_kw  = [
            "tarif","prix","dt","offre","activation","forfait","go","mo",
            "minute","mois","gratuit","\u062a\u0639\u0631\u064a\u0641\u0629","\u0639\u0631\u0636","\u062a\u0641\u0639\u064a\u0644","\u0628\u0627\u0642\u0629","\u0645\u062c\u0627\u0646\u064a"
        ]
        scored = []
        for s in sentences:
            s_norm = normalize_query(s.lower())
            score  = sum(1 for w in q_words if w in s_norm)
            score += sum(0.5 for kw in bonus_kw if kw in s.lower() or kw in s)
            if not any(s.strip().startswith(b) for b in BRUIT_DEBUT_PHRASE):
                scored.append((score, s))
        scored.sort(reverse=True)
        all_sentences.extend(s for sc, s in scored[:2] if sc > 0)
    if not all_sentences:
        return get_fallback(lang)
    best = all_sentences[0]
    if len(best) > 300:
        best = re.split(r'(?<=[.!?\u061f])\s+', best[:350])[0]
    return clean_jsonl_artifacts(best)


# ─────────────────────────────────────────────────────────────
# [FIX-v1.6] NETTOYAGE OUTPUT — supprime fragments question et "Response:"
# ─────────────────────────────────────────────────────────────
def _clean_phi_output(text: str) -> str:
    # Supprime les tokens speciaux
    text = re.sub(r"<\|.*?\|>", "", text).strip()
    # Supprime "assistant:" en debut
    text = re.sub(r"^(?:assistant|Assistant)\s*:\s*", "", text).strip()
    # Supprime "Response:" / "Reponse:" en debut
    text = re.sub(r"^(?:Response|Reponse|Reponds|Answer|Repondre)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    # Supprime les fragments de question avant la reponse (pattern: "...? Response: ...")
    # Detecte si le texte commence par un fragment de question
    match = re.match(r"^[^.!?\u061f]{0,120}[?\u061f]\s*(?:Response|Reponse)?\s*:?\s*", text, flags=re.DOTALL)
    if match and match.end() < len(text) * 0.6:
        text = text[match.end():].strip()
    # Supprime "..." en fin
    text = re.sub(r'\s*\.\.\.\s*$', '.', text).strip()
    return clean_jsonl_artifacts(text)


# ─────────────────────────────────────────────────────────────
# GENERATION PHI-3.5 AVEC RETRY
# ─────────────────────────────────────────────────────────────
def _run_single_generation(
    model, tokenizer, messages: list,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 0.1,
) -> tuple:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to("cuda:0")

    n_input_tokens = inputs["input_ids"].shape[-1]
    logger.info("[GEN-PHI] Tokens input : %d / max %d", n_input_tokens, MAX_INPUT_LENGTH)

    # [FIX-v1.6] Ne pas passer temperature quand do_sample=False (evite warning)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        min_new_tokens=5,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    t_start = time.time()
    with _cuda_generation_lock:
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)
    elapsed    = time.time() - t_start
    new_tokens = output_ids[0][n_input_tokens:]
    n_new      = len(new_tokens)
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw_answer, n_new, elapsed


def _generate_sync(question: str, chunks: List[dict], lang: str = 'fr') -> tuple:
    is_multiline = _is_multiline_query(question)

    if lang == 'ar':
        sys_prompt  = GENERATIVE_SYSTEM_PROMPT_AR_MULTILINE if is_multiline else GENERATIVE_SYSTEM_PROMPT_AR
        instruction = "Reponds en 2-3 phrases en arabe en te basant UNIQUEMENT sur le contexte."
    else:
        sys_prompt  = GENERATIVE_SYSTEM_PROMPT_MULTILINE if is_multiline else GENERATIVE_SYSTEM_PROMPT
        instruction = "Reponds en 2-3 phrases en francais en te basant UNIQUEMENT sur le contexte."

    context = _build_rag_context(chunks, is_multiline=is_multiline)

    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                f"{instruction}"
            )
        },
    ]

    if not _gen_manager.is_loaded():
        logger.error("[GEN-PHI] Modele non charge -> extractif")
        return build_extractive_answer(chunks, question, lang), "extractive_fallback"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        free_vram = _get_free_vram_gb()
        logger.info("[GEN-PHI] VRAM libre : %.2f GB (mode=%s)", free_vram, _gen_manager.quant_mode())
        if free_vram < MIN_FREE_VRAM_GB:
            logger.warning("[GEN-PHI] VRAM insuffisante -> extractif")
            with _metrics_lock:
                _global_metrics["generative_vram_skip"] += 1
            return build_extractive_answer(chunks, question, lang), "extractive_oom"

    try:
        model, tokenizer = _gen_manager.get()

        last_reason = "unknown"
        # tentative 1 : greedy, tentative 2 : sampling
        configs = [(False, 0.1), (True, 0.3)]

        for attempt in range(1, GEN_MAX_RETRIES + 1):
            do_sample, temp = configs[min(attempt - 1, len(configs) - 1)]
            logger.info("[GEN-PHI] Tentative %d/%d | do_sample=%s", attempt, GEN_MAX_RETRIES, do_sample)

            try:
                raw_answer, n_new_tokens, elapsed = _run_single_generation(
                    model, tokenizer, messages,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=do_sample,
                    temperature=temp,
                )
            except torch.cuda.OutOfMemoryError:
                logger.error("[GEN-PHI] OOM tentative %d", attempt)
                torch.cuda.empty_cache()
                gc.collect()
                with _metrics_lock:
                    _global_metrics["generative_error_oom"] += 1
                break

            if elapsed > GEN_TIMEOUT_S:
                logger.warning("[GEN-PHI] Timeout (%.1fs) tentative %d", elapsed, attempt)
                with _metrics_lock:
                    _global_metrics["generative_timeout_count"] += 1
                break

            response = _clean_phi_output(raw_answer)
            logger.info(
                "[GEN-PHI] Tentative %d — %.2fs | %d tokens | '%s'...",
                attempt, elapsed, n_new_tokens, response[:150]
            )

            sentences = re.split(r'(?<=[.!?\u061f])\s+', response)
            if len(sentences) > 6:
                response = " ".join(sentences[:6])

            valid, reason = _validate_phi_response(
                response, question, context, chunks=chunks, attempt=attempt
            )

            if valid:
                if attempt > 1:
                    logger.info("[GEN-PHI] Succes au retry (tentative %d)", attempt)
                    with _metrics_lock:
                        _global_metrics["generative_retry_success"] += 1
                with _metrics_lock:
                    _global_metrics["generative_count"] += 1
                logger.info("[GEN-PHI] Reponse generee tentative=%d | %.2fs", attempt, elapsed)
                return response, "generative"
            else:
                last_reason = reason
                logger.warning(
                    "[GEN-PHI] Rejet tentative %d — raison='%s' | reponse='%s'...",
                    attempt, reason, response[:100]
                )
                with _metrics_lock:
                    _global_metrics["validation_rejected_count"] += 1
                    _global_metrics["validation_rejection_reasons"][reason] += 1

        logger.warning("[GEN-PHI] %d tentative(s) echouee(s) — raison='%s' -> extractif",
                       GEN_MAX_RETRIES, last_reason)
        with _metrics_lock:
            _global_metrics["generative_retry_fail"] += 1
            _global_metrics["extractive_count"]       += 1
        return build_extractive_answer(chunks, question, lang), "extractive_validation"

    except RuntimeError as e:
        logger.error("[GEN-PHI] RuntimeError : %s\n%s", e, traceback.format_exc())
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        with _metrics_lock:
            _global_metrics["generative_error_runtime"] += 1
            _global_metrics["extractive_count"]         += 1
        return build_extractive_answer(chunks, question, lang), "extractive_runtime_error"

    except Exception as e:
        logger.error("[GEN-PHI] Exception inattendue :\n%s", traceback.format_exc())
        with _metrics_lock:
            _global_metrics["generative_error_other"] += 1
            _global_metrics["extractive_count"]       += 1
        return build_extractive_answer(chunks, question, lang), "extractive_fallback"


async def generate_answer(question: str, chunks: List[dict], lang: str = 'fr') -> tuple:
    return await asyncio.to_thread(_generate_sync, question, chunks, lang)


# ─────────────────────────────────────────────────────────────
# RAG — ChromaDB
# ─────────────────────────────────────────────────────────────
_collection  = None
_embed_model = None
_embed_lock  = threading.Lock()

def _get_embed_model():
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("[EMBED] Chargement encodeur : %s", EMBED_MODEL)
            _embed_model = SentenceTransformer(EMBED_MODEL)
            logger.info("[EMBED] Encodeur pret.")
        return _embed_model

def init_rag():
    global _collection
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    _collection = client.get_collection(name=COLLECTION_NAME)
    _get_embed_model()
    logger.info("ChromaDB — %d chunks / '%s'", _collection.count(), COLLECTION_NAME)

def rag_search(query: str, top_k: int = TOP_K) -> List[dict]:
    encoder = _get_embed_model()
    q_emb   = encoder.encode([query], convert_to_numpy=True).tolist()
    results = _collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k + 1, _collection.count()),
        include=["documents","metadatas","distances"]
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        hits.append({
            "text":     doc,
            "score":    round(1 - dist, 4),
            "filename": meta.get("file_name", meta.get("filename", "")),
            "year":     meta.get("year",  ""),
            "theme":    meta.get("theme", ""),
        })
    seen = {}
    for h in hits:
        if h["text"] not in seen or h["score"] > seen[h["text"]]["score"]:
            seen[h["text"]] = h
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]

def rag_search_best(query_original: str, query_processed: str, top_k: int = TOP_K) -> List[dict]:
    hits1      = rag_search(query_original, top_k)
    similarity = fuzz.ratio(query_original.lower(), query_processed)
    hits2      = rag_search(query_processed, top_k) if similarity < 85 else []
    seen = {}
    for h in hits1 + hits2:
        if h["text"] not in seen or h["score"] > seen[h["text"]]["score"]:
            seen[h["text"]] = h
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────
# SQLITE
# ─────────────────────────────────────────────────────────────
_db_lock = threading.Lock()

@contextmanager
def get_db():
    os.makedirs(
        os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".",
        exist_ok=True
    )
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn; conn.commit()
    except Exception:
        conn.rollback(); raise
    finally:
        conn.close()

def init_db():
    os.makedirs(
        os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".",
        exist_ok=True
    )
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
            CREATE TABLE IF NOT EXISTS unanswered_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                reason TEXT NOT NULL,
                is_telecom INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                msg_id TEXT DEFAULT '',
                rating INTEGER NOT NULL CHECK(rating IN (1,-1)),
                comment TEXT DEFAULT '',
                user_question TEXT DEFAULT '',
                bot_answer TEXT DEFAULT '',
                timestamp TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS rag_enrichments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL UNIQUE,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                theme TEXT NOT NULL DEFAULT 'enrichissement_admin',
                source_note TEXT DEFAULT '',
                added_by TEXT DEFAULT 'admin',
                timestamp TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );
        """)
        for sql in [
            "ALTER TABLE feedback ADD COLUMN user_question TEXT DEFAULT ''",
            "ALTER TABLE feedback ADD COLUMN bot_answer TEXT DEFAULT ''",
            "ALTER TABLE unanswered_questions ADD COLUMN is_telecom INTEGER NOT NULL DEFAULT 0",
        ]:
            try: conn.execute(sql)
            except Exception: pass
    logger.info("DB initialisee : %s", DB_PATH)

def save_message(sid, role, content):
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (?,?,?)",
                (sid, role, content)
            )

def get_history(sid, limit=MAX_HISTORY):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations "
            "WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (sid, limit * 2)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def log_unanswered(sid, question, reason, is_telecom: bool = False):
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO unanswered_questions "
                "(session_id, question, reason, is_telecom) VALUES (?,?,?,?)",
                (sid, question, reason, int(is_telecom))
            )

def delete_session(sid):
    with _db_lock:
        with get_db() as conn:
            return conn.execute(
                "DELETE FROM conversations WHERE session_id=?", (sid,)
            ).rowcount


# ─────────────────────────────────────────────────────────────
# SCHEMAS PYDANTIC
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: Optional[str] = None
    mode:       Optional[str] = None

class ChatResponse(BaseModel):
    session_id:       str
    answer:           str
    sources:          List[dict]
    is_telecom:       bool
    rag_used:         bool
    confidence:       float
    response_time_ms: int
    mode:             str
    cache_hit:        bool = False
    lang:             str  = "fr"

class FeedbackRequest(BaseModel):
    session_id:    str
    msg_id:        Optional[str] = None
    rating:        int
    comment:       Optional[str] = ""
    user_question: Optional[str] = ""
    bot_answer:    Optional[str] = ""
    timestamp:     Optional[str] = None

class RagEntryRequest(BaseModel):
    question:    str
    answer:      str
    theme:       Optional[str] = "enrichissement_admin"
    source_note: Optional[str] = ""


# ─────────────────────────────────────────────────────────────
# FASTAPI — lifespan
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Chatbot Tunisie Telecom — Phi-3.5-mini-instruct v1.6")
    logger.info("  MODEL_DIR=%s", MODEL_DIR)
    logger.info("  USE_4BIT=%s | MAX_INPUT_LENGTH=%d", USE_4BIT, MAX_INPUT_LENGTH)
    logger.info("  RAG_CTX=%d/%d | GEN_MAX_RETRIES=%d",
                RAG_CONTEXT_CHARS_SIMPLE, RAG_CONTEXT_CHARS_MULTILINE, GEN_MAX_RETRIES)
    logger.info("=" * 60)
    init_db()
    init_auth_db()
    init_rag()
    _semantic_cache.clear()

    logger.info("[STARTUP] Chargement Phi-3.5...")
    try:
        _gen_manager.load()
        logger.info("[STARTUP] Modele charge (mode=%s)", _gen_manager.quant_mode())
        if torch.cuda.is_available():
            vram_used  = torch.cuda.memory_allocated(0) / 1e9
            vram_free  = _get_free_vram_gb()
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(
                "[STARTUP] VRAM : utilisee=%.2f GB | libre=%.2f GB | total=%.1f GB",
                vram_used, vram_free, vram_total
            )
    except Exception as e:
        logger.error("[STARTUP] ECHEC chargement : %s", e)
    logger.info("[STARTUP] API operationnelle — v1.6")
    yield
    logger.info("API arretee")

app = FastAPI(
    title="Chatbot Tunisie Telecom — Phi-3.5 v1.6",
    description="RAG + Phi-3.5-mini-instruct 4-bit NF4 | Mode generatif | FR/AR",
    version="1.6.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────
# ROUTES AUTH
# ─────────────────────────────────────────────────────────────
@app.post("/auth/register", tags=["Auth"])
async def register(req: RegisterRequest):
    return register_route(req)

@app.post("/auth/verify", tags=["Auth"])
async def verify(req: VerifyRequest):
    return verify_route(req)

@app.post("/auth/login", tags=["Auth"])
async def login(req: LoginRequest):
    return login_route(req)

@app.post("/auth/resend", tags=["Auth"])
async def resend(req: ResendRequest):
    return resend_route(req)

@app.get("/auth/me", tags=["Auth"])
async def me(user=Depends(require_auth)):
    return me_route(user)

@app.post("/auth/logout", tags=["Auth"])
async def logout(authorization: str = Header(None)):
    return logout_route(authorization)

@app.post("/auth/check-email", tags=["Auth"])
async def check_email(req: CheckEmailRequest):
    return await check_email_route(req)


# ─────────────────────────────────────────────────────────────
# ROUTE CHAT
# ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    t0         = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    query      = request.message.strip()
    cache_hit  = False

    if not query:
        raise HTTPException(400, "Message vide.")

    lang = detect_language(query)
    logger.info("[LANG] detectee=%s | query='%s'", lang, query[:60])

    if lang == 'other':
        answer  = LANG_NOT_SUPPORTED_MSG
        elapsed = int((time.time() - t0) * 1000)
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", answer)
        log_unanswered(session_id, query, "langue_non_supportee", is_telecom=False)
        with _metrics_lock:
            _global_metrics["total_requests"]      += 1
            _global_metrics["lang_rejected_count"] += 1
        return ChatResponse(
            session_id=session_id, answer=answer, sources=[],
            is_telecom=False, rag_used=False, confidence=0.0,
            response_time_ms=elapsed, mode="lang_rejected", lang=lang)

    if lang == 'ar':
        with _metrics_lock:
            _global_metrics["arabic_query_count"] += 1

    if is_greeting(query):
        answer  = get_greeting_response(lang)
        elapsed = int((time.time() - t0) * 1000)
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", answer)
        with _metrics_lock:
            _global_metrics["total_requests"] += 1
            _global_metrics["greeting_count"] += 1
        return ChatResponse(
            session_id=session_id, answer=answer, sources=[],
            is_telecom=True, rag_used=False, confidence=1.0,
            response_time_ms=elapsed, mode="greeting", lang=lang)

    if not is_telecom_related(query):
        log_unanswered(session_id, query, "hors_sujet", is_telecom=False)
        answer  = get_hors_sujet(lang)
        elapsed = int((time.time() - t0) * 1000)
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", answer)
        with _metrics_lock:
            _global_metrics["total_requests"]   += 1
            _global_metrics["hors_sujet_count"] += 1
        return ChatResponse(
            session_id=session_id, answer=answer, sources=[],
            is_telecom=False, rag_used=False, confidence=0.0,
            response_time_ms=elapsed, mode="hors_sujet", lang=lang)

    cached = _semantic_cache.get(query)
    if cached:
        elapsed = int((time.time() - t0) * 1000)
        save_message(session_id, "user", query)
        save_message(session_id, "assistant", cached["answer"])
        with _metrics_lock:
            _global_metrics["total_requests"]  += 1
            _global_metrics["cache_hit_count"] += 1
        return ChatResponse(
            session_id=session_id, answer=cached["answer"],
            sources=cached["sources"], is_telecom=True,
            rag_used=cached["rag_used"], confidence=cached["confidence"],
            response_time_ms=elapsed, mode=cached["mode"]+"_cached",
            cache_hit=True, lang=cached.get("lang", lang))

    query_processed = preprocess_query(query)
    hits            = rag_search_best(query, query_processed)
    confidence      = hits[0]["score"] if hits else 0.0
    rag_relevant    = is_rag_relevant(hits, query)

    logger.info("[RAG] conf=%.3f | pertinent=%s | chunks=%d", confidence, rag_relevant, len(hits))
    for i, h in enumerate(hits[:3]):
        logger.info("[CHUNK-%d] %.3f | %s | '%s'...", i+1, h["score"], h["filename"], h["text"][:80])

    if hits and confidence >= CONF_LOW and rag_relevant:
        rag_used = True
        answer, answer_mode = await generate_answer(query, hits, lang)
    else:
        reason      = "confiance_faible" if confidence < CONF_LOW else "chunk_non_pertinent"
        log_unanswered(session_id, query, reason, is_telecom=True)
        answer      = get_fallback(lang)
        answer_mode = "no_info"
        rag_used    = False
        hits        = []
        with _metrics_lock:
            _global_metrics["no_info_count"] += 1

    if any(sig in answer.lower() for sig in HALLUCINATION_SIGNALS):
        log_unanswered(session_id, query, "hallucination_detectee", is_telecom=True)
        answer      = get_fallback(lang)
        answer_mode = "fallback_securite"
        with _metrics_lock:
            _global_metrics["hallucination_count"] += 1

    save_message(session_id, "user", query)
    save_message(session_id, "assistant", answer)
    elapsed = int((time.time() - t0) * 1000)

    sources = [
        {
            "filename": h["filename"], "year": h["year"],
            "theme": h["theme"], "score": h["score"],
            "text": h["text"][:300]
        }
        for h in (hits if rag_used else [])
    ]

    if SEMANTIC_CACHE_ENABLED and rag_used and answer_mode == "generative":
        _semantic_cache.set(query, {
            "answer": answer, "sources": sources,
            "rag_used": rag_used, "confidence": confidence,
            "mode": answer_mode, "lang": lang,
        })

    with _metrics_lock:
        _global_metrics["total_requests"]   += 1
        _global_metrics["rag_used_count"]   += int(rag_used)
        _global_metrics["total_latency_ms"] += elapsed
        _global_metrics["confidence_scores"].append(confidence)

    logger.info(
        "[%s] %dms | conf=%.3f | rag=%s | lang=%s | quant=%s",
        answer_mode, elapsed, confidence, rag_used, lang, _gen_manager.quant_mode()
    )
    return ChatResponse(
        session_id=session_id, answer=answer, sources=sources,
        is_telecom=True, rag_used=rag_used, confidence=confidence,
        response_time_ms=elapsed, mode=answer_mode,
        cache_hit=cache_hit, lang=lang)


# ─────────────────────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────────────────────
@app.post("/feedback", tags=["Feedback"])
async def feedback_endpoint(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(400, "rating doit etre 1 ou -1")
    ts = req.timestamp or datetime.now().isoformat()
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO feedback "
                "(session_id, msg_id, rating, comment, user_question, bot_answer, timestamp) "
                "VALUES (?,?,?,?,?,?,?)",
                (req.session_id or "unknown", req.msg_id or "", req.rating,
                 req.comment or "", req.user_question or "", req.bot_answer or "", ts)
            )
    with _metrics_lock:
        key = "positive_feedback" if req.rating == 1 else "negative_feedback"
        _global_metrics[key] = _global_metrics.get(key, 0) + 1
    return {"status": "ok"}

@app.get("/feedback/stats", tags=["Feedback"])
async def feedback_stats():
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) as positive, "
            "SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) as negative "
            "FROM feedback"
        ).fetchone()
        recent = conn.execute(
            "SELECT session_id, msg_id, comment, user_question, bot_answer, timestamp "
            "FROM feedback WHERE rating=-1 ORDER BY timestamp DESC LIMIT 15"
        ).fetchall()
    pos, neg, total = row["positive"] or 0, row["negative"] or 0, row["total"] or 0
    return {
        "positive": pos, "negative": neg, "total": total,
        "satisfaction_rate": round(100*pos/total, 1) if total > 0 else 0,
        "recent_negative": [dict(r) for r in recent],
    }


# ─────────────────────────────────────────────────────────────
# HISTORIQUE
# ─────────────────────────────────────────────────────────────
@app.get("/history/{session_id}", tags=["Historique"])
async def history(session_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM conversations "
            "WHERE session_id=? ORDER BY id",
            (session_id,)).fetchall()
    if not rows:
        raise HTTPException(404, "Session introuvable.")
    return [dict(r) for r in rows]

@app.delete("/history/{session_id}", tags=["Historique"])
async def clear_history(session_id: str):
    n = delete_session(session_id)
    if n == 0:
        raise HTTPException(404, "Session introuvable.")
    return {"deleted": n}


# ─────────────────────────────────────────────────────────────
# RAG ADMIN
# ─────────────────────────────────────────────────────────────
@app.post("/admin/rag/add", tags=["RAG Admin"])
async def rag_add(req: RagEntryRequest, user=Depends(require_auth)):
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(400, "question et answer sont obligatoires")
    chunk_id   = f"admin_{uuid.uuid4().hex[:12]}"
    chunk_text = (
        f"Pour la question : {req.question.strip()}, "
        f"voici la reponse : {req.answer.strip()}."
    )
    try:
        _collection.add(
            documents=[chunk_text],
            metadatas=[{
                "file_name": "admin_enrichment",
                "filename":  "admin_enrichment",
                "year":      str(datetime.now().year),
                "theme":     req.theme or "enrichissement_admin",
                "question":  req.question[:200],
            }],
            ids=[chunk_id]
        )
    except Exception as e:
        raise HTTPException(500, f"Erreur ajout ChromaDB : {e}")
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_enrichments "
                "(chunk_id, question, answer, theme, source_note) VALUES (?,?,?,?,?)",
                (chunk_id, req.question, req.answer,
                 req.theme or "enrichissement_admin", req.source_note or "")
            )
    _semantic_cache.clear()
    return {"status": "ok", "chunk_id": chunk_id, "n_chunks": _collection.count()}

@app.get("/admin/rag/entries", tags=["RAG Admin"])
async def rag_entries(limit: int = 100, user=Depends(require_auth)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, chunk_id, question, answer, theme, timestamp "
            "FROM rag_enrichments ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"total": len(rows), "entries": [dict(r) for r in rows]}

@app.delete("/admin/cache/clear", tags=["Admin"])
async def clear_cache(user=Depends(require_auth)):
    _semantic_cache.clear()
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────
# METRIQUES & HEALTH
# ─────────────────────────────────────────────────────────────
@app.get("/metrics", tags=["Admin"])
async def get_metrics(user=Depends(require_auth)):
    with _metrics_lock:
        m = dict(_global_metrics)
    n    = m["total_requests"] or 1
    conf = m.pop("confidence_scores", [])
    m.pop("response_lengths", [])
    rejection_reasons = dict(m.pop("validation_rejection_reasons", {}))
    vram_used = (torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0)
    vram_free = _get_free_vram_gb()
    return {
        **m,
        "rag_rate":                    m["rag_used_count"] / n,
        "avg_latency_ms":              m["total_latency_ms"] / n,
        "avg_confidence":              statistics.mean(conf) if conf else 0,
        "generative_rate":             m["generative_count"] / n,
        "no_info_rate":                m["no_info_count"] / n,
        "cache_hit_rate":              m["cache_hit_count"] / n,
        "validation_rejected_rate":    m["validation_rejected_count"] / n,
        "lang_rejected_rate":          m["lang_rejected_count"] / n,
        "arabic_query_rate":           m["arabic_query_count"] / n,
        "validation_rejection_reasons": rejection_reasons,
        "generative_errors": {
            "oom":       m["generative_error_oom"],
            "runtime":   m["generative_error_runtime"],
            "other":     m["generative_error_other"],
            "vram_skip": m["generative_vram_skip"],
        },
        "generative_retries": {
            "success": m["generative_retry_success"],
            "fail":    m["generative_retry_fail"],
        },
        "model_loaded":                _gen_manager.is_loaded(),
        "quant_mode":                  _gen_manager.quant_mode(),
        "use_4bit":                    USE_4BIT,
        "version":                     "1.6.0",
        "model_dir":                   MODEL_DIR,
        "collection":                  COLLECTION_NAME,
        "conf_low":                    CONF_LOW,
        "min_context_coverage":        MIN_CONTEXT_COVERAGE,
        "max_new_tokens":              MAX_NEW_TOKENS,
        "max_input_length":            MAX_INPUT_LENGTH,
        "gen_timeout_s":               GEN_TIMEOUT_S,
        "gen_max_retries":             GEN_MAX_RETRIES,
        "gpu_vram_used_gb":            round(vram_used, 2),
        "gpu_vram_free_gb":            round(vram_free, 2),
        "rag_context_chars_simple":    RAG_CONTEXT_CHARS_SIMPLE,
        "rag_context_chars_multiline": RAG_CONTEXT_CHARS_MULTILINE,
        "semantic_cache_size":         _semantic_cache.size(),
        "supported_languages":         ["fr", "ar"],
        "timestamp":                   datetime.now().isoformat(),
    }

@app.get("/health", tags=["Systeme"])
async def health():
    n_chunks = _collection.count() if _collection else 0
    with get_db() as conn:
        n_msgs  = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
        n_sess  = conn.execute("SELECT COUNT(DISTINCT session_id) as c FROM conversations").fetchone()["c"]
        n_fb    = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
        n_rag   = conn.execute("SELECT COUNT(*) as c FROM rag_enrichments").fetchone()["c"]
        n_unans = conn.execute("SELECT COUNT(*) as c FROM unanswered_questions").fetchone()["c"]
    vram_used  = (torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0)
    vram_free  = _get_free_vram_gb()
    vram_total = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available() else 0
    )
    model_loaded = _gen_manager.is_loaded()
    status = "ok" if (model_loaded and n_chunks > 0 and vram_free >= 0.3) else (
        "vram_critical" if vram_free < 0.3 else "degraded"
    )
    return {
        "status":               status,
        "version":              "1.6.0",
        "model":                "Phi-3.5-mini-instruct",
        "model_loaded":         model_loaded,
        "quant_mode":           _gen_manager.quant_mode(),
        "use_4bit":             USE_4BIT,
        "gpu":                  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "gpu_vram_used_gb":     round(vram_used, 2),
        "gpu_vram_free_gb":     round(vram_free, 2),
        "gpu_vram_total_gb":    round(vram_total, 1),
        "rag_chunks":           n_chunks,
        "collection":           COLLECTION_NAME,
        "total_messages":       n_msgs,
        "total_sessions":       n_sess,
        "total_feedback":       n_fb,
        "rag_enrichments":      n_rag,
        "unanswered":           n_unans,
        "cache_size":           _semantic_cache.size(),
        "min_context_coverage": MIN_CONTEXT_COVERAGE,
        "max_new_tokens":       MAX_NEW_TOKENS,
        "max_input_length":     MAX_INPUT_LENGTH,
        "gen_max_retries":      GEN_MAX_RETRIES,
        "timestamp":            datetime.now().isoformat(),
    }

@app.get("/logs/unanswered", tags=["Admin"])
async def unanswered(
    limit: int = 50,
    reason: Optional[str] = None,
    user=Depends(require_auth)
):
    with get_db() as conn:
        if reason:
            rows = conn.execute(
                "SELECT id, session_id, question, reason, is_telecom, timestamp "
                "FROM unanswered_questions WHERE reason=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (reason, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, session_id, question, reason, is_telecom, timestamp "
                "FROM unanswered_questions ORDER BY timestamp DESC LIMIT ?",
                (limit,)).fetchall()
    return [dict(r) for r in rows]

@app.get("/admin/conversations/stats", tags=["Admin"])
async def conversations_stats(limit: int = 1000, user=Depends(require_auth)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT content, session_id, timestamp FROM conversations "
            "WHERE role='user' ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    questions  = [dict(r) for r in rows]
    total_q    = len(questions)
    q_counter  = Counter()
    q_original = {}
    for r in questions:
        key = r["content"].strip().lower()[:100]
        q_counter[key] += 1
        if key not in q_original:
            q_original[key] = r["content"].strip()
    top5 = [
        {"question": q_original[k], "count": c}
        for k, c in q_counter.most_common(5)
    ]
    theme_counter = Counter()
    for r in questions:
        theme_counter[detect_theme(r["content"])] += 1
    return {
        "total_questions": total_q,
        "top5_questions":  top5,
        "top_themes": [
            {"theme": t, "count": c}
            for t, c in theme_counter.most_common(10) if t
        ],
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/", include_in_schema=False)
async def root():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path)
    return {"message": "Chatbot Tunisie Telecom — Phi-3.5-mini-instruct v1.6"}


# ─────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Chatbot Tunisie Telecom — Phi-3.5-mini-instruct v1.6")
    print("=" * 60)
    print(f"  MODEL_DIR        : {MODEL_DIR}")
    print(f"  MAX_INPUT_LENGTH : {MAX_INPUT_LENGTH}")
    print(f"  RAG_CONTEXT      : {RAG_CONTEXT_CHARS_SIMPLE}/{RAG_CONTEXT_CHARS_MULTILINE} chars")
    print(f"  GEN_MAX_RETRIES  : {GEN_MAX_RETRIES}")
    print(f"  ChromaDB         : {CHROMA_DB_DIR} / {COLLECTION_NAME}")
    print(f"  DB               : {DB_PATH}")
    print(f"  Port             : {API_PORT}")
    print("=" * 60)
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False, log_level="info")