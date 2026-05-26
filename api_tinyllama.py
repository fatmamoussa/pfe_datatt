"""
=============================================================
API v2.16 — Chatbot Tunisie Telecom — TinyLlama 1.1B
CORRECTIONS v2.16 :
  [FIX-1] Routes auth montées (/auth/login, /auth/register, etc.)
  [FIX-2] init_auth_db() appelé dans lifespan
  [FIX-3] Routes manquantes ajoutées :
            GET /admin/conversations/stats
            GET /feedback/stats
  [FIX-4] Header importé depuis fastapi (manquait pour require_auth)
=============================================================
"""

import os, re, time, uuid, sqlite3, logging, threading, statistics, gc, json
import unicodedata
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import unicodedata as _ud
import re as _re
import torch
from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from rapidfuzz import fuzz, process

# =============================================================
# [FIX-1] Import auth_sqlite
# =============================================================
from auth_sqlite import (
    init_auth_db,
    register_route,
    verify_route,
    login_route,
    resend_route,
    require_auth,
    me_route,
    logout_route,
    check_email_route,
    RegisterRequest,
    VerifyRequest,
    LoginRequest,
    ResendRequest,
    CheckEmailRequest,
)

# =============================================================
# CONFIG — chemins relatifs au projet rag_fromscratch
# =============================================================
MODEL_DIR       = os.environ.get("MODEL_DIR_TINY", "models/tinyllama-tt-merged")
BASE_MODEL      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EXTRACTIVE_MODE = os.environ.get("EXTRACTIVE_MODE", "false").lower() == "true"

MAX_NEW_TOKENS     = int(os.environ.get("MAX_NEW_TOKENS",     "120"))
MIN_NEW_TOKENS     = int(os.environ.get("MIN_NEW_TOKENS",     "40"))
MAX_NEW_TOKENS_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "280"))

DO_SAMPLE          = False
REPETITION_PENALTY = 1.3

CHROMA_DB_DIR   = os.environ.get("CHROMA_DB_DIR",   "chroma_tt_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "tt_train")
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K           = int(os.environ.get("TOP_K", "3"))

ADMIN_SCORE_BOOST      = float(os.environ.get("ADMIN_SCORE_BOOST",      "0.10"))
ADMIN_DIRECT_THRESHOLD = float(os.environ.get("ADMIN_DIRECT_THRESHOLD", "0.72"))

CONFIDENCE_THRESHOLD  = float(os.environ.get("CONFIDENCE_THRESHOLD",  "0.45"))
THRESHOLD_SHORT_QUERY = float(os.environ.get("THRESHOLD_SHORT_QUERY", "0.38"))
THRESHOLD_LONG_QUERY  = float(os.environ.get("THRESHOLD_LONG_QUERY",  "0.50"))
SHORT_QUERY_MAX_WORDS = 4
MIN_CONTEXT_COVERAGE  = float(os.environ.get("MIN_CONTEXT_COVERAGE",  "0.20"))

MODEL_IDLE_TIMEOUT         = int(os.environ.get("MODEL_IDLE_TIMEOUT",         "1800"))
STRUCTURED_CHUNK_MAX_CHARS = int(os.environ.get("STRUCTURED_CHUNK_MAX_CHARS", "600"))

TORCH_COMPILE_ENABLED = os.environ.get("TORCH_COMPILE", "false").lower() == "true"

DB_PATH    = os.environ.get("DB_PATH",    "data/chatbot.db")
STATIC_DIR = os.environ.get("STATIC_DIR", "static")
API_HOST   = "0.0.0.0"
API_PORT   = 8001
MAX_HISTORY = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_metrics_lock = threading.Lock()
_global_metrics = {
    "total_requests": 0, "rag_used_count": 0, "hors_sujet_count": 0,
    "greeting_count": 0, "hallucination_count": 0, "extractive_count": 0,
    "low_coverage_fallback": 0, "reformulation_count": 0, "generative_count": 0,
    "total_latency_ms": 0, "confidence_scores": [], "response_lengths": [],
    "positive_feedback": 0, "negative_feedback": 0,
    "truncated_fallback_count": 0, "structured_chunk_count": 0,
    "foreign_lang_rejected": 0, "foreign_context_skip": 0,
    "url_truncated_count": 0, "admin_boost_applied": 0,
    "admin_boost_adaptive": 0, "admin_chunk_selected": 0,
    "admin_direct_count": 0, "admin_direct_fuzzy_count": 0,
    "token_estimates": [], "prefill_times_ms": [],
    "generation_times_ms": [], "tokens_generated": [],
    "stream_requests": 0, "raw_table_filtered_count": 0,
    "icons_stripped_count": 0, "multi_criteria_count": 0,
    "fusion_used_count": 0,
}

_live_metrics_lock    = threading.Lock()
_live_metrics_history = deque(maxlen=50)

# =============================================================
# CONSTANTES
# =============================================================
GREETINGS = [
    "bonjour","bonsoir","salut","hello","hi","salam","bjr","bsr",
    "coucou","hey","allo","bj","good morning"
]
GREETING_RESPONSE = (
    "Bonjour ! Je suis l'assistant virtuel de Tunisie Telecom. "
    "Je suis la pour vous aider concernant nos offres et services.\n"
    "- Les offres mobiles (Hayya, forfaits 4G/5G)\n"
    "- L'internet fixe (ADSL, Fibre, NetBox)\n"
    "- Le roaming international\n"
    "- Les tarifs et recharges\n\n"
    "Comment puis-je vous aider ?"
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
    "140","540","6eme",
    "afghanistan","african","agences","alaska","albanie","algerie",
    "allemagne","angola","arabia","argentina","armenia","australie",
    "autriche","azerbaijan","bahrain","bangladesh","belarus","belgique",
    "belize","benin","bosnie","bresil","canada","denmark","espagne",
    "estonie","europe","france","greece","italie","lybie","maroc",
    "mauritanie","monde","portugal","saudi","usa",
    "apn","bancaire","cadeau","cash","changement","client","compatibilite",
    "compte","conditions","configuration","consolide","consommation",
    "consultation","contact","controle","depannage","destination",
    "deverrouillage","disponibilite","entrante","epuise","etudiant",
    "facturation","gestion","gratuit","illimite","inactive",
    "itinerance","jour","mega","money","nperf","nuit","numeros",
    "parental","partout","pays","perdue","position","postpayees",
    "presentation","resilier","reste","social","solutions","sortante",
    "tabdil","tarifs","telephone","trophee","ttcash","urgence",
    "utilisation","via","voix","weekend","zoom",
]

TELECOM_PRODUCT_NAMES = [
    "activation","advanced","anti","anti ddos","appel","audiotex",
    "avantages","big bonus","bleu","bonus","box 5gtt pro","cession",
    "cession prepayee","cloud","cloud pbx","cloud vdc","codes",
    "connect","conso","corporate","couverture","data","ddos",
    "desactivation","dim connect","dim net corporate","dim@connect",
    "double","double appel","duo","easy","easy saff","ehdia",
    "ehdia net","eleve","eligibilite","energy","entreprise","esim",
    "esports","esports by tt","fancy","fast","fast link","forfait",
    "forfait partage","forfaits","forfaits el 3echra","forfaits fancy",
    "forfaits internet","forfaits internet mobiles","forfaits partages",
    "freeze","hadranet","hajj","housing","hybride","iaas","inscription",
    "inscription eleve","inscription en ligne","international",
    "internet","internet fixe","internet mobile","jaweknet",
    "joignabilite","kallemni","lights","ligne","link","marhaba",
    "messagerie","messagerie vocale","microsoft","microsoft office 365",
    "mms","mobile","mobile postpaid","mobile prepaid","mobiles",
    "mobiracid","mobirif","mobirif post paye","musique","musique vod",
    "my tt","national","numero","numero audiotex","numero bleu",
    "numero platine","numero vert","office","offre hajj","offres",
    "offres prepayees","one connect","optimum","optimum plus","options",
    "options tt","pack","pack pro","partage","partages","pass",
    "pass marhaba","pass roaming data","pass weekend","paye","platine",
    "portabilite","post","prepayee","prepayees","presse","privilege",
    "prix","profix","prolongation","prolongation de validite",
    "rapides","rapido","rapido pro","recharge","reseau","resiliation",
    "roaming","saff","sajalni","select","select plus","services",
    "services rapides","smart","smart energy","smart freeze",
    "smart lights","smart roaming","sms appel manque","sms joignabilite",
    "sms plus","solde","sos bip","sos solde","souscription","suivi",
    "suivi conso","support","tabba3ni","tarif","tfadhal","trankil",
    "transfert","transfert d appel","transfert internet","tt presse",
    "tunisie telecom","ussd","validite","vas","vert","vocale",
    "vpn international","vpn national","waffi",
]

HALLUCINATION_SIGNALS = ["ooredoo","myoredoo","orange tunisie","tunisiana","vodafone"]
GENERATION_NOISE      = ["casino","tirage au sort","el jem","karting","festival","sayyefi"]
PROMPT_LEAKS = [
    "donne toujours une reponse","reponse precise en 1","reponds uniquement",
    "assistant officiel","selon le contexte","d'apres le contexte",
    "reformule en une phrase","voici la reponse de tunisie","flash commercial",
]

FALLBACK_NO_INFO = (
    "Je n'ai pas trouve d'information precise sur ce sujet. "
    "Contactez le service client au 1298."
)

VOCAB_TELECOM = [
    "hayya","haya","haiya","forfait","internet","recharge","abonnement","facture",
    "roaming","netbox","adsl","fibre","activation","solde","credit",
    "tarif","offre","mobile","sim","reseau","couverture","debit",
    "illimite","gratuit","appel","sms","4g","5g","wifi","ussd",
    "activer","desactiver","souscrire","telecom","tunisie","prix",
    "cout","minute","mois","go","mo","data","forfaits","pack",
    "esim","vdsl","elissa","fixe","prepaye","postpaye","numero",
    "signal","connexion","hotspot","telechargement","streaming",
    "international","etranger","appels","messages","gigaoctet",
    "forfet","abonement","rechargement","facturation",
    "recette","taraji","mouzikti","tourist","waffi","weekend",
    "dimconnect","corporate","entreprise","partout",
    "disponible","gratuit","illimite","debit","stabilite",
]

CORRECTIONS_DIRECTES = {
    "haya":"hayya","haiya":"hayya","hayia":"hayya",
    "forfet":"forfait","internt":"internet","internrt":"internet",
    "aboneman":"abonnement","abonement":"abonnement",
    "recharg":"recharge","recharje":"recharge",
    "netboc":"netbox","netbok":"netbox",
    "dimconnect":"dim connect","dim@connect":"dim connect",
    "passweekend":"pass weekend",
}

_JSONL_ARTIFACTS = [
    (r'[\d.,]+\s*(?:DT|H|Go|Mo|min|TND|millimes?|dinars?)\s*(?:\|\s*[\d.,]+\s*(?:DT|H|Go|Mo|min|TND|millimes?|dinars?)\s*){2,}\.?', ''),
    (r'(?:[^|]{1,20}\|){3,}[^|]{1,20}', ''),
    (r'\bavec\s+est\b',   'est'),
    (r'\best\s+de\s+de\b','est de'),
    (r'\bde\s+de\b',      'de'),
    (r'voici\s+la\s+r[eé]ponse\s+de\s+Tunisie\s+Telecom\s*:\s*', ''),
    (r'\s{2,}', ' '),
]

_MULTILINE_KEYWORDS = [
    "inscrire","inscription","souscrire","souscription","activer","activation",
    "acceder","acces","telecharger","telechargement","comment","etapes",
    "possibilites","plusieurs","manieres","facons",
]

_LIST_TRIGGERS = [
    "avantages","caracteristiques","difference","comparer","comparaison",
    "quels sont","liste","offres disponibles","options disponibles",
    "pourquoi choisir","qu est ce que","presentation","decrire",
    "inclus","comprend","contient","fonctionnalites",
]

_SIMPLE_TRIGGERS = [
    "prix","coute","cout","code ussd","numero","quand","combien",
    "quel tarif","quel prix","c est quoi","definition","duree",
    "validite","expire",
]

_MULTI_CRITERIA_PATTERNS = [
    r"avantage.{0,30}prix",       r"prix.{0,30}avantage",
    r"avantage.{0,30}disponib",   r"disponib.{0,30}avantage",
    r"prix.{0,30}disponib",       r"disponib.{0,30}prix",
    r"cout.{0,30}disponib",       r"tarif.{0,30}disponib",
    r"combien.{0,30}disponib",    r"ainsi\s+que",
    r"et\s+aussi",
    r"et\s+(?:le|la|les|son|sa|ses)\s+(?:prix|cout|tarif|disponib|avantage)",
    r"(?:quels?|quelles?)\s+sont.{0,30}(?:avantage|service|option|inclus|compris)",
    r"tout\s+(?:savoir|connaitre|ce\s+que)",
    r"(?:avantage|service|option).{0,20}(?:validite|duree|periode)",
    r"(?:validite|duree).{0,20}(?:prix|cout|tarif)",
]
_MULTI_CRITERIA_COMPILED = [re.compile(p, re.IGNORECASE) for p in _MULTI_CRITERIA_PATTERNS]

_FUSION_CATEGORIES = {
    "prix":          ["dt","tnd","millimes","dinars","prix","cout","tarif","coute","payant","gratuit","offert"],
    "avantages":     ["go","mo","gigaoctet","mega","illimite","appel","sms","minutes","inclus","offre","beneficier","propose","comprend","contient"],
    "disponibilite": ["disponible","espace","agence","partout","tous","boutique","en ligne","mytt","application"],
    "validite":      ["valable","validite","jours","mois","an","expire","partir","duree","periode"],
    "activation":    ["activer","activation","composer","code","ussd","*","#","appuyez","souscrire"],
    "suivi":         ["suivre","suivi","consulter","*235","*200","code","verifier","conso","consommation"],
}

_VALID_ENDINGS = frozenset(['.', '!', '?', ':', '»', ';'])

_FOREIGN_WORDS = {
    "the","and","you","can","your","with","for","this","that","are",
    "high","speed","visit","unique","solutions","only","also",
}

_STRUCTURED_MARKERS = [
    r'\b(Appels?|Internet|Mixte|Data|SMS)\s*:',
    r'^\s*[-•]\s+',
    r'\bà partir de\b.*\bDT\b',
    r'\bDes forfaits allant\b',
    r'\d+\s*(?:Go|Mo|min|minutes?|heures?)\b',
    r'\bforfaits?\b.*\bDT\b',
]

_RAW_TABLE_UNIT_PATTERN = re.compile(
    r'^\s*[\d.,]+\s*(?:DT|H|Go|Mo|min|minutes?|heures?|TND|millimes?|dinars?|%)\s*$',
    re.IGNORECASE
)

_NOISE_PATTERNS = [
    r'(?:[\d.,]+\s*(?:DT|H|Go|Mo|min|minutes?|heures?|TND|millimes?|dinars?)\s*\|[\s]*){1,}[\d.,]+\s*(?:DT|H|Go|Mo|min|minutes?|heures?|TND|millimes?|dinars?)[^\n]*',
    r"Flash [Cc]ommercial \d{1,2}/\d{1,2}/\d{4}",
    r"Flash [Ii]nfo \d{1,2}/\d{1,2}/\d{4}",
    r"Source d.information\s+Direction\s+\S+(\s+\S+){0,3}",
    r"DCCM\s+Page\s+\d+/\d+",
    r"Supports de communication.*",
    r"\(cid:\d+\)",
    r"PAR MOIS EN DTHT.*",
    r"Cible\s*:?\s*Toute la clientele\s*(de\s*)?Tunisie Telecom",
    r"Date de lancement\s*:?\s*\d{2}/\d{2}/\d{4}",
    r"Direction (?:Marketing|VAS|Commerciale|Reseau).*",
    r"Strictement confidentiel.*",
    r"l occasion de mois de Ramadan.*",
]

_ICON_PATTERN = re.compile(
    r'[•·▸▶►✓✔★☆◆◇●○▪▫]\s*'
    r'|(?<!\w)[-–—]\s+(?=[A-ZÀ-Ža-zà-ž0-9])'
    r'|[^\x00-\x7F\u00C0-\u024F\u0600-\u06FF\u2019\u2018\u00AB\u00BB\n\r\t]',
    re.UNICODE
)

TELECOM_THEME_KEYWORDS = {
    "entreprise":      ["entreprise","corporate","b2b","business","cloud","pbx","vdc","housing","iaas","profix","professionnel"],
    "offre":           ["vas","musique","taraji","sport","jeu","streaming","presse","mobiracid","mobirif","audiotex","fancy","waffi"],
    "internet_fixe":   ["adsl","vdsl","fixe","fibre","elissa","netbox","waffi","jaweknet","hadranet","ehdia"],
    "internet_mobile": ["internet mobile","data mobile","4g","5g","forfait internet","pass internet","go","mo","debit","hotspot","forfait","hayya","rapido"],
    "roaming":         ["roaming","itinerance","international","etranger","roam","marhaba","hajj","tourist","pass roaming"],
    "mobile_prepaid":  ["prepaye","prepaid","recharge","solde","credit","sos","bip","transfert internet","bonus","cession"],
    "mobile_postpaid": ["postpaye","postpaid","facture","abonnement","hybride","select","optimum","platine","dim connect"],
    "general":         ["sim","esim","numero","activation","desactivation","ussd","mytt","1298","agence","couverture","reseau","signal"],
}


# =============================================================
# UTILITAIRES
# =============================================================

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
        if match and match[0] != word:
            corrected.append(match[0])
        else:
            corrected.append(word)
    return " ".join(corrected)

def preprocess_query(query: str) -> str:
    corrected  = correct_telecom_keywords(query)
    normalized = normalize_query(corrected)
    return normalized

def get_adaptive_threshold(query: str) -> float:
    return (THRESHOLD_SHORT_QUERY
            if len(query.strip().split()) <= SHORT_QUERY_MAX_WORDS
            else THRESHOLD_LONG_QUERY)

def detect_theme(question: str) -> str:
    q = normalize_query(question.lower())
    for theme, kws in TELECOM_THEME_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return theme
    return "general"

def detect_language(text: str) -> str:
    arabic = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', text))
    latin  = len(re.findall(r'[a-zA-Z]', text))
    total  = arabic + latin
    if total == 0: return 'fr'
    ratio  = arabic / total
    if ratio >= 0.60: return 'ar'
    if ratio >= 0.20: return 'mixed'
    return 'fr'

def is_greeting(query: str) -> bool:
    q = query.lower().strip()
    return len(q) <= 40 and any(g in q for g in GREETINGS)

def _normalize_for_kw(text: str) -> str:
    text = text.lower()
    text = _ud.normalize('NFD', text)
    text = ''.join(c for c in text if _ud.category(c) != 'Mn')
    text = _re.sub(r'[^\w\s]', ' ', text)
    return _re.sub(r'\s+', ' ', text).strip()

def is_telecom_related(query: str) -> bool:
    q_orig  = query.lower()
    q_norm  = _normalize_for_kw(query)
    q_words = set(q_norm.split())
    for kw in TELECOM_KEYWORDS:
        if kw in q_words: return True
    for product in TELECOM_PRODUCT_NAMES:
        if product in q_norm or product in q_orig: return True
    return False

def _is_multiline_query(query: str) -> bool:
    q = normalize_query(query.lower())
    return any(kw in q for kw in _MULTILINE_KEYWORDS)

def _is_multi_criteria_query(query: str) -> bool:
    q_norm  = normalize_query(query.lower())
    matched = any(pat.search(q_norm) for pat in _MULTI_CRITERIA_COMPILED)
    if matched:
        logger.info("[MCRIT] Question multi-criteres : '%s'", query[:80])
        with _metrics_lock:
            _global_metrics["multi_criteria_count"] += 1
    return matched

def _strip_icons(text: str) -> str:
    cleaned = _ICON_PATTERN.sub('', text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    stripped = cleaned.strip()
    if stripped != text.strip():
        with _metrics_lock:
            _global_metrics["icons_stripped_count"] += 1
    return stripped


# =============================================================
# DETECTION TABLEAU BRUT
# =============================================================

def _contains_raw_table(text: str) -> bool:
    pipe_segments = re.split(r'\s*\|\s*', text)
    if len(pipe_segments) < 3: return False
    matching = sum(1 for seg in pipe_segments if _RAW_TABLE_UNIT_PATTERN.match(seg.strip()))
    is_table = matching >= 3
    if is_table:
        with _metrics_lock:
            _global_metrics["raw_table_filtered_count"] += 1
    return is_table

def _strip_raw_tables(text: str) -> str:
    if _contains_raw_table(text):
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if not _contains_raw_table(line):
                cleaned = re.sub(
                    r'[\d.,]+\s*(?:DT|H|Go|Mo|min|TND|millimes?|dinars?)\s*(?:\|\s*[\d.,]+\s*(?:DT|H|Go|Mo|min|TND|millimes?|dinars?)\s*)+',
                    '', line, flags=re.IGNORECASE
                ).strip()
                if cleaned:
                    clean_lines.append(cleaned)
        return ' '.join(clean_lines).strip()
    return text


# =============================================================
# FUSION MULTI-CHUNKS
# =============================================================

def _categorize_sentence(sentence: str) -> str:
    s = normalize_query(sentence.lower())
    for cat, keywords in _FUSION_CATEGORIES.items():
        if any(kw in s for kw in keywords):
            return cat
    return "autre"

def _sentences_are_duplicate(s1: str, s2: str, threshold: float = 0.72) -> bool:
    w1 = set(normalize_query(s1.lower()).split())
    w2 = set(normalize_query(s2.lower()).split())
    if not w1 or not w2: return False
    overlap = len(w1 & w2) / max(len(w1), len(w2))
    return overlap >= threshold

def _extract_informative_sentences(text: str, max_sentences: int = 4) -> List[str]:
    BRUIT_DEBUT = ["Marketing","Contexte","Description","Concept","Source","Flash","DCCM","Cible"]
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    result = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) < 20: continue
        if any(s.startswith(b) for b in BRUIT_DEBUT): continue
        if _contains_raw_table(s): continue
        s_norm = normalize_query(s.lower())
        has_info = any(any(kw in s_norm for kw in kws) for kws in _FUSION_CATEGORIES.values())
        if has_info or len(s) > 60:
            result.append(s)
        if len(result) >= max_sentences:
            break
    return result

def _detect_requested_criteria(query: str) -> List[str]:
    q = normalize_query(query.lower())
    criteria = []
    criteria_map = {
        "avantages":     ["avantage","benefice","inclus","comprend","offre","propose","service"],
        "prix":          ["prix","cout","coute","tarif","combien","dt","payant","gratuit"],
        "disponibilite": ["disponib","ou","agence","espace","boutique","trouver","obtenir"],
        "validite":      ["validite","valable","duree","expire","periode","combien de temps"],
        "activation":    ["activer","activation","comment","souscrire","inscrire"],
        "suivi":         ["suivre","suivi","verifier","consulter","conso"],
    }
    for crit, kws in criteria_map.items():
        if any(kw in q for kw in kws):
            criteria.append(crit)
    if not criteria:
        criteria = ["avantages", "prix"]
    return criteria

def fuse_chunks_answer(chunks: List[dict], query: str) -> str:
    if not chunks:
        return FALLBACK_NO_INFO
    requested_criteria = _detect_requested_criteria(query)
    all_sentences_by_cat: dict = {cat: [] for cat in _FUSION_CATEGORIES}
    all_sentences_by_cat["autre"] = []
    for chunk in chunks[:TOP_K]:
        raw_text    = clean_chunk(chunk["text"])
        answer_part = _extract_answer_from_chunk(raw_text)
        if _contains_raw_table(answer_part):
            answer_part = _strip_raw_tables(answer_part)
        sentences = _extract_informative_sentences(answer_part, max_sentences=5)
        for s in sentences:
            cat = _categorize_sentence(s)
            if cat not in all_sentences_by_cat:
                all_sentences_by_cat[cat] = []
            all_sentences_by_cat[cat].append(s)
    selected_sentences = []
    used_texts = []
    for crit in requested_criteria:
        candidates = all_sentences_by_cat.get(crit, [])
        for candidate in candidates:
            is_dup = any(_sentences_are_duplicate(candidate, used) for used in used_texts)
            if not is_dup and len(candidate) > 20:
                selected_sentences.append(candidate)
                used_texts.append(candidate)
                break
    if len(selected_sentences) < 2:
        for s in all_sentences_by_cat.get("autre", []):
            is_dup = any(_sentences_are_duplicate(s, used) for used in used_texts)
            if not is_dup and len(s) > 30:
                selected_sentences.append(s)
                used_texts.append(s)
                if len(selected_sentences) >= 3:
                    break
    if not selected_sentences:
        return build_extractive_answer(chunks, query)
    answer = " ".join(selected_sentences)
    answer = clean_jsonl_artifacts(answer)
    answer = _strip_icons(answer)
    if answer and answer[-1] not in '.!?':
        answer = answer.rstrip(',;:') + '.'
    with _metrics_lock:
        _global_metrics["fusion_used_count"] += 1
    return answer

def _build_fused_context(chunks: List[dict], max_chars_per_chunk: int = 500) -> str:
    parts = []
    seen_content = []
    for chunk in chunks[:3]:
        cleaned = clean_chunk(chunk["text"])
        answer_only = (clean_jsonl_artifacts(cleaned) if chunk.get("is_admin")
                       else _extract_answer_from_chunk(cleaned))
        if _contains_raw_table(answer_only):
            answer_only = _strip_raw_tables(answer_only)
        if not answer_only or len(answer_only) < 15: continue
        is_dup = any(_sentences_are_duplicate(answer_only[:100], seen[:100]) for seen in seen_content)
        if not is_dup:
            parts.append(answer_only[:max_chars_per_chunk])
            seen_content.append(answer_only)
    return " | ".join(parts)


# =============================================================
# ESTIMATION DYNAMIQUE DES TOKENS
# =============================================================

def estimate_max_tokens(query: str, chunks: List[dict]) -> int:
    query_norm = normalize_query(query.lower())
    if _is_multi_criteria_query(query):
        estimated, reason = 280, "multi_criteres"
    elif _is_multiline_query(query):
        estimated, reason = 220, "multiline_procedure"
    elif any(kw in query_norm for kw in _LIST_TRIGGERS):
        estimated, reason = 180, "liste_avantages"
    elif chunks:
        context_word_count = len(_build_clean_context(chunks, max_chars=220).split())
        if context_word_count > 80:
            estimated, reason = 200, f"contexte_long_{context_word_count}mots"
        elif context_word_count > 50:
            estimated, reason = 160, f"contexte_moyen_{context_word_count}mots"
        else:
            estimated, reason = MAX_NEW_TOKENS, "contexte_court"
    elif any(kw in query_norm for kw in _SIMPLE_TRIGGERS):
        estimated, reason = 100, "question_simple"
    else:
        estimated, reason = MAX_NEW_TOKENS, "defaut"
    final = max(MIN_NEW_TOKENS, min(MAX_NEW_TOKENS_CAP, estimated))
    logger.info("[TOK] max_new_tokens=%d (raison=%s)", final, reason)
    with _metrics_lock:
        _global_metrics["token_estimates"].append({"query_preview": query[:60], "estimated": final, "reason": reason})
    return final


# =============================================================
# DETECTION REPONSE TRONQUEE
# =============================================================

def is_truncated(response: str) -> bool:
    stripped = response.strip()
    if len(stripped) < 10: return True
    if stripped.endswith('...') or stripped.endswith('\u2026'): return True
    if stripped[-1] not in _VALID_ENDINGS: return True
    if len(stripped) > 2 and stripped[-2] in (',', ';'): return True
    sentences = re.split(r'(?<=[.!?])\s+', stripped)
    if len(sentences) > 1:
        last = sentences[-1].strip()
        if last and last[-1] not in _VALID_ENDINGS and len(last.split()) < 4:
            return True
    return False

def is_structured_list(text: str) -> bool:
    matches = sum(1 for pat in _STRUCTURED_MARKERS if re.search(pat, text, re.IGNORECASE | re.MULTILINE))
    return matches >= 2


# =============================================================
# NETTOYAGE
# =============================================================

def clean_jsonl_artifacts(text: str) -> str:
    text = _strip_raw_tables(text)
    for pattern, replacement in _JSONL_ARTIFACTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = _strip_icons(text)
    text = text.strip()
    if text and text[-1] not in '.!?':
        text = text.rstrip(',;:') + '.'
    return text

def _extract_answer_from_chunk(text: str) -> str:
    idx = text.find('?')
    if idx != -1 and idx < len(text) * 0.6:
        after = text[idx + 1:].strip()
        if len(after) > 10:
            return clean_jsonl_artifacts(after)
    return clean_jsonl_artifacts(text)

def clean_chunk(text: str) -> str:
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


# =============================================================
# EXTRACTIF
# =============================================================

def find_relevant_sentences(text: str, query: str, n: int = 3) -> List[str]:
    stop = {
        "les","des","de","la","le","un","une","est","que","quel","quels","comment",
        "sont","pour","dans","du","au","et","en","je","il","elle","ce","ou","par","avec","qui","sur"
    }
    q_norm  = normalize_query(query.lower())
    q_words = {w for w in q_norm.split() if len(w) > 2 and w not in stop}
    sentences = [s.strip() for s in re.split(r'[.!?;]\s+|\n', text) if len(s.strip()) > 25]
    if not sentences: return []
    bonus_kw = [
        "tarif","prix","dt","millimes","offre","avantage","permet","beneficier",
        "activation","forfait","donnees","go","mo","minute","mois","remise","gratuit",
        "composez","allez","activez","appelez","portail","application","telechargez",
        "couverture","reseau","disponible","zone","region","nationale","4g","3g",
    ]
    scored = []
    for s in sentences:
        s_norm = normalize_query(s.lower())
        score  = sum(1 for w in q_words if w in s_norm)
        score += sum(0.5 for kw in bonus_kw if kw in s.lower())
        scored.append((score, s))
    scored.sort(reverse=True)
    top = [s for sc, s in scored[:n] if sc > 0]
    return top if top else [s for _, s in scored[:2]]

def extract_answer_part(chunk_text: str, query: str) -> str:
    text = chunk_text.strip()
    if not text: return text
    idx_q = text.find('?')
    if idx_q != -1 and idx_q < len(text) * 0.5:
        after = text[idx_q + 1:].strip()
        if len(after) > 20:
            return after
    q_norm  = normalize_query(query.lower())
    q_words = {w for w in q_norm.split() if len(w) > 3}
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        for i, s in enumerate(sentences):
            overlap = sum(1 for w in q_words if w in normalize_query(s.lower()))
            if overlap < max(1, len(q_words) * 0.5) and len(s.strip()) > 20:
                return " ".join(sentences[i:]).strip()
        return " ".join(sentences[1:]).strip()
    return text

def build_extractive_answer(chunks: List[dict], query: str) -> str:
    if not chunks:
        return FALLBACK_NO_INFO
    BRUIT_DEBUT = ["Marketing","Contexte","Description","Concept","Source","Flash","DCCM","Cible"]
    def phrase_propre(s: str) -> bool:
        return not any(s.strip().startswith(mot) for mot in BRUIT_DEBUT)
    for chunk in chunks:
        raw_text    = clean_chunk(chunk["text"])
        answer_part = extract_answer_part(raw_text, query)
        if _contains_raw_table(answer_part): continue
        if is_structured_list(answer_part) and phrase_propre(answer_part):
            clean = clean_jsonl_artifacts(answer_part[:STRUCTURED_CHUNK_MAX_CHARS])
            with _metrics_lock:
                _global_metrics["structured_chunk_count"] += 1
            return clean
    all_sentences = []
    for chunk in chunks:
        raw_text    = clean_chunk(chunk["text"])
        answer_part = extract_answer_part(raw_text, query)
        if _contains_raw_table(answer_part): continue
        if chunk.get("is_admin") and len(answer_part) > 20:
            return clean_jsonl_artifacts(answer_part)
        for s in find_relevant_sentences(answer_part, query, n=2):
            if phrase_propre(s):
                all_sentences.append(s)
    if not all_sentences:
        for chunk in chunks:
            raw_text    = clean_chunk(chunk["text"])
            answer_part = extract_answer_part(raw_text, query)
            if _contains_raw_table(answer_part): continue
            parts = [
                s.strip() for s in answer_part.split(".")
                if len(s.strip()) > 30 and phrase_propre(s.strip())
            ]
            if parts:
                return clean_jsonl_artifacts(parts[0].rstrip(",;:") + ".")
        return FALLBACK_NO_INFO
    best = all_sentences[0]
    if len(best) > 300:
        sentences_split = re.split(r'(?<=[.!?])\s+', best[:350])
        best = sentences_split[0] if sentences_split else best[:300]
    return clean_jsonl_artifacts(best)


# =============================================================
# MODELE — fp16 sans bitsandbytes
# =============================================================

_model           = None
_tokenizer       = None
_model_lock      = threading.Lock()
_model_last_used = 0.0

def _load_model():
    global _model, _tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = MODEL_DIR if os.path.isdir(MODEL_DIR) else BASE_MODEL
    logger.info("[MODEL] Chargement TinyLlama fp16 : %s", model_path)
    _tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    _tokenizer.pad_token = _tokenizer.eos_token
    device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    _model.eval()
    logger.info("[MODEL] Charge | device=%s | dtype=%s",
                next(_model.parameters()).device,
                next(_model.parameters()).dtype)
    if TORCH_COMPILE_ENABLED and torch.cuda.is_available():
        try:
            _model = torch.compile(_model, mode="reduce-overhead")
            logger.info("[MODEL] torch.compile OK")
        except Exception as e:
            logger.warning("[MODEL] torch.compile non disponible : %s", e)

def _warmup_model():
    global _model_last_used
    if _model is None or _tokenizer is None: return
    try:
        dummy = "<|system|>\nTu es un assistant.</s>\n<|user|>\nBonjour.</s>\n<|assistant|>\n"
        inputs = _tokenizer(dummy, return_tensors="pt").to(next(_model.parameters()).device)
        with torch.no_grad():
            _ = _model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                pad_token_id=_tokenizer.eos_token_id, use_cache=True)
        _model_last_used = time.time()
        logger.info("[MODEL] Warm-up termine")
    except Exception as e:
        logger.warning("[MODEL] Warm-up echoue : %s", e)

def _unload_model():
    global _model, _tokenizer
    del _model; del _tokenizer
    _model = None; _tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model():
    global _model_last_used
    with _model_lock:
        _model_last_used = time.time()
        if _model is None:
            _load_model()
            _warmup_model()
        return _model, _tokenizer

def model_is_loaded() -> bool:
    with _model_lock:
        return _model is not None

def model_watchdog():
    while True:
        time.sleep(60)
        with _model_lock:
            if _model is not None and (time.time() - _model_last_used) > MODEL_IDLE_TIMEOUT:
                _unload_model()
                logger.info("[MODEL] Dechargé après inactivité")

def init_model():
    with _model_lock:
        _load_model()
        _warmup_model()


# =============================================================
# PROMPT TINYLLAMA
# =============================================================

def _build_clean_context(chunks: List[dict], max_chars: int = 250) -> str:
    parts = []
    for chunk in chunks[:2]:
        cleaned = clean_chunk(chunk["text"])
        answer_only = (clean_jsonl_artifacts(cleaned) if chunk.get("is_admin")
                       else _extract_answer_from_chunk(cleaned))
        if _contains_raw_table(answer_only):
            answer_only = _strip_raw_tables(answer_only)
        if answer_only and len(answer_only) > 10:
            parts.append(answer_only[:max_chars])
    return " ".join(parts)

def build_generative_prompt(query: str, chunks: List[dict]) -> str:
    is_multiline  = _is_multiline_query(query)
    is_list       = any(kw in normalize_query(query.lower()) for kw in _LIST_TRIGGERS)
    is_multi_crit = _is_multi_criteria_query(query)
    if is_multi_crit:
        context     = _build_fused_context(chunks, max_chars_per_chunk=500)
        instruction = "Reponds en 3 phrases courtes et completes, une par critere demande."
    elif is_multiline:
        context     = _build_clean_context(chunks, max_chars=400)
        instruction = "Explique en 3 phrases maximum comment faire cela."
    elif is_list:
        context     = _build_clean_context(chunks, max_chars=400)
        instruction = "Liste les points principaux en 3 phrases maximum."
    else:
        context     = _build_clean_context(chunks, max_chars=250)
        instruction = "Reponds en 1 ou 2 phrases courtes et completes."
    system = (
        "Tu es l'assistant de Tunisie Telecom. "
        "Utilise UNIQUEMENT l'information ci-dessous. "
        "N'invente rien. Reponds en francais. "
        "N'utilise aucun emoji, bullet ou icone dans ta reponse."
    )
    user = (
        f"Information disponible : {context}\n\n"
        f"Question : {query}\n\n"
        f"{instruction}"
    )
    return (
        f"<|system|>\n{system}</s>\n"
        f"<|user|>\n{user}</s>\n"
        f"<|assistant|>\n"
    )

def _token_overlap(text1: str, text2: str) -> float:
    words1 = set(normalize_query(text1.lower()).split())
    words2 = set(normalize_query(text2.lower()).split())
    if not words1 or not words2: return 0.0
    return len(words1 & words2) / max(len(words1), len(words2))

def _validate_tinyllama_response(response: str, query: str, chunks: List[dict], context: str) -> tuple:
    if len(response.strip()) < 10:                          return False, "trop_court"
    if _contains_raw_table(response):                       return False, "tableau_brut"
    if any(sig in response.lower() for sig in HALLUCINATION_SIGNALS): return False, "hallucination"
    resp_norm = normalize_query(response.lower())
    if any(leak in resp_norm for leak in PROMPT_LEAKS):     return False, "fuite_prompt"
    if any(n in response.lower() for n in GENERATION_NOISE): return False, "bruit"
    resp_lower    = f" {response.lower()} "
    context_lower = f" {context.lower()} "
    foreign_in_response = sum(1 for w in _FOREIGN_WORDS if f" {w} " in resp_lower)
    if foreign_in_response >= 2:
        foreign_in_context = sum(1 for w in _FOREIGN_WORDS if f" {w} " in context_lower)
        if (foreign_in_response - foreign_in_context) >= 2:
            with _metrics_lock:
                _global_metrics["foreign_lang_rejected"] += 1
            return False, "langue_etrangere"
    overlap = _token_overlap(response, context)
    if overlap > 0.85 and len(response) > 50: return False, f"copie_contexte_{overlap:.2f}"
    context_words  = set(context.lower().split())
    response_words = [w for w in response.lower().split() if len(w) > 4]
    if response_words:
        coverage = sum(1 for w in response_words if w in context_words) / len(response_words)
        if coverage < MIN_CONTEXT_COVERAGE:
            return False, f"couverture_faible_{coverage:.2f}"
    return True, "ok"


# =============================================================
# GENERATION
# =============================================================

def _prepare_generation_inputs(query: str, chunks: List[dict]):
    model, tokenizer = get_model()
    max_tokens    = estimate_max_tokens(query, chunks)
    is_multi_crit = _is_multi_criteria_query(query)
    is_multiline  = _is_multiline_query(query)
    if is_multi_crit:
        context = _build_fused_context(chunks, max_chars_per_chunk=500)
    else:
        context = _build_clean_context(chunks, max_chars=400 if is_multiline else 250)
    prompt = build_generative_prompt(query, chunks)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512, padding=False,
    ).to(model.device)
    return model, tokenizer, inputs, max_tokens, context

@torch.inference_mode()
def generate_llm_answer(query: str, chunks: List[dict]) -> str:
    model, tokenizer = get_model()
    if model is None or tokenizer is None:
        return build_extractive_answer(chunks, query)
    if not chunks:
        return FALLBACK_NO_INFO
    if chunks[0].get("is_admin"):
        admin_text   = clean_chunk(chunks[0]["text"])
        admin_answer = _strip_icons(clean_jsonl_artifacts(admin_text))
        if len(admin_answer) > 20:
            with _metrics_lock:
                _global_metrics["admin_direct_count"] += 1
            return admin_answer
    if _is_multi_criteria_query(query):
        fused = fuse_chunks_answer(chunks, query)
        if fused and fused != FALLBACK_NO_INFO:
            return fused
    try:
        model, tokenizer, inputs, max_tokens, context = _prepare_generation_inputs(query, chunks)
        t_prefill = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=DO_SAMPLE, repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True,
            )
        gen_ms = int((time.time() - t_prefill) * 1000)
        tokens_out = out.shape[1] - inputs["input_ids"].shape[1]
        with _metrics_lock:
            _global_metrics["generation_times_ms"].append(gen_ms)
            _global_metrics["tokens_generated"].append(tokens_out)
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        response = re.sub(r"<\|.*?\|>", "", response).strip()
        response = re.sub(r"^(?:assistant|Assistant)\s*:\s*", "", response).strip()
        response = clean_jsonl_artifacts(response)
        response = _strip_icons(response)
        is_list      = any(kw in normalize_query(query.lower()) for kw in _LIST_TRIGGERS)
        is_multiline = _is_multiline_query(query)
        is_mc        = _is_multi_criteria_query(query)
        max_sent     = 5 if (is_multiline or is_list or is_mc) else 3
        sentences    = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > max_sent:
            response = " ".join(sentences[:max_sent])
        if is_truncated(response):
            with _metrics_lock:
                _global_metrics["truncated_fallback_count"] += 1
            if is_mc:
                fused = fuse_chunks_answer(chunks, query)
                if fused and fused != FALLBACK_NO_INFO:
                    return fused
            return build_extractive_answer(chunks, query)
        valid, reason = _validate_tinyllama_response(response, query, chunks, context)
        if not valid:
            with _metrics_lock:
                _global_metrics["low_coverage_fallback"] += 1
            return build_extractive_answer(chunks, query)
        with _metrics_lock:
            _global_metrics["reformulation_count"] += 1
        return response
    except Exception as e:
        logger.error("[GEN] Erreur : %s -> extractif", e)
        return build_extractive_answer(chunks, query)


# =============================================================
# RAG — ChromaDB
# =============================================================

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
    client      = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    _collection = client.get_collection(name=COLLECTION_NAME)
    _get_embed_model()
    logger.info("ChromaDB — %d chunks / '%s'", _collection.count(), COLLECTION_NAME)


def rag_search(query: str, top_k: int = TOP_K) -> List[dict]:
    encoder = _get_embed_model()
    q_emb   = encoder.encode([query], convert_to_numpy=True).tolist()

    results = _collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k * 3, _collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    admin_entries     = []
    non_admin_entries = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        raw_score = round(1 - dist, 4)
        fname    = meta.get("file_name", meta.get("filename", ""))
        is_admin = fname == "admin_enrichment"
        entry = {
            "text":      doc,
            "score_raw": raw_score,
            "score":     raw_score,
            "filename":  fname,
            "year":      meta.get("year",  ""),
            "theme":     meta.get("theme", ""),
            "is_admin":  is_admin,
        }
        if is_admin: admin_entries.append(entry)
        else:        non_admin_entries.append(entry)
    non_admin_entries.sort(key=lambda x: x["score_raw"], reverse=True)
    topk_threshold = (non_admin_entries[top_k-1]["score_raw"]
                      if len(non_admin_entries) >= top_k else 0.0)
    for entry in admin_entries:
        raw         = entry["score_raw"]
        fixed_boost = min(1.0, raw + ADMIN_SCORE_BOOST)
        if raw < topk_threshold:
            adaptive_score = min(1.0, topk_threshold + 0.05)
            final_score    = max(fixed_boost, adaptive_score)
        else:
            final_score = fixed_boost
        entry["score"] = final_score
    all_hits = admin_entries + non_admin_entries
    seen = {}
    for h in all_hits:
        if h["text"] not in seen or h["score"] > seen[h["text"]]["score"]:
            seen[h["text"]] = h
    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    for h in ranked:
        h.pop("score_raw", None)
    return ranked

def rag_search_best(query_original: str, query_processed: str, top_k: int = TOP_K) -> List[dict]:
    hits1 = rag_search(query_original, top_k)
    hits2 = (rag_search(query_processed, top_k)
             if query_processed != query_original.lower() else [])
    seen = {}
    for h in hits1 + hits2:
        if h["text"] not in seen or h["score"] > seen[h["text"]]["score"]:
            seen[h["text"]] = h
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]

def rag_search_multi_criteria(query_original: str, query_processed: str, top_k: int = None) -> List[dict]:
    extended_k = (TOP_K * 2) if top_k is None else top_k
    hits1 = rag_search(query_original, extended_k)
    hits2 = (rag_search(query_processed, extended_k)
             if query_processed != query_original.lower() else [])
    seen = {}
    for h in hits1 + hits2:
        if h["text"] not in seen or h["score"] > seen[h["text"]]["score"]:
            seen[h["text"]] = h
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:extended_k]

def check_admin_enrichment_direct(query: str, threshold: float = ADMIN_DIRECT_THRESHOLD) -> Optional[str]:
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT question, answer FROM rag_enrichments ORDER BY timestamp DESC"
            ).fetchall()
    except Exception:
        return None
    if not rows: return None
    query_norm  = normalize_query(query.lower().strip())
    best_score  = 0.0
    best_answer = None
    for row in rows:
        stored_q_norm   = normalize_query(row["question"].lower().strip())
        score_ratio     = fuzz.ratio(query_norm, stored_q_norm) / 100.0
        score_token_set = fuzz.token_set_ratio(query_norm, stored_q_norm) / 100.0
        score           = max(score_ratio, score_token_set)
        if score > best_score:
            best_score  = score
            best_answer = row["answer"]
    if best_score >= threshold:
        with _metrics_lock:
            _global_metrics["admin_direct_fuzzy_count"] += 1
        return best_answer
    return None

def _select_chunks_for_query(query: str, query_processed: str, lang: str):
    if _is_multi_criteria_query(query):
        hits = rag_search_multi_criteria(query, query_processed)
    else:
        hits = rag_search_best(query, query_processed)
    confidence         = hits[0]["score"] if hits else 0.0
    adaptive_threshold = get_adaptive_threshold(query)
    rag_used           = confidence >= adaptive_threshold
    return hits, confidence, adaptive_threshold, rag_used


# =============================================================
# SQLITE
# =============================================================

_db_lock = threading.Lock()

@contextmanager
def get_db():
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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

def init_db():
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
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
                (sid, role, content))

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
                (sid, question, reason, int(is_telecom)))

def delete_session(sid):
    with _db_lock:
        with get_db() as conn:
            return conn.execute(
                "DELETE FROM conversations WHERE session_id=?", (sid,)).rowcount


# =============================================================
# SCHEMAS PYDANTIC
# =============================================================

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

class FeedbackRequest(BaseModel):
    session_id:    str
    msg_id:        Optional[str] = None
    rating:        int
    comment:       Optional[str] = ""
    user_question: Optional[str] = ""
    bot_answer:    Optional[str] = ""

class RagEntryRequest(BaseModel):
    question:    str
    answer:      str
    theme:       Optional[str] = "enrichissement_admin"
    source_note: Optional[str] = ""


# =============================================================
# FASTAPI — lifespan
# [FIX-2] init_auth_db() appelé ici
# =============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("  Chatbot Tunisie Telecom — TinyLlama fp16")
    logger.info("  ChromaDB : %s / collection : %s", CHROMA_DB_DIR, COLLECTION_NAME)
    logger.info("  Modele   : %s", MODEL_DIR)
    logger.info("  DB       : %s", DB_PATH)
    logger.info("=" * 55)
    init_db()
    init_auth_db()   # [FIX-2]
    init_rag()
    if not EXTRACTIVE_MODE:
        logger.info("Chargement TinyLlama fp16...")
        try:
            init_model()
            threading.Thread(target=model_watchdog, daemon=True).start()
        except Exception as e:
            logger.error("Echec chargement modele : %s — mode extractif", e)
    yield
    logger.info("API arretee")

app = FastAPI(
    title="Chatbot Tunisie Telecom — TinyLlama",
    version="2.16.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =============================================================
# ROUTE PRINCIPALE — /chat
# =============================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    t0         = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    query      = request.message.strip()

    if not query:
        raise HTTPException(400, "Message vide.")

    # 1. Greeting
    if is_greeting(query):
        answer  = GREETING_RESPONSE
        elapsed = int((time.time() - t0) * 1000)
        save_message(session_id, "user",      query)
        save_message(session_id, "assistant", answer)
        with _metrics_lock:
            _global_metrics["total_requests"]   += 1
            _global_metrics["greeting_count"]   += 1
            _global_metrics["total_latency_ms"] += elapsed
        return ChatResponse(
            session_id=session_id, answer=answer, sources=[],
            is_telecom=True, rag_used=False, confidence=1.0,
            response_time_ms=elapsed, mode="greeting")

    # 2. Hors sujet
    if not is_telecom_related(query):
        log_unanswered(session_id, query, "hors_sujet", is_telecom=False)
        answer  = ("Je suis l'assistant Tunisie Telecom et je reponds uniquement "
                   "aux questions sur nos offres et services.")
        elapsed = int((time.time() - t0) * 1000)
        save_message(session_id, "user",      query)
        save_message(session_id, "assistant", answer)
        with _metrics_lock:
            _global_metrics["total_requests"]   += 1
            _global_metrics["hors_sujet_count"] += 1
            _global_metrics["total_latency_ms"] += elapsed
        return ChatResponse(
            session_id=session_id, answer=answer, sources=[],
            is_telecom=False, rag_used=False, confidence=0.0,
            response_time_ms=elapsed, mode="hors_sujet")

    # 3. Admin direct fuzzy
    admin_fuzzy = await run_in_threadpool(check_admin_enrichment_direct, query)
    if admin_fuzzy:
        answer  = _strip_icons(admin_fuzzy)
        elapsed = int((time.time() - t0) * 1000)
        save_message(session_id, "user",      query)
        save_message(session_id, "assistant", answer)
        with _metrics_lock:
            _global_metrics["total_requests"]   += 1
            _global_metrics["rag_used_count"]   += 1
            _global_metrics["total_latency_ms"] += elapsed
        return ChatResponse(
            session_id=session_id, answer=answer, sources=[],
            is_telecom=True, rag_used=True, confidence=1.0,
            response_time_ms=elapsed, mode="admin_direct")

    # 4. RAG
    lang         = detect_language(query)
    search_query = preprocess_query(query)
    hits, confidence, adaptive_threshold, rag_used = await run_in_threadpool(
        _select_chunks_for_query, query, search_query, lang
    )

    if not rag_used:
        log_unanswered(session_id, query, "confiance_faible", is_telecom=True)

    chunks_for_answer = hits if rag_used else []

    for i, h in enumerate(hits[:3]):
        logger.info("[CHUNK-%d] score=%.3f | %s | '%s'",
                    i+1, h["score"], h["filename"], h["text"][:60])

    # 5. Generation de la reponse
    if rag_used and chunks_for_answer and chunks_for_answer[0].get("is_admin"):
        answer      = _strip_icons(clean_jsonl_artifacts(clean_chunk(chunks_for_answer[0]["text"])))
        answer_mode = "admin_direct"
        with _metrics_lock:
            _global_metrics["admin_direct_count"] += 1

    elif rag_used and _is_multi_criteria_query(query):
        answer      = _strip_icons(await run_in_threadpool(fuse_chunks_answer, chunks_for_answer, query))
        answer_mode = "fusion_multi_criteria"
        with _metrics_lock:
            _global_metrics["fusion_used_count"] += 1

    elif rag_used and not EXTRACTIVE_MODE:
        answer      = _strip_icons(await run_in_threadpool(generate_llm_answer, query, chunks_for_answer))
        answer_mode = "generative_rag"
        with _metrics_lock:
            _global_metrics["generative_count"] += 1

    elif rag_used:
        answer      = _strip_icons(await run_in_threadpool(build_extractive_answer, chunks_for_answer, query))
        answer_mode = "extractive"
        with _metrics_lock:
            _global_metrics["extractive_count"] += 1

    else:
        answer      = FALLBACK_NO_INFO
        answer_mode = "no_context"

    # 6. Anti-hallucination
    if any(sig in answer.lower() for sig in HALLUCINATION_SIGNALS):
        log_unanswered(session_id, query, "hallucination_detectee", is_telecom=True)
        answer      = _strip_icons(await run_in_threadpool(build_extractive_answer, chunks_for_answer, query))
        answer_mode = "extractive_fallback"
        with _metrics_lock:
            _global_metrics["hallucination_count"] += 1

    save_message(session_id, "user",      query)
    save_message(session_id, "assistant", answer)
    elapsed = int((time.time() - t0) * 1000)

    with _metrics_lock:
        _global_metrics["total_requests"]   += 1
        _global_metrics["rag_used_count"]   += int(rag_used)
        _global_metrics["total_latency_ms"] += elapsed
        _global_metrics["confidence_scores"].append(confidence)
        _global_metrics["response_lengths"].append(len(answer))

    sources = [
        {
            "filename": h["filename"], "year": h["year"],
            "theme":    h["theme"],    "score": h["score"],
            "text":     h["text"][:300],
            "is_admin": h.get("is_admin", False),
        }
        for h in (hits if rag_used else [])
    ]

    logger.info("[%s] %dms | conf=%.3f | seuil=%.2f | rag=%s",
                answer_mode, elapsed, confidence, adaptive_threshold, rag_used)

    return ChatResponse(
        session_id=session_id, answer=answer, sources=sources,
        is_telecom=True, rag_used=rag_used, confidence=confidence,
        response_time_ms=elapsed, mode=answer_mode)


# =============================================================
# FEEDBACK
# =============================================================

@app.post("/feedback", tags=["Feedback"])
async def feedback_endpoint(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(400, "rating doit etre 1 ou -1")
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO feedback "
                "(session_id, msg_id, rating, comment, user_question, bot_answer) "
                "VALUES (?,?,?,?,?,?)",
                (req.session_id or "unknown", req.msg_id or "", req.rating,
                 req.comment or "", req.user_question or "", req.bot_answer or ""))
    with _metrics_lock:
        key = "positive_feedback" if req.rating == 1 else "negative_feedback"
        _global_metrics[key] = _global_metrics.get(key, 0) + 1
    return {"status": "ok"}


# [FIX-3] Route /feedback/stats manquante
@app.get("/feedback/stats", tags=["Feedback"])
async def feedback_stats():
    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) as c FROM feedback"
        ).fetchone()["c"]
        pos = conn.execute(
            "SELECT COUNT(*) as c FROM feedback WHERE rating=1"
        ).fetchone()["c"]
        neg = conn.execute(
            "SELECT COUNT(*) as c FROM feedback WHERE rating=-1"
        ).fetchone()["c"]
        recent = conn.execute(
            "SELECT session_id, rating, comment, user_question, "
            "bot_answer, timestamp FROM feedback ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
    return {
        "total":             total,
        "positive":          pos,
        "negative":          neg,
        "satisfaction_rate": round(pos / total * 100, 1) if total > 0 else 0,
        "recent":            [dict(r) for r in recent],
    }


# =============================================================
# HISTORIQUE
# =============================================================

@app.get("/history/{session_id}", tags=["Historique"])
async def history(session_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM conversations "
            "WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
    if not rows:
        raise HTTPException(404, "Session introuvable.")
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
            for r in rows]

@app.delete("/history/{session_id}", tags=["Historique"])
async def clear_history(session_id: str):
    n = delete_session(session_id)
    if n == 0:
        raise HTTPException(404, "Session introuvable.")
    return {"deleted": n}


# =============================================================
# AUTH ROUTES
# [FIX-1] Routes auth montées dans l'app
# =============================================================

@app.post("/auth/register", tags=["Auth"])
async def auth_register(req: RegisterRequest):
    return await run_in_threadpool(register_route, req)

@app.post("/auth/verify", tags=["Auth"])
async def auth_verify(req: VerifyRequest):
    return await run_in_threadpool(verify_route, req)

@app.post("/auth/login", tags=["Auth"])
async def auth_login(req: LoginRequest):
    return await run_in_threadpool(login_route, req)

@app.post("/auth/resend", tags=["Auth"])
async def auth_resend(req: ResendRequest):
    return await run_in_threadpool(resend_route, req)

@app.post("/auth/check-email", tags=["Auth"])
async def auth_check_email(req: CheckEmailRequest):
    return await check_email_route(req)

@app.get("/auth/me", tags=["Auth"])
async def auth_me(user: dict = Depends(require_auth)):
    return me_route(user)

@app.post("/auth/logout", tags=["Auth"])
async def auth_logout(authorization: str = Header(None)):
    return await run_in_threadpool(logout_route, authorization)


# =============================================================
# RAG ADMIN
# =============================================================

@app.post("/admin/rag/add", tags=["RAG Admin"])
async def rag_add(req: RagEntryRequest):
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(400, "question et answer sont obligatoires")
    chunk_id   = f"admin_{uuid.uuid4().hex[:12]}"
    chunk_text = req.answer.strip()
    encoder = _get_embed_model()
    emb     = encoder.encode([chunk_text], convert_to_numpy=True).tolist()
    try:
        _collection.add(
            documents  = [chunk_text],
            embeddings = emb,
            metadatas  = [{
                "file_name": "admin_enrichment",
                "filename":  "admin_enrichment",
                "year":      str(datetime.now().year),
                "theme":     req.theme or "enrichissement_admin",
                "question":  req.question[:200],
                "answer":    req.answer[:500],
            }],
            ids=[chunk_id])
    except Exception as e:
        raise HTTPException(500, f"Erreur ajout ChromaDB : {e}")
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_enrichments "
                "(chunk_id, question, answer, theme, source_note) VALUES (?,?,?,?,?)",
                (chunk_id, req.question, req.answer,
                 req.theme or "enrichissement_admin", req.source_note or ""))
    return {"status": "ok", "chunk_id": chunk_id, "n_chunks": _collection.count()}

@app.get("/admin/rag/entries", tags=["RAG Admin"])
async def rag_entries(limit: int = 100):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, chunk_id, question, answer, theme, timestamp "
            "FROM rag_enrichments ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return {"total": len(rows), "entries": [dict(r) for r in rows]}


# =============================================================
# ADMIN — CONVERSATIONS STATS
# [FIX-3] Route /admin/conversations/stats manquante
# =============================================================

@app.get("/admin/conversations/stats", tags=["Admin"])
async def conversation_stats(limit: int = Query(default=1000, ge=1, le=5000)):
    with get_db() as conn:
        total_msgs = conn.execute(
            "SELECT COUNT(*) as c FROM conversations"
        ).fetchone()["c"]
        total_sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) as c FROM conversations"
        ).fetchone()["c"]
        # Distribution par role
        role_dist = conn.execute(
            "SELECT role, COUNT(*) as c FROM conversations GROUP BY role"
        ).fetchall()
        # Questions hors sujet
        hors_sujet = conn.execute(
            "SELECT COUNT(*) as c FROM unanswered_questions WHERE reason='hors_sujet'"
        ).fetchone()["c"]
        unanswered_total = conn.execute(
            "SELECT COUNT(*) as c FROM unanswered_questions"
        ).fetchone()["c"]
        # Sessions récentes avec comptage
        recent = conn.execute(
            "SELECT session_id, COUNT(*) as msg_count, MAX(timestamp) as last_msg "
            "FROM conversations GROUP BY session_id "
            "ORDER BY last_msg DESC LIMIT ?", (limit,)
        ).fetchall()
    return {
        "total_messages":    total_msgs,
        "total_sessions":    total_sessions,
        "role_distribution": {r["role"]: r["c"] for r in role_dist},
        "unanswered_total":  unanswered_total,
        "hors_sujet_total":  hors_sujet,
        "sessions":          [dict(r) for r in recent],
    }


# =============================================================
# METRICS & HEALTH
# =============================================================

@app.get("/metrics", tags=["Admin"])
async def get_metrics():
    with _metrics_lock:
        m = dict(_global_metrics)
    n    = m["total_requests"]
    conf = m.pop("confidence_scores", [])
    m.pop("response_lengths", [])
    m.pop("token_estimates", [])
    m.pop("prefill_times_ms", [])
    m.pop("generation_times_ms", [])
    m.pop("tokens_generated", [])
    return {
        **m,
        "rag_rate":           m["rag_used_count"] / n if n > 0 else 0,
        "avg_latency_ms":     m["total_latency_ms"] / n if n > 0 else 0,
        "avg_confidence":     sum(conf) / len(conf) if conf else 0,
        "model_loaded":       model_is_loaded(),
        "extractive_mode":    EXTRACTIVE_MODE,
        "collection":         COLLECTION_NAME,
        "chroma_dir":         CHROMA_DB_DIR,
        "model_dir":          MODEL_DIR,
        "timestamp":          datetime.now().isoformat(),
    }

@app.get("/health", tags=["Systeme"])
async def health():
    n_chunks = _collection.count() if _collection else 0
    with get_db() as conn:
        n_msgs  = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
        n_sess  = conn.execute("SELECT COUNT(DISTINCT session_id) as c FROM conversations").fetchone()["c"]
        n_fb    = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
        n_rag   = conn.execute("SELECT COUNT(*) as c FROM rag_enrichments").fetchone()["c"]
    return {
        "status":          "ok" if n_chunks > 0 else "degraded",
        "model_loaded":    model_is_loaded(),
        "extractive_mode": EXTRACTIVE_MODE,
        "rag_chunks":      n_chunks,
        "collection":      COLLECTION_NAME,
        "chroma_dir":      CHROMA_DB_DIR,
        "model_dir":       MODEL_DIR,
        "total_messages":  n_msgs,
        "total_sessions":  n_sess,
        "total_feedback":  n_fb,
        "rag_enrichments": n_rag,
        "timestamp":       datetime.now().isoformat(),
    }

@app.get("/logs/unanswered", tags=["Admin"])
async def unanswered(limit: int = 50, reason: Optional[str] = None):
    with get_db() as conn:
        if reason:
            rows = conn.execute(
                "SELECT id, session_id, question, reason, is_telecom, timestamp "
                "FROM unanswered_questions WHERE reason=? "
                "ORDER BY timestamp DESC LIMIT ?", (reason, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, session_id, question, reason, is_telecom, timestamp "
                "FROM unanswered_questions ORDER BY timestamp DESC LIMIT ?",
                (limit,)).fetchall()
    return [dict(r) for r in rows]

@app.get("/", include_in_schema=False)
async def root():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path)
    return {"message": "Chatbot Tunisie Telecom API v2.16 — TinyLlama fp16"}


# =============================================================
# LANCEMENT
# =============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("  Chatbot Tunisie Telecom — TinyLlama fp16  v2.16")
    print(f"  ChromaDB  : {CHROMA_DB_DIR} / {COLLECTION_NAME}")
    print(f"  Modele    : {MODEL_DIR}")
    print(f"  DB        : {DB_PATH}")
    print(f"  Mode      : {'extractif' if EXTRACTIVE_MODE else 'generatif'}")
    print(f"  Port      : {API_PORT}")
    print("=" * 55)
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False, log_level="info")