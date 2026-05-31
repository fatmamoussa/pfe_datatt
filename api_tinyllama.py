"""
=============================================================
API v2.18 — Chatbot Tunisie Telecom — TinyLlama 1.1B
FUSION v2.15 (features complètes) + v2.17 (correctifs + fp16)

NOUVEAUTÉS v2.18 (fusion) :
  [v2.17-FIX1]  correct_telecom_keywords : whitelist mots FR + score_cutoff 82
  [v2.17-FIX2]  _chunks_are_coherent() : filtre overlap lexical post-retrieval
  [v2.17-FIX3]  TOP_K default 5
  [v2.17-MODEL] TinyLlama fp16 (sans bitsandbytes) + encodeur SentenceTransformer direct
  [v2.15-FUSION] fuse_chunks_answer() — agrégation multi-chunks sans LLM
  [v2.15-MCRIT]  _is_multi_criteria_query() — 16 patterns
  [v2.15-CTX2]   _build_fused_context() — 3 chunks x 500 chars
  [v2.15-TOK2]   multi_criteres → 280 tokens

ROUTES COMPLÈTES :
  POST /chat                        — réponse classique
  POST /chat/stream                 — streaming SSE
  POST /feedback                    — soumettre un avis
  GET  /feedback/stats              — statistiques feedback
  GET  /history/{session_id}        — historique session
  DELETE /history/{session_id}      — supprimer session
  POST /admin/rag/add               — ajouter chunk admin
  PUT  /admin/rag/update/{id}       — modifier chunk admin
  DELETE /admin/rag/delete/{id}     — supprimer chunk admin
  GET  /admin/rag/entries           — lister chunks admin
  GET  /admin/rag/test              — tester RAG
  GET  /admin/metrics/live          — métriques temps réel
  GET  /admin/metrics/perf          — métriques perf LLM
  GET  /admin/metrics/tokens        — estimation tokens
  GET  /admin/conversations/stats   — stats conversations
  GET  /logs/unanswered             — questions sans réponse
  GET  /metrics                     — métriques globales
  GET  /health                      — état du système
  POST /auth/register|verify|login|resend|logout|check-email
  GET  /auth/me
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

from auth_sqlite import (
    init_auth_db,
    register_route, verify_route, login_route,
    resend_route, require_auth, me_route, logout_route,
    check_email_route,
    RegisterRequest, VerifyRequest, LoginRequest,
    ResendRequest, CheckEmailRequest,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
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

# [v2.17-FIX3] TOP_K default 5
TOP_K = int(os.environ.get("TOP_K", "5"))

ADMIN_SCORE_BOOST      = float(os.environ.get("ADMIN_SCORE_BOOST",      "0.10"))
ADMIN_DIRECT_THRESHOLD = float(os.environ.get("ADMIN_DIRECT_THRESHOLD", "0.72"))

CONFIDENCE_THRESHOLD  = float(os.environ.get("CONFIDENCE_THRESHOLD",  "0.45"))
THRESHOLD_SHORT_QUERY = float(os.environ.get("THRESHOLD_SHORT_QUERY", "0.38"))
THRESHOLD_LONG_QUERY  = float(os.environ.get("THRESHOLD_LONG_QUERY",  "0.50"))
SHORT_QUERY_MAX_WORDS = 4
MIN_CONTEXT_COVERAGE  = float(os.environ.get("MIN_CONTEXT_COVERAGE",  "0.20"))

# [v2.17-FIX2] Seuil overlap lexical minimum pour validation chunk
COHERENCE_OVERLAP_MIN = float(os.environ.get("COHERENCE_OVERLAP_MIN", "0.10"))

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
    # v2.17
    "coherence_rejected_count": 0,
    "fuzzy_correction_blocked": 0,
}

_live_metrics_lock    = threading.Lock()
_live_metrics_history = deque(maxlen=50)

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
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

AR_GREETINGS = [
    "\u0627\u0644\u0633\u0644\u0627\u0645", "\u0645\u0631\u062d\u0628\u0627",
    "\u0635\u0628\u0627\u062d \u0627\u0644\u062e\u064a\u0631",
    "\u0645\u0633\u0627\u0621 \u0627\u0644\u062e\u064a\u0631",
    "\u0623\u0647\u0644\u0627", "\u0633\u0644\u0627\u0645", "\u0647\u0644\u0627",
]
GREETING_RESPONSE_AR = (
    "\u0645\u0631\u062d\u0628\u0627! \u0623\u0646\u0627 \u0627\u0644\u0645\u0633\u0627\u0639\u062f "
    "\u0627\u0644\u0627\u0641\u062a\u0631\u0627\u0636\u064a \u0644\u0640 \u062a\u0648\u0646\u0633 "
    "\u0644\u0644\u0627\u062a\u0635\u0627\u0644\u0627\u062a.\n"
    "\u0643\u064a\u0641 \u064a\u0645\u0643\u0646\u0646\u064a \u0645\u0633\u0627\u0639\u062f\u062a\u0643\u061f"
)
HORS_SUJET_RESPONSE_AR = (
    "\u0623\u0646\u0627 \u0645\u0633\u0627\u0639\u062f \u062a\u0648\u0646\u0633 "
    "\u0644\u0644\u0627\u062a\u0635\u0627\u0644\u0627\u062a \u0648\u0644\u0627 "
    "\u0623\u0633\u062a\u0637\u064a\u0639 \u0627\u0644\u0625\u062c\u0627\u0628\u0629 "
    "\u0625\u0644\u0627 \u0639\u0644\u0649 \u0623\u0633\u0626\u0644\u0629 \u062a\u062a\u0639\u0644\u0642 "
    "\u0628\u062e\u062f\u0645\u0627\u062a\u0646\u0627."
)
NO_INFO_RESPONSE_AR = (
    "\u0644\u0645 \u0623\u062c\u062f \u0645\u0639\u0644\u0648\u0645\u0627\u062a \u062f\u0642\u064a\u0642\u0629. "
    "\u064a\u0631\u062c\u0649 \u0627\u0644\u0627\u062a\u0635\u0627\u0644 \u0639\u0644\u0649 1298."
)
FALLBACK_NO_INFO = (
    "Je n'ai pas trouve d'information precise sur ce sujet. "
    "Contactez le service client au 1298."
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

AR_TELECOM_KEYWORDS = [
    "\u0627\u0646\u062a\u0631\u0646\u062a", "\u0625\u0646\u062a\u0631\u0646\u062a",
    "\u0647\u0627\u062a\u0641", "\u0631\u0635\u064a\u062f", "\u0634\u062d\u0646",
    "\u0639\u0631\u0636", "\u0639\u0631\u0648\u0636", "\u0627\u0634\u062a\u0631\u0627\u0643",
    "\u062a\u062c\u0648\u0627\u0644", "\u062e\u062f\u0645\u0629",
    "\u062a\u0648\u0646\u0633 \u0644\u0644\u0627\u062a\u0635\u0627\u0644\u0627\u062a",
    "\u062a\u0641\u0639\u064a\u0644", "\u0634\u0628\u0643\u0629", "1298",
]

HALLUCINATION_SIGNALS = ["ooredoo","myoredoo","orange tunisie","tunisiana","vodafone"]
GENERATION_NOISE      = ["casino","tirage au sort","el jem","karting","festival","sayyefi"]
PROMPT_LEAKS = [
    "donne toujours une reponse","reponse precise en 1","reponds uniquement",
    "assistant officiel","selon le contexte","d'apres le contexte",
    "reformule en une phrase","voici la reponse de tunisie","flash commercial",
]

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
    "waffi","weekend","dimconnect","corporate","entreprise",
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

# [v2.17-FIX1] Whitelist mots français courants à ne PAS corriger
_COMMON_WORDS_FR = {
    "taper","voir","reste","combien","faut","pour","comment",
    "quel","quels","quelle","quelles","mon","ma","mes","je","me",
    "il","elle","nous","vous","ils","elles","qui","que","quoi",
    "dont","avoir","etre","faire","vouloir","pouvoir","savoir",
    "aller","venir","mettre","prendre","donner","trouver","parler",
    "avec","dans","sans","sous","sur","par","vers","chez","entre",
    "depuis","pendant","avant","apres","encore","toujours","jamais",
    "souvent","parfois","bien","tres","plus","moins","aussi","donc",
    "mais","car","puis","ainsi","alors","cela","ceci","tout","tous",
    "cette","leur","leurs","notre","votre","son","ses",
    "chaque","autre","autres","nouveau","nouvelle","grand","petit",
    "meme","comme","quand","si","non","oui","pas","peu",
    "trop","assez","beaucoup","plusieurs","certains","certaines",
}

TELECOM_THEME_KEYWORDS = {
    "entreprise":      ["entreprise","corporate","b2b","business","cloud","pbx","vdc","housing","iaas","profix","professionnel","dim@connect","dimconnect"],
    "offre":           ["vas","musique","mouzikti","taraji","sport","jeu","game","streaming","svod","presse","mobiracid","mobirif","audiotex","fancy","waffi"],
    "internet_fixe":   ["adsl","vdsl","fixe","fibre","elissa","netbox","waffi","jaweknet","hadranet","ehdia"],
    "internet_mobile": ["internet mobile","data mobile","4g","5g","forfait internet","pass internet","go","mo","debit","hotspot","forfait","hayya","rapido"],
    "roaming":         ["roaming","itinerance","international","etranger","roam","marhaba","hajj","tourist","pass roaming"],
    "mobile_prepaid":  ["prepaye","prepaid","recharge","solde","credit","sos","bip","transfert internet","bonus","cession"],
    "mobile_postpaid": ["postpaye","postpaid","facture","abonnement","hybride","select","optimum","platine","dim connect"],
    "general":         ["sim","esim","numero","activation","desactivation","ussd","mytt","1298","agence","couverture","reseau","signal"],
}

_JSONL_ARTIFACTS = [
    (r'[\d.,]+\s*(?:DT|H|Go|Mo|min|TND|millimes?|dinars?)\s*(?:\|\s*[\d.,]+\s*(?:DT|H|Go|Mo|min|TND|millimes?|dinars?)\s*){2,}\.?', ''),
    (r'(?:[^|]{1,20}\|){3,}[^|]{1,20}', ''),
    (r'\bavec\s+est\b', 'est'),
    (r'\best\s+de\s+de\b', 'est de'),
    (r'\bde\s+de\b', 'de'),
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

# [v2.15-MCRIT] 16 patterns multi-critères
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


# ═══════════════ UTILITAIRES ═══════════════

def normalize_query(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


# [v2.17-FIX1] correct_telecom_keywords amélioré
def correct_telecom_keywords(query: str) -> str:
    words = query.lower().split()
    corrected = []
    for word in words:
        if len(word) <= 3:
            corrected.append(word)
            continue
        if word in _COMMON_WORDS_FR:
            corrected.append(word)
            continue
        if word in CORRECTIONS_DIRECTES:
            corrected.append(CORRECTIONS_DIRECTES[word])
            continue
        # score_cutoff relevé 70 → 82
        match = process.extractOne(
            word, VOCAB_TELECOM, scorer=fuzz.ratio, score_cutoff=82
        )
        if match and match[0] != word:
            logger.debug("[FIX1] Correction fuzzy : '%s' → '%s' (score=%d)", word, match[0], match[1])
            corrected.append(match[0])
        else:
            corrected.append(word)
    return " ".join(corrected)


def preprocess_query(query: str) -> str:
    corrected  = correct_telecom_keywords(query)
    normalized = normalize_query(corrected)
    if normalized != query.lower():
        logger.info("[PREPROCESS] '%s' -> '%s'", query, normalized)
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
    if len(q) <= 40 and any(g in q for g in GREETINGS): return True
    if len(query) <= 50 and any(g in query for g in AR_GREETINGS): return True
    return False


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
    try:
        if any(kw in query for kw in AR_TELECOM_KEYWORDS): return True
    except Exception:
        pass
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


# ═══════════════ DETECTION TABLEAU BRUT ═══════════════

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


# ═══════════════ NETTOYAGE ═══════════════

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


# ═══════════════ FUSION MULTI-CHUNKS ═══════════════

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


# ═══════════════ ESTIMATION TOKENS ═══════════════

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
        _global_metrics["token_estimates"].append({
            "query_preview": query[:60], "estimated": final, "reason": reason
        })
    return final


# ═══════════════ DETECTION REPONSE TRONQUEE ═══════════════

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
    matches = sum(1 for pat in _STRUCTURED_MARKERS
                  if re.search(pat, text, re.IGNORECASE | re.MULTILINE))
    return matches >= 2


# ═══════════════ EXTRACTIF ═══════════════

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


# ═══════════════ MODELE — fp16 sans bitsandbytes ═══════════════

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


# ═══════════════ PROMPT TINYLLAMA ═══════════════

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
        else:
            with _metrics_lock:
                _global_metrics["foreign_context_skip"] += 1
    overlap = _token_overlap(response, context)
    if overlap > 0.85 and len(response) > 50: return False, f"copie_contexte_{overlap:.2f}"
    context_words  = set(context.lower().split())
    response_words = [w for w in response.lower().split() if len(w) > 4]
    if response_words:
        coverage = sum(1 for w in response_words if w in context_words) / len(response_words)
        if coverage < MIN_CONTEXT_COVERAGE:
            return False, f"couverture_faible_{coverage:.2f}"
    return True, "ok"


# ═══════════════ GENERATION ═══════════════

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
        gen_ms     = int((time.time() - t_prefill) * 1000)
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


@torch.inference_mode()
def generate_llm_answer_stream(query: str, chunks: List[dict]):
    from transformers import TextIteratorStreamer
    if not chunks:
        yield FALLBACK_NO_INFO
        return
    if chunks[0].get("is_admin"):
        admin_text   = clean_chunk(chunks[0]["text"])
        admin_answer = _strip_icons(clean_jsonl_artifacts(admin_text))
        if len(admin_answer) > 20:
            with _metrics_lock:
                _global_metrics["admin_direct_count"] += 1
            yield admin_answer
            return
    if _is_multi_criteria_query(query):
        fused = fuse_chunks_answer(chunks, query)
        if fused and fused != FALLBACK_NO_INFO:
            yield fused
            return
    try:
        model, tokenizer, inputs, max_tokens, context = _prepare_generation_inputs(query, chunks)
        streamer = TextIteratorStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=True, timeout=60.0)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=DO_SAMPLE,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            streamer=streamer,
        )
        t_start    = time.time()
        gen_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        gen_thread.start()
        first_token      = True
        generated_tokens = []
        for new_text in streamer:
            if not new_text: continue
            clean_text = re.sub(r"<\|.*?\|>", "", new_text)
            clean_text = re.sub(r"^(?:assistant|Assistant)\s*:\s*", "", clean_text)
            clean_text = _strip_icons(clean_text)
            if clean_text:
                if first_token:
                    with _metrics_lock:
                        _global_metrics["prefill_times_ms"].append(
                            int((time.time() - t_start) * 1000))
                    first_token = False
                generated_tokens.append(clean_text)
                yield clean_text
        gen_thread.join(timeout=120)
        full_response = "".join(generated_tokens).strip()
        full_response = clean_jsonl_artifacts(full_response)
        with _metrics_lock:
            _global_metrics["reformulation_count"] += 1
    except Exception as e:
        logger.error("[STREAM] Erreur stream : %s → fallback extractif", e)
        yield build_extractive_answer(chunks, query)


# ═══════════════ RAG — ChromaDB ═══════════════

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
        fname     = meta.get("file_name", meta.get("filename", ""))
        is_admin  = fname == "admin_enrichment"
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
    topk_threshold = (non_admin_entries[TOP_K - 1]["score_raw"]
                      if len(non_admin_entries) >= TOP_K else 0.0)
    admin_boosted = 0
    adaptive_used = 0
    for entry in admin_entries:
        raw         = entry["score_raw"]
        fixed_boost = min(1.0, raw + ADMIN_SCORE_BOOST)
        if raw < topk_threshold:
            adaptive_score = min(1.0, topk_threshold + 0.05)
            final_score    = max(fixed_boost, adaptive_score)
            adaptive_used += 1
        else:
            final_score = fixed_boost
        entry["score"] = final_score
        admin_boosted += 1
    if admin_boosted > 0:
        with _metrics_lock:
            _global_metrics["admin_boost_applied"] += admin_boosted
            _global_metrics["admin_boost_adaptive"] += adaptive_used
    all_hits = admin_entries + non_admin_entries
    seen = {}
    for h in all_hits:
        if h["text"] not in seen or h["score"] > seen[h["text"]]["score"]:
            seen[h["text"]] = h
    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    for h in ranked:
        h.pop("score_raw", None)
    admin_selected = sum(1 for h in ranked if h.get("is_admin"))
    if admin_selected > 0:
        with _metrics_lock:
            _global_metrics["admin_chunk_selected"] += admin_selected
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


def rag_search_multi_criteria(query_original: str, query_processed: str,
                               top_k: int = None) -> List[dict]:
    extended_k = (TOP_K * 2) if top_k is None else top_k
    hits1 = rag_search(query_original, extended_k)
    hits2 = (rag_search(query_processed, extended_k)
             if query_processed != query_original.lower() else [])
    seen = {}
    for h in hits1 + hits2:
        if h["text"] not in seen or h["score"] > seen[h["text"]]["score"]:
            seen[h["text"]] = h
    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:extended_k]
    logger.info("[FUSION] RAG elargi : %d chunks récupérés", len(ranked))
    return ranked


# ═══════════════ ADMIN DIRECT LOOKUP ═══════════════

def check_admin_enrichment_direct(query: str,
                                   threshold: float = ADMIN_DIRECT_THRESHOLD) -> Optional[str]:
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
    best_q      = ""
    for row in rows:
        stored_q_norm   = normalize_query(row["question"].lower().strip())
        score_ratio     = fuzz.ratio(query_norm, stored_q_norm) / 100.0
        score_token_set = fuzz.token_set_ratio(query_norm, stored_q_norm) / 100.0
        score           = max(score_ratio, score_token_set)
        if score > best_score:
            best_score  = score
            best_answer = row["answer"]
            best_q      = row["question"]
    if best_score >= threshold:
        logger.info("[ADMIN-FUZZY] Match (score=%.3f) | q_stored='%s'", best_score, best_q[:80])
        with _metrics_lock:
            _global_metrics["admin_direct_fuzzy_count"] += 1
        return best_answer
    return None


# ═══════════════ [v2.17-FIX2] Filtre cohérence post-retrieval ═══════════════

def _chunks_are_coherent(hits: List[dict], query: str) -> bool:
    if not hits: return False
    _stop = {"pour","dans","avec","comment","quel","quels","quelle","quelles",
             "faut","taper","voir","faire","avoir","etre","veux","vouloir"}
    query_words = {
        w for w in normalize_query(query.lower()).split()
        if len(w) > 3 and w not in _stop
    }
    if not query_words: return True
    top_text = normalize_query(hits[0]["text"].lower())
    overlap  = sum(1 for w in query_words if w in top_text) / len(query_words)
    if overlap < COHERENCE_OVERLAP_MIN:
        logger.warning(
            "[FIX2-COHERENCE] Chunk rejeté (overlap=%.2f) pour query='%s'",
            overlap, query[:60]
        )
        with _metrics_lock:
            _global_metrics["coherence_rejected_count"] += 1
        return False
    return True


def _select_chunks_for_query(query: str, query_processed: str,
                              lang: str) -> tuple:
    if _is_multi_criteria_query(query):
        hits = rag_search_multi_criteria(query, query_processed)
    else:
        hits = rag_search_best(query, query_processed)
    confidence         = hits[0]["score"] if hits else 0.0
    adaptive_threshold = get_adaptive_threshold(query)
    rag_used           = confidence >= adaptive_threshold
    # [v2.17-FIX2] Filtre cohérence
    if rag_used and hits and not hits[0].get("is_admin"):
        if not _chunks_are_coherent(hits, query):
            rag_used = False
    return hits, confidence, adaptive_threshold, rag_used


# ═══════════════ MÉTRIQUES TEMPS RÉEL ═══════════════

def _stop_words():
    return {
        "les","des","de","la","le","un","une","est","que","quel","quels",
        "comment","sont","pour","dans","du","au","et","en","je","il","elle",
        "ce","ou","par","avec","qui","sur","vous","nos","tu","nous"
    }

def _q_words(question: str):
    stop = _stop_words()
    return {w for w in question.lower().split() if len(w) > 2 and w not in stop}

def _rt_hit_rate(sources, question, k=3):
    if not sources: return 0.0
    qw = _q_words(question)
    if not qw: return 0.0
    for src in sources[:k]:
        combined = (src.get("text","") + " " + src.get("filename","")).lower()
        if any(w in combined for w in qw): return 1.0
    return 0.0

def _rt_mrr(sources, question):
    if not sources: return 0.0
    qw = _q_words(question)
    if not qw: return 0.0
    for i, src in enumerate(sources):
        combined = (src.get("text","") + " " + src.get("filename","")).lower()
        if any(w in combined for w in qw):
            return round(1.0 / (i + 1), 4)
    return 0.0

def _rt_precision_k(sources, question, k=3):
    if not sources: return 0.0
    qw = _q_words(question)
    if not qw: return 0.0
    rel = sum(1 for s in sources[:k]
              if any(w in (s.get("text","") + s.get("filename","")).lower() for w in qw))
    return round(rel / min(k, len(sources)), 4)

def _rt_faithfulness(answer: str, sources: list):
    if not sources: return 0.5
    context = " ".join(s.get("text","") for s in sources).lower()
    if not context.strip(): return 0.5
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words: return 0.5
    return round(sum(1 for w in words if w in context) / len(words), 4)

def _rt_keyword_coverage(answer: str, question: str):
    qw = list(_q_words(question))
    if not qw: return 1.0
    al = answer.lower()
    return round(sum(1 for kw in qw if kw in al) / len(qw), 4)

def _rt_answer_length_score(answer: str):
    n = len(answer)
    if n < 50:   return 0.2
    if n < 100:  return 0.6
    if n <= 500: return 1.0
    if n <= 800: return 0.8
    return 0.5

def compute_realtime_metrics(question, answer, sources, rag_used, confidence,
                              response_time_ms, mode):
    context_texts = [s.get("text","") for s in sources]
    if rag_used and sources:
        rag = {
            "hit_rate":          _rt_hit_rate(sources, question),
            "mrr":               _rt_mrr(sources, question),
            "precision_k":       _rt_precision_k(sources, question),
        }
    else:
        rag = {"hit_rate": None, "mrr": None, "precision_k": None}
    gen = {
        "faithfulness":     _rt_faithfulness(answer, sources),
        "keyword_coverage": _rt_keyword_coverage(answer, question),
        "length_score":     _rt_answer_length_score(answer),
    }
    scores = [confidence]
    if rag_used and sources:
        scores += [v for v in rag.values() if v is not None]
    scores += [gen["faithfulness"], gen["keyword_coverage"]]
    global_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    metrics = {
        "rag": rag, "generation": gen, "global_score": global_score,
        "confidence": round(confidence, 4), "rag_used": rag_used,
        "mode": mode, "latency_ms": response_time_ms, "answer_len": len(answer),
    }
    with _live_metrics_lock:
        _live_metrics_history.appendleft({
            "question":       question[:120],
            "answer_preview": answer[:100],
            "metrics":        metrics,
            "timestamp":      datetime.now().isoformat(),
        })
    return metrics


# ═══════════════ SQLITE ═══════════════

_db_lock = threading.Lock()

@contextmanager
def get_db():
    db_dir = os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else "."
    os.makedirs(db_dir, exist_ok=True)
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
    db_dir = os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else "."
    os.makedirs(db_dir, exist_ok=True)
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
            CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
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
            CREATE INDEX IF NOT EXISTS idx_rag_ts ON rag_enrichments(timestamp);
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


# ═══════════════ SCHEMAS PYDANTIC ═══════════════

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
    metrics:          Optional[dict] = None

class FeedbackRequest(BaseModel):
    session_id:    str
    msg_id:        Optional[str] = None
    rating:        int
    comment:       Optional[str] = ""
    user_question: Optional[str] = ""
    bot_answer:    Optional[str] = ""
    timestamp:     Optional[str] = None

class RagEntryRequest(BaseModel):
    question:      str
    answer:        str
    theme:         Optional[str] = "enrichissement_admin"
    source_note:   Optional[str] = ""
    unanswered_id: Optional[int] = None


# ═══════════════ FASTAPI — LIFESPAN ═══════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Chatbot Tunisie Telecom — TinyLlama fp16  v2.18")
    logger.info("  ChromaDB : %s / collection : %s", CHROMA_DB_DIR, COLLECTION_NAME)
    logger.info("  Modele   : %s", MODEL_DIR)
    logger.info("  DB       : %s", DB_PATH)
    logger.info("  [v2.17] fp16 | whitelist FR | coherence filter | TOP_K=5")
    logger.info("  [v2.15] fusion multi-chunks | %d patterns multi-criteres", len(_MULTI_CRITERIA_PATTERNS))
    logger.info("=" * 60)
    init_db()
    init_auth_db()
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
    title="Chatbot Tunisie Telecom — TinyLlama v2.18",
    description="RAG + TinyLlama fp16 — fusion multi-chunks, filtre cohérence, TOP_K=5",
    version="2.18.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ═══════════════ ROUTES AUTH ═══════════════

@app.post("/auth/register",    tags=["Auth"])
async def auth_register(req: RegisterRequest):
    return await run_in_threadpool(register_route, req)

@app.post("/auth/verify",      tags=["Auth"])
async def auth_verify(req: VerifyRequest):
    return await run_in_threadpool(verify_route, req)

@app.post("/auth/login",       tags=["Auth"])
async def auth_login(req: LoginRequest):
    return await run_in_threadpool(login_route, req)

@app.post("/auth/resend",      tags=["Auth"])
async def auth_resend(req: ResendRequest):
    return await run_in_threadpool(resend_route, req)

@app.post("/auth/check-email", tags=["Auth"])
async def auth_check_email(req: CheckEmailRequest):
    return await check_email_route(req)

@app.get("/auth/me",           tags=["Auth"])
async def auth_me(user: dict = Depends(require_auth)):
    return me_route(user)

@app.post("/auth/logout",      tags=["Auth"])
async def auth_logout(authorization: str = Header(None)):
    return await run_in_threadpool(logout_route, authorization)


# ═══════════════ ROUTE CHAT CLASSIQUE ═══════════════

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Endpoint classique non-stream."""
    t0         = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    query      = request.message.strip()
    lang       = detect_language(query)

    if not query:
        raise HTTPException(400, "Message vide.")

    # 1. Greeting
    if is_greeting(query):
        answer  = GREETING_RESPONSE_AR if lang == "ar" else GREETING_RESPONSE
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
            response_time_ms=elapsed, mode="greeting", metrics=None)

    # 2. Hors sujet
    if not is_telecom_related(query):
        log_unanswered(session_id, query, "hors_sujet", is_telecom=False)
        answer  = (HORS_SUJET_RESPONSE_AR if lang == "ar" else
                   "Je suis l'assistant Tunisie Telecom et je reponds uniquement "
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
            response_time_ms=elapsed, mode="hors_sujet", metrics=None)

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
            _global_metrics["confidence_scores"].append(1.0)
        rt_metrics = compute_realtime_metrics(
            question=query, answer=answer, sources=[],
            rag_used=True, confidence=1.0,
            response_time_ms=elapsed, mode="admin_direct")
        return ChatResponse(
            session_id=session_id, answer=answer, sources=[],
            is_telecom=True, rag_used=True, confidence=1.0,
            response_time_ms=elapsed, mode="admin_direct", metrics=rt_metrics)

    # 4. RAG
    search_query = preprocess_query(query) if lang != "ar" else query
    hits, confidence, adaptive_threshold, rag_used = await run_in_threadpool(
        _select_chunks_for_query, query, search_query, lang
    )

    if not rag_used:
        log_unanswered(session_id, query, "confiance_faible", is_telecom=True)

    chunks_for_answer = hits if rag_used else []

    for i, h in enumerate(hits[:3]):
        admin_tag = " [ADMIN]" if h.get("is_admin") else ""
        logger.info("[CHUNK-%d] score=%.3f%s | '%s'",
                    i + 1, h["score"], admin_tag, h["text"][:60])

    # 5. Génération
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
        answer      = NO_INFO_RESPONSE_AR if lang == "ar" else FALLBACK_NO_INFO
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

    rt_metrics = compute_realtime_metrics(
        question=query, answer=answer,
        sources=hits if rag_used else [],
        rag_used=rag_used, confidence=confidence,
        response_time_ms=elapsed, mode=answer_mode,
    )

    logger.info("[%s] %dms | conf=%.3f | seuil=%.2f | lang=%s | rag=%s",
                answer_mode, elapsed, confidence, adaptive_threshold, lang, rag_used)

    return ChatResponse(
        session_id=session_id, answer=answer, sources=sources,
        is_telecom=True, rag_used=rag_used, confidence=confidence,
        response_time_ms=elapsed, mode=answer_mode, metrics=rt_metrics)


# ═══════════════ ROUTE CHAT STREAMING ═══════════════

@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Endpoint streaming SSE.
    Événements : start | token | end | error
    """
    t0         = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    query      = request.message.strip()
    lang       = detect_language(query)

    if not query:
        raise HTTPException(400, "Message vide.")

    with _metrics_lock:
        _global_metrics["stream_requests"] += 1

    async def event_generator():
        yield f'data: {json.dumps({"type":"start","session_id":session_id})}\n\n'
        try:
            # 1. Greeting
            if is_greeting(query):
                answer = GREETING_RESPONSE_AR if lang == "ar" else GREETING_RESPONSE
                yield f'data: {json.dumps({"type":"token","text":answer})}\n\n'
                elapsed = int((time.time() - t0) * 1000)
                yield f'data: {json.dumps({"type":"end","mode":"greeting","latency_ms":elapsed,"rag_used":False,"confidence":1.0})}\n\n'
                save_message(session_id, "user",      query)
                save_message(session_id, "assistant", answer)
                with _metrics_lock:
                    _global_metrics["total_requests"]   += 1
                    _global_metrics["greeting_count"]   += 1
                    _global_metrics["total_latency_ms"] += elapsed
                return

            # 2. Hors sujet
            if not is_telecom_related(query):
                log_unanswered(session_id, query, "hors_sujet", is_telecom=False)
                answer = (HORS_SUJET_RESPONSE_AR if lang == "ar" else
                          "Je suis l'assistant Tunisie Telecom et je reponds uniquement "
                          "aux questions sur nos offres et services.")
                yield f'data: {json.dumps({"type":"token","text":answer})}\n\n'
                elapsed = int((time.time() - t0) * 1000)
                yield f'data: {json.dumps({"type":"end","mode":"hors_sujet","latency_ms":elapsed,"rag_used":False,"confidence":0.0})}\n\n'
                save_message(session_id, "user",      query)
                save_message(session_id, "assistant", answer)
                with _metrics_lock:
                    _global_metrics["total_requests"]   += 1
                    _global_metrics["hors_sujet_count"] += 1
                    _global_metrics["total_latency_ms"] += elapsed
                return

            # 3. Admin direct fuzzy
            admin_fuzzy = await run_in_threadpool(check_admin_enrichment_direct, query)
            if admin_fuzzy:
                answer = _strip_icons(admin_fuzzy)
                yield f'data: {json.dumps({"type":"token","text":answer})}\n\n'
                elapsed = int((time.time() - t0) * 1000)
                yield f'data: {json.dumps({"type":"end","mode":"admin_direct","latency_ms":elapsed,"rag_used":True,"confidence":1.0})}\n\n'
                save_message(session_id, "user",      query)
                save_message(session_id, "assistant", answer)
                with _metrics_lock:
                    _global_metrics["total_requests"]   += 1
                    _global_metrics["rag_used_count"]   += 1
                    _global_metrics["total_latency_ms"] += elapsed
                return

            # 4. RAG
            search_query = preprocess_query(query) if lang != "ar" else query
            hits, confidence, adaptive_threshold, rag_used = await run_in_threadpool(
                _select_chunks_for_query, query, search_query, lang
            )
            if not rag_used:
                log_unanswered(session_id, query, "confiance_faible", is_telecom=True)
            chunks_for_answer = hits if rag_used else []

            # 5. Génération
            answer_mode       = "no_context"
            full_answer_parts = []

            if rag_used and chunks_for_answer and chunks_for_answer[0].get("is_admin"):
                answer      = _strip_icons(clean_jsonl_artifacts(clean_chunk(chunks_for_answer[0]["text"])))
                answer_mode = "admin_direct"
                yield f'data: {json.dumps({"type":"token","text":answer})}\n\n'
                full_answer_parts = [answer]
                with _metrics_lock:
                    _global_metrics["admin_direct_count"] += 1

            elif rag_used and _is_multi_criteria_query(query):
                fused = await run_in_threadpool(fuse_chunks_answer, chunks_for_answer, query)
                answer_mode = "fusion_multi_criteria"
                yield f'data: {json.dumps({"type":"token","text":fused})}\n\n'
                full_answer_parts = [fused]
                with _metrics_lock:
                    _global_metrics["fusion_used_count"] += 1

            elif rag_used and not EXTRACTIVE_MODE:
                import asyncio
                token_queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def _run_stream():
                    try:
                        for token in generate_llm_answer_stream(query, chunks_for_answer):
                            loop.call_soon_threadsafe(token_queue.put_nowait, token)
                    except Exception as e:
                        loop.call_soon_threadsafe(token_queue.put_nowait, f"[ERR:{e}]")
                    finally:
                        loop.call_soon_threadsafe(token_queue.put_nowait, None)

                stream_thread = threading.Thread(target=_run_stream, daemon=True)
                stream_thread.start()
                while True:
                    token = await token_queue.get()
                    if token is None: break
                    if token.startswith("[ERR:"): break
                    full_answer_parts.append(token)
                    yield f'data: {json.dumps({"type":"token","text":token})}\n\n'
                stream_thread.join(timeout=5)
                answer_mode = "generative_rag_stream"
                with _metrics_lock:
                    _global_metrics["generative_count"] += 1

            elif rag_used:
                answer      = _strip_icons(await run_in_threadpool(
                    build_extractive_answer, chunks_for_answer, query))
                answer_mode = "extractive"
                yield f'data: {json.dumps({"type":"token","text":answer})}\n\n'
                full_answer_parts = [answer]
                with _metrics_lock:
                    _global_metrics["extractive_count"] += 1

            else:
                answer      = NO_INFO_RESPONSE_AR if lang == "ar" else FALLBACK_NO_INFO
                answer_mode = "no_context"
                yield f'data: {json.dumps({"type":"token","text":answer})}\n\n'
                full_answer_parts = [answer]

            full_answer = "".join(full_answer_parts).strip()

            # Anti-hallucination
            if any(sig in full_answer.lower() for sig in HALLUCINATION_SIGNALS):
                log_unanswered(session_id, query, "hallucination_detectee", is_telecom=True)
                full_answer = _strip_icons(await run_in_threadpool(
                    build_extractive_answer, chunks_for_answer, query))
                answer_mode = "extractive_fallback"
                with _metrics_lock:
                    _global_metrics["hallucination_count"] += 1

            save_message(session_id, "user",      query)
            save_message(session_id, "assistant", full_answer)
            elapsed = int((time.time() - t0) * 1000)

            with _metrics_lock:
                _global_metrics["total_requests"]   += 1
                _global_metrics["rag_used_count"]   += int(rag_used)
                _global_metrics["total_latency_ms"] += elapsed
                _global_metrics["confidence_scores"].append(confidence)
                _global_metrics["response_lengths"].append(len(full_answer))

            yield f'data: {json.dumps({"type":"end","mode":answer_mode,"latency_ms":elapsed,"rag_used":rag_used,"confidence":round(confidence,4)})}\n\n'

        except Exception as e:
            logger.error("[STREAM] Erreur inattendue : %s", e)
            yield f'data: {json.dumps({"type":"error","message":str(e)})}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ═══════════════ FEEDBACK ═══════════════

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
                 req.comment or "", req.user_question or "", req.bot_answer or "", ts))
            fb_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    with _metrics_lock:
        key = "positive_feedback" if req.rating == 1 else "negative_feedback"
        _global_metrics[key] = _global_metrics.get(key, 0) + 1
    return {"status": "ok", "feedback_id": fb_id}


@app.get("/feedback/stats", tags=["Feedback"])
async def feedback_stats():
    with get_db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN rating=1  THEN 1 ELSE 0 END) as positive,
                   SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) as negative
            FROM feedback""").fetchone()
        recent_neg = conn.execute("""
            SELECT session_id, msg_id, comment, user_question, bot_answer, timestamp
            FROM feedback WHERE rating=-1 ORDER BY timestamp DESC LIMIT 15""").fetchall()
    pos   = row["positive"] or 0
    neg   = row["negative"] or 0
    total = row["total"]    or 0
    rate  = round(100 * pos / total, 1) if total > 0 else 0
    return {
        "positive":  pos, "negative": neg, "total": total,
        "satisfaction_rate": rate,
        "recent_negative": [
            {
                "session_id":    r["session_id"],
                "msg_id":        r["msg_id"] or "",
                "comment":       r["comment"] or "(pas de commentaire)",
                "user_question": r["user_question"] or "",
                "bot_answer":    r["bot_answer"] or "",
                "timestamp":     r["timestamp"],
            }
            for r in recent_neg
        ],
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════ HISTORIQUE ═══════════════

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


# ═══════════════ LOGS ═══════════════

@app.get("/logs/unanswered", tags=["Admin"])
async def unanswered(
    limit: int = 50,
    reason: Optional[str] = None,
    telecom_only: Optional[int] = Query(None),
    user=Depends(require_auth),
):
    with get_db() as conn:
        if telecom_only == 1:
            rows = conn.execute(
                "SELECT id, session_id, question, reason, is_telecom, timestamp "
                "FROM unanswered_questions WHERE is_telecom=1 "
                "ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        elif reason:
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


# ═══════════════ RAG ADMIN ═══════════════

def _save_enrichment_db(chunk_id, question, answer, theme, source_note):
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_enrichments "
                "(chunk_id, question, answer, theme, source_note) VALUES (?,?,?,?,?)",
                (chunk_id, question, answer, theme, source_note or ""))


@app.post("/admin/rag/add", tags=["RAG Admin"])
async def rag_add(req: RagEntryRequest, user=Depends(require_auth)):
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
    _save_enrichment_db(
        chunk_id, req.question, req.answer,
        req.theme or "enrichissement_admin", req.source_note or "")
    if req.unanswered_id:
        with _db_lock:
            with get_db() as conn:
                conn.execute(
                    "DELETE FROM unanswered_questions WHERE id=?", (req.unanswered_id,))
    return {
        "status": "ok", "chunk_id": chunk_id,
        "n_chunks": _collection.count(),
        "fix_y_threshold": ADMIN_DIRECT_THRESHOLD,
    }


@app.put("/admin/rag/update/{chunk_id}", tags=["RAG Admin"])
async def rag_update(chunk_id: str, req: RagEntryRequest, user=Depends(require_auth)):
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(400, "question et answer sont obligatoires")
    chunk_text = req.answer.strip()
    encoder = _get_embed_model()
    emb     = encoder.encode([chunk_text], convert_to_numpy=True).tolist()
    try:
        _collection.update(
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
        raise HTTPException(500, f"Erreur update ChromaDB : {e}")
    _save_enrichment_db(
        chunk_id, req.question, req.answer,
        req.theme or "enrichissement_admin", req.source_note or "")
    return {"status": "ok", "chunk_id": chunk_id}


@app.delete("/admin/rag/delete/{chunk_id}", tags=["RAG Admin"])
async def rag_delete(chunk_id: str, user=Depends(require_auth)):
    try:
        _collection.delete(ids=[chunk_id])
    except Exception as e:
        raise HTTPException(500, f"Erreur suppression ChromaDB : {e}")
    with _db_lock:
        with get_db() as conn:
            conn.execute("DELETE FROM rag_enrichments WHERE chunk_id=?", (chunk_id,))
    return {"status": "ok", "chunk_id": chunk_id}


@app.get("/admin/rag/entries", tags=["RAG Admin"])
async def rag_entries(limit: int = 100, user=Depends(require_auth)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, chunk_id, question, answer, theme, source_note, added_by, timestamp "
            "FROM rag_enrichments ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return {"total": len(rows), "entries": [dict(r) for r in rows]}


@app.get("/admin/rag/test", tags=["RAG Admin"])
async def rag_test(q: str = Query(...), user=Depends(require_auth)):
    if not _collection:
        raise HTTPException(503, "ChromaDB non initialisé")

    fuzzy_result = None
    fuzzy_score  = 0.0
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT question, answer FROM rag_enrichments ORDER BY timestamp DESC"
            ).fetchall()
        q_norm = normalize_query(q.lower().strip())
        for row in rows:
            sq    = normalize_query(row["question"].lower().strip())
            score = max(fuzz.ratio(q_norm, sq) / 100.0,
                        fuzz.token_set_ratio(q_norm, sq) / 100.0)
            if score > fuzzy_score:
                fuzzy_score  = score
                fuzzy_result = {
                    "question": row["question"],
                    "answer":   row["answer"][:200],
                    "score":    round(fuzzy_score, 4)
                }
    except Exception as e:
        fuzzy_result = {"error": str(e)}

    fix_y_fires   = fuzzy_score >= ADMIN_DIRECT_THRESHOLD
    is_multi_crit = _is_multi_criteria_query(q)

    encoder = _get_embed_model()
    q_emb   = encoder.encode([q], convert_to_numpy=True).tolist()
    results = _collection.query(
        query_embeddings=q_emb,
        n_results=min(TOP_K * 3, _collection.count()),
        include=["documents", "metadatas", "distances"])

    admin_hits     = []
    non_admin_hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        raw_score = round(1 - dist, 4)
        fname    = meta.get("file_name", meta.get("filename", ""))
        is_admin = fname == "admin_enrichment"
        entry = {
            "is_admin":      is_admin,
            "has_raw_table": _contains_raw_table(doc),
            "score_raw":     raw_score,
            "filename":      fname,
            "question":      meta.get("question", "")[:100],
            "text_stored":   doc[:150],
        }
        if is_admin: admin_hits.append(entry)
        else:        non_admin_hits.append(entry)

    non_admin_hits.sort(key=lambda x: x["score_raw"], reverse=True)
    admin_hits.sort(key=lambda x:    x["score_raw"], reverse=True)
    topk_threshold = (non_admin_hits[TOP_K - 1]["score_raw"]
                      if len(non_admin_hits) >= TOP_K else 0.0)

    for h in admin_hits:
        raw   = h["score_raw"]
        fixed = round(min(1.0, raw + ADMIN_SCORE_BOOST), 4)
        if raw < topk_threshold:
            adaptive = round(min(1.0, topk_threshold + 0.05), 4)
            h["score_final"]          = max(fixed, adaptive)
            h["adaptive_boost_needed"] = True
        else:
            h["score_final"]          = fixed
            h["adaptive_boost_needed"] = False
        h["would_enter_topk"] = h["score_final"] >= (topk_threshold or 0)

    return {
        "query": q, "top_k": TOP_K,
        "v215_fusion": {
            "is_multi_criteria": is_multi_crit,
            "would_use_fusion":  is_multi_crit,
            "extended_k":        TOP_K * 2 if is_multi_crit else TOP_K,
        },
        "fix_y_fuzzy": {
            "threshold":  ADMIN_DIRECT_THRESHOLD,
            "best_score": round(fuzzy_score, 4),
            "fires":      fix_y_fires,
            "best_match": fuzzy_result,
        },
        "admin_chunks":    admin_hits,
        "top3_non_admin":  non_admin_hits[:3],
        "diagnosis": {
            "any_admin_enters_topk":    any(h["would_enter_topk"] for h in admin_hits),
            "fix_y_fires":              fix_y_fires,
            "raw_table_filtered_count": _global_metrics.get("raw_table_filtered_count", 0),
            "icons_stripped_count":     _global_metrics.get("icons_stripped_count", 0),
            "fusion_used_count":        _global_metrics.get("fusion_used_count", 0),
            "multi_criteria_count":     _global_metrics.get("multi_criteria_count", 0),
            "coherence_rejected_count": _global_metrics.get("coherence_rejected_count", 0),
        },
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════ MÉTRIQUES ═══════════════

@app.get("/admin/metrics/live", tags=["Admin"])
async def metrics_live(limit: int = 20, user=Depends(require_auth)):
    with _live_metrics_lock:
        hist = list(_live_metrics_history)[:limit]
    return {"count": len(hist), "history": hist, "timestamp": datetime.now().isoformat()}


@app.get("/admin/metrics/perf", tags=["Admin"])
async def metrics_perf(user=Depends(require_auth)):
    with _metrics_lock:
        prefill_ms  = list(_global_metrics.get("prefill_times_ms", []))
        gen_ms      = list(_global_metrics.get("generation_times_ms", []))
        tokens_gen  = list(_global_metrics.get("tokens_generated", []))
        stream_reqs = _global_metrics.get("stream_requests", 0)

    def stats(lst):
        if not lst: return {}
        return {
            "count": len(lst), "min": min(lst), "max": max(lst),
            "avg": round(sum(lst) / len(lst), 1),
            "p50": sorted(lst)[len(lst) // 2],
            "p95": sorted(lst)[int(len(lst) * 0.95)] if len(lst) >= 20 else max(lst),
        }

    return {
        "prefill_ms":       stats(prefill_ms),
        "generation_ms":    stats(gen_ms),
        "tokens_generated": stats(tokens_gen),
        "stream_requests":  stream_reqs,
        "config": {
            "MAX_NEW_TOKENS":           MAX_NEW_TOKENS,
            "MIN_NEW_TOKENS":           MIN_NEW_TOKENS,
            "MAX_NEW_TOKENS_CAP":       MAX_NEW_TOKENS_CAP,
            "TORCH_COMPILE":            TORCH_COMPILE_ENABLED,
            "tokenizer_max_length":     512,
            "TOP_K":                    TOP_K,
            "COHERENCE_OVERLAP_MIN":    COHERENCE_OVERLAP_MIN,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/admin/metrics/tokens", tags=["Admin"])
async def metrics_tokens(limit: int = 50, user=Depends(require_auth)):
    with _metrics_lock:
        estimates = list(_global_metrics.get("token_estimates", []))[-limit:]
    if not estimates:
        return {"count": 0, "estimates": [], "stats": {}}
    values  = [e["estimated"] for e in estimates]
    reasons = Counter(e["reason"] for e in estimates)
    return {
        "count": len(estimates), "estimates": estimates,
        "stats": {
            "min": min(values), "max": max(values),
            "avg": round(sum(values) / len(values), 1),
            "by_reason":                dict(reasons),
            "admin_direct_count":       _global_metrics.get("admin_direct_count", 0),
            "admin_direct_fuzzy_count": _global_metrics.get("admin_direct_fuzzy_count", 0),
            "raw_table_filtered_count": _global_metrics.get("raw_table_filtered_count", 0),
            "icons_stripped_count":     _global_metrics.get("icons_stripped_count", 0),
            "fusion_used_count":        _global_metrics.get("fusion_used_count", 0),
            "multi_criteria_count":     _global_metrics.get("multi_criteria_count", 0),
            "coherence_rejected_count": _global_metrics.get("coherence_rejected_count", 0),
        },
        "config": {
            "MIN_NEW_TOKENS":     MIN_NEW_TOKENS,
            "MAX_NEW_TOKENS_CAP": MAX_NEW_TOKENS_CAP,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/admin/conversations/stats", tags=["Admin"])
async def conversations_stats(limit: int = 1000, user=Depends(require_auth)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT content, session_id, timestamp FROM conversations "
            "WHERE role='user' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    questions  = [dict(r) for r in rows]
    total_q    = len(questions)
    q_counter  = Counter()
    q_original = {}
    for r in questions:
        key = r["content"].strip().lower()[:100]
        q_counter[key] += 1
        if key not in q_original:
            q_original[key] = r["content"].strip()
    top5 = [{"question": q_original[k], "count": c}
            for k, c in q_counter.most_common(5)]
    theme_counter  = Counter()
    theme_questions = {}
    for r in questions:
        theme = detect_theme(r["content"])
        theme_counter[theme] += 1
        qkey  = r["content"].strip().lower()[:100]
        if theme not in theme_questions:
            theme_questions[theme] = Counter()
        theme_questions[theme][qkey] += 1
    top_themes = [{"theme": t, "count": c}
                  for t, c in theme_counter.most_common(12) if t]
    top2_per_theme = {
        theme: [{"question": q_original.get(k, k), "count": c}
                for k, c in qcounter.most_common(2)]
        for theme, qcounter in theme_questions.items()
        if theme and theme_counter[theme] >= 1
    }
    now    = datetime.now()
    hourly = {str(h): 0 for h in range(24)}
    for r in questions:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
            if (now - ts).total_seconds() < 86400:
                hourly[str(ts.hour)] += 1
        except Exception:
            pass
    return {
        "total_questions": total_q,
        "top5_questions":  top5,
        "top_themes":      top_themes,
        "top2_per_theme":  top2_per_theme,
        "hourly_distribution": [
            {"hour": int(h), "count": c}
            for h, c in sorted(hourly.items(), key=lambda x: int(x[0]))
        ],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics", tags=["Admin"])
async def get_metrics(user=Depends(require_auth)):
    with _metrics_lock:
        m = dict(_global_metrics)
    n    = m["total_requests"]
    conf = m.pop("confidence_scores", [])
    m.pop("response_lengths", [])
    token_estimates = m.pop("token_estimates", [])
    prefill_ms      = m.pop("prefill_times_ms", [])
    gen_ms_list     = m.pop("generation_times_ms", [])
    tokens_gen_list = m.pop("tokens_generated", [])
    token_values    = [e["estimated"] for e in token_estimates] if token_estimates else []
    return {
        **m,
        "rag_rate":                   m["rag_used_count"] / n if n > 0 else 0,
        "avg_latency_ms":             m["total_latency_ms"] / n if n > 0 else 0,
        "avg_confidence":             sum(conf) / len(conf) if conf else 0,
        "reformulation_rate":         m.get("reformulation_count", 0) / n if n > 0 else 0,
        "avg_max_tokens":             round(sum(token_values) / len(token_values), 1) if token_values else 0,
        "avg_prefill_ms":             round(sum(prefill_ms) / len(prefill_ms), 1) if prefill_ms else 0,
        "avg_generation_ms":          round(sum(gen_ms_list) / len(gen_ms_list), 1) if gen_ms_list else 0,
        "avg_tokens_generated":       round(sum(tokens_gen_list) / len(tokens_gen_list), 1) if tokens_gen_list else 0,
        "stream_requests":            m.get("stream_requests", 0),
        "admin_direct_count":         m.get("admin_direct_count", 0),
        "admin_direct_fuzzy_count":   m.get("admin_direct_fuzzy_count", 0),
        "raw_table_filtered_count":   m.get("raw_table_filtered_count", 0),
        "icons_stripped_count":       m.get("icons_stripped_count", 0),
        "fusion_used_count":          m.get("fusion_used_count", 0),
        "multi_criteria_count":       m.get("multi_criteria_count", 0),
        "coherence_rejected_count":   m.get("coherence_rejected_count", 0),
        "mode":                       "hybrid" if not EXTRACTIVE_MODE else "extractive",
        "model_currently_loaded":     model_is_loaded(),
        "version":                    "2.18.0",
        "optimizations": {
            "v2.17_fp16":       "TinyLlama float16 — sans bitsandbytes",
            "v2.17_whitelist":  f"{len(_COMMON_WORDS_FR)} mots FR non corrigés",
            "v2.17_coherence":  f"overlap min={COHERENCE_OVERLAP_MIN}",
            "v2.17_topk":       f"TOP_K={TOP_K}",
            "v2.15_fusion":     "fuse_chunks_answer() — agrégation multi-chunks sans LLM",
            "v2.15_multi_crit": f"{len(_MULTI_CRITERIA_PATTERNS)} patterns",
            "v2.15_tokens":     f"multi_criteres → 280 tokens | CAP={MAX_NEW_TOKENS_CAP}",
        },
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════ HEALTH ═══════════════

@app.get("/health", tags=["Systeme"])
async def health():
    n_chunks = _collection.count() if _collection else 0
    with get_db() as conn:
        n_msgs  = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
        n_sess  = conn.execute("SELECT COUNT(DISTINCT session_id) as c FROM conversations").fetchone()["c"]
        n_unans = conn.execute("SELECT COUNT(*) as c FROM unanswered_questions").fetchone()["c"]
        n_fb    = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
        n_pos   = conn.execute("SELECT COUNT(*) as c FROM feedback WHERE rating=1").fetchone()["c"]
        n_rag   = conn.execute("SELECT COUNT(*) as c FROM rag_enrichments").fetchone()["c"]
    rate = round(100 * n_pos / n_fb, 1) if n_fb > 0 else 0
    return {
        "status":          "ok" if n_chunks > 0 else "degraded",
        "version":         "2.18.0",
        "mode":            "hybrid" if not EXTRACTIVE_MODE else "extractive",
        "model_loaded":    model_is_loaded(),
        "rag_chunks":      n_chunks,
        "collection":      COLLECTION_NAME,
        "chroma_dir":      CHROMA_DB_DIR,
        "model_dir":       MODEL_DIR,
        "total_messages":  n_msgs,
        "total_sessions":  n_sess,
        "unanswered_logged": n_unans,
        "total_feedback":  n_fb,
        "satisfaction_rate": rate,
        "rag_enrichments": n_rag,
        "v218_features": {
            "fp16_model":          True,
            "whitelist_fr":        f"{len(_COMMON_WORDS_FR)} mots",
            "coherence_filter":    f"overlap >= {COHERENCE_OVERLAP_MIN}",
            "top_k":               TOP_K,
            "fusion_multi_chunks": True,
            "multi_criteria_patterns": len(_MULTI_CRITERIA_PATTERNS),
            "admin_rag_routes":    ["add","update","delete","entries","test"],
            "metrics_routes":      ["live","perf","tokens","conversations/stats"],
            "feedback_routes":     ["POST /feedback","GET /feedback/stats"],
            "streaming":           "POST /chat/stream SSE",
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/", include_in_schema=False)
async def root():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(html_path):
        return FileResponse(html_path)
    return {"message": "Chatbot Tunisie Telecom API TinyLlama v2.18"}


# ═══════════════ LANCEMENT ═══════════════

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Chatbot Tunisie Telecom — TinyLlama fp16  v2.18")
    print(f"  ChromaDB  : {CHROMA_DB_DIR} / {COLLECTION_NAME}")
    print(f"  Modele    : {MODEL_DIR}")
    print(f"  DB        : {DB_PATH}")
    print(f"  Mode      : {'extractif' if EXTRACTIVE_MODE else 'generatif fp16'}")
    print(f"  TOP_K     : {TOP_K}")
    print(f"  Port      : {API_PORT}")
    print()
    print("  [v2.17] fp16 | whitelist FR | coherence filter | TOP_K=5")
    print("  [v2.15] fusion multi-chunks | multi-criteres | tokens adaptes")
    print("  [ROUTES] RAG admin (add/update/delete/entries/test)")
    print("           Metrics (live/perf/tokens/conversations)")
    print("           Feedback (POST/stats)")
    print("           Streaming SSE (/chat/stream)")
    print("=" * 60)
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False, log_level="info")