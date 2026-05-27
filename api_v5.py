"""
=============================================================
API v3.0 — Chatbot Tunisie Telecom — Phi-3.5-mini-instruct
CORRECTIONS v3.0 (basées sur l'analyse TinyLlama qui marche) :
  [FIX-1] Prompt système EN FRANÇAIS (était en anglais → confusion multilingue)
  [FIX-2] CONFIDENCE_THRESHOLD abaissé à 0.45 (était 0.55 → trop de no_info)
  [FIX-3] MIN_CONTEXT_COVERAGE = 0.20 (était 0.04 → trop permissif)
  [FIX-4] Seuils adaptatifs court/long query comme TinyLlama
  [FIX-5] Nettoyage post-génération renforcé (_strip_raw_tables, _strip_icons)
  [FIX-6] Validation allégée (moins de faux rejets)
  [FIX-7] temperature=None quand do_sample=False
=============================================================
"""

import os, re, time, uuid, sqlite3, logging, threading, statistics, gc, asyncio
import traceback, unicodedata
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
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from rapidfuzz import fuzz, process

from auth_sqlite import (
    init_auth_db, require_auth,
    RegisterRequest, VerifyRequest, LoginRequest, ResendRequest, CheckEmailRequest,
    register_route, verify_route, login_route,
    resend_route, me_route, logout_route, check_email_route,
)

# =============================================================
# CONFIG
# =============================================================
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
MODEL_DIR  = os.environ.get("MODEL_DIR", "models/phi35-tt-merged")

# [FIX-2] Seuils abaissés comme TinyLlama
CONFIDENCE_THRESHOLD  = float(os.environ.get("CONFIDENCE_THRESHOLD",  "0.45"))
THRESHOLD_SHORT_QUERY = float(os.environ.get("THRESHOLD_SHORT_QUERY", "0.38"))
THRESHOLD_LONG_QUERY  = float(os.environ.get("THRESHOLD_LONG_QUERY",  "0.50"))
SHORT_QUERY_MAX_WORDS = 4

# [FIX-3] Coverage comme TinyLlama
MIN_CONTEXT_COVERAGE = float(os.environ.get("MIN_CONTEXT_COVERAGE", "0.20"))

MAX_NEW_TOKENS     = int(os.environ.get("MAX_NEW_TOKENS",     "150"))
MIN_NEW_TOKENS     = int(os.environ.get("MIN_NEW_TOKENS",     "20"))
MAX_NEW_TOKENS_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "280"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.3"))
GEN_TIMEOUT_S      = float(os.environ.get("GEN_TIMEOUT_S", "60.0"))
GEN_MAX_RETRIES    = int(os.environ.get("GEN_MAX_RETRIES", "2"))
USE_4BIT           = os.environ.get("USE_4BIT", "true").lower() == "true"

CHROMA_DB_DIR   = os.environ.get("CHROMA_DB_DIR",   "chroma_tt_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "tt_train")
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K           = int(os.environ.get("TOP_K", "3"))

DB_PATH    = os.environ.get("DB_PATH",    "data/chatbot.db")
STATIC_DIR = os.environ.get("STATIC_DIR", "static")
API_HOST   = "0.0.0.0"
API_PORT   = 8000
MAX_HISTORY = 10

STRUCTURED_CHUNK_MAX_CHARS = int(os.environ.get("STRUCTURED_CHUNK_MAX_CHARS", "600"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_cuda_generation_lock = threading.Lock()
_metrics_lock = threading.Lock()
_global_metrics = {
    "total_requests": 0, "rag_used_count": 0, "hors_sujet_count": 0,
    "greeting_count": 0, "hallucination_count": 0, "extractive_count": 0,
    "generative_count": 0, "generative_timeout_count": 0,
    "validation_rejected_count": 0, "no_info_count": 0,
    "total_latency_ms": 0, "confidence_scores": [],
    "positive_feedback": 0, "negative_feedback": 0,
    "validation_rejection_reasons": Counter(),
    "generative_error_oom": 0, "generative_error_runtime": 0,
    "generative_vram_skip": 0,
}

# =============================================================
# CONSTANTES
# =============================================================
GREETINGS_FR = [
    "bonjour","bonsoir","salut","hello","hi","bjr","bsr",
    "coucou","hey","allo","bj","good morning",
]
GREETINGS_AR = [
    "سلام","السلام عليكم","مرحبا","اهلا","صباح الخير","مساء الخير",
    "أهلا","أهلاً وسهلاً","يسعد صباحك","يسعد مساك","هلا","salam",
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
    "مرحباً! أنا المساعد الافتراضي لـ Tunisie Telecom.\n"
    "يمكنني مساعدتك في:\n"
    "- عروض الهاتف المحمول (Hayya، باقات 4G/5G)\n"
    "- الإنترنت الثابت (ADSL، Fibre، NetBox)\n"
    "- الشحن والرصيد\n"
    "- التجوال الدولي\n\n"
    "كيف يمكنني مساعدتك؟"
)
FALLBACK_NO_INFO_FR = (
    "Je n'ai pas trouve d'information precise sur ce sujet. "
    "Contactez le service client au 1298."
)
FALLBACK_NO_INFO_AR = (
    "لم أجد معلومات دقيقة حول هذا الموضوع. "
    "يرجى الاتصال بخدمة العملاء على الرقم 1298."
)
HORS_SUJET_FR = (
    "Je suis l'assistant Tunisie Telecom et je reponds uniquement "
    "aux questions sur nos offres et services."
)
HORS_SUJET_AR = (
    "أنا مساعد Tunisie Telecom ولا أجيب إلا على الأسئلة المتعلقة "
    "بعروضنا وخدماتنا."
)

HALLUCINATION_SIGNALS = ["ooredoo","myoredoo","orange tunisie","tunisiana","vodafone"]
GENERATION_NOISE      = ["casino","karting"]

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
    "vpn","waffi","4g","5g","3g","wifi","signal","debit","fibre","adsl","vdsl",
    "sim","esim","credit","facture","recharge","solde","activer",
    "souscrire","abonnement","abonner","inscrire","paiement","payer",
    "1298","mytt","agence","prepaye","postpaye","etranger","itinerance",
    "minutes","go","gigaoctet","mb","gb","panne","depannage","technique",
    "assistance","carte","fonctionne","disponible","cout","coute",
    "انترنت","شبكة","رصيد","شحن","عرض","عروض","اشتراك","تفعيل",
    "باقة","باقات","مكالمة","رسالة","بيانات","تجوال","فاتورة",
    "خدمة","خدمات","هاتف","موبايل","سيم","رقم","تحويل",
]

TELECOM_PRODUCT_NAMES = [
    "hayya","forfait","internet","recharge","abonnement","roaming",
    "netbox","adsl","fibre","activation","solde","tarif","offre",
    "mobile","sim","reseau","ussd","mytt","rapido","waffi","elissa",
]

_MULTILINE_KEYWORDS = [
    "inscrire","inscription","souscrire","souscription","activer","activation",
    "acceder","acces","telecharger","telechargement","comment","etapes",
    "possibilites","plusieurs","manieres","facons",
    "كيفية","خطوات","طريقة","كيف","تفعيل","اشتراك","تسجيل",
]

_LIST_TRIGGERS = [
    "avantages","caracteristiques","difference","comparer","comparaison",
    "quels sont","liste","offres disponibles","options disponibles",
    "pourquoi choisir","qu est ce que","presentation","decrire",
    "inclus","comprend","contient","fonctionnalites",
]

_STRUCTURED_MARKERS = [
    r'\b(Appels?|Internet|Mixte|Data|SMS)\s*:',
    r'^\s*[-•]\s+',
    r'\bà partir de\b.*\bDT\b',
    r'\d+\s*(?:Go|Mo|min|minutes?|heures?)\b',
    r'\bforfaits?\b.*\bDT\b',
]

_RAW_TABLE_UNIT_PATTERN = re.compile(
    r'^\s*[\d.,]+\s*(?:DT|H|Go|Mo|min|minutes?|heures?|TND|millimes?|dinars?|%)\s*$',
    re.IGNORECASE
)
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
]
_JSONL_ARTIFACTS = [
    (r'\bavec\s+est\b',                 'est'),
    (r'\best\s+de\s+de\b',             'est de'),
    (r'\bde\s+de\b',                   'de'),
    (r'voici\s+la\s+r[ee]ponse\s+de\s+Tunisie\s+Telecom\s*:\s*', ''),
    (r'\s{2,}',                        ' '),
]
_ICON_PATTERN = re.compile(
    r'[•·▸▶►✓✔★☆◆◇●○▪▫]\s*'
    r'|(?<!\w)[-–—]\s+(?=[A-ZÀ-Ža-zà-ž0-9])'
    r'|[^\x00-\x7F\u00C0-\u024F\u0600-\u06FF\u2019\u2018\u00AB\u00BB\n\r\t]',
    re.UNICODE
)
_VALID_ENDINGS = frozenset(['.', '!', '?', ':', '»', ';', '؟'])
BRUIT_DEBUT = ["Marketing","Contexte","Description","Concept","Source","Flash","DCCM","Cible"]

VOCAB_TELECOM = [
    "hayya","forfait","internet","recharge","abonnement","facture",
    "roaming","netbox","adsl","fibre","activation","solde","credit",
    "tarif","offre","mobile","sim","reseau","couverture","debit",
    "illimite","gratuit","appel","sms","4g","5g","wifi","ussd",
    "activer","desactiver","souscrire","telecom","tunisie","prix",
    "cout","minute","mois","go","mo","data","forfaits","pack",
]
CORRECTIONS_DIRECTES = {
    "haya":"hayya","haiya":"hayya","hayia":"hayya",
    "forfet":"forfait","internt":"internet",
    "aboneman":"abonnement","abonement":"abonnement",
    "recharg":"recharge","recharje":"recharge",
    "netboc":"netbox","netbok":"netbox",
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
        corrected.append(match[0] if match and match[0] != word else word)
    return " ".join(corrected)

def preprocess_query(query: str) -> str:
    return normalize_query(correct_telecom_keywords(query))

def get_adaptive_threshold(query: str) -> float:
    return (THRESHOLD_SHORT_QUERY
            if len(query.strip().split()) <= SHORT_QUERY_MAX_WORDS
            else THRESHOLD_LONG_QUERY)

def detect_language(text: str) -> str:
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F]', text))
    latin_chars  = len(re.findall(r'[a-zA-Z\u00C0-\u00FF]', text))
    total = arabic_chars + latin_chars
    if total == 0: return 'fr'
    if arabic_chars / total >= 0.40: return 'ar'
    return 'fr'

def is_greeting(query: str) -> bool:
    q = query.strip()
    if len(q) <= 60 and any(g in q for g in GREETINGS_AR): return True
    return len(q.lower().strip()) <= 40 and any(g in q.lower() for g in GREETINGS_FR)

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
    q = normalize_query(query.lower())
    return (any(kw in q for kw in _MULTILINE_KEYWORDS) or
            any(kw in query for kw in _MULTILINE_KEYWORDS if '\u0600' <= kw[0] <= '\u06FF'))

def is_structured_list(text: str) -> bool:
    return sum(1 for pat in _STRUCTURED_MARKERS
               if re.search(pat, text, re.IGNORECASE | re.MULTILINE)) >= 2

def _contains_raw_table(text: str) -> bool:
    pipe_segments = re.split(r'\s*\|\s*', text)
    if len(pipe_segments) < 3: return False
    return sum(1 for seg in pipe_segments if _RAW_TABLE_UNIT_PATTERN.match(seg.strip())) >= 3

def _strip_raw_tables(text: str) -> str:
    if not _contains_raw_table(text): return text
    lines = text.split('\n')
    clean = []
    for line in lines:
        if not _contains_raw_table(line):
            cleaned = re.sub(
                r'[\d.,]+\s*(?:DT|H|Go|Mo|min|TND|millimes?|dinars?)\s*'
                r'(?:\|\s*[\d.,]+\s*(?:DT|H|Go|Mo|min|TND|millimes?|dinars?)\s*)+',
                '', line, flags=re.IGNORECASE
            ).strip()
            if cleaned: clean.append(cleaned)
    return ' '.join(clean).strip()

def _strip_icons(text: str) -> str:
    cleaned = _ICON_PATTERN.sub('', text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return re.sub(r' {2,}', ' ', cleaned).strip()

def clean_chunk(text: str) -> str:
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()

def clean_jsonl_artifacts(text: str) -> str:
    text = _strip_raw_tables(text)
    for pattern, replacement in _JSONL_ARTIFACTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = _strip_icons(text).strip()
    if text and text[-1] not in '.!?؟':
        text = text.rstrip(',;:') + '.'
    return text

def _extract_answer_from_chunk(text: str) -> str:
    for marker in ['?', '؟']:
        idx = text.find(marker)
        if idx != -1 and idx < len(text) * 0.6:
            after = text[idx + 1:].strip()
            if len(after) > 10:
                return clean_jsonl_artifacts(after)
    return clean_jsonl_artifacts(text)

def is_truncated(response: str) -> bool:
    s = response.strip()
    if len(s) < 10: return True
    if s.endswith('...') or s.endswith('…'): return True
    if s[-1] not in _VALID_ENDINGS: return True
    return False

def get_fallback(lang: str) -> str:
    return FALLBACK_NO_INFO_AR if lang == 'ar' else FALLBACK_NO_INFO_FR

def get_hors_sujet(lang: str) -> str:
    return HORS_SUJET_AR if lang == 'ar' else HORS_SUJET_FR

def get_greeting_response(lang: str) -> str:
    return GREETING_RESPONSE_AR if lang == 'ar' else GREETING_RESPONSE_FR

def _get_free_vram_gb() -> float:
    if not torch.cuda.is_available(): return 0.0
    try:
        return (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)) / 1e9
    except Exception:
        return 0.0

# =============================================================
# EXTRACTIF
# =============================================================

def find_relevant_sentences(text: str, query: str, n: int = 3) -> List[str]:
    stop = {"les","des","de","la","le","un","une","est","que","quel","quels","comment",
            "sont","pour","dans","du","au","et","en","je","il","elle","ce","ou","par","avec","qui","sur"}
    q_norm  = normalize_query(query.lower())
    q_words = {w for w in q_norm.split() if len(w) > 2 and w not in stop}
    sentences = [s.strip() for s in re.split(r'[.!?؟;]\s+|\n', text) if len(s.strip()) > 25]
    if not sentences: return []
    bonus_kw = ["tarif","prix","dt","offre","avantage","activation","forfait","go","mo",
                "minute","mois","gratuit","composez","activer","disponible"]
    scored = []
    for s in sentences:
        s_norm = normalize_query(s.lower())
        score  = sum(1 for w in q_words if w in s_norm)
        score += sum(0.5 for kw in bonus_kw if kw in s.lower())
        scored.append((score, s))
    scored.sort(reverse=True)
    top = [s for sc, s in scored[:n] if sc > 0]
    return top if top else [s for _, s in scored[:2]]

def build_extractive_answer(chunks: List[dict], query: str, lang: str = 'fr') -> str:
    if not chunks: return get_fallback(lang)
    for chunk in chunks:
        raw_text    = clean_chunk(chunk["text"])
        answer_part = _extract_answer_from_chunk(raw_text)
        if _contains_raw_table(answer_part): continue
        if is_structured_list(answer_part) and not any(answer_part.startswith(b) for b in BRUIT_DEBUT):
            return clean_jsonl_artifacts(answer_part[:STRUCTURED_CHUNK_MAX_CHARS])
    all_sentences = []
    for chunk in chunks:
        raw_text    = clean_chunk(chunk["text"])
        answer_part = _extract_answer_from_chunk(raw_text)
        if _contains_raw_table(answer_part): continue
        for s in find_relevant_sentences(answer_part, query, n=2):
            if not any(s.startswith(b) for b in BRUIT_DEBUT):
                all_sentences.append(s)
    if not all_sentences: return get_fallback(lang)
    best = all_sentences[0]
    if len(best) > 300:
        best = re.split(r'(?<=[.!?؟])\s+', best[:350])[0]
    return clean_jsonl_artifacts(best)

# =============================================================
# GESTIONNAIRE MODELE PHI-3.5
# =============================================================

class PhiModelManager:
    def __init__(self):
        self._model     = None
        self._tokenizer = None
        self._lock      = threading.RLock()
        self._loaded    = False
        self._quant     = "unknown"

    def load(self):
        if self._loaded: return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise SystemExit("CUDA non disponible.")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("[GPU] %s | VRAM: %.1f GB", torch.cuda.get_device_name(0), vram)
        model_path = MODEL_DIR if os.path.isdir(MODEL_DIR) else BASE_MODEL
        logger.info("[MODEL] Chargement Phi-3.5 : %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token    = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        if USE_4BIT:
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path, quantization_config=bnb,
                    device_map={"": "cuda:0"}, trust_remote_code=True,
                    attn_implementation="eager",
                )
                self._quant = "4bit_nf4"
            except Exception as e:
                logger.warning("[MODEL] 4-bit échoué (%s) -> fp16", e)
                self._load_fp16(model_path)
        else:
            self._load_fp16(model_path)
        self._model.eval()
        self._model.config.use_cache = True
        self._warmup()
        logger.info("[MODEL] Pret | quant=%s | VRAM=%.2f GB", self._quant,
                    torch.cuda.memory_allocated(0) / 1e9)
        self._loaded = True

    def _load_fp16(self, model_path):
        from transformers import AutoModelForCausalLM
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map={"": "cuda:0"}, trust_remote_code=True,
            attn_implementation="eager", low_cpu_mem_usage=True,
        )
        self._quant = "fp16"

    def _warmup(self):
        try:
            dummy = self._tokenizer("Test.", return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                self._model.generate(**dummy, max_new_tokens=5, do_sample=False,
                                     pad_token_id=self._tokenizer.eos_token_id)
            torch.cuda.empty_cache()
            logger.info("[MODEL] Warmup OK")
        except Exception as e:
            logger.warning("[MODEL] Warmup échoué : %s", e)

    def get(self):
        if not self._loaded: raise RuntimeError("Modèle non chargé.")
        return self._model, self._tokenizer

    def is_loaded(self) -> bool: return self._loaded
    def quant_mode(self) -> str: return self._quant

_gen_manager = PhiModelManager()

# =============================================================
# PROMPT PHI-3.5 — [FIX-1] EN FRANÇAIS comme TinyLlama
# =============================================================

def _build_clean_context(chunks: List[dict], max_chars: int = 300) -> str:
    parts = []
    for chunk in chunks[:2]:
        cleaned     = clean_chunk(chunk["text"])
        answer_only = _extract_answer_from_chunk(cleaned)
        if _contains_raw_table(answer_only):
            answer_only = _strip_raw_tables(answer_only)
        if answer_only and len(answer_only) > 10:
            parts.append(answer_only[:max_chars])
    return " ".join(parts)

def estimate_max_tokens(query: str, chunks: List[dict]) -> int:
    q_norm = normalize_query(query.lower())
    if _is_multiline_query(query):
        return min(MAX_NEW_TOKENS_CAP, 220)
    if any(kw in q_norm for kw in _LIST_TRIGGERS):
        return min(MAX_NEW_TOKENS_CAP, 180)
    return MAX_NEW_TOKENS

def build_generative_prompt_phi(query: str, chunks: List[dict], lang: str = 'fr') -> str:
    is_multiline = _is_multiline_query(query)
    is_list      = any(kw in normalize_query(query.lower()) for kw in _LIST_TRIGGERS)

    if is_multiline or is_list:
        context     = _build_clean_context(chunks, max_chars=400)
        instruction = "Explique en 3 phrases courtes et completes."
    else:
        context     = _build_clean_context(chunks, max_chars=250)
        instruction = "Reponds en 1 ou 2 phrases courtes et completes."

    # [FIX-1] Prompt EN FRANÇAIS — même approche que TinyLlama
    if lang == 'ar':
        system = (
            "أنت مساعد Tunisie Telecom الرسمي. "
            "استخدم فقط المعلومات المقدمة أدناه. "
            "لا تخترع أي شيء. أجب باللغة العربية. "
            "لا تستخدم رموز تعبيرية أو نقاط."
        )
        user = (
            f"المعلومات المتاحة : {context}\n\n"
            f"السؤال : {query}\n\n"
            f"أجب في جملة أو جملتين قصيرتين وكاملتين."
        )
    else:
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

    # Template Phi-3.5
    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
    ]
    return messages

# =============================================================
# VALIDATION — [FIX-6] Allégée
# =============================================================

def _validate_phi_response(response: str, query: str, context: str) -> tuple:
    if len(response.strip()) < 10:
        return False, "trop_court"
    if any(sig in response.lower() for sig in HALLUCINATION_SIGNALS):
        return False, "hallucination"
    if any(n in response.lower() for n in GENERATION_NOISE):
        return False, "bruit"
    if _contains_raw_table(response):
        return False, "tableau_brut"
    # [FIX-6] Suppression des PROMPT_LEAKS trop agressifs
    # Garder seulement les vraies fuites
    real_leaks = ["as an ai language model", "je suis un modele ia"]
    resp_norm = normalize_query(response.lower())
    for leak in real_leaks:
        if leak in resp_norm:
            return False, "fuite_prompt"
    # [FIX-3] Coverage 0.20 comme TinyLlama
    context_words  = set(normalize_query(context.lower()).split())
    response_words = [w for w in normalize_query(response.lower()).split() if len(w) > 4]
    if response_words:
        coverage = sum(1 for w in response_words if w in context_words) / len(response_words)
        if coverage < MIN_CONTEXT_COVERAGE:
            return False, f"couverture_faible_{coverage:.2f}"
    return True, "ok"

# =============================================================
# NETTOYAGE OUTPUT PHI
# =============================================================

def _clean_phi_output(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", "", text).strip()
    text = re.sub(r"^(?:assistant|Assistant)\s*:\s*", "", text).strip()
    text = re.sub(r"^(?:Response|Reponse|Answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    # Supprime fragment de question en début
    match = re.match(r"^[^.!?؟]{0,120}[?؟]\s*(?:Response|Reponse)?\s*:?\s*", text, re.DOTALL)
    if match and match.end() < len(text) * 0.6:
        text = text[match.end():].strip()
    text = re.sub(r'\s*\.\.\.\s*$', '.', text).strip()
    return clean_jsonl_artifacts(text)

# =============================================================
# GENERATION PHI
# =============================================================

def _generate_sync(question: str, chunks: List[dict], lang: str = 'fr') -> tuple:
    if not _gen_manager.is_loaded():
        return build_extractive_answer(chunks, question, lang), "extractive_fallback"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        free_vram = _get_free_vram_gb()
        if free_vram < 0.2:
            with _metrics_lock:
                _global_metrics["generative_vram_skip"] += 1
            return build_extractive_answer(chunks, question, lang), "extractive_oom"

    try:
        model, tokenizer = _gen_manager.get()
        messages         = build_generative_prompt_phi(question, chunks, lang)
        context          = _build_clean_context(chunks, max_chars=300)
        max_tokens       = estimate_max_tokens(question, chunks)

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to("cuda:0")

        last_reason = "unknown"
        configs = [(False, 0.1), (True, 0.3)]

        for attempt in range(1, GEN_MAX_RETRIES + 1):
            do_sample, temp = configs[min(attempt - 1, 1)]
            try:
                t0 = time.time()
                with _cuda_generation_lock:
                    with torch.no_grad():
                        gen_kwargs = dict(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_tokens,
                            do_sample=do_sample,
                            repetition_penalty=REPETITION_PENALTY,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                        )
                        if do_sample:
                            gen_kwargs["temperature"] = temp
                        output_ids = model.generate(**gen_kwargs)
                elapsed = time.time() - t0
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                with _metrics_lock: _global_metrics["generative_error_oom"] += 1
                break

            if elapsed > GEN_TIMEOUT_S:
                with _metrics_lock: _global_metrics["generative_timeout_count"] += 1
                break

            n_input   = inputs["input_ids"].shape[1]
            raw       = tokenizer.decode(output_ids[0][n_input:], skip_special_tokens=True).strip()
            response  = _clean_phi_output(raw)

            # Tronquer à max 5 phrases
            sentences = re.split(r'(?<=[.!?؟])\s+', response)
            if len(sentences) > 5:
                response = " ".join(sentences[:5])

            if is_truncated(response):
                return build_extractive_answer(chunks, question, lang), "extractive_truncated"

            valid, reason = _validate_phi_response(response, question, context)
            if valid:
                with _metrics_lock: _global_metrics["generative_count"] += 1
                logger.info("[GEN-PHI] OK tentative=%d | %.2fs | '%s'...",
                            attempt, elapsed, response[:80])
                return response, "generative"
            else:
                last_reason = reason
                logger.warning("[GEN-PHI] Rejet tentative=%d raison=%s", attempt, reason)
                with _metrics_lock:
                    _global_metrics["validation_rejected_count"] += 1
                    _global_metrics["validation_rejection_reasons"][reason] += 1

        with _metrics_lock: _global_metrics["extractive_count"] += 1
        return build_extractive_answer(chunks, question, lang), "extractive_validation"

    except RuntimeError as e:
        logger.error("[GEN-PHI] RuntimeError : %s", e)
        torch.cuda.empty_cache(); gc.collect()
        with _metrics_lock: _global_metrics["generative_error_runtime"] += 1
        return build_extractive_answer(chunks, question, lang), "extractive_runtime"

    except Exception as e:
        logger.error("[GEN-PHI] Exception : %s", traceback.format_exc())
        return build_extractive_answer(chunks, question, lang), "extractive_fallback"

async def generate_answer(question: str, chunks: List[dict], lang: str = 'fr') -> tuple:
    return await asyncio.to_thread(_generate_sync, question, chunks, lang)

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
            logger.info("[EMBED] Chargement : %s", EMBED_MODEL)
            _embed_model = SentenceTransformer(EMBED_MODEL)
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
        n_results=min(top_k + 2, _collection.count()),
        include=["documents","metadatas","distances"]
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
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
    hits1 = rag_search(query_original, top_k)
    hits2 = rag_search(query_processed, top_k) if fuzz.ratio(query_original.lower(), query_processed) < 85 else []
    seen  = {}
    for h in hits1 + hits2:
        if h["text"] not in seen or h["score"] > seen[h["text"]]["score"]:
            seen[h["text"]] = h
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]

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
        yield conn; conn.commit()
    except Exception:
        conn.rollback(); raise
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
# SCHEMAS
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
    lang:             str = "fr"

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
# FASTAPI
# =============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Chatbot Tunisie Telecom — Phi-3.5 v3.0")
    logger.info("  MODEL_DIR=%s | USE_4BIT=%s", MODEL_DIR, USE_4BIT)
    logger.info("  CONF_THRESHOLD=%.2f | MIN_COVERAGE=%.2f",
                CONFIDENCE_THRESHOLD, MIN_CONTEXT_COVERAGE)
    logger.info("=" * 60)
    init_db()
    init_auth_db()
    init_rag()
    logger.info("[STARTUP] Chargement Phi-3.5...")
    try:
        _gen_manager.load()
        logger.info("[STARTUP] Modele pret (quant=%s)", _gen_manager.quant_mode())
    except Exception as e:
        logger.error("[STARTUP] Echec : %s", e)
    yield
    logger.info("API arretee")

app = FastAPI(
    title="Chatbot Tunisie Telecom — Phi-3.5 v3.0",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =============================================================
# ROUTES AUTH
# =============================================================
@app.post("/auth/register", tags=["Auth"])
async def register(req: RegisterRequest): return register_route(req)

@app.post("/auth/verify", tags=["Auth"])
async def verify(req: VerifyRequest): return verify_route(req)

@app.post("/auth/login", tags=["Auth"])
async def login(req: LoginRequest): return login_route(req)

@app.post("/auth/resend", tags=["Auth"])
async def resend(req: ResendRequest): return resend_route(req)

@app.get("/auth/me", tags=["Auth"])
async def me(user=Depends(require_auth)): return me_route(user)

@app.post("/auth/logout", tags=["Auth"])
async def logout(authorization: str = Header(None)): return logout_route(authorization)

@app.post("/auth/check-email", tags=["Auth"])
async def check_email(req: CheckEmailRequest): return await check_email_route(req)

# =============================================================
# ROUTE CHAT
# =============================================================
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    t0         = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    query      = request.message.strip()

    if not query:
        raise HTTPException(400, "Message vide.")

    lang = detect_language(query)

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

    query_processed = preprocess_query(query)
    hits            = rag_search_best(query, query_processed)
    confidence      = hits[0]["score"] if hits else 0.0
    adaptive_thresh = get_adaptive_threshold(query)
    rag_used        = confidence >= adaptive_thresh

    logger.info("[RAG] conf=%.3f seuil=%.2f chunks=%d", confidence, adaptive_thresh, len(hits))
    for i, h in enumerate(hits[:3]):
        logger.info("[CHUNK-%d] %.3f | %s | '%s'...", i+1, h["score"], h["filename"], h["text"][:60])

    if rag_used:
        answer, answer_mode = await generate_answer(query, hits, lang)
    else:
        log_unanswered(session_id, query, "confiance_faible", is_telecom=True)
        answer      = get_fallback(lang)
        answer_mode = "no_info"
        hits        = []
        with _metrics_lock: _global_metrics["no_info_count"] += 1

    if any(sig in answer.lower() for sig in HALLUCINATION_SIGNALS):
        log_unanswered(session_id, query, "hallucination", is_telecom=True)
        answer      = get_fallback(lang)
        answer_mode = "fallback_securite"
        with _metrics_lock: _global_metrics["hallucination_count"] += 1

    save_message(session_id, "user", query)
    save_message(session_id, "assistant", answer)
    elapsed = int((time.time() - t0) * 1000)

    sources = [{"filename": h["filename"], "year": h["year"],
                "theme": h["theme"], "score": h["score"], "text": h["text"][:300]}
               for h in (hits if rag_used else [])]

    with _metrics_lock:
        _global_metrics["total_requests"]   += 1
        _global_metrics["rag_used_count"]   += int(rag_used)
        _global_metrics["total_latency_ms"] += elapsed
        _global_metrics["confidence_scores"].append(confidence)

    logger.info("[%s] %dms | conf=%.3f | rag=%s | lang=%s",
                answer_mode, elapsed, confidence, rag_used, lang)

    return ChatResponse(
        session_id=session_id, answer=answer, sources=sources,
        is_telecom=True, rag_used=rag_used, confidence=confidence,
        response_time_ms=elapsed, mode=answer_mode, lang=lang)

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

@app.get("/feedback/stats", tags=["Feedback"])
async def feedback_stats():
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) as positive, "
            "SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) as negative "
            "FROM feedback").fetchone()
        recent = conn.execute(
            "SELECT session_id, comment, user_question, bot_answer, timestamp "
            "FROM feedback WHERE rating=-1 ORDER BY timestamp DESC LIMIT 15"
        ).fetchall()
    pos, neg, total = row["positive"] or 0, row["negative"] or 0, row["total"] or 0
    return {"positive": pos, "negative": neg, "total": total,
            "satisfaction_rate": round(100*pos/total, 1) if total > 0 else 0,
            "recent_negative": [dict(r) for r in recent]}

# =============================================================
# HISTORIQUE
# =============================================================
@app.get("/history/{session_id}", tags=["Historique"])
async def history(session_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM conversations "
            "WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
    if not rows: raise HTTPException(404, "Session introuvable.")
    return [dict(r) for r in rows]

@app.delete("/history/{session_id}", tags=["Historique"])
async def clear_history(session_id: str):
    n = delete_session(session_id)
    if n == 0: raise HTTPException(404, "Session introuvable.")
    return {"deleted": n}

# =============================================================
# RAG ADMIN
# =============================================================
@app.post("/admin/rag/add", tags=["RAG Admin"])
async def rag_add(req: RagEntryRequest, user=Depends(require_auth)):
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(400, "question et answer sont obligatoires")
    chunk_id   = f"admin_{uuid.uuid4().hex[:12]}"
    chunk_text = f"Pour la question : {req.question.strip()}, voici la reponse : {req.answer.strip()}."
    try:
        _collection.add(
            documents=[chunk_text],
            metadatas=[{"file_name": "admin_enrichment", "filename": "admin_enrichment",
                        "year": str(datetime.now().year), "theme": req.theme or "enrichissement_admin",
                        "question": req.question[:200]}],
            ids=[chunk_id])
    except Exception as e:
        raise HTTPException(500, f"Erreur ChromaDB : {e}")
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rag_enrichments "
                "(chunk_id, question, answer, theme, source_note) VALUES (?,?,?,?,?)",
                (chunk_id, req.question, req.answer, req.theme or "enrichissement_admin",
                 req.source_note or ""))
    return {"status": "ok", "chunk_id": chunk_id, "n_chunks": _collection.count()}

@app.get("/admin/rag/entries", tags=["RAG Admin"])
async def rag_entries(limit: int = 100, user=Depends(require_auth)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, chunk_id, question, answer, theme, timestamp "
            "FROM rag_enrichments ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return {"total": len(rows), "entries": [dict(r) for r in rows]}

@app.delete("/admin/cache/clear", tags=["Admin"])
async def clear_cache(user=Depends(require_auth)):
    return {"status": "ok", "message": "Pas de cache sémantique dans cette version"}

# =============================================================
# METRICS & HEALTH
# =============================================================
@app.get("/metrics", tags=["Admin"])
async def get_metrics(user=Depends(require_auth)):
    with _metrics_lock:
        m = dict(_global_metrics)
    n    = m["total_requests"] or 1
    conf = m.pop("confidence_scores", [])
    rejection_reasons = dict(m.pop("validation_rejection_reasons", {}))
    vram_used = torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0
    vram_free = _get_free_vram_gb()
    return {
        **m,
        "rag_rate":            m["rag_used_count"] / n,
        "avg_latency_ms":      m["total_latency_ms"] / n,
        "avg_confidence":      sum(conf)/len(conf) if conf else 0,
        "generative_rate":     m["generative_count"] / n,
        "no_info_rate":        m["no_info_count"] / n,
        "validation_rejection_reasons": rejection_reasons,
        "model_loaded":        _gen_manager.is_loaded(),
        "quant_mode":          _gen_manager.quant_mode(),
        "version":             "3.0.0",
        "model_dir":           MODEL_DIR,
        "conf_threshold":      CONFIDENCE_THRESHOLD,
        "min_context_coverage": MIN_CONTEXT_COVERAGE,
        "gpu_vram_used_gb":    round(vram_used, 2),
        "gpu_vram_free_gb":    round(vram_free, 2),
        "timestamp":           datetime.now().isoformat(),
    }

@app.get("/health", tags=["Systeme"])
async def health():
    n_chunks = _collection.count() if _collection else 0
    with get_db() as conn:
        n_msgs = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
        n_sess = conn.execute("SELECT COUNT(DISTINCT session_id) as c FROM conversations").fetchone()["c"]
        n_fb   = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
    vram_free = _get_free_vram_gb()
    status    = "ok" if (_gen_manager.is_loaded() and n_chunks > 0 and vram_free >= 0.3) else "degraded"
    return {
        "status":          status,
        "version":         "3.0.0",
        "model":           "Phi-3.5-mini-instruct",
        "model_loaded":    _gen_manager.is_loaded(),
        "quant_mode":      _gen_manager.quant_mode(),
        "gpu":             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_vram_free_gb": round(vram_free, 2),
        "rag_chunks":      n_chunks,
        "total_messages":  n_msgs,
        "total_sessions":  n_sess,
        "total_feedback":  n_fb,
        "conf_threshold":  CONFIDENCE_THRESHOLD,
        "min_coverage":    MIN_CONTEXT_COVERAGE,
        "timestamp":       datetime.now().isoformat(),
    }

@app.get("/logs/unanswered", tags=["Admin"])
async def unanswered(limit: int = 50, reason: Optional[str] = None,
                     user=Depends(require_auth)):
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

@app.get("/admin/conversations/stats", tags=["Admin"])
async def conversations_stats(limit: int = 1000, user=Depends(require_auth)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT content FROM conversations WHERE role='user' "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    from collections import Counter
    q_counter = Counter(r["content"].strip().lower()[:100] for r in rows)
    return {
        "total_questions": len(rows),
        "top5_questions":  [{"question": k, "count": v}
                            for k, v in q_counter.most_common(5)],
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/", include_in_schema=False)
async def root():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(html_path): return FileResponse(html_path)
    return {"message": "Chatbot Tunisie Telecom — Phi-3.5 v3.0"}

# =============================================================
# LANCEMENT
# =============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Chatbot Tunisie Telecom — Phi-3.5-mini-instruct v3.0")
    print(f"  MODEL_DIR        : {MODEL_DIR}")
    print(f"  CONF_THRESHOLD   : {CONFIDENCE_THRESHOLD}")
    print(f"  MIN_COVERAGE     : {MIN_CONTEXT_COVERAGE}")
    print(f"  USE_4BIT         : {USE_4BIT}")
    print(f"  ChromaDB         : {CHROMA_DB_DIR} / {COLLECTION_NAME}")
    print(f"  Port             : {API_PORT}")
    print("=" * 60)
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False, log_level="info")