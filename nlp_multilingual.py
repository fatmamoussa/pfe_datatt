"""
Module NLP Multilingue pour Détection/Traduction AR ↔ FR
Détecte la langue automatiquement et traduit si nécessaire

CORRECTIONS v2 :
  - Dictionnaire AR→FR enrichi pour couvrir les vraies requêtes clients TT
  - Détection de langue plus robuste (turc/latin arabe dialectal inclus)
  - Normalisation arabe étendue (diacritiques complets, tatweel)
  - Support arabe dialectal tunisien enrichi
  - Traduction directe de phrases arabes fréquentes (pas seulement mot à mot)
"""

import re
from typing import Tuple, Optional, Dict, List
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================
# DÉTECTION DE LANGUE (Pattern-based + Unicode)
# ============================================================

class Language(Enum):
    """Langues supportées"""
    ARABIC = "ar"
    FRENCH = "fr"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"

# Plages Unicode
ARABIC_RANGE = (0x0600, 0x06FF)  # Arabic block
ARABIC_EXTENDED_A = (0x0750, 0x077F)
ARABIC_PRESENTATION_A = (0xFB50, 0xFDFF)
ARABIC_PRESENTATION_B = (0xFE70, 0xFEFF)

FRENCH_ACCENTS = set('àâäéèêëïîôöùûüœæç')

def is_arabic_char(char: str) -> bool:
    """Vérifie si c'est un caractère arabe (toutes plages Unicode)"""
    code = ord(char)
    return (
        ARABIC_RANGE[0] <= code <= ARABIC_RANGE[1] or
        ARABIC_EXTENDED_A[0] <= code <= ARABIC_EXTENDED_A[1] or
        ARABIC_PRESENTATION_A[0] <= code <= ARABIC_PRESENTATION_A[1] or
        ARABIC_PRESENTATION_B[0] <= code <= ARABIC_PRESENTATION_B[1]
    )

def is_french_char(char: str) -> bool:
    """Vérifie si c'est un caractère français (latin + accents)"""
    return not is_arabic_char(char) and char.isalpha()

def detect_language(text: str) -> Language:
    """
    Détecte la langue du texte.
    Retourne: Language.ARABIC, Language.FRENCH, Language.MIXED, Language.UNKNOWN
    """
    if not text or len(text.strip()) < 2:
        return Language.UNKNOWN

    text = text.strip()

    arabic_chars = sum(1 for c in text if is_arabic_char(c))
    french_chars = sum(1 for c in text if is_french_char(c))
    total_chars = len([c for c in text if c.isalpha()])

    if total_chars == 0:
        return Language.UNKNOWN

    ar_ratio = arabic_chars / total_chars
    fr_ratio = french_chars / total_chars

    # Arabe dominant (>70%)
    if ar_ratio > 0.70:
        return Language.ARABIC

    # Mélange AR+FR (les deux présents)
    if arabic_chars > 0 and french_chars > 0:
        return Language.MIXED

    # Français dominant (>70%)
    if fr_ratio > 0.70:
        return Language.FRENCH

    # Cas particulier: chiffres + quelques lettres (ex: "*140# svp")
    if fr_ratio > 0.30:
        return Language.FRENCH

    return Language.UNKNOWN


# ============================================================
# PHRASES ENTIÈRES AR→FR  (traductions directes, prioritaires)
# ============================================================
# Ces phrases sont cherchées EN PREMIER avant la traduction mot-à-mot.
# Ordre : du plus long au plus court.

AR_PHRASES_FR = [
    # Questions fréquentes clients
    ("ما هي عروض الأنترنت المنزلي", "quelles sont les offres internet fixe"),
    ("ما هي عروض الانترنت المنزلي", "quelles sont les offres internet fixe"),
    ("ما هو عرض هيا من تونس تيليكوم", "quelle est l'offre hayya de tunisie telecom"),
    ("ما هو عرض هايا من تونس تيليكوم", "quelle est l'offre hayya de tunisie telecom"),
    ("كم يكلف باقة الأنترنت الشهرية", "quel est le prix du forfait internet mensuel"),
    ("كم يكلف باقة الانترنت الشهرية", "quel est le prix du forfait internet mensuel"),
    ("كم سعر الفايبر", "quel est le prix de la fibre"),
    ("كيف أشتري باقة هايا", "comment acheter un forfait hayya"),
    ("كيف اشتري باقة هايا", "comment acheter un forfait hayya"),
    ("هل يوجد رومينج دولي", "est-ce qu'il y a du roaming international"),
    ("هل يوجد رومينغ دولي", "est-ce qu'il y a du roaming international"),
    ("ما هي خدمات تونس للاتصالات", "quels sont les services de tunisie telecom"),
    ("ما هي خدمات تونس تيليكوم", "quels sont les services de tunisie telecom"),
    ("كيف أشحن رصيدي", "comment recharger mon crédit"),
    ("كيف اشحن رصيدي", "comment recharger mon crédit"),
    ("كيف أعرف رصيدي", "comment consulter mon solde"),
    ("كيف اعرف رصيدي", "comment consulter mon solde"),
    ("ما هو رقم خدمة العملاء", "quel est le numéro du service client"),
    ("ما هو رقم الخدمة", "quel est le numéro du service"),
    ("كيف أستخدم سوس سولد", "comment utiliser SOS solde"),
    ("كيف استخدم سوس سولد", "comment utiliser SOS solde"),
    ("ما هي أسعار الروامينج", "quels sont les prix du roaming"),
    ("ما هي اسعار الروامينج", "quels sont les prix du roaming"),
    ("ما هي عروض الحج", "quelles sont les offres hajj"),
    ("ما هو عرض الحج", "quelle est l'offre hajj"),
    ("ما هي باقات الانترنت", "quels sont les forfaits internet"),
    ("ما هي باقات الأنترنت", "quels sont les forfaits internet"),
    ("كيف أفعل الفايبر", "comment activer la fibre"),
    ("كيف افعل الفايبر", "comment activer la fibre"),
    ("كيف أشترك في وافي", "comment souscrire à waffi"),
    ("كيف اشترك في وافي", "comment souscrire à waffi"),
    ("ما هو تطبيق ماي تي تي", "qu'est-ce que l'application my tt"),
    ("كيف أحمل تطبيق ماي تي تي", "comment télécharger l'application my tt"),
    ("ما هو رقم الشكاوى", "quel est le numéro des réclamations"),
    ("كيف أغير رقمي", "comment changer mon numéro"),
    ("ما هي عروض الحج من تونس تيليكوم", "quelles sont les offres hajj de tunisie telecom"),
    ("ما هي شروط الاشتراك في وافي", "quelles sont les conditions d'abonnement à waffi"),
    ("كيف أفعل الانترنت على هاتفي", "comment activer internet sur mon mobile"),
    ("كيف افعل الانترنت على هاتفي", "comment activer internet sur mon mobile"),
    ("ما هو الكود للاشتراك في انترنت", "quel est le code pour souscrire à internet"),
    ("ما هو كود الانترنت", "quel est le code internet"),
]

# ============================================================
# DICTIONNAIRE ARABE → FRANÇAIS (Tunisie Telecom)
# ============================================================

AR_FR_DICT = {
    # === Produits TT ===
    "هايا": "HAYYA",
    "هيا": "HAYYA",
    "حايا": "HAYYA",
    "وافي": "WAFFI",
    "رابيدو": "RAPIDO",
    "فايبر": "fibre",
    "الفايبر": "fibre",
    "نت بوكس": "NetBox",
    "ماي تي تي": "My TT",
    "بو9": "Po9",
    "أوميغا": "OHMega",
    "او ميغا": "OHMega",
    "سمارت": "Smart",
    "ريد بول": "RedBull",

    # === Termes télécom généraux ===
    "عرض": "offre",
    "عروض": "offres",
    "باقة": "forfait",
    "باقات": "forfaits",
    "اشتراك": "abonnement",
    "الاشتراك": "abonnement",
    "خدمة": "service",
    "خدمات": "services",
    "سعر": "prix",
    "اسعار": "prix",
    "أسعار": "prix",
    "تكلفة": "coût",
    "رسوم": "frais",
    "تسعير": "tarification",
    "التسعير": "tarification",
    "تعرفة": "tarif",
    "رصيد": "solde crédit",
    "شحن": "recharge",
    "إعادة شحن": "recharge",
    "شحن الرصيد": "recharge crédit",
    "تفعيل": "activation",
    "تفعيل الخدمة": "activation service",
    "إلغاء": "annulation",
    "الغاء": "annulation",
    "إلغاء الاشتراك": "résiliation abonnement",
    "تجديد": "renouvellement",
    "تحويل": "transfert",
    "نقل الرقم": "portabilité",
    "البورتابيليتي": "portabilité",
    "كود ريو": "code RIO",

    # === Internet ===
    "إنترنت": "internet",
    "انترنت": "internet",
    "الإنترنت": "internet",
    "الانترنت": "internet",
    "أنترنت": "internet",
    "الأنترنت": "internet",
    "إنترنت مجاني": "internet gratuit",
    "إنترنت منزلي": "internet fixe",
    "إنترنت محمول": "internet mobile",
    "الإنترنت المحمول": "internet mobile",
    "جيغا": "Go",
    "غيغا": "Go",
    "ميغا": "Mo",
    "سرعة": "vitesse débit",
    "سرعة الانترنت": "vitesse internet débit",
    "الادسل": "ADSL",
    "الفي دي اس ال": "VDSL",
    "جي بون": "GPON fibre",

    # === Données/DATA ===
    "بيانات": "données data",
    "داتا": "data",
    "فورفي": "forfait",
    "دقائق": "minutes",
    "المكالمات": "appels",
    "مكالمة": "appel",
    "صوت": "voix",
    "رسالة نصية": "SMS",
    "رسالة": "SMS message",
    "رسائل": "SMS messages",

    # === Roaming ===
    "تجوال": "roaming",
    "التجوال": "roaming",
    "رومينج": "roaming",
    "الرومينج": "roaming",
    "رومينغ": "roaming",
    "الرومينغ": "roaming",
    "دولي": "international",
    "الدولي": "international",
    "المغرب العربي": "Maghreb",
    "أوروبا": "Europe",
    "فرنسا": "France",
    "المملكة العربية السعودية": "Arabie Saoudite",
    "السعودية": "Arabie Saoudite",
    "الجزائر": "Algérie",
    "مصر": "Egypte",

    # === Hajj/Omra ===
    "حج": "Hajj",
    "الحج": "Hajj",
    "عمرة": "Omra",
    "العمرة": "Omra",
    "باس رومينج": "pass roaming",
    "باس بيانات": "pass data",

    # === Codes USSD ===
    "كود": "code",
    "رمز": "code",
    "قائمة": "menu",
    "قائمة الخدمات": "menu services",

    # === Paiement/Recharge ===
    "دفع": "paiement",
    "دفع إلكتروني": "paiement électronique",
    "بطاقة شحن": "carte recharge",
    "قسيمة": "voucher",
    "إيفوشر": "E-Voucher",
    "سبا": "SABBA internet",
    "تاكتيك": "taktik",

    # === Appareils ===
    "هاتف": "téléphone mobile",
    "هاتف ذكي": "smartphone",
    "جهاز": "appareil terminal",
    "جهاز المستخدم": "CPE terminal",
    "سيم": "SIM",
    "بطاقة سيم": "carte SIM",
    "إي سيم": "eSIM",
    "مودم": "modem",
    "روتر": "routeur",

    # === Offres fixes ===
    "خط ثابت": "ligne fixe",
    "الهاتف الثابت": "téléphone fixe",
    "الخط الثابت": "ligne fixe",
    "ارضي": "fixe",
    "فاتورة": "facture",
    "فاتورتي": "ma facture",
    "الفاتورة": "la facture",

    # === Services ===
    "تطبيق": "application",
    "تطبيق ماي تي تي": "application My TT",
    "موزيكتي": "Mouzikti",
    "الموزيكتي": "Mouzikti",
    "برنامج الولاء": "programme fidélité",
    "برنامج الوفاء": "programme fidélité",
    "كلمة": "KELMA",
    "نقاط": "points",

    # === Adjectifs/adverbes manquants ===
    "منزلي": "fixe résidentiel domicile",
    "منزلية": "fixe résidentielle",
    "شهرية": "mensuelle",
    "يكلف": "coûte",
    "تكلف": "coûte",
    "يكلّف": "coûte",
    "يكلفني": "me coûte",
    "يستغرق": "prend dure",
    "يمكن": "possible peut",
    "تتوفر": "disponible",
    "يتوفر": "disponible",
    "أريد": "je veux",
    "نحب": "je veux",
    "باغي": "je veux",
    "عايز": "je veux",
    "محتاج": "j'ai besoin",

    # === Client ===
    "عميل": "client",
    "العميل": "client",
    "مستخدم": "utilisateur",
    "مشترك": "abonné",
    "المشترك": "l'abonné",
    "شركة": "entreprise",
    "الشركة": "l'entreprise",

    # === TT en arabe ===
    "تونس للاتصالات": "Tunisie Telecom",
    "تونس تيليكوم": "Tunisie Telecom",
    "تونس تليكوم": "Tunisie Telecom",
    "المتعامل": "l'opérateur",
    "الشبكة": "réseau",
    "شبكة تونس": "réseau Tunisie Telecom",

    # === Temporal ===
    "يوم": "jour",
    "أسبوع": "semaine",
    "شهر": "mois",
    "شهري": "mensuel",
    "أسبوعي": "hebdomadaire",
    "يومي": "journalier",
    "سنة": "an année",
    "سنوي": "annuel",

    # === Qualificatifs ===
    "مجاني": "gratuit",
    "مجانية": "gratuite",
    "مجانا": "gratuitement",
    "إضافي": "supplémentaire",
    "لا محدود": "illimité",
    "غير محدود": "illimité",
    "محدود": "limité",
    "تنافسي": "compétitif",
    "حصري": "exclusif",
    "مميز": "premium",
    "جديد": "nouveau",
    "جديدة": "nouvelle",

    # === Unités monétaires ===
    "دينار": "dinar",
    "دت": "DT",
    "مليم": "millime",
    "مليمات": "millimes",

    # === Verbes courants ===
    "اشتري": "acheter souscrire",
    "أشتري": "acheter souscrire",
    "شراء": "achat souscription",
    "فعّل": "activer",
    "فعّلت": "activé",
    "أريد": "je veux",
    "يريد": "il veut",
    "نريد": "nous voulons",
    "احتاج": "j'ai besoin",
    "أحتاج": "j'ai besoin",
    "يحتاج": "besoin",
    "أبحث": "je cherche",
    "يبحث": "cherche",
    "أريد أن أعرف": "je veux savoir",
    "أريد أن أفعل": "je veux activer",
    "كيفاش": "comment",
    "برك": "seulement",
    "بحال": "comme",

    # === Questions ===
    "كيف": "comment",
    "ما هو": "qu'est-ce que quel est",
    "ما هي": "quelles sont",
    "أين": "où",
    "متى": "quand",
    "لماذا": "pourquoi",
    "كم": "combien",
    "هل": "est-ce que",
    "هل يوجد": "est-ce qu'il y a",
    "هل يمكن": "est-il possible",
    "من": "qui",
    "ما": "quoi quel",

    # === Particules ===
    "نعم": "oui",
    "لا": "non",
    "مع": "avec",
    "بدون": "sans",
    "على": "sur pour",
    "في": "dans en",
    "إلى": "vers à",
    "من": "de depuis",
    "و": "et",
    "أو": "ou",
    "لكن": "mais",
    "إذا": "si",
    "فقط": "seulement",
    "أيضا": "aussi",
    "أيضاً": "aussi",

    # === SOS ===
    "سوس سولد": "SOS solde",
    "سوس انترنت": "SOS internet",
    "سوس": "SOS",
    "بلا رصيد": "sans crédit solde vide",
    "رصيد فارغ": "solde vide",
    "رصيدي خلص": "mon solde est épuisé",
}

def translate_arabic_to_french(text_ar: str) -> str:
    """
    Traduit le texte arabe en français.
    Étapes: phrases entières → article défini → dictionnaire mot-à-mot → préfixes verbaux.
    """
    text_fr = text_ar

    # Étape 0: Traduction de phrases entières (prioritaire)
    text_lower = text_fr.strip()
    for ar_phrase, fr_phrase in AR_PHRASES_FR:
        # Correspondance souple (ignorer ponctuation finale)
        clean_input = re.sub(r'[؟?!،,\.]+$', '', text_lower).strip()
        clean_phrase = re.sub(r'[؟?!،,\.]+$', '', ar_phrase).strip()
        if clean_input == clean_phrase:
            logger.info(f"[NLP] Phrase exacte: '{ar_phrase}' → '{fr_phrase}'")
            return fr_phrase

    # Étape 1: Enlever l'article défini arabe (ال)
    text_fr = re.sub(r'(?<!\w)ال', '', text_fr, flags=re.UNICODE)

    # Étape 2: Traductions directes (du plus long au plus court)
    sorted_keys = sorted(AR_FR_DICT.keys(), key=len, reverse=True)
    for ar_term in sorted_keys:
        fr_term = AR_FR_DICT[ar_term]
        # Word boundary adapté à l'arabe (espaces ou début/fin)
        pattern = r'(?<!\w)' + re.escape(ar_term) + r'(?!\w)'
        text_fr = re.sub(pattern, f' {fr_term} ', text_fr, flags=re.IGNORECASE | re.UNICODE)

    # Étape 3: Nettoyer espaces multiples
    text_fr = re.sub(r'\s+', ' ', text_fr).strip()

    return text_fr


# ============================================================
# NORMALISATION DU TEXTE
# ============================================================

def normalize_arabic(text_ar: str) -> str:
    """
    Normalise le texte arabe :
    - Supprime diacritiques (harakats)
    - Normalise les variantes de l'alif
    - Supprime tatweel (ـ)
    """
    # Supprimer diacritiques
    diacritics_range = (0x064B, 0x065F)
    diacritics_extra = ['\u0610', '\u0611', '\u0612', '\u0613', '\u0614',
                        '\u0615', '\u0616', '\u0617', '\u0618', '\u0619',
                        '\u061A', '\u06D4', '\u06D6', '\u06D7', '\u06D8',
                        '\u06D9', '\u06DA', '\u06DB', '\u06DC', '\u06DF',
                        '\u06E0', '\u06E1', '\u06E2', '\u06E3', '\u06E4',
                        '\u06E7', '\u06E8', '\u06EA', '\u06EB', '\u06EC',
                        '\u06ED', '\u0670']
    result = []
    for c in text_ar:
        code = ord(c)
        if diacritics_range[0] <= code <= diacritics_range[1]:
            continue
        if c in diacritics_extra:
            continue
        result.append(c)
    text_ar = ''.join(result)

    # Supprimer tatweel (extension de lettre)
    text_ar = text_ar.replace('\u0640', '')

    # Normaliser variantes de l'alif
    text_ar = text_ar.replace('أ', 'ا')
    text_ar = text_ar.replace('إ', 'ا')
    text_ar = text_ar.replace('آ', 'ا')
    text_ar = text_ar.replace('ٱ', 'ا')

    # Normaliser teh marbuta (ة → ه) — optionnel, peut casser certains mots
    # text_ar = text_ar.replace('ة', 'ه')

    return text_ar


def normalize_french(text_fr: str) -> str:
    """Normalise le français"""
    text_fr = text_fr.lower()
    text_fr = re.sub(r'\s+', ' ', text_fr)
    return text_fr.strip()


# ============================================================
# CORRECTION ARABE DIALECTAL → STANDARD
# ============================================================

DIALECTAL_TO_STANDARD = {
    # Tunisien → Standard
    "وش": "ما هو",
    "أش": "ما هو",
    "أشنو": "ما هو",
    "آش": "ما هو",
    "فاش": "كيف",
    "وقتاش": "متى",
    "كيفاش": "كيف",
    "علاش": "لماذا",
    "قعدت": "جلست",
    "خذ": "أخذ",
    "جاب": "أحضر",
    "نقول": "أقول",
    "بحال": "مثل",
    "برك": "فقط",
    "عندي": "لدي",
    "عندك": "لديك",
    "هيا": "هايا",     # HAYYA dialectal → standard TT
    "شنو": "ما هو",
    "بش": "لكي",
    "ماهو": "ما هو",
    "ماهي": "ما هي",
    "نحب": "أريد",
    "نبغي": "أريد",
    "باغي": "أريد",
    "عايز": "أريد",
}

def normalize_dialectal_arabic(text_ar: str) -> str:
    """Normalise l'arabe dialectal tunisien en arabe standard"""
    text = text_ar
    for dialectal, standard in DIALECTAL_TO_STANDARD.items():
        pattern = r'(?<!\w)' + re.escape(dialectal) + r'(?!\w)'
        text = re.sub(pattern, standard, text, flags=re.UNICODE | re.IGNORECASE)
    return text


# ============================================================
# PREPROCESSING MULTILINGUE PRINCIPAL
# ============================================================

def preprocess_query(query: str) -> Tuple[str, Language]:
    """
    Prétraite une requête utilisateur.

    Processus:
    1. Détecte la langue
    2. Si arabe: dialectal → standard → normalise → traduit en FR
    3. Si mixte: traduit les parties arabes
    4. Normalise le tout
    5. Retourne (query_fr, langue_originale)
    """
    lang = detect_language(query)

    if lang == Language.ARABIC:
        # Dialectal → Standard → Normalise → Traduit
        normalized = normalize_dialectal_arabic(query)
        normalized = normalize_arabic(normalized)
        translated = translate_arabic_to_french(normalized)
        final_query = normalize_french(translated)
        logger.info(f"[NLP] AR→FR: '{query[:60]}' → '{final_query[:60]}'")
        return final_query, Language.ARABIC

    elif lang == Language.MIXED:
        # Traduit les parties arabes
        normalized = normalize_dialectal_arabic(query)
        normalized = normalize_arabic(normalized)
        translated = translate_arabic_to_french(normalized)
        final_query = normalize_french(translated)
        logger.info(f"[NLP] MIXED→FR: '{query[:60]}' → '{final_query[:60]}'")
        return final_query, Language.MIXED

    elif lang == Language.FRENCH:
        final_query = normalize_french(query)
        return final_query, Language.FRENCH

    else:
        # Unknown — garder tel quel (peut être anglais ou code USSD)
        return normalize_french(query), Language.UNKNOWN


# ============================================================
# DÉTECTION LANGUE DE RÉPONSE
# ============================================================

def detect_response_language(query_lang: Language) -> str:
    """
    Détermine la langue de réponse.
    Retourne "ar" si la requête était en arabe, "fr" sinon.
    """
    if query_lang == Language.ARABIC:
        return "ar"
    return "fr"


# ============================================================
# DICTIONNAIRE INVERSE FR → AR
# ============================================================

FR_AR_DICT = {}
for ar_k, fr_v in AR_FR_DICT.items():
    # Prendre seulement le premier mot FR comme clé inverse
    first_word = fr_v.split()[0]
    if first_word not in FR_AR_DICT:
        FR_AR_DICT[first_word] = ar_k

def translate_french_to_arabic(text_fr: str) -> str:
    """Traduit le texte français en arabe (dictionnaire inverse)"""
    text_ar = text_fr
    sorted_keys = sorted(FR_AR_DICT.keys(), key=len, reverse=True)
    for fr_term in sorted_keys:
        ar_term = FR_AR_DICT[fr_term]
        pattern = r'\b' + re.escape(fr_term) + r'\b'
        text_ar = re.sub(pattern, ar_term, text_ar, flags=re.IGNORECASE | re.UNICODE)
    return text_ar


# ============================================================
# UTILITAIRES
# ============================================================

def is_query_numeric(query: str) -> bool:
    """Détecte si la requête est principalement numérique"""
    digits = sum(1 for c in query if c.isdigit())
    total = len([c for c in query if c.isalnum()])
    return total > 0 and digits / total > 0.5

def extract_keywords(query: str, lang: Language) -> List[str]:
    """Extrait les mots-clés d'une requête"""
    text = re.sub(r'[^\w\s]', ' ', query)

    stopwords_fr = {'le', 'la', 'les', 'de', 'des', 'un', 'une', 'et', 'ou', 'à', 'est', 'je', 'tu', 'il'}
    stopwords_ar = {'في', 'من', 'إلى', 'هو', 'هي', 'و', 'أو', 'هل', 'ما', 'على'}

    stopwords = stopwords_ar if lang == Language.ARABIC else stopwords_fr

    words = text.split()
    keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
    return keywords


# ============================================================
# TESTS INTÉGRÉS
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("=== TEST DÉTECTION DE LANGUE ===")
    tests_lang = [
        ("C'est quoi l'offre Hayya ?", Language.FRENCH),
        ("ما هو عرض هايا ؟", Language.ARABIC),
        ("Je veux عروض HAYYA", Language.MIXED),
        ("*140#", Language.UNKNOWN),
        ("أريد باقة انترنت", Language.ARABIC),
        ("كيفاش نشري باقة", Language.ARABIC),
    ]
    all_ok = True
    for text, expected in tests_lang:
        result = detect_language(text)
        ok = result == expected
        status = "✅" if ok else "❌"
        print(f"  {status} '{text[:40]}' → {result.name} (attendu: {expected.name})")
        if not ok:
            all_ok = False

    print("\n=== TEST TRADUCTION AR→FR ===")
    tests_trad = [
        ("ما هي عروض الأنترنت المنزلي لتونس تيليكوم", "internet fixe"),
        ("كم يكلف باقة الأنترنت الشهرية في تونس تيليكوم", "forfait"),
        ("ما هو عرض هايا من تونس تيليكوم", "hayya"),
        ("هل يوجد رومينج دولي", "roaming"),
        ("كيف أشحن رصيدي", "recharge"),
        ("أريد باقة انترنت شهرية", "forfait"),
        ("كم سعر الفايبر", "fibre"),
        ("كيف أشتري باقة هايا", "hayya"),
    ]
    for ar, expected_kw in tests_trad:
        translated, lang = preprocess_query(ar)
        ok = expected_kw.lower() in translated.lower()
        status = "✅" if ok else "❌"
        print(f"  {status} AR: '{ar}'")
        print(f"       FR: '{translated}'")
        if not ok:
            print(f"       ⚠️  Mot-clé attendu '{expected_kw}' absent")
            all_ok = False
        print()

    print("=" * 60)
    if all_ok:
        print("✅ Tous les tests passés")
    else:
        print("❌ Certains tests ont échoué")
        sys.exit(1)