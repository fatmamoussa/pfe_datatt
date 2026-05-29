"""
=============================================================
Patch_arabic_translation.py — Traduction FR↔AR
Chatbot Tunisie Telecom
=============================================================

CORRECTIONS v3 :
  [1] Nom du fichier harmonisé : Patch_arabic_translation.py
      (P majuscule = cohérent avec les imports dans les APIs)
  [2] Import deep_translator déplacé en haut (une seule fois)
      au lieu d'être répété à chaque appel de fonction
  [3] Cache FR→AR limité à 500 entrées (évite fuite mémoire)

FONCTIONNEMENT :
  translate_to_french(text)  → AR → FR (pour la recherche RAG)
    1. Google Translate si internet disponible
    2. Dictionnaire local (toujours disponible, 0ms latence)

  translate_to_arabic(text)  → FR → AR (pour afficher la réponse)
    1. Google Translate si internet disponible
    2. Fallback : réponse française + note arabe
=============================================================
"""

import re
import threading
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ─── FIX [2] : import deep_translator une seule fois au démarrage ─
_GOOGLE_TRANSLATE_AVAILABLE = False
try:
    from deep_translator import GoogleTranslator
    _GOOGLE_TRANSLATE_AVAILABLE = True
    logger.info("[ARABIC] deep_translator disponible — Google Translate actif")
except ImportError:
    logger.info("[ARABIC] deep_translator absent — mode dictionnaire local uniquement")
    logger.info("[ARABIC] Pour activer Google Translate : pip install deep-translator")


# ─── FIX [3] : cache limité à 500 entrées (LRU simple) ───────────
_MAX_CACHE_SIZE   = 500
_translation_cache = OrderedDict()
_cache_lock       = threading.Lock()

def _cache_get(key: str):
    with _cache_lock:
        if key in _translation_cache:
            _translation_cache.move_to_end(key)  # LRU : remonter l'entrée
            return _translation_cache[key]
    return None

def _cache_set(key: str, value: str):
    with _cache_lock:
        if key in _translation_cache:
            _translation_cache.move_to_end(key)
        else:
            if len(_translation_cache) >= _MAX_CACHE_SIZE:
                _translation_cache.popitem(last=False)  # Supprimer le plus ancien
        _translation_cache[key] = value


# ─── DICTIONNAIRE AR → FR ─────────────────────────────────────────
# Termes télécom tunisiens les plus fréquents.
# Suffisant pour trouver les bons chunks ChromaDB sans internet.
AR_TO_FR_DICT = {
    # Offres et services
    "هايا":              "hayya",
    "هايا رمضان":        "hayya ramadan",
    "عرض":               "offre",
    "عروض":              "offres",
    "باقة":              "forfait",
    "باقات":             "forfaits",
    "اشتراك":            "abonnement",
    # Internet
    "انترنت":            "internet",
    "إنترنت":            "internet",
    "نت بوكس":           "netbox",
    "ادسل":              "adsl",
    "ألياف":             "fibre",
    "واي فاي":           "wifi",
    "ويفي":              "wifi",
    "الجيل الرابع":      "4g",
    "الجيل الخامس":      "5g",
    "الجيل الثالث":      "3g",
    # Mobile
    "هاتف":              "mobile téléphone",
    "شبكة":              "réseau",
    "بيانات":            "data internet",
    "سيم":               "sim",
    "خط":                "ligne",
    "رقم":               "numéro",
    "بطاقة سيم":         "carte sim",
    # Recharge et solde
    "شحن":               "recharge",
    "رصيد":              "solde crédit",
    "صوص سولد":          "sos solde",
    "sos رصيد":          "sos solde",
    # Roaming
    "تجوال":             "roaming international",
    "التجوال":           "roaming",
    # Services et tarifs
    "خدمة":              "service",
    "خدمة العملاء":      "service client",
    "تعريفة":            "tarif prix",
    "فاتورة":            "facture",
    "تفعيل":             "activer activation",
    "تنشيط":             "activer",
    "إلغاء":             "désactiver",
    "تعطيل":             "désactiver",
    "مكالمة":            "appel",
    "مكالمات":           "appels",
    "رسالة":             "sms message",
    "رسائل":             "sms messages",
    # USSD
    "رمز":               "code ussd",
    "كود":               "code",
    # Corporate / IoT
    "شركات":             "entreprises corporate",
    "لوراوان":           "lorawan iot",
    "تتبع":              "tracking gps",
    # Autres offres TT
    "وافي":              "waffi",
    "صابا":              "sabba",
    "ديوان":             "diwan sport",
    "إليسا":             "elissa",
    # Questions fréquentes
    "كيف":               "comment",
    "ما هو":             "qu'est-ce que",
    "ما هي":             "quelles sont",
    "أين":               "où",
    "متى":               "quand",
    "كم":                "combien",
    "هل":                "est-ce que",
}


# ═══════════════ TRADUCTION LOCALE AR→FR ═══════════════

def translate_ar_to_fr_local(text: str) -> str:
    """
    Traduction locale AR→FR par dictionnaire télécom.
    100% offline, instantané, 0 dépendance réseau.
    Suffisant pour trouver les bons chunks dans ChromaDB.
    """
    result = text

    # Remplacer du plus long au plus court (évite remplacements partiels)
    for ar, fr in sorted(AR_TO_FR_DICT.items(), key=lambda x: -len(x[0])):
        result = result.replace(ar, fr)

    # Supprimer les caractères arabes résiduels
    result = re.sub(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF]+', ' ', result)
    result = re.sub(r'\s+', ' ', result).strip()

    # Si résultat trop court → garder l'original
    if len(result.strip()) < 3:
        return text

    return result


# ═══════════════ TRANSLATE TO FRENCH ═══════════════

def translate_to_french(text: str) -> str:
    """
    Traduit une question arabe en français pour la recherche RAG.
    1. Google Translate si disponible et internet actif
    2. Fallback : dictionnaire local (toujours disponible)
    """
    if not text:
        return text

    # FIX [2] : utiliser la variable globale, pas un import répété
    if _GOOGLE_TRANSLATE_AVAILABLE:
        try:
            translated = GoogleTranslator(source='ar', target='fr').translate(text)
            if translated and len(translated.strip()) > 3:
                logger.info("[ARABIC] Google Translate AR→FR : '%s' → '%s'",
                            text[:30], translated[:30])
                return translated
        except Exception as e:
            logger.warning("[ARABIC] Google Translate échoué (%s) — fallback dictionnaire", e)

    # Dictionnaire local
    result = translate_ar_to_fr_local(text)
    logger.info("[ARABIC] Dictionnaire local AR→FR : '%s' → '%s'",
                text[:30], result[:30])
    return result


# ═══════════════ TRANSLATE TO ARABIC ═══════════════

def translate_to_arabic(text: str) -> str:
    """
    Traduit une réponse française en arabe pour l'affichage.
    1. Vérifier le cache (FIX [3] : limité à 500 entrées LRU)
    2. Google Translate si disponible
    3. Fallback : réponse française + note arabe
    """
    if not text or len(text.strip()) < 5:
        return text

    # FIX [3] : cache LRU limité
    cached = _cache_get(text)
    if cached:
        return cached

    # FIX [2] : utiliser la variable globale
    if _GOOGLE_TRANSLATE_AVAILABLE:
        try:
            translated = GoogleTranslator(source='fr', target='ar').translate(text)
            if translated and len(translated.strip()) > 5:
                _cache_set(text, translated)
                logger.info("[ARABIC] Google Translate FR→AR OK (%d chars)", len(translated))
                return translated
        except Exception as e:
            logger.warning("[ARABIC] Google Translate FR→AR échoué (%s) — fallback", e)

    # Fallback : français + note arabe
    note   = "المعلومات متوفرة باللغة الفرنسية:\n"
    result = note + text
    _cache_set(text, result)
    return result


# ═══════════════ TEST ═══════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  TEST Patch_arabic_translation.py v3")
    print(f"  Google Translate : {'disponible' if _GOOGLE_TRANSLATE_AVAILABLE else 'absent (dictionnaire local)'}")
    print("=" * 60)

    tests_ar_fr = [
        "ما هو عرض هايا ؟",
        "كيف أشحن رصيدي ؟",
        "ما هي خدمة التجوال ؟",
        "كيف أفعّل باقة الانترنت ؟",
        "ما هي عروض نت بوكس ؟",
        "ما هو رمز تفعيل الخدمة ؟",
        "ما هي عروض الشركات ؟",
        "كيف أوقف خدمة غير مرغوبة ؟",
    ]

    print("\n── AR → FR (pour recherche RAG) ──")
    for q in tests_ar_fr:
        result = translate_to_french(q)
        print(f"  AR : {q}")
        print(f"  FR : {result}")
        print()

    print("\n── FR → AR (pour affichage réponse) ──")
    fr_tests = [
        "Offre prépayé HAYYA destinée aux clients TT.",
        "Chaque recharge a une validité de 30 jours.",
        "Contactez le service client au 1298.",
    ]
    for fr in fr_tests:
        result = translate_to_arabic(fr)
        print(f"  FR : {fr}")
        print(f"  AR : {result[:100]}")
        print()

    print(f"Cache actuel : {len(_translation_cache)} / {_MAX_CACHE_SIZE} entrées")
    print("✅ Test terminé")