"""
╔══════════════════════════════════════════════════════════════════════╗
║   TEST FINAL DE ROBUSTESSE — Chatbot Tunisie Telecom  (PFE)          ║
║   ─────────────────────────────────────────────────────────────────  ║
║   • 21 sujets × 6 variantes = 126 requêtes au total                  ║
║   • Questions tirées des VRAIS chunks (prix, codes, tarifs exacts)   ║
║   • Reformulations naturelles en français (style conversation réelle)║
║   • Questions en arabe + fautes de frappe + argot SMS                ║
║   • Tests hors-sujet, concurrents, multi-critères                    ║
║   • Rapport HTML interactif généré automatiquement                   ║
╚══════════════════════════════════════════════════════════════════════╝

UTILISATION :
    pip install requests
    python test_chatbot_tt_final.py

    # Avec un compte spécifique :
    python test_chatbot_tt_final.py --email admin@tt.tn --password monpass

    # Pour tester seulement certaines catégories :
    python test_chatbot_tt_final.py --cat hajj waffi hayya

RÉSULTAT : génère rapport_test_final_chatbot.html dans le dossier courant
"""

import requests
import json
import time
import argparse
import sys
import os
import uuid
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_BASE    = "http://localhost:8002"
CHAT_ROUTE  = "/chat"
LOGIN_ROUTE = "/auth/login"
DELAY_MS    = 600
TIMEOUT     = 60

CONFIDENCE_OK      = 0.40
MIN_ANSWER_LEN     = 20
HALLUCINATION_SIGS = ["ooredoo", "orange tunisie", "tunisiana", "vodafone"]

# ─────────────────────────────────────────────
# BANQUE DE QUESTIONS — 21 sujets × 6 variantes
# Toutes les questions sont ancrées sur des données
# réelles des chunks (tarifs, codes, noms exacts).
# ─────────────────────────────────────────────
TEST_BANK = [

  # ══════════════════════════════════════════════
  # 1. SALUTATIONS
  # ══════════════════════════════════════════════
  {
    "subject": "Salutation",
    "category": "greeting",
    "expected_mode": "greeting",
    "expected_keywords": ["bonjour", "assistant", "tunisie telecom", "services"],
    "variants": [
      "Bonjour",
      "Salam, comment ça va ?",
      "Bonsoir !",
      "Hello, t'es là ?",
      "مرحبا",
      "أهلا وسهلا",
    ]
  },

  # ══════════════════════════════════════════════
  # 2. OFFRE HAYYA
  # Source chunk : FCOffreHayya.pdf
  # Tarif réel : 50mil/min + 25mil/SMS, bonus 50%
  # sur forfait internet via *140#
  # ══════════════════════════════════════════════
  {
    "subject": "Offre Hayya — tarif & caractéristiques",
    "category": "hayya",
    "expected_mode": "rag",
    "expected_keywords": ["hayya", "millimes", "forfait", "internet"],
    "variants": [
      # Style conversation réelle (comme dans tes logs)
      "Haya ofre TT, c koi exatement ?",
      "C'est quoi l'offre Hayya de Tunisie Telecom ?",
      "Quels sont les avantages et le prix de l'offre Hayya ?",
      # Ancré sur tarif précis du chunk : 50 millimes/min
      "Quel est le tarif à la minute avec l'offre Hayya ?",
      # Ancré sur bonus 50% via *140#
      "Est-ce qu'il y a un bonus internet quand j'active un forfait avec *140# sur Hayya ?",
      # Arabe
      "ما هو عرض هيا من تونس تيليكوم ؟",
    ]
  },

  # ══════════════════════════════════════════════
  # 3. OFFRE WAFFI — INTERNET FIXE
  # Source chunk : FC-WAFFI.pdf + Argumentaire-de-vente-WAFFI.pdf
  # Tarif réel : 39,9 DT/mois (10M), jusqu'à 100M
  # Forfait voix : 30h/mois vers fixe national
  # Frais raccordement : 20 DT, modem gratuit
  # ══════════════════════════════════════════════
  {
    "subject": "Offre Waffi internet fixe",
    "category": "waffi",
    "expected_mode": "rag",
    "expected_keywords": ["waffi", "internet", "dt"],
    "variants": [
      # Style conversation réelle (comme dans tes logs)
      "cest quoi offre waffi",
      "je veux que tu définisses l'offre Waffi svp",
      # Ancré sur tarif précis chunk : 39,9 DT
      "Quel est le prix mensuel de l'offre Waffi chez Tunisie Telecom ?",
      # Ancré sur débit chunk : 10M à 100M
      "Quels débits internet sont disponibles avec Waffi ?",
      # Ancré sur forfait voix chunk : 30h/mois
      "Est-ce que l'offre Waffi inclut des appels téléphoniques fixes ?",
      # Arabe
      "ما هي عروض الأنترنت المنزلي لتونس تيليكوم ؟",
    ]
  },

  # ══════════════════════════════════════════════
  # 4. PROMOTION HAJJ
  # Source chunk : FC-Promo-Hajj-2019-VF-002.pdf + Flash info 2024
  # Données réelles :
  #   - 50% remise roaming Arabie Saoudite
  #   - 40% remise appels depuis Tunisie vers AS
  #   - Activation prépayé : *142# (gratuit)
  #   - Pass Hajj 2024 : 4 DT (250Mo/2j), 9 DT (600Mo/7j), 24 DT (2Go/15j)
  #   - Service Tfadhel : *114*numéro#
  # ══════════════════════════════════════════════
  {
    "subject": "Promotion Hajj Tunisie Telecom",
    "category": "hajj",
    "expected_mode": "rag",
    "expected_keywords": ["hajj", "arabie saoudite", "roaming", "remise"],
    "variants": [
      # Question exacte de ton log (réponse confirmée)
      "Quels sont le prix et les avantages de l'offre du Hajj proposée par Tunisie Telecom ?",
      # Ancré sur 50% remise roaming
      "Quel est le pourcentage de réduction sur le roaming en Arabie Saoudite pendant le Hajj ?",
      # Ancré sur 40% depuis fixe/mobile Tunisie
      "Est-ce qu'il y a une réduction sur les appels depuis la Tunisie vers l'Arabie Saoudite pour le Hajj ?",
      # Ancré sur Pass Data 2024 : 4 DT / 9 DT / 24 DT
      "Combien coûtent les pass internet Hajj de Tunisie Telecom en Arabie Saoudite ?",
      # Ancré sur code *142# prépayés
      "Comment activer la promo Hajj sur mon mobile prépayé TT, c'est quel code ?",
      # Ancré sur Tfadhel : appel sans crédit
      "Je suis en Arabie Saoudite pour le Hajj et je n'ai plus de crédit, est-ce que je peux quand même appeler la Tunisie avec TT ?",
    ]
  },

  # ══════════════════════════════════════════════
  # 5. FORFAITS INTERNET MOBILE / *140#
  # Source chunk : FCOffreHayya.pdf + packs data
  # Code réel : *140# pour souscrire, *122# pour suivi
  # ══════════════════════════════════════════════
  {
    "subject": "Forfaits Internet Mobile & code *140#",
    "category": "internet_mobile",
    "expected_mode": "rag",
    "expected_keywords": ["forfait", "internet", "dt", "go"],
    "variants": [
      "Quels sont les forfaits internet mobile Tunisie Telecom ?",
      "Je veux activer internet sur mon portable TT, comment je fais ?",
      # Ancré sur code *140# du chunk
      "C'est quel code pour souscrire un forfait internet sur mon mobile TT ?",
      "Prix des packs data 4G chez TT",
      "كم يكلف باقة الأنترنت الشهرية في تونس تيليكوم ؟",
      "forfet internett mobile tt prix",
    ]
  },

  # ══════════════════════════════════════════════
  # 6. RECHARGE / CONSULTATION SOLDE
  # ══════════════════════════════════════════════
  {
    "subject": "Recharge et consultation solde",
    "category": "recharge",
    "expected_mode": "rag",
    "expected_keywords": ["recharge", "solde", "crédit"],
    "variants": [
      "Comment recharger mon crédit Tunisie Telecom ?",
      "Je veux consulter mon solde, quel est le code ?",
      "Recharge en ligne TT, c'est possible ? comment faire ?",
      "recharg solde comment vérifier crédit",
      "كيف أشحن رصيدي في تونس تيليكوم ؟",
      "Comment savoir combien de crédit il me reste sur mon mobile TT ?",
    ]
  },

  # ══════════════════════════════════════════════
  # 7. SOS INTERNET
  # Source chunk : SOS-Internet_Flash-Commercial_15-01-2020.pdf
  # Code réel : *150# ou SMS "D" au 85159
  # Suivi : *122*47#
  # Avance dès 50 Mo jusqu'à 55 Go
  # ══════════════════════════════════════════════
  {
    "subject": "SOS Internet / SOS Solde",
    "category": "sos",
    "expected_mode": "rag",
    "expected_keywords": ["sos", "internet", "avance"],
    "variants": [
      "Comment faire un SOS internet sur TT ?",
      "Mon forfait internet est épuisé, comment obtenir une avance de data ?",
      # Ancré sur code *150# du chunk
      "C'est quel code pour demander un SOS internet sur mon mobile TT ?",
      # Ancré sur remboursement automatique à la recharge
      "Comment est remboursé le SOS internet Tunisie Telecom ?",
      "code sos internnet tt comment activer",
      "كيف أطلب سوس أنترنت في تونس تيليكوم ؟",
    ]
  },

  # ══════════════════════════════════════════════
  # 8. ROAMING INTERNATIONAL — TARIFS
  # ══════════════════════════════════════════════
  {
    "subject": "Roaming international — tarifs & activation",
    "category": "roaming",
    "expected_mode": "rag",
    "expected_keywords": ["roaming", "zone", "international"],
    "variants": [
      "Comment activer le roaming Tunisie Telecom ?",
      "Je pars en France, est-ce que mon téléphone TT fonctionne à l'étranger ?",
      "C'est combien le roaming en Europe avec TT ?",
      "roamig france tunisie telecom prix appel",
      "كيف أفعّل التجوال الدولي في تونس تيليكوم ؟",
      "Tarif appel depuis l'étranger vers la Tunisie avec mon numéro TT",
    ]
  },

  # ══════════════════════════════════════════════
  # 9. PASS ROAMING DATA
  # Source chunk : données comparatives roaming
  # Pass France : 1 Go à 4,5 DT / 2j ; 5 Go à 20 DT / 7j
  # Pass Grand Maghreb : 100 Mo / 3 DT / 48h
  # ══════════════════════════════════════════════
  {
    "subject": "Pass Roaming Data",
    "category": "roaming_pass",
    "expected_mode": "rag",
    "expected_keywords": ["pass", "roaming", "data"],
    "variants": [
      "C'est quoi le Pass Roaming Data Tunisie Telecom ?",
      "Existe-t-il un forfait internet pour utiliser à l'étranger avec TT ?",
      # Ancré sur Pass France du chunk : 1 Go à 4,5 DT / 2 jours
      "Quel est le prix du pass internet pour la France chez TT ?",
      # Ancré sur Pass Grand Maghreb
      "Est-ce qu'il y a un pass internet pour les pays du Maghreb chez TT ?",
      "ما هو باقة التجوال للبيانات من تونس تيليكوم ؟",
      "forfait internet etranger tunisie telecom pas cher",
    ]
  },

  # ══════════════════════════════════════════════
  # 10. NETBOX 4G / OFFRE 5G
  # Source chunk : FlashcommercialNetBox4GDataOnlyfin.pdf
  # Pack Clé 4G prépayé : 40 DT pour 5 Go / 2 mois
  # ══════════════════════════════════════════════
  {
    "subject": "NetBox 4G / Offre internet à domicile sans fil",
    "category": "5g",
    "expected_mode": "rag",
    "expected_keywords": ["dt", "go", "internet"],
    "variants": [
      "Tunisie Telecom a une offre internet à domicile sans fil ?",
      "C'est quoi la NetBox 4G de TT et combien ça coûte ?",
      "Je veux internet à la maison sans ligne fixe, c'est possible avec TT ?",
      # Ancré sur Pack Clé 4G : 40 DT / 5 Go / 2 mois
      "Quel est le prix du pack clé 4G prépayé chez Tunisie Telecom ?",
      "هل تونس تيليكوم تملك عرض أنترنت منزلي بدون خط ثابت ؟",
      "5gtt tunisie telecom ofre prix abonnement",
    ]
  },

  # ══════════════════════════════════════════════
  # 11. OFFRE OHMega
  # ══════════════════════════════════════════════
  {
    "subject": "Offre OHMega postpayée",
    "category": "ohmega",
    "expected_mode": "rag",
    "expected_keywords": ["ohmega", "forfait", "dt"],
    "variants": [
      "C'est quoi l'offre OHMega de Tunisie Telecom ?",
      "Quels sont les avantages des forfaits OHMega TT ?",
      "OHMega postpayé, tarif et volume data inclus ?",
      "ofre ohmega tunisie telecom caracteristiques",
      "ما هو عرض أوميغا من تونس تيليكوم ؟",
      "Comment migrer vers l'offre OHMega chez TT ?",
    ]
  },

  # ══════════════════════════════════════════════
  # 12. FIBRE OPTIQUE
  # ══════════════════════════════════════════════
  {
    "subject": "Fibre optique Tunisie Telecom",
    "category": "fibre",
    "expected_mode": "rag",
    "expected_keywords": ["fibre", "internet", "mbps"],
    "variants": [
      "La fibre optique Tunisie Telecom, c'est disponible où ?",
      "Quelles zones sont couvertes par la fibre TT ?",
      "Prix abonnement fibre optique chez TT",
      "fibre optiqe tunisie telecom disopnible zone",
      "هل الألياف الضوئية متوفرة في منطقتي عند تونس تيليكوم ؟",
      "Débit garanti avec la fibre TT et coût mensuel",
    ]
  },

  # ══════════════════════════════════════════════
  # 13. OFFRE RAPIDO prépayée
  # Source chunk : FCPromoAlgerie.pdf + divers
  # ══════════════════════════════════════════════
  {
    "subject": "Offre Rapido prépayée",
    "category": "rapido",
    "expected_mode": "rag",
    "expected_keywords": ["rapido", "dt"],
    "variants": [
      "C'est quoi l'offre Rapido de Tunisie Telecom ?",
      "Rapido prépayé TT, comment s'abonner ?",
      "Tarif appels et SMS avec la carte Rapido",
      "rapdio tt offre prepaye caracteristique",
      "ما هو عرض رابيدو من تونس تيليكوم للمدفوع المسبق ؟",
      "Quelles sont les recharges disponibles pour Rapido ?",
    ]
  },

  # ══════════════════════════════════════════════
  # 14. APPLICATION MY TT
  # ══════════════════════════════════════════════
  {
    "subject": "Application My TT",
    "category": "mytt",
    "expected_mode": "rag",
    "expected_keywords": ["my tt", "application"],
    "variants": [
      "Comment télécharger l'application My TT ?",
      "À quoi sert l'appli My TT de Tunisie Telecom ?",
      "Je veux gérer mon compte depuis mon téléphone, c'est possible avec TT ?",
      "appli mytt tunisie telecom comment installer",
      "كيف أحمّل تطبيق ماي تي تي ؟",
      "My TT application, quelles fonctionnalités sont disponibles ?",
    ]
  },

  # ══════════════════════════════════════════════
  # 15. CONTACT / SERVICE CLIENT
  # ══════════════════════════════════════════════
  {
    "subject": "Contact service client TT",
    "category": "contact",
    "expected_mode": "rag",
    "expected_keywords": ["contact", "service", "client"],
    "variants": [
      "Comment contacter le service client Tunisie Telecom ?",
      "Quel est le numéro de téléphone du support TT ?",
      "Je veux joindre un conseiller TT, comment faire ?",
      "numéro servic client tt contacter comment",
      "كيف أتصل بخدمة عملاء تونس تيليكوم ؟",
      "Service client TT disponible 24h/24 ?",
    ]
  },

  # ══════════════════════════════════════════════
  # 16. PORTABILITÉ
  # Source chunk : FCBonusMAJWelcomeBonusPortageINV04052018.pdf
  # Bonus de bienvenue pour les portés IN
  # ══════════════════════════════════════════════
  {
    "subject": "Portabilité numéro mobile",
    "category": "portabilite",
    "expected_mode": "rag",
    "expected_keywords": ["portabilité", "numéro"],
    "variants": [
      "Comment garder mon numéro en passant à Tunisie Telecom ?",
      "La portabilité chez TT, ça prend combien de temps ?",
      "Je veux venir chez TT en conservant mon numéro actuel",
      # Ancré sur bonus de bienvenue du chunk
      "Est-ce qu'il y a un bonus si je porte mon numéro vers Tunisie Telecom ?",
      "كيف أحتفظ برقمي عند الانتقال إلى تونس تيليكوم ؟",
      "portabilite nuemro mobile tunisie telecom comment",
    ]
  },

  # ══════════════════════════════════════════════
  # 17. QUESTION COMPLEXE MULTI-CRITÈRES
  # ══════════════════════════════════════════════
  {
    "subject": "Question complexe multi-critères",
    "category": "multi_criteres",
    "expected_mode": "rag",
    "expected_keywords": ["dt", "forfait", "internet"],
    "variants": [
      "Quels sont les avantages, le prix et l'activation de l'offre Hayya ?",
      "Comparez les offres prépayées Tunisie Telecom avec leurs prix respectifs",
      "Hayya vs OHMega, lequel choisir selon mon budget et mes besoins data ?",
      "tout savoir sur les forfaits internet 4G TT prix avantages activation",
      "ما الفرق بين عروض تونس تيليكوم المختلفة وأسعارها ؟",
      "Parlez-moi du prix, des avantages et de la disponibilité des offres internet mobile TT",
    ]
  },

  # ══════════════════════════════════════════════
  # 18. QUESTION FLOUE / AMBIGUË
  # ══════════════════════════════════════════════
  {
    "subject": "Question floue ou incomplète",
    "category": "flou",
    "expected_mode": "rag",
    "expected_keywords": ["tunisie telecom", "offres", "services"],
    "variants": [
      "Vous avez des offres ?",
      "C'est combien ?",
      "J'ai un problème avec mon téléphone TT",
      "Koi de neuf chez TT ?",
      "ما عندكم ؟",
      "TT, vous faites quoi exactement comme services ?",
    ]
  },

  # ══════════════════════════════════════════════
  # 19. HORS SUJET — doit REFUSER poliment
  # ══════════════════════════════════════════════
  {
    "subject": "Hors sujet (doit refuser)",
    "category": "hors_sujet",
    "expected_mode": "hors_sujet",
    "expected_keywords": ["assistant", "tunisie telecom", "services"],
    "must_not_contain": ["ooredoo", "orange", "recette", "météo", "couscous"],
    "variants": [
      "Qui a gagné la Coupe du Monde 2022 ?",
      "Donne-moi une recette de couscous tunisien",
      "Quelle est la météo à Tunis aujourd'hui ?",
      "Comment apprendre Python rapidement ?",
      "Prix voiture Tunisie 2024 occasion",
      "Ooredoo, c'est quoi leurs offres 4G ?",
    ]
  },

  # ══════════════════════════════════════════════
  # 20. ROBUSTESSE ORTHOGRAPHE / ARGOT / SMS
  # Inspiré directement de tes logs de conversation
  # ══════════════════════════════════════════════
  {
    "subject": "Robustesse orthographe / argot / SMS",
    "category": "typo",
    "expected_mode": "rag",
    "expected_keywords": ["tunisie telecom", "offres", "services"],
    "variants": [
      # Style exact de tes logs
      "Haya ofre TT, c koi exatement ?",
      "cest quoi offre waffi",
      "je veux que du definir l offre waffi svp",
      "jai pb avec mon fornfait data tt koi faire",
      "tunisi telecom internt comment activr",
      "je veu m abonné a tt c possible comment",
    ]
  },

  # ══════════════════════════════════════════════
  # 21. CODE USSD — activation & suivi
  # Source chunks : *140# (forfait), *150# (SOS),
  #                 *122# (suivi), *142# (promo Hajj)
  # ══════════════════════════════════════════════
  {
    "subject": "Codes USSD utiles TT",
    "category": "ussd",
    "expected_mode": "rag",
    "expected_keywords": ["*140", "code"],
    "variants": [
      # Ancré sur *140# du chunk Hayya
      "Quel est le code USSD pour activer un forfait internet TT ?",
      # Ancré sur *150# du chunk SOS Internet
      "C'est quel code pour le SOS internet sur TT ?",
      # Ancré sur *122# suivi conso
      "Comment vérifier ma consommation internet sur TT, c'est quel code ?",
      # Ancré sur *142# promo Hajj prépayé
      "Quel code composer pour activer la promo Hajj sur mon prépayé TT ?",
      "code ussd internt tunisie telecom comment faire",
      "ما هو الكود للاشتراك في باقة الأنترنت ؟",
    ]
  },

]

# ─────────────────────────────────────────────
# COULEURS CONSOLE
# ─────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    GREY   = "\033[90m"

def ok(s):      return f"{C.GREEN}✅ {s}{C.RESET}"
def fail(s):    return f"{C.RED}❌ {s}{C.RESET}"
def warn(s):    return f"{C.YELLOW}⚠️  {s}{C.RESET}"
def info(s):    return f"{C.CYAN}{s}{C.RESET}"
def bold(s):    return f"{C.BOLD}{s}{C.RESET}"

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def login(email: str, password: str) -> str:
    url = API_BASE + LOGIN_ROUTE
    try:
        r = requests.post(url, json={"email": email, "password": password}, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        token = (data.get("access_token") or data.get("token") or
                 data.get("data", {}).get("token") or "")
        if not token:
            print(fail(f"Login OK mais pas de token trouvé. Réponse : {data}"))
            sys.exit(1)
        print(ok(f"Connecté en tant que {email}"))
        return token
    except requests.exceptions.ConnectionError:
        print(fail(f"Impossible de joindre l'API sur {API_BASE}"))
        print(f"  → Vérifie que ton serveur tourne : uvicorn api:app --port 8002")
        sys.exit(1)
    except Exception as e:
        print(fail(f"Erreur login : {e}"))
        sys.exit(1)

# ─────────────────────────────────────────────
# ENVOI D'UN MESSAGE
# ─────────────────────────────────────────────
def send_chat(message: str, token: str, session_id: str = None, delay_ms: int = DELAY_MS) -> dict:
    url = API_BASE + CHAT_ROUTE
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "message": message,
        "session_id": session_id or str(uuid.uuid4()),
    }
    try:
        t0 = time.time()
        r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        elapsed = int((time.time() - t0) * 1000)
        r.raise_for_status()
        data = r.json()
        data["_elapsed_ms"] = elapsed
        return data
    except requests.exceptions.Timeout:
        return {"error": "timeout", "answer": "", "_elapsed_ms": TIMEOUT * 1000}
    except Exception as e:
        return {"error": str(e), "answer": "", "_elapsed_ms": 0}

# ─────────────────────────────────────────────
# ÉVALUATION AUTOMATIQUE
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD_WARN = 0.38

def evaluate(response: dict, test_case: dict) -> dict:
    if "error" in response:
        return {"status": "ERREUR", "reasons": [f"Erreur API : {response['error']}"]}

    answer      = (response.get("answer") or "").lower()
    mode        = response.get("mode", "")
    rag_used    = response.get("rag_used", False)
    confidence  = response.get("confidence", 0)
    expected    = test_case["expected_mode"]
    reasons     = []
    status      = "OK"

    # 1. Réponse vide ou trop courte
    if len(answer.strip()) < MIN_ANSWER_LEN:
        reasons.append(f"Réponse trop courte ({len(answer)} chars)")
        status = "ECHEC"

    # 2. Vérification mode attendu
    if expected == "greeting":
        if mode != "greeting":
            reasons.append(f"Mode attendu 'greeting', reçu '{mode}'")
            status = "ECHEC"
    elif expected == "hors_sujet":
        if mode not in ("hors_sujet", "no_context"):
            reasons.append(f"Devait refuser mais mode='{mode}' — possible hallucination")
            status = "ECHEC"
        for bad in test_case.get("must_not_contain", []):
            if bad in answer:
                reasons.append(f"Mentionne '{bad}' alors que c'est hors sujet")
                status = "ECHEC"
    elif expected == "rag":
        if not rag_used and mode not in ("admin_direct", "admin_direct_fuzzy",
                                          "fusion_multi_criteria", "generative_rag",
                                          "extractive", "extractive_rag"):
            reasons.append(f"RAG non utilisé (mode='{mode}', conf={confidence:.2f})")
            status = "PARTIEL" if status != "ECHEC" else "ECHEC"
        if confidence < CONFIDENCE_THRESHOLD_WARN:
            reasons.append(f"Confiance faible ({confidence:.2f})")
            if status == "OK":
                status = "PARTIEL"

    # 3. Mots-clés attendus
    expected_kw = test_case.get("expected_keywords", [])
    missing_kw  = [kw for kw in expected_kw if kw not in answer]
    found_kw    = [kw for kw in expected_kw if kw in answer]

    if missing_kw and expected not in ("hors_sujet", "greeting"):
        pct = len(found_kw) / len(expected_kw) if expected_kw else 1
        if pct < 0.5:
            reasons.append(f"Mots-clés manquants : {missing_kw[:3]}")
            if status == "OK":
                status = "PARTIEL"

    # 4. Hallucinations concurrents
    for sig in HALLUCINATION_SIGS:
        if sig in answer:
            reasons.append(f"Hallucination détectée : mentionne '{sig}'")
            status = "ECHEC"

    if not reasons:
        reasons = ["Réponse cohérente ✓"]

    return {"status": status, "reasons": reasons, "found_kw": found_kw, "missing_kw": missing_kw}

# ─────────────────────────────────────────────
# TEST D'UN SUJET COMPLET
# ─────────────────────────────────────────────
def test_subject(test_case: dict, token: str, delay_ms: int) -> dict:
    subject  = test_case["subject"]
    variants = test_case["variants"]

    print(f"\n{bold('━'*64)}")
    print(f"  {bold(subject)}  [{test_case['category']}]")
    print(f"{'━'*64}")

    results = []
    for i, variant in enumerate(variants):
        time.sleep(delay_ms / 1000)
        sid = str(uuid.uuid4())

        print(f"  [{i+1}/{len(variants)}] {C.GREY}{variant[:75]}{C.RESET}")
        response = send_chat(variant, token, sid, delay_ms)

        if "error" in response:
            print(f"        {fail('ERREUR API')} : {response['error']}")
            results.append({
                "variant": variant, "status": "ERREUR",
                "answer": "", "mode": "error", "confidence": 0,
                "rag_used": False, "elapsed_ms": 0,
                "reasons": [response["error"]], "eval": {"status": "ERREUR", "reasons": [response["error"]]}
            })
            continue

        eval_result = evaluate(response, test_case)
        status      = eval_result["status"]
        mode        = response.get("mode", "?")
        conf        = response.get("confidence", 0)
        answer      = response.get("answer", "")
        elapsed     = response.get("_elapsed_ms", 0)

        color = C.GREEN if status == "OK" else (C.YELLOW if status == "PARTIEL" else C.RED)
        icon  = "✅" if status == "OK" else ("⚠️ " if status == "PARTIEL" else "❌")
        print(f"        {icon} {color}{status}{C.RESET}  mode={mode}  conf={conf:.2f}  {elapsed}ms")
        print(f"        {C.GREY}→ {answer[:110].strip()}...{C.RESET}")
        if eval_result["reasons"] != ["Réponse cohérente ✓"]:
            for r_msg in eval_result["reasons"]:
                print(f"        {C.YELLOW}  ! {r_msg}{C.RESET}")

        results.append({
            "variant":    variant,
            "status":     status,
            "answer":     answer,
            "mode":       mode,
            "confidence": conf,
            "rag_used":   response.get("rag_used", False),
            "elapsed_ms": elapsed,
            "sources":    response.get("sources", []),
            "reasons":    eval_result["reasons"],
            "eval":       eval_result,
        })

    ok_count      = sum(1 for r in results if r["status"] == "OK")
    partial_count = sum(1 for r in results if r["status"] == "PARTIEL")
    fail_count    = sum(1 for r in results if r["status"] in ("ECHEC", "ERREUR"))
    score         = ok_count / len(results) * 100 if results else 0

    print(f"\n  Score : {ok_count}/{len(results)} OK  |  {partial_count} partiel(s)  |  {fail_count} échec(s)  [{score:.0f}%]")

    return {
        "subject":  subject,
        "category": test_case["category"],
        "results":  results,
        "ok":       ok_count,
        "partial":  partial_count,
        "fail":     fail_count,
        "score":    score,
    }

# ─────────────────────────────────────────────
# RAPPORT HTML
# ─────────────────────────────────────────────
def generate_html_report(all_results: list, email: str, duration: float) -> str:
    total_variants = sum(len(s["results"]) for s in all_results)
    total_ok       = sum(s["ok"]      for s in all_results)
    total_partial  = sum(s["partial"] for s in all_results)
    total_fail     = sum(s["fail"]    for s in all_results)
    global_score   = total_ok / total_variants * 100 if total_variants > 0 else 0

    verdict       = ("✅ PRÊT POUR LA SOUTENANCE" if global_score >= 80
                     else "⚠️ AMÉLIORATIONS NÉCESSAIRES" if global_score >= 60
                     else "❌ NON PRÊT — RÉVISION REQUISE")
    verdict_color = ("#2e7d32" if global_score >= 80
                     else "#e65100" if global_score >= 60
                     else "#c62828")

    cat_stats = defaultdict(lambda: {"ok": 0, "partial": 0, "fail": 0, "total": 0})
    for s in all_results:
        cat_stats[s["category"]]["ok"]      += s["ok"]
        cat_stats[s["category"]]["partial"] += s["partial"]
        cat_stats[s["category"]]["fail"]    += s["fail"]
        cat_stats[s["category"]]["total"]   += len(s["results"])

    def status_badge(status):
        colors = {"OK": ("#e8f5e9","#2e7d32"), "PARTIEL": ("#fff3e0","#e65100"),
                  "ECHEC": ("#ffebee","#c62828"), "ERREUR": ("#fce4ec","#880e4f")}
        icons  = {"OK": "✅", "PARTIEL": "⚠️", "ECHEC": "❌", "ERREUR": "💥"}
        bg, fg = colors.get(status, ("#f5f5f5","#333"))
        ic     = icons.get(status, "?")
        return f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;">{ic} {status}</span>'

    def escape(s):
        return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    subjects_html = ""
    for s in all_results:
        sub_score  = s["score"]
        bar_color  = "#4caf50" if sub_score >= 75 else "#ff9800" if sub_score >= 50 else "#f44336"
        subject_row = f"""
        <div class="subject-block">
          <div class="subject-header" onclick="toggleBlock(this)">
            <span class="subject-title">{escape(s['subject'])}</span>
            <span class="cat-tag">{s['category']}</span>
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:{sub_score:.0f}%;background:{bar_color}"></div>
            </div>
            <span class="score-pct" style="color:{bar_color}">{sub_score:.0f}%</span>
            <span class="toggle-btn">▼</span>
          </div>
          <div class="table-wrap">
          <table class="variants-table">
            <thead><tr><th>#</th><th>Question posée au chatbot</th><th>Statut</th><th>Mode</th><th>Conf.</th><th>Latence</th><th>Réponse (extrait)</th></tr></thead>
            <tbody>"""

        for i, r in enumerate(s["results"]):
            reasons_str    = "; ".join(r["reasons"])
            answer_preview = escape(r["answer"][:130]) + ("..." if len(r["answer"]) > 130 else "")
            subject_row += f"""
              <tr class="variant-row">
                <td style="color:#888;font-size:12px;">{i+1}</td>
                <td class="variant-text" dir="auto">"{escape(r['variant'])}"</td>
                <td>{status_badge(r['status'])}<br><small style="color:#888;font-size:10px;">{escape(reasons_str[:70])}</small></td>
                <td><code>{escape(r['mode'])}</code></td>
                <td>{r['confidence']:.2f}</td>
                <td>{r['elapsed_ms']}ms</td>
                <td class="answer-preview">{answer_preview}</td>
              </tr>"""

        subject_row += "</tbody></table></div></div>"
        subjects_html += subject_row

    cats_html = ""
    for cat, st in sorted(cat_stats.items()):
        pct = st["ok"] / st["total"] * 100 if st["total"] > 0 else 0
        col = "#4caf50" if pct >= 75 else "#ff9800" if pct >= 50 else "#f44336"
        cats_html += f"""
          <div class="cat-card">
            <div class="cat-name">{escape(cat)}</div>
            <div class="cat-score" style="color:{col}">{pct:.0f}%</div>
            <div class="cat-bar-bg"><div class="cat-bar-fill" style="width:{pct:.0f}%;background:{col}"></div></div>
            <div class="cat-detail">{st['ok']} ✅ · {st['partial']} ⚠️ · {st['fail']} ❌ / {st['total']}</div>
          </div>"""

    all_latencies = [r["elapsed_ms"] for s in all_results for r in s["results"] if r["elapsed_ms"] > 0]
    avg_latency   = int(sum(all_latencies) / len(all_latencies)) if all_latencies else 0

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rapport Test Final — Chatbot Tunisie Telecom (PFE)</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#f0f2f5; color:#1a1a2e; }}
  .header {{ background:linear-gradient(135deg,#1a1a2e 0%,#0f3460 60%,#16213e 100%); color:white; padding:28px 36px; }}
  .header h1 {{ font-size:24px; margin-bottom:6px; }}
  .header p  {{ font-size:13px; opacity:0.7; }}
  .container {{ max-width:1280px; margin:0 auto; padding:28px 24px; }}
  .verdict-box {{ background:white; border-radius:16px; padding:32px; margin-bottom:28px;
                  text-align:center; border-top:5px solid {verdict_color};
                  box-shadow:0 2px 12px rgba(0,0,0,.07); }}
  .verdict-score {{ font-size:64px; font-weight:900; color:{verdict_color}; line-height:1; }}
  .verdict-text  {{ font-size:20px; font-weight:700; color:{verdict_color}; margin-top:8px; }}
  .stats-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:28px; }}
  .stat-card  {{ background:white; border-radius:12px; padding:20px; text-align:center;
                 box-shadow:0 2px 8px rgba(0,0,0,.05); }}
  .stat-num   {{ font-size:30px; font-weight:800; }}
  .stat-label {{ font-size:12px; color:#888; margin-top:4px; }}
  .section-title {{ font-size:16px; font-weight:700; margin-bottom:16px; color:#333; }}
  .cats-grid  {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(155px,1fr)); gap:12px; margin-bottom:32px; }}
  .cat-card   {{ background:white; border-radius:12px; padding:14px;
                 box-shadow:0 2px 8px rgba(0,0,0,.05); }}
  .cat-name   {{ font-size:11px; color:#666; margin-bottom:4px; font-weight:700;
                 text-transform:uppercase; letter-spacing:0.5px; }}
  .cat-score  {{ font-size:24px; font-weight:800; margin-bottom:6px; }}
  .cat-bar-bg {{ height:4px; background:#eee; border-radius:2px; margin-bottom:6px; overflow:hidden; }}
  .cat-bar-fill {{ height:100%; border-radius:2px; }}
  .cat-detail {{ font-size:11px; color:#888; }}
  .subject-block  {{ background:white; border-radius:12px; margin-bottom:14px; overflow:hidden;
                     box-shadow:0 2px 8px rgba(0,0,0,.05); }}
  .subject-header {{ display:flex; align-items:center; gap:10px; padding:14px 18px;
                     cursor:pointer; background:#fafbff; border-bottom:1px solid #eee;
                     user-select:none; }}
  .subject-header:hover {{ background:#f0f4ff; }}
  .subject-title  {{ font-weight:700; font-size:14px; flex:1; }}
  .cat-tag {{ background:#e8eaf6; color:#3949ab; font-size:11px; font-weight:700;
              padding:3px 9px; border-radius:10px; white-space:nowrap; }}
  .score-bar-wrap {{ width:80px; height:5px; background:#eee; border-radius:3px; overflow:hidden; }}
  .score-bar  {{ height:100%; border-radius:3px; transition:width .4s; }}
  .score-pct  {{ font-size:13px; font-weight:700; min-width:38px; text-align:right; }}
  .toggle-btn {{ font-size:12px; color:#999; cursor:pointer; transition:transform .2s; }}
  .table-wrap {{ overflow-x:auto; }}
  .variants-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .variants-table thead {{ background:#f7f9ff; }}
  .variants-table th {{ padding:9px 11px; text-align:left; font-size:11px;
                        text-transform:uppercase; letter-spacing:0.5px; color:#777;
                        font-weight:700; border-bottom:1px solid #eee; }}
  .variant-row:hover {{ background:#fafafa; }}
  .variant-row td {{ padding:9px 11px; border-bottom:1px solid #f5f5f5; vertical-align:top; }}
  .variant-text   {{ font-style:italic; color:#444; max-width:220px; word-break:break-word; }}
  .answer-preview {{ color:#555; font-size:12px; max-width:300px; line-height:1.5; }}
  code {{ background:#f0f0f0; padding:2px 6px; border-radius:4px; font-size:11px; color:#444; }}
  .table-wrap.collapsed {{ display:none; }}
  .meta-info {{ background:white; border-radius:12px; padding:16px 20px; margin-bottom:28px;
                font-size:13px; color:#555; display:flex; gap:28px; flex-wrap:wrap;
                box-shadow:0 2px 8px rgba(0,0,0,.05); }}
  .meta-item span {{ font-weight:700; color:#333; }}
  .print-note {{ font-size:11px; color:#aaa; text-align:right; margin-top:8px; }}
  @media print {{
    .toggle-btn {{ display:none; }}
    .table-wrap.collapsed {{ display:block !important; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>🤖 Rapport Test Final — Chatbot Tunisie Telecom (PFE)</h1>
  <p>Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")} · Compte : {escape(email)} · Durée : {duration:.1f}s</p>
</div>
<div class="container">

  <div class="meta-info">
    <div class="meta-item">API : <span>{API_BASE}/chat</span></div>
    <div class="meta-item">Sujets testés : <span>{len(all_results)}</span></div>
    <div class="meta-item">Requêtes totales : <span>{total_variants}</span></div>
    <div class="meta-item">Latence moyenne : <span>{avg_latency} ms</span></div>
    <div class="meta-item">Durée totale : <span>{duration:.1f}s</span></div>
  </div>

  <div class="verdict-box">
    <div class="verdict-score">{global_score:.0f}%</div>
    <div class="verdict-text">{verdict}</div>
    <p style="color:#888;font-size:13px;margin-top:10px;">
      {total_ok} correctes · {total_partial} partielles · {total_fail} échecs
      sur <strong>{total_variants}</strong> variantes testées
    </p>
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-num" style="color:#2e7d32">{total_ok}</div><div class="stat-label">✅ Correctes</div></div>
    <div class="stat-card"><div class="stat-num" style="color:#e65100">{total_partial}</div><div class="stat-label">⚠️ Partielles</div></div>
    <div class="stat-card"><div class="stat-num" style="color:#c62828">{total_fail}</div><div class="stat-label">❌ Échecs</div></div>
    <div class="stat-card"><div class="stat-num" style="color:#1565c0">{total_variants}</div><div class="stat-label">📨 Requêtes</div></div>
    <div class="stat-card"><div class="stat-num" style="color:#6a1b9a">{avg_latency}ms</div><div class="stat-label">⏱ Latence moy.</div></div>
  </div>

  <p class="section-title">📊 Résultats par catégorie</p>
  <div class="cats-grid">{cats_html}</div>

  <p class="section-title">📋 Détail par sujet <span style="font-size:12px;font-weight:400;color:#999;">(cliquer pour dérouler / replier)</span></p>
  {subjects_html}

  <p class="print-note">Rapport généré automatiquement — Chatbot PFE Tunisie Telecom</p>
</div>
<script>
function toggleBlock(header) {{
  const wrap = header.nextElementSibling;
  const btn  = header.querySelector('.toggle-btn');
  if (wrap.classList.contains('collapsed')) {{
    wrap.classList.remove('collapsed');
    btn.textContent = '▲';
  }} else {{
    wrap.classList.add('collapsed');
    btn.textContent = '▼';
  }}
}}
document.addEventListener('DOMContentLoaded', function() {{
  document.querySelectorAll('.table-wrap').forEach(w => {{
    w.classList.add('collapsed');
  }});
}});
</script>
</body>
</html>"""
    return html

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test final robustesse chatbot Tunisie Telecom — PFE")
    parser.add_argument("--email",    default="test@tunisietelecom.tn", help="Email de connexion")
    parser.add_argument("--password", default="Test@1234",              help="Mot de passe")
    parser.add_argument("--cat",      nargs="*", default=None,          help="Catégories à tester")
    parser.add_argument("--output",   default="rapport_test_final_chatbot.html", help="Fichier HTML de sortie")
    parser.add_argument("--delay",    type=int, default=DELAY_MS,       help="Délai entre requêtes (ms)")
    args = parser.parse_args()

    delay_ms = args.delay

    print(f"\n{bold('═'*64)}")
    print(f"  {bold('TEST FINAL ROBUSTESSE — Chatbot Tunisie Telecom (PFE)')}")
    print(f"  API : {API_BASE}")
    print(f"{'═'*64}\n")

    token = login(args.email, args.password)

    tests = TEST_BANK
    if args.cat:
        tests = [t for t in TEST_BANK if t["category"] in args.cat]
        if not tests:
            print(fail(f"Aucune catégorie trouvée parmi : {args.cat}"))
            print(f"Catégories disponibles : {[t['category'] for t in TEST_BANK]}")
            sys.exit(1)
        print(info(f"Catégories sélectionnées : {args.cat}"))

    total_variants_count = sum(len(t["variants"]) for t in tests)
    print(info(f"Lancement : {len(tests)} sujets · {total_variants_count} variantes au total\n"))

    all_results = []
    t_start     = time.time()

    for test_case in tests:
        result = test_subject(test_case, token, delay_ms)
        all_results.append(result)

    duration = time.time() - t_start

    total_v  = sum(len(s["results"]) for s in all_results)
    total_ok = sum(s["ok"]      for s in all_results)
    total_p  = sum(s["partial"] for s in all_results)
    total_f  = sum(s["fail"]    for s in all_results)
    score    = total_ok / total_v * 100 if total_v > 0 else 0

    print(f"\n{bold('═'*64)}")
    print(f"  {bold('RÉSUMÉ FINAL')}")
    print(f"{'═'*64}")
    print(f"  Score global  : {bold(f'{score:.1f}%')}")
    print(f"  Correctes     : {ok(str(total_ok))} / {total_v}")
    print(f"  Partielles    : {warn(str(total_p))}")
    print(f"  Échecs        : {fail(str(total_f))}")
    print(f"  Durée totale  : {duration:.1f}s")

    if score >= 80:
        print(f"\n  {ok('VERDICT : PRÊT POUR LA SOUTENANCE ✅')}")
    elif score >= 60:
        print(f"\n  {warn('VERDICT : QUELQUES AMÉLIORATIONS NÉCESSAIRES')}")
    else:
        print(f"\n  {fail('VERDICT : NON PRÊT — RÉVISION REQUISE')}")

    html = generate_html_report(all_results, args.email, duration)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  {info(f'Rapport HTML : {os.path.abspath(args.output)}')}")
    print(f"{'═'*64}\n")

    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(args.output)}")
        print("  (rapport ouvert dans le navigateur)")
    except Exception:
        pass

if __name__ == "__main__":
    main()