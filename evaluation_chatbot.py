#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
evaluation_chatbot.py — Évaluation Phi-3.5 vs TinyLlama vs Qwen1.5-7B
Projet : Chatbot Tunisie Telecom — PFE Cycle Ingénieur IA

ADAPTÉ AU FORMAT CHUNKS (texte brut) :
  - test.json  : liste de chunks PDF  { chunk_id, file_name,
                   year, month, chunk_index, text }
  - Questions  : générées automatiquement depuis chaque chunk
  - Référence  : le texte du chunk lui-même
  - Métriques  : Hit-Rate RAG, ROUGE-L, Faithfulness, etc.

UTILISATION :
  python evaluation_chatbot.py --all                    # évaluer les 3 modèles
  python evaluation_chatbot.py --both                   # Phi vs TinyLlama (compat. ancien)
  python evaluation_chatbot.py --model tinyllama
  python evaluation_chatbot.py --model qwen
  python evaluation_chatbot.py --all --n 50
  python evaluation_chatbot.py --all --no-injected
  python evaluation_chatbot.py --all \
      --phi-url   http://localhost:8000 \
      --tiny-url  http://localhost:8001 \
      --qwen-url  http://localhost:8002 \
      --test-file data/test.json
=============================================================
"""

import os, re, json, time, random, argparse, statistics
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
API_PHI        = "http://localhost:8000"
API_TINYLLAMA  = "http://localhost:8001"
API_QWEN       = "http://localhost:8002"   # ← Qwen1.5-7B
TEST_FILE      = "data/test.json"
OUTPUT_REPORT  = "data/evaluation_report.json"
TOP_K          = 3
N_QUESTIONS    = 999

# ─────────────────────────────────────────────────────────────
# QUESTIONS HORS-SUJET (classification is_telecom)
# ─────────────────────────────────────────────────────────────
OUT_OF_SCOPE_QUESTIONS = [
    "Quelle est la météo à Tunis aujourd'hui ?",
    "Qui est le président actuel de la Tunisie ?",
    "Comment cuisiner un couscous traditionnel ?",
    "Quel est le score du dernier match de l'Espérance ?",
    "Quels sont les meilleurs restaurants à Carthage ?",
    "Comment apprendre l'arabe rapidement ?",
    "Quel est le taux de change Euro/Dinar aujourd'hui ?",
    "Combien de kilomètres entre Tunis et Sfax ?",
    "Quel est le meilleur médicament contre la migraine ?",
    "Comment réussir mon entretien d'embauche ?",
    "Qui a écrit Les Misérables ?",
    "Quelle est la capitale de la France ?",
    "Comment perdre du poids rapidement ?",
    "Quels sont les meilleurs films de 2024 ?",
    "Comment investir en bourse ?",
]
N_INJECTED_OUTSCOPE = 15

# ─────────────────────────────────────────────────────────────
# COULEURS
# ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def color(val: float, good: float, warn: float) -> str:
    if val >= good:  return GREEN
    if val >= warn:  return YELLOW
    return RED


# ═══════════════════════════════════════════════════════════════
# GÉNÉRATION DE QUESTIONS DEPUIS LES CHUNKS
# ═══════════════════════════════════════════════════════════════

_STOP = {
    "les","des","de","la","le","un","une","est","que","quel","quels",
    "comment","sont","pour","dans","du","au","et","en","je","il","elle",
    "ce","ou","par","avec","qui","sur","vous","nos","votre","flash",
    "commercial","source","information","direction","marketing","contexte",
    "description","clients","client","tous","tout","toute","leurs","leur",
    "cette","plus","mais","aussi","donc","ainsi","alors","après","avant",
    "selon","sans","lors","entre","depuis","chaque","dont","dont","comme",
    "très","même","fait","faire","peut","doit","sera","sont","avoir",
    "être","avoir","page","note","date","lancement","version","offre",
}

def _parse_filename_topic(file_name: str) -> str:
    name = file_name.replace(".pdf", "").replace(".PDF", "")
    for prefix in [
        "Flash-commercial-", "Flash_commercial_", "Flash-Commercial-",
        "Fiche-Commerciale-", "Fiche_Commerciale_",
        "Fiche-Produit-",    "Fiche_Produit_",
        "FC-MAJ-",  "FC_MAJ_", "FCMAJ_", "FCMAJ-",
        "FC-",      "FC_",     "FC",
        "MAJ-",     "MAJ_",    "MAJ",
        "Final-_-FC-_-",
    ]:
        if name.upper().startswith(prefix.upper()):
            name = name[len(prefix):]
            break
    name = re.sub(r'[-_]\d{2}[-_]\d{2}[-_]\d{4}.*$', '', name)
    name = re.sub(r'[-_]\d{8}.*$', '',                  name)
    name = re.sub(r'[-_][Vv]\d+.*$', '',                 name)
    name = re.sub(r'[-_]+', ' ', name)
    name = re.sub(r'([a-zàâéèêëîïôùûüç])([A-ZÀÂÉÈÊËÎÏÔÙÛÜ])', r'\1 \2', name)
    name = re.sub(r'([A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{2,})([A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûüç])', r'\1 \2', name)
    name = name.strip()
    return name if name else file_name.replace(".pdf", "")


def _extract_keywords(file_name: str, text: str) -> List[str]:
    topic = _parse_filename_topic(file_name)
    fname_words = [
        w.lower() for w in re.split(r'\W+', topic)
        if len(w) > 3 and w.lower() not in _STOP and w.isalpha()
    ]
    text_words = [
        w.lower() for w in re.split(r'\W+', text[:400])
        if len(w) > 4 and w.lower() not in _STOP and w.isalpha()
    ]
    seen, kws = set(), []
    for w in fname_words + text_words:
        if w not in seen:
            seen.add(w)
            kws.append(w)
    return kws[:10]


def _generate_question(file_name: str, text: str) -> str:
    topic = _parse_filename_topic(file_name)
    tl    = text.lower()
    if re.search(r'\*\d{3}[#*]', text):
        return f"Quels sont les codes USSD et forfaits disponibles pour {topic} ?"
    if any(w in tl for w in ["recharge", "bonus sur recharge", "doubliha"]):
        return f"Quelles sont les conditions et avantages de l'offre {topic} ?"
    if any(w in tl for w in ["forfait internet", "go", "mo", "data", "navigation"]):
        return f"Quels sont les forfaits internet disponibles pour {topic} ?"
    if any(w in tl for w in ["jeu", "tirage", "gagner", "points", "score"]):
        return f"Comment fonctionne le jeu {topic} et comment participer ?"
    if any(w in tl for w in ["souscription", "souscrire", "activer", "activation"]):
        return f"Comment souscrire à {topic} et quelles sont les conditions d'éligibilité ?"
    if any(w in tl for w in ["fidélité", "kelma", "bons d'achat", "points kelma"]):
        return f"Comment fonctionne le programme de fidélité {topic} ?"
    if any(w in tl for w in ["tarif", "prix", "dt", "dinar", "facturation"]):
        return f"Quels sont les tarifs et la tarification de {topic} ?"
    if any(w in tl for w in ["désinscription", "désactiver", "arrêt"]):
        return f"Comment se désinscrire ou désactiver {topic} ?"
    if any(w in tl for w in ["4g", "5g", "box", "fibre", "adsl", "fixe"]):
        return f"Quelles sont les caractéristiques et conditions de l'offre {topic} ?"
    if any(w in tl for w in ["roaming", "international", "étranger"]):
        return f"Quelles sont les conditions d'utilisation de {topic} à l'international ?"
    return f"Donnez-moi des informations détaillées sur {topic}."


def _derive_category(file_name: str) -> str:
    fn = file_name.lower()
    if any(w in fn for w in ["jeu", "challenge", "quiz", "game"]):
        return "jeux_concours"
    if any(w in fn for w in ["fidel", "kelma"]):
        return "fidelite"
    if any(w in fn for w in ["facebook", "social", "digital", "jawek"]):
        return "digital"
    if any(w in fn for w in ["internet", "data", "140", "4g", "5g", "box"]):
        return "internet_data"
    if any(w in fn for w in ["roaming", "international"]):
        return "roaming"
    if any(w in fn for w in ["forfait", "pack", "offre", "hayya", "taraji",
                              "elissa", "hybrid", "css"]):
        return "offres_mobiles"
    if any(w in fn for w in ["nokia", "samsung", "iphone", "terminal", "produit"]):
        return "terminaux"
    if any(w in fn for w in ["panorama", "maj", "menu"]):
        return "catalogue"
    if any(w in fn for w in ["entreprise", "pme", "b2b", "fh"]):
        return "entreprise"
    return "general"


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DU DATASET
# ═══════════════════════════════════════════════════════════════

def load_test_dataset(
    path: str,
    n: int = N_QUESTIONS,
    add_outscope: bool = True,
) -> List[Dict]:
    random.seed(42)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFichier introuvable : {path}\n"
            f"Vérifie le chemin ou utilise --test-file <chemin>"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("test.json doit être une liste JSON de chunks.")

    examples = []
    skipped  = 0

    for chunk in raw:
        text = chunk.get("text", chunk.get("chunk_text", "")).strip()
        fn   = chunk.get("file_name", "").strip()

        if not text or not fn:
            skipped += 1
            continue
        if len(text) < 40:
            skipped += 1
            continue

        question  = _generate_question(fn, text)
        keywords  = _extract_keywords(fn, text)
        category  = _derive_category(fn)

        examples.append({
            "id":               len(examples) + 1,
            "question":         question,
            "expected_answer":  text,
            "is_telecom":       True,
            "source_keywords":  keywords,
            "expected_keywords":keywords[:6],
            "category":         category,
            "sub_category":     chunk.get("chunk_index", 0),
            "chunk_id":         chunk.get("chunk_id", ""),
            "file_name":        fn,
            "year":             chunk.get("year", ""),
            "month":            chunk.get("month", ""),
        })

    if skipped:
        print(f"  [INFO] {skipped} chunk(s) ignoré(s) (texte trop court ou vide)")

    if len(examples) > n:
        random.shuffle(examples)
        examples = examples[:n]

    if add_outscope:
        n_inject = min(N_INJECTED_OUTSCOPE, len(OUT_OF_SCOPE_QUESTIONS))
        for q in random.sample(OUT_OF_SCOPE_QUESTIONS, n_inject):
            examples.append({
                "id":               0,
                "question":         q,
                "expected_answer":  "[HORS-SUJET — l'assistant doit refuser]",
                "is_telecom":       False,
                "source_keywords":  [],
                "expected_keywords":[],
                "category":         "out_of_scope",
                "sub_category":     "",
                "chunk_id":         "",
                "file_name":        "",
                "year":             "",
                "month":            "",
            })
        random.shuffle(examples)

    for i, ex in enumerate(examples, 1):
        ex["id"] = i

    n_tc  = sum(1 for e in examples if e["is_telecom"])
    n_oos = len(examples) - n_tc
    print(
        f"  Dataset : {len(examples)} questions générées "
        f"({n_tc} télécom + {n_oos} hors-sujet) "
        f"— {os.path.basename(path)}"
    )
    return examples


# ═══════════════════════════════════════════════════════════════
# APPEL API
# ═══════════════════════════════════════════════════════════════

def call_api(question: str, api_base: str, timeout: int = 60) -> Optional[Dict]:
    import requests
    try:
        resp = requests.post(
            f"{api_base}/chat",
            json={"message": question},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "answer":           data.get("answer",           ""),
            "sources":          data.get("sources",          []),
            "rag_used":         data.get("rag_used",         False),
            "confidence":       data.get("confidence",       0.0),
            "is_telecom":       data.get("is_telecom",       False),
            "mode":             data.get("mode",             "?"),
            "response_time_ms": data.get("response_time_ms", 0),
            "cache_hit":        data.get("cache_hit",        False),
            "model":            data.get("model",            "unknown"),
        }
    except requests.exceptions.ConnectionError:
        print(f"  [ERREUR] Impossible de joindre {api_base} — API démarrée ?")
        return None
    except requests.exceptions.Timeout:
        print(f"  [ERREUR] Timeout ({timeout}s) sur {api_base}")
        return None
    except Exception as e:
        print(f"  [ERREUR] {api_base} : {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# MÉTRIQUES RAG
# ═══════════════════════════════════════════════════════════════

def _source_text(src: Dict) -> str:
    parts = [
        src.get("text",         ""),
        src.get("chunk_text",   ""),
        src.get("filename",     ""),
        src.get("file_name",    ""),
        src.get("theme",        ""),
        src.get("sub_category", ""),
        src.get("chunk_id",     ""),
    ]
    tags = src.get("tags", "")
    if isinstance(tags, list):
        parts.append(" ".join(tags))
    elif isinstance(tags, str):
        parts.append(tags)
    return " ".join(p for p in parts if p).lower()


def hit_rate(sources: List[Dict], kws: List[str], k: int = TOP_K) -> float:
    if not kws:
        return 1.0
    for src in sources[:k]:
        txt = _source_text(src)
        if any(kw.lower() in txt for kw in kws):
            return 1.0
    return 0.0


def mrr(sources: List[Dict], kws: List[str]) -> float:
    if not kws:
        return 1.0
    for i, src in enumerate(sources):
        txt = _source_text(src)
        if any(kw.lower() in txt for kw in kws):
            return 1.0 / (i + 1)
    return 0.0


def precision_k(sources: List[Dict], kws: List[str], k: int = TOP_K) -> float:
    if not kws or not sources:
        return 0.0
    rel = sum(
        1 for s in sources[:k]
        if any(kw.lower() in _source_text(s) for kw in kws)
    )
    return rel / min(k, len(sources))


def context_relevance(sources: List[Dict], question: str) -> float:
    if not sources:
        return 0.0
    q_words = {
        w for w in question.lower().split()
        if len(w) > 2 and w not in _STOP
    }
    if not q_words:
        return 0.0
    rel = sum(
        1 for src in sources
        if any(w in (_source_text(src)) for w in q_words)
    )
    return rel / len(sources)


# ═══════════════════════════════════════════════════════════════
# MÉTRIQUES GÉNÉRATION
# ═══════════════════════════════════════════════════════════════

def rouge_l(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer as rs
        return rs.RougeScorer(["rougeL"], use_stemmer=False)\
                 .score(ref, pred)["rougeL"].fmeasure
    except ImportError:
        pass
    p_tok, r_tok = pred.lower().split(), ref.lower().split()
    m, n = len(p_tok), len(r_tok)
    if not m or not n:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i-1][j-1] + 1
                if p_tok[i-1] == r_tok[j-1]
                else max(dp[i-1][j], dp[i][j-1])
            )
    lcs  = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def keyword_coverage(answer: str, kws: List[str]) -> float:
    if not kws:
        return 1.0
    al = answer.lower()
    return sum(1 for kw in kws if kw.lower() in al) / len(kws)


def faithfulness(answer: str, sources: List[Dict]) -> float:
    if not sources:
        return 0.5
    ctx = " ".join(_source_text(s) for s in sources)
    if not ctx.strip():
        return 0.5
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words:
        return 0.5
    return sum(1 for w in words if w in ctx) / len(words)


def chunk_exact_hit(sources: List[Dict], chunk_id: str) -> float:
    if not chunk_id or not sources:
        return 0.0
    for src in sources[:TOP_K]:
        if src.get("chunk_id", "") == chunk_id:
            return 1.0
    return 0.0


def length_score(answer: str) -> float:
    n = len(answer)
    if n < 30:   return 0.1
    if n < 80:   return 0.4
    if n < 140:  return 0.7
    if n <= 350: return 1.0
    if n <= 600: return 0.85
    if n <= 900: return 0.6
    return 0.4


def hallucination_check(answer: str) -> bool:
    signals = ["ooredoo", "myoredoo", "orange tunisie", "tunisiana", "vodafone"]
    al = answer.lower()
    return any(s in al for s in signals)


def bert_score_batch(preds: List[str], refs: List[str]) -> Dict:
    try:
        from bert_score import score as bscore
        P, R, F1 = bscore(
            preds, refs,
            model_type="distilbert-base-multilingual-cased",
            lang="fr", verbose=False, device="cpu",
        )
        return {
            "f1":        float(F1.mean()),
            "precision": float(P.mean()),
            "recall":    float(R.mean()),
        }
    except ImportError:
        print("  [INFO] bert-score non installé — BERTScore ignoré")
        return {"f1": None, "precision": None, "recall": None}
    except Exception as e:
        print(f"  [WARN] BERTScore échoué : {e}")
        return {"f1": None, "precision": None, "recall": None}


# ═══════════════════════════════════════════════════════════════
# AFFICHAGE
# ═══════════════════════════════════════════════════════════════

def pm(label: str, val, unit: str = "", good: float = 0.7, warn: float = 0.5):
    if isinstance(val, float):
        c    = color(val, good, warn)
        icon = "OK" if val >= good else "~~" if val >= warn else "!!"
        print(f"  [{icon}] {label:<42} {c}{val:.4f}{RESET} {unit}")
    else:
        print(f"  [  ] {label:<42} {val} {unit}")


# ═══════════════════════════════════════════════════════════════
# ÉVALUATION D'UN MODÈLE
# ═══════════════════════════════════════════════════════════════

def evaluate_model(
    test_data:  List[Dict],
    api_base:   str,
    model_name: str,
) -> Dict:

    print(f"\n{'='*65}")
    print(f"  ÉVALUATION — {model_name}")
    print(f"{'='*65}")
    print(f"  API : {api_base}  |  Questions : {len(test_data)}")

    if call_api("bonjour", api_base, timeout=10) is None:
        print(f"\n  {RED}ABANDON : API inaccessible sur {api_base}{RESET}")
        return {}

    results        = []
    rag_ms         = []
    gen_ms         = []
    times          = []
    rag_used_n     = 0
    telecom_ok     = 0
    hallucin_n     = 0
    exact_hits     = []
    preds_tc       = []
    refs_tc        = []
    cat_stats      = {}

    gen_n     = 0
    ext_n     = 0
    no_info_n = 0

    for item in test_data:
        q_preview = item["question"][:65]
        print(f"\n  [Q{item['id']:03d}] {q_preview}{'...' if len(item['question'])>65 else ''}")
        if item["file_name"]:
            print(f"         Chunk : {item['file_name']}  [idx={item['sub_category']}]")

        t0   = time.time()
        data = call_api(item["question"], api_base)
        wall = int((time.time() - t0) * 1000)

        if data is None:
            print("         → Réponse vide (API inaccessible)")
            continue

        times.append(wall)

        answer   = data["answer"]
        sources  = data["sources"]
        rag_used = data["rag_used"]
        conf     = data["confidence"]
        is_tc    = data["is_telecom"]
        mode     = data["mode"]

        if mode == "generative" or mode == "generative_rag":
            gen_n += 1
        elif mode.startswith("extractive"):
            ext_n += 1
        elif mode == "no_info":
            no_info_n += 1

        if rag_used:
            rag_used_n += 1
        if is_tc == item["is_telecom"]:
            telecom_ok += 1
        if hallucination_check(answer):
            hallucin_n += 1

        if item["is_telecom"]:
            preds_tc.append(answer)
            refs_tc.append(item["expected_answer"])

        print(
            f"         Mode: {mode:<22} | Conf: {conf:.3f} | "
            f"RAG: {'Oui' if rag_used else 'Non'} | {wall}ms"
        )
        print(
            f"         Télécom attendu: {item['is_telecom']} | "
            f"Détecté: {is_tc} | "
            f"Hallu: {'OUI ⚠' if hallucination_check(answer) else 'non'}"
        )
        print(f"         Réponse: {answer[:120]}{'...' if len(answer)>120 else ''}")

        rm = {}
        if item["is_telecom"] and sources:
            eh = chunk_exact_hit(sources, item.get("chunk_id", ""))
            exact_hits.append(eh)

            if item["source_keywords"]:
                rm = {
                    "hit_rate":          hit_rate(sources,  item["source_keywords"]),
                    "mrr":               mrr(sources,       item["source_keywords"]),
                    "precision_k":       precision_k(sources,item["source_keywords"]),
                    "context_relevance": context_relevance(sources, item["question"]),
                    "chunk_exact_hit":   eh,
                }
                rag_ms.append(rm)

        gm = {}
        if item["is_telecom"]:
            gm = {
                "rouge_l":          rouge_l(answer,          item["expected_answer"]),
                "keyword_coverage": keyword_coverage(answer, item["expected_keywords"]),
                "faithfulness":     faithfulness(answer,     sources),
                "length_score":     length_score(answer),
                "hallucination":    int(hallucination_check(answer)),
            }
            gen_ms.append(gm)

        cat = item.get("category", "unknown") or "unknown"
        if cat not in cat_stats:
            cat_stats[cat] = {
                "n": 0, "classif_ok": 0, "rag_used": 0,
                "rouge_l_sum": 0.0, "kw_cov_sum": 0.0, "hallucin": 0,
                "gen_n": 0, "ext_n": 0, "no_info_n": 0,
            }
        cs = cat_stats[cat]
        cs["n"]          += 1
        cs["classif_ok"] += int(is_tc == item["is_telecom"])
        cs["rag_used"]   += int(rag_used)
        if mode in ("generative", "generative_rag"):
            cs["gen_n"] += 1
        elif mode.startswith("extractive"):
            cs["ext_n"] += 1
        elif mode == "no_info":
            cs["no_info_n"] += 1
        if gm:
            cs["rouge_l_sum"] += gm.get("rouge_l", 0)
            cs["kw_cov_sum"]  += gm.get("keyword_coverage", 0)
            cs["hallucin"]    += gm.get("hallucination", 0)

        results.append({
            "id":               item["id"],
            "question":         item["question"],
            "file_name":        item.get("file_name", ""),
            "chunk_id":         item.get("chunk_id", ""),
            "category":         item.get("category", ""),
            "expected":         item["expected_answer"][:300] + "...",
            "answer":           answer,
            "is_telecom":       is_tc,
            "expected_telecom": item["is_telecom"],
            "rag_used":         rag_used,
            "confidence":       conf,
            "response_time_ms": wall,
            "mode":             mode,
            "rag_metrics":      rm,
            "gen_metrics":      gm,
        })

    # BERTScore
    bert = {"f1": None, "precision": None, "recall": None}
    if preds_tc:
        print(f"\n  Calcul BERTScore sur {len(preds_tc)} réponses télécom...")
        bert = bert_score_batch(preds_tc, refs_tc)

    n   = len(test_data)
    acc = telecom_ok / n if n else 0

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS — {model_name}")
    print(f"{'='*65}")

    print(f"\n  Classification télécom / hors-sujet")
    pm("Accuracy classification",    acc,                         good=0.90, warn=0.75)
    pm("Taux RAG activé",            rag_used_n/n if n else 0,    good=0.70, warn=0.50)

    hr = hallucin_n/n if n else 0
    c  = GREEN if hr <= 0.02 else YELLOW if hr <= 0.05 else RED
    print(f"  [{'OK' if hr<=0.02 else '!!' }] {'Taux hallucinations':<42} {c}{hr:.4f}{RESET}  (↓ souhaité)")

    n_answered = len(results)
    print(f"\n  Modes de génération  (sur {n_answered} réponses obtenues)")
    print(f"  {'─'*55}")

    gen_rate = gen_n / n_answered if n_answered else 0
    c_gen    = GREEN if gen_rate >= 0.60 else YELLOW if gen_rate >= 0.40 else RED
    print(
        f"  [{'OK' if gen_rate>=0.60 else '~~' if gen_rate>=0.40 else '!!'}] "
        f"{'Génératif (LLM réussi)':<42} "
        f"{c_gen}{gen_n:>4} réponses  ({gen_rate*100:5.1f}%){RESET}"
    )

    ext_rate = ext_n / n_answered if n_answered else 0
    c_ext    = GREEN if ext_rate <= 0.20 else YELLOW if ext_rate <= 0.40 else RED
    print(
        f"  [{'OK' if ext_rate<=0.20 else '~~' if ext_rate<=0.40 else '!!'}] "
        f"{'Extractif (fallback)':<42} "
        f"{c_ext}{ext_n:>4} réponses  ({ext_rate*100:5.1f}%){RESET}  (↓ souhaité)"
    )

    noi_rate = no_info_n / n_answered if n_answered else 0
    c_noi    = GREEN if noi_rate <= 0.10 else YELLOW if noi_rate <= 0.20 else RED
    print(
        f"  [{'OK' if noi_rate<=0.10 else '~~' if noi_rate<=0.20 else '!!'}] "
        f"{'No-info (conf RAG trop faible)':<42} "
        f"{c_noi}{no_info_n:>4} réponses  ({noi_rate*100:5.1f}%){RESET}  (↓ souhaité)"
    )

    other_n    = n_answered - gen_n - ext_n - no_info_n
    other_rate = other_n / n_answered if n_answered else 0
    print(
        f"  [  ] {'Autres (greeting, hors_sujet…)':<42} "
        f"{other_n:>4} réponses  ({other_rate*100:5.1f}%)"
    )
    print(f"  {'─'*55}")

    ext_subtypes = {}
    for r in results:
        m = r["mode"]
        if m.startswith("extractive"):
            ext_subtypes[m] = ext_subtypes.get(m, 0) + 1
    if ext_subtypes:
        print(f"  Détail fallbacks extractifs :")
        for sub, cnt in sorted(ext_subtypes.items(), key=lambda x: -x[1]):
            print(f"       {sub:<35} {cnt:>4} fois")

    if times:
        print(f"\n  Latence")
        print(f"  [  ] {'Temps moyen':<42} {statistics.mean(times):.0f} ms")
        print(f"  [  ] {'Temps médian':<42} {statistics.median(times):.0f} ms")
        print(f"  [  ] {'Temps max':<42} {max(times):.0f} ms")

    if rag_ms:
        print(f"\n  Métriques RAG  (sur {len(rag_ms)} questions avec mots-clés)")
        for k, label, g, w in [
            ("hit_rate",          "Hit Rate@3",        0.70, 0.50),
            ("mrr",               "MRR",               0.65, 0.45),
            ("precision_k",       "Precision@3",       0.55, 0.35),
            ("context_relevance", "Context Relevance", 0.60, 0.40),
            ("chunk_exact_hit",   "Chunk Exact Hit",   0.40, 0.20),
        ]:
            vs = [m[k] for m in rag_ms if k in m]
            if vs:
                pm(label, statistics.mean(vs), good=g, warn=w)

    if gen_ms:
        print(f"\n  Qualité de génération  (sur {len(gen_ms)} questions télécom)")
        for k, label, g, w in [
            ("rouge_l",          "ROUGE-L",            0.25, 0.12),
            ("keyword_coverage", "Keyword Coverage",    0.55, 0.35),
            ("faithfulness",     "Faithfulness",        0.40, 0.25),
            ("length_score",     "Length Score",        0.80, 0.50),
        ]:
            vs = [m[k] for m in gen_ms if k in m]
            if vs:
                pm(label, statistics.mean(vs), good=g, warn=w)

    if bert["f1"] is not None:
        print(f"\n  Sémantique (BERTScore)")
        pm("BERTScore F1",        bert["f1"],        good=0.65, warn=0.50)
        pm("BERTScore Precision", bert["precision"], good=0.65, warn=0.50)
        pm("BERTScore Recall",    bert["recall"],    good=0.65, warn=0.50)

    if len(cat_stats) > 1:
        print(f"\n  Détail par catégorie")
        print(
            f"  {'Catégorie':<22} {'N':>4} {'Class.%':>8} "
            f"{'RAG%':>6} {'Gen':>5} {'Ext':>5} {'NoInf':>6} "
            f"{'ROUGE-L':>8} {'KW Cov':>8} {'Halluc':>7}"
        )
        print(f"  {'─'*22} {'─'*4} {'─'*8} {'─'*6} {'─'*5} {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*7}")
        for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["n"]):
            nc      = s["n"]
            cls_pct = 100 * s["classif_ok"] / nc if nc else 0
            rag_pct = 100 * s["rag_used"]   / nc if nc else 0
            rl      = s["rouge_l_sum"]  / nc if nc else 0
            kw      = s["kw_cov_sum"]   / nc if nc else 0
            ha      = s["hallucin"]
            gn      = s["gen_n"]
            en      = s["ext_n"]
            nin     = s["no_info_n"]
            print(
                f"  {cat:<22} {nc:>4} {cls_pct:>7.0f}% "
                f"{rag_pct:>5.0f}% {gn:>5} {en:>5} {nin:>6} "
                f"{rl:>8.3f} {kw:>8.3f} {ha:>7d}"
            )

    scores = [acc]
    for k in ["hit_rate", "mrr", "context_relevance"]:
        vs = [m[k] for m in rag_ms if k in m]
        if vs: scores.append(statistics.mean(vs))
    for k in ["keyword_coverage", "faithfulness"]:
        vs = [m[k] for m in gen_ms if k in m]
        if vs: scores.append(statistics.mean(vs))
    if bert["f1"]:
        scores.append(bert["f1"])

    gs    = statistics.mean(scores) if scores else 0
    grade = (
        "Excellent"          if gs >= 0.70 else
        "Très satisfaisant"  if gs >= 0.60 else
        "Satisfaisant"       if gs >= 0.55 else
        "À améliorer"
    )

    print(f"\n  Score global")
    pm("Score composite", gs, good=0.70, warn=0.55)
    print(f"\n  Appréciation PFE : {BOLD}{grade}{RESET}")

    def avg(lst, key):
        vs = [m[key] for m in lst if key in m]
        return statistics.mean(vs) if vs else 0.0

    rag_avg = {k: avg(rag_ms, k) for k in
               ["hit_rate","mrr","precision_k","context_relevance","chunk_exact_hit"]}
    gen_avg = {k: avg(gen_ms, k) for k in
               ["rouge_l","keyword_coverage","faithfulness","length_score"]}

    return {
        "model_name":         model_name,
        "api_base":           api_base,
        "n_questions":        n,
        "n_answered":         len(results),
        "n_telecom":          sum(1 for x in test_data if x["is_telecom"]),
        "n_outscope":         sum(1 for x in test_data if not x["is_telecom"]),
        "accuracy":           acc,
        "rag_rate":           rag_used_n / n if n else 0,
        "hallucination_rate": hallucin_n  / n if n else 0,
        "avg_latency_ms":     statistics.mean(times)   if times else 0,
        "median_latency_ms":  statistics.median(times) if times else 0,
        "max_latency_ms":     max(times) if times else 0,
        "global_score":       gs,
        "grade":              grade,
        "generation_modes": {
            "generative_n":    gen_n,
            "extractive_n":    ext_n,
            "no_info_n":       no_info_n,
            "other_n":         other_n,
            "generative_rate": round(gen_rate, 4),
            "extractive_rate": round(ext_rate, 4),
            "no_info_rate":    round(noi_rate, 4),
            "extractive_subtypes": ext_subtypes,
        },
        "rag_avg":            rag_avg,
        "gen_avg":            gen_avg,
        "bert_score":         bert,
        "category_stats":     cat_stats,
        "per_question":       results,
    }


# ═══════════════════════════════════════════════════════════════
# TABLEAU COMPARATIF — 2 MODÈLES
# ═══════════════════════════════════════════════════════════════

def compare_two(r1: Dict, r2: Dict):
    n1, n2 = r1["model_name"], r2["model_name"]
    W = 18

    print(f"\n{'='*72}")
    print(f"  TABLEAU COMPARATIF — {n1}  vs  {n2}")
    print(f"{'='*72}")
    print(f"\n  {'Métrique':<34} {n1:>{W}} {n2:>{W}}")
    print(f"  {'─'*34} {'─'*W} {'─'*W}")

    def get(d, *keys):
        for k in keys:
            d = d.get(k, 0) if isinstance(d, dict) else 0
        return d or 0

    def row(label, v1, v2, good=0.7, warn=0.5, fmt=".4f", lower_better=False):
        if lower_better:
            b1, b2 = v1 < v2, v2 < v1
        else:
            b1, b2 = v1 > v2, v2 > v1
        s1, s2 = " ◀" if b1 else "", " ◀" if b2 else ""
        e1 = (1 - v1/max(v1,v2,0.001)) if lower_better else v1
        e2 = (1 - v2/max(v1,v2,0.001)) if lower_better else v2
        c1, c2 = color(e1, good, warn), color(e2, good, warn)
        r = RESET
        print(
            f"  {label:<34} "
            f"{c1}{f'{v1:{fmt}}{s1}':>{W}}{r} "
            f"{c2}{f'{v2:{fmt}}{s2}':>{W}}{r}"
        )

    print(f"\n  ── Métriques RAG ──────────────────────────────────")
    row("Hit Rate@3",         get(r1,"rag_avg","hit_rate"),          get(r2,"rag_avg","hit_rate"),          good=0.70, warn=0.50)
    row("MRR",                get(r1,"rag_avg","mrr"),               get(r2,"rag_avg","mrr"),               good=0.65, warn=0.45)
    row("Precision@3",        get(r1,"rag_avg","precision_k"),       get(r2,"rag_avg","precision_k"),       good=0.55, warn=0.35)
    row("Context Relevance",  get(r1,"rag_avg","context_relevance"), get(r2,"rag_avg","context_relevance"), good=0.60, warn=0.40)
    row("Chunk Exact Hit",    get(r1,"rag_avg","chunk_exact_hit"),   get(r2,"rag_avg","chunk_exact_hit"),   good=0.40, warn=0.20)
    print(f"\n  ── Qualité de génération ──────────────────────────")
    row("ROUGE-L",            get(r1,"gen_avg","rouge_l"),           get(r2,"gen_avg","rouge_l"),           good=0.25, warn=0.12)
    row("Keyword Coverage",   get(r1,"gen_avg","keyword_coverage"),  get(r2,"gen_avg","keyword_coverage"),  good=0.55, warn=0.35)
    row("Faithfulness",       get(r1,"gen_avg","faithfulness"),      get(r2,"gen_avg","faithfulness"),      good=0.40, warn=0.25)
    row("Length Score",       get(r1,"gen_avg","length_score"),      get(r2,"gen_avg","length_score"),      good=0.80, warn=0.50)
    print(f"\n  ── Sémantique (BERTScore) ─────────────────────────")
    row("BERTScore F1",       get(r1,"bert_score","f1"),             get(r2,"bert_score","f1"),             good=0.65, warn=0.50)
    row("BERTScore Prec.",    get(r1,"bert_score","precision"),      get(r2,"bert_score","precision"),      good=0.65, warn=0.50)
    row("BERTScore Recall",   get(r1,"bert_score","recall"),         get(r2,"bert_score","recall"),         good=0.65, warn=0.50)
    print(f"\n  ── Modes de génération ────────────────────────────")
    row("Taux génératif (%)",
        get(r1,"generation_modes","generative_rate")*100,
        get(r2,"generation_modes","generative_rate")*100,
        good=60.0, warn=40.0, fmt=".1f")
    row("Taux extractif / fallback (%)",
        get(r1,"generation_modes","extractive_rate")*100,
        get(r2,"generation_modes","extractive_rate")*100,
        good=80.0, warn=60.0, fmt=".1f", lower_better=True)
    row("Taux no-info (%)",
        get(r1,"generation_modes","no_info_rate")*100,
        get(r2,"generation_modes","no_info_rate")*100,
        good=90.0, warn=80.0, fmt=".1f", lower_better=True)
    print(f"\n  ── Système ────────────────────────────────────────")
    row("Accuracy classif.",  r1["accuracy"],           r2["accuracy"],           good=0.90, warn=0.75)
    row("Taux RAG activé",    r1["rag_rate"],           r2["rag_rate"],           good=0.70, warn=0.50)
    row("Taux hallucinations",r1["hallucination_rate"], r2["hallucination_rate"], good=0.98, warn=0.95, lower_better=True)
    row("Latence moy. (ms)",  r1["avg_latency_ms"],     r2["avg_latency_ms"],     good=0.5,  warn=0.3,  fmt=".0f", lower_better=True)
    print(f"\n  ── Score global ───────────────────────────────────")
    row("Score composite",    r1["global_score"],        r2["global_score"],        good=0.65, warn=0.55)

    winner = n1 if r1["global_score"] >= r2["global_score"] else n2
    delta  = abs(r1["global_score"] - r2["global_score"])
    print(f"\n{'='*72}")
    print(f"  Meilleur : {BOLD}{winner}{RESET}  (+{delta:.4f})")
    print(f"  {n1:<30} : {r1['grade']}")
    print(f"  {n2:<30} : {r2['grade']}")
    print(f"{'='*72}")


# ═══════════════════════════════════════════════════════════════
# TABLEAU COMPARATIF — 3 MODÈLES  ← NOUVEAU
# ═══════════════════════════════════════════════════════════════

def compare_three(r1: Dict, r2: Dict, r3: Dict):
    """
    Tableau comparatif à 3 colonnes : Phi-3.5 | TinyLlama | Qwen1.5-7B
    Le meilleur sur chaque ligne est indiqué par ◀
    """
    n1, n2, n3 = r1["model_name"], r2["model_name"], r3["model_name"]
    W = 16

    print(f"\n{'='*80}")
    print(f"  TABLEAU COMPARATIF — 3 MODÈLES — SOUTENANCE PFE")
    print(f"{'='*80}")
    print(f"\n  {'Métrique':<32} {n1:>{W}} {n2:>{W}} {n3:>{W}}")
    print(f"  {'─'*32} {'─'*W} {'─'*W} {'─'*W}")

    def get(d, *keys):
        for k in keys:
            d = d.get(k, 0) if isinstance(d, dict) else 0
        return d or 0

    def row3(label, v1, v2, v3, good=0.7, warn=0.5, fmt=".4f", lower_better=False):
        vals   = [v1, v2, v3]
        best   = min(vals) if lower_better else max(vals)
        marks  = [" ◀" if v == best else "" for v in vals]
        # Normalise pour la couleur
        if lower_better:
            enorm = [(1 - v / max(max(vals), 0.001)) for v in vals]
        else:
            enorm = vals
        cols = [color(e, good, warn) for e in enorm]
        cells = [
            f"{cols[i]}{f'{vals[i]:{fmt}}{marks[i]}':>{W}}{RESET}"
            for i in range(3)
        ]
        print(f"  {label:<32} {cells[0]} {cells[1]} {cells[2]}")

    print(f"\n  ── Métriques RAG ──────────────────────────────────────────")
    row3("Hit Rate@3",
         get(r1,"rag_avg","hit_rate"),          get(r2,"rag_avg","hit_rate"),          get(r3,"rag_avg","hit_rate"),
         good=0.70, warn=0.50)
    row3("MRR",
         get(r1,"rag_avg","mrr"),               get(r2,"rag_avg","mrr"),               get(r3,"rag_avg","mrr"),
         good=0.65, warn=0.45)
    row3("Precision@3",
         get(r1,"rag_avg","precision_k"),       get(r2,"rag_avg","precision_k"),       get(r3,"rag_avg","precision_k"),
         good=0.55, warn=0.35)
    row3("Context Relevance",
         get(r1,"rag_avg","context_relevance"), get(r2,"rag_avg","context_relevance"), get(r3,"rag_avg","context_relevance"),
         good=0.60, warn=0.40)
    row3("Chunk Exact Hit",
         get(r1,"rag_avg","chunk_exact_hit"),   get(r2,"rag_avg","chunk_exact_hit"),   get(r3,"rag_avg","chunk_exact_hit"),
         good=0.40, warn=0.20)

    print(f"\n  ── Qualité de génération ──────────────────────────────────")
    row3("ROUGE-L",
         get(r1,"gen_avg","rouge_l"),           get(r2,"gen_avg","rouge_l"),           get(r3,"gen_avg","rouge_l"),
         good=0.25, warn=0.12)
    row3("Keyword Coverage",
         get(r1,"gen_avg","keyword_coverage"),  get(r2,"gen_avg","keyword_coverage"),  get(r3,"gen_avg","keyword_coverage"),
         good=0.55, warn=0.35)
    row3("Faithfulness",
         get(r1,"gen_avg","faithfulness"),      get(r2,"gen_avg","faithfulness"),      get(r3,"gen_avg","faithfulness"),
         good=0.40, warn=0.25)
    row3("Length Score",
         get(r1,"gen_avg","length_score"),      get(r2,"gen_avg","length_score"),      get(r3,"gen_avg","length_score"),
         good=0.80, warn=0.50)

    print(f"\n  ── Sémantique (BERTScore) ─────────────────────────────────")
    row3("BERTScore F1",
         get(r1,"bert_score","f1"),             get(r2,"bert_score","f1"),             get(r3,"bert_score","f1"),
         good=0.65, warn=0.50)
    row3("BERTScore Precision",
         get(r1,"bert_score","precision"),      get(r2,"bert_score","precision"),      get(r3,"bert_score","precision"),
         good=0.65, warn=0.50)
    row3("BERTScore Recall",
         get(r1,"bert_score","recall"),         get(r2,"bert_score","recall"),         get(r3,"bert_score","recall"),
         good=0.65, warn=0.50)

    print(f"\n  ── Modes de génération ────────────────────────────────────")
    row3("Taux génératif (%)",
         get(r1,"generation_modes","generative_rate")*100,
         get(r2,"generation_modes","generative_rate")*100,
         get(r3,"generation_modes","generative_rate")*100,
         good=60.0, warn=40.0, fmt=".1f")
    row3("Taux extractif / fallback (%)",
         get(r1,"generation_modes","extractive_rate")*100,
         get(r2,"generation_modes","extractive_rate")*100,
         get(r3,"generation_modes","extractive_rate")*100,
         good=80.0, warn=60.0, fmt=".1f", lower_better=True)
    row3("Taux no-info (%)",
         get(r1,"generation_modes","no_info_rate")*100,
         get(r2,"generation_modes","no_info_rate")*100,
         get(r3,"generation_modes","no_info_rate")*100,
         good=90.0, warn=80.0, fmt=".1f", lower_better=True)

    print(f"\n  ── Système ────────────────────────────────────────────────")
    row3("Accuracy classif.",
         r1["accuracy"],           r2["accuracy"],           r3["accuracy"],
         good=0.90, warn=0.75)
    row3("Taux RAG activé",
         r1["rag_rate"],           r2["rag_rate"],           r3["rag_rate"],
         good=0.70, warn=0.50)
    row3("Taux hallucinations",
         r1["hallucination_rate"], r2["hallucination_rate"], r3["hallucination_rate"],
         good=0.98, warn=0.95, lower_better=True)
    row3("Latence moy. (ms)",
         r1["avg_latency_ms"],     r2["avg_latency_ms"],     r3["avg_latency_ms"],
         good=0.5, warn=0.3, fmt=".0f", lower_better=True)
    row3("Latence médiane (ms)",
         r1["median_latency_ms"],  r2["median_latency_ms"],  r3["median_latency_ms"],
         good=0.5, warn=0.3, fmt=".0f", lower_better=True)

    print(f"\n  ── Score global ───────────────────────────────────────────")
    row3("Score composite",
         r1["global_score"],       r2["global_score"],       r3["global_score"],
         good=0.65, warn=0.55)

    # Classement final
    ranked = sorted(
        [(r1["global_score"], n1, r1["grade"]),
         (r2["global_score"], n2, r2["grade"]),
         (r3["global_score"], n3, r3["grade"])],
        reverse=True
    )
    print(f"\n{'='*80}")
    print(f"  CLASSEMENT FINAL")
    print(f"  {'─'*78}")
    medals = ["🥇", "🥈", "🥉"]
    for i, (score, name, grade) in enumerate(ranked):
        medal = medals[i] if i < len(medals) else f"  {i+1}."
        delta = score - ranked[-1][0]
        delta_str = f"  +{delta:.4f} vs dernier" if i < len(ranked)-1 else ""
        print(f"  {medal}  {name:<30}  Score: {score:.4f}  |  {grade}{delta_str}")

    print(f"\n  Gagnant absolu : {BOLD}{ranked[0][1]}{RESET}")
    print(f"{'='*80}")

    # Résumé modes par modèle
    print(f"\n  RÉSUMÉ MODES DE GÉNÉRATION")
    print(f"  {'─'*78}")
    print(f"  {'Mode':<30} {n1:>{W}} {n2:>{W}} {n3:>{W}}")
    print(f"  {'─'*30} {'─'*W} {'─'*W} {'─'*W}")
    for label, key in [
        ("Génératif (n)",   "generative_n"),
        ("Extractif (n)",   "extractive_n"),
        ("No-info (n)",     "no_info_n"),
        ("Autres (n)",      "other_n"),
    ]:
        v1 = get(r1, "generation_modes", key)
        v2 = get(r2, "generation_modes", key)
        v3 = get(r3, "generation_modes", key)
        print(f"  {label:<30} {str(v1):>{W}} {str(v2):>{W}} {str(v3):>{W}}")
    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Évaluation Chatbot TT — Phi-3.5 vs TinyLlama vs Qwen1.5-7B"
    )
    parser.add_argument("--all",         action="store_true",
                        help="Évaluer et comparer les 3 modèles (Phi + TinyLlama + Qwen)")
    parser.add_argument("--both",        action="store_true",
                        help="Évaluer et comparer Phi vs TinyLlama (compatibilité ancienne version)")
    parser.add_argument("--model",       choices=["phi", "tinyllama", "qwen"], default="phi",
                        help="Évaluer un seul modèle")
    parser.add_argument("--n",           type=int, default=N_QUESTIONS,
                        help="Nombre de chunks à évaluer (défaut : tout)")
    parser.add_argument("--no-injected", action="store_true",
                        help="Ne pas injecter les questions hors-sujet")
    parser.add_argument("--test-file",   default=TEST_FILE)
    parser.add_argument("--phi-url",     default=API_PHI)
    parser.add_argument("--tiny-url",    default=API_TINYLLAMA)
    parser.add_argument("--qwen-url",    default=API_QWEN,
                        help=f"URL de l'API Qwen1.5-7B (défaut: {API_QWEN})")
    parser.add_argument("--output",      default=OUTPUT_REPORT)
    args = parser.parse_args()

    print("\n" + "="*72)
    print("  Évaluation Chatbot Tunisie Telecom — PFE")
    print("  (Dataset : chunks texte brut)")
    print("="*72)
    print(f"  Phi-3.5      : {args.phi_url}")
    print(f"  TinyLlama    : {args.tiny_url}")
    print(f"  Qwen1.5-7B   : {args.qwen_url}  ← NOUVEAU")
    print(f"  Dataset      : {args.test_file}")
    print(f"  Rapport      : {args.output}")
    print("="*72)

    try:
        import rouge_score  # noqa
        print("  [OK] rouge-score installé")
    except ImportError:
        print("  [WARN] rouge-score non installé — pip install rouge-score")

    test_data = load_test_dataset(
        args.test_file,
        n=args.n,
        add_outscope=not args.no_injected,
    )

    print(f"\n  Aperçu des questions générées :")
    for ex in test_data[:3]:
        if ex["is_telecom"]:
            print(f"    [{ex['file_name']}]")
            print(f"    → {ex['question']}")
            print(f"    Mots-clés : {ex['source_keywords']}\n")

    final = {}

    # ─────────────────────────────────────────
    # Cas 1 : --all  → évaluer les 3 modèles
    # ─────────────────────────────────────────
    if args.all:
        r_phi  = evaluate_model(test_data, args.phi_url,  "Phi-3.5-mini-FT")
        r_tiny = evaluate_model(test_data, args.tiny_url, "TinyLlama-1.1B-FT")
        r_qwen = evaluate_model(test_data, args.qwen_url, "Qwen1.5-7B-FT")

        valid = {k: v for k, v in {
            "phi":  r_phi,
            "tiny": r_tiny,
            "qwen": r_qwen,
        }.items() if v}

        # Comparaisons 2 à 2 si au moins 2 modèles ont répondu
        available = list(valid.values())
        if len(available) >= 2:
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    compare_two(available[i], available[j])

        # Comparatif 3 colonnes si tous disponibles
        if len(available) == 3:
            compare_three(r_phi, r_tiny, r_qwen)

        # Classement global simplifié si certains modèles sont absents
        if len(available) == 2:
            m1, m2 = available
            winner = m1["model_name"] if m1["global_score"] >= m2["global_score"] else m2["model_name"]
            delta  = abs(m1["global_score"] - m2["global_score"])
            print(f"\n  Meilleur modèle : {BOLD}{winner}{RESET}  (+{delta:.4f})")

        ranked = sorted(valid.values(), key=lambda x: x["global_score"], reverse=True)
        final = {
            "phi35":       r_phi,
            "tinyllama":   r_tiny,
            "qwen15":      r_qwen,
            "ranking": [
                {"rank": i+1, "model": r["model_name"],
                 "score": r["global_score"], "grade": r["grade"]}
                for i, r in enumerate(ranked)
            ],
            "winner":       ranked[0]["model_name"] if ranked else "",
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        }

    # ─────────────────────────────────────────
    # Cas 2 : --both  → Phi vs TinyLlama (compat.)
    # ─────────────────────────────────────────
    elif args.both:
        r_phi  = evaluate_model(test_data, args.phi_url,  "Phi-3.5-mini-FT")
        r_tiny = evaluate_model(test_data, args.tiny_url, "TinyLlama-1.1B-FT")
        if r_phi and r_tiny:
            compare_two(r_phi, r_tiny)
            final = {
                "phi35":     r_phi,
                "tinyllama": r_tiny,
                "comparison": {
                    "winner":     "Phi-3.5-mini-FT" if r_phi["global_score"] >= r_tiny["global_score"] else "TinyLlama-1.1B-FT",
                    "phi_score":  r_phi["global_score"],
                    "tiny_score": r_tiny["global_score"],
                    "delta":      abs(r_phi["global_score"] - r_tiny["global_score"]),
                },
                "generated_at": __import__("datetime").datetime.now().isoformat(),
            }
        else:
            final = {"phi35": r_phi, "tinyllama": r_tiny}

    # ─────────────────────────────────────────
    # Cas 3 : --model <nom>  → un seul modèle
    # ─────────────────────────────────────────
    elif args.model == "tinyllama":
        final = evaluate_model(test_data, args.tiny_url, "TinyLlama-1.1B-FT") or {}
    elif args.model == "qwen":
        final = evaluate_model(test_data, args.qwen_url, "Qwen1.5-7B-FT") or {}
    else:
        final = evaluate_model(test_data, args.phi_url, "Phi-3.5-mini-FT") or {}

    if final:
        os.makedirs(
            os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
            exist_ok=True
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        print(f"\n  Rapport sauvegardé : {args.output}\n")
    else:
        print(f"\n  Aucun résultat à sauvegarder.\n")


if __name__ == "__main__":
    main()