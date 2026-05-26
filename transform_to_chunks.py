"""
transform_to_chunks.py
======================
Transforme Extract_Doc_TT_Biblio_clean.json → chunks.json

Structure de l'entrée (par document) :
  {
    "year": "2018", "month": "01",
    "filename": "FCOffreHayya.pdf",
    "text": "...",
    "tables": [...],
    "chunks": [{"chunk_index": 0, "chunk_text": "..."}, ...]
  }

Structure de la sortie (un objet par chunk) :
  {
    "chunk_id": "<sha1>",
    "source": "app/data/data/2018/01/FCOffreHayya.pdf",
    "file_name": "FCOffreHayya.pdf",
    "year": "2018", "month": "01",
    "document_type": "pdf",
    "chunk_index": 0,
    "total_chunks": 4,
    "confidentiality": "shareable" | "confidential",
    "confidentiality_reason": "rule:<kw>" | "hard_rule:<kw>" | "path_marker:<kw>" | "default:shareable",
    "shareable_with_client": true | false,
    "chunk_text": "..."
  }

Usage :
    python transform_to_chunks.py \\
        --input  Extract_Doc_TT_Biblio_clean.json \\
        --output chunks.json \\
        [--base-path app/data/data]
"""

import json
import hashlib
import re
import argparse
from pathlib import Path


# ══════════════════════════════════════════════════════════
#  Règles de confidentialité
# ══════════════════════════════════════════════════════════

# 1. Hard rules → toujours confidentiel (priorité absolue)
HARD_RULES_CONFIDENTIAL = [
    "argumentaire",
    "benchmark",
    "procedure",
    "segmentation",
    "use case",
]

# 2. Règles par mot-clé (cherchées dans le nom de fichier puis dans le texte)
#    Ordre : du plus spécifique (multi-mots) au plus générique (1 mot)
#    Le premier match l'emporte.
KEYWORD_RULES = [
    # ── Confidentiel ──────────────────────────────────────
    ("argumentaire de vente", "confidential"),
    ("back office",           "confidential"),
    ("confidentiel",          "confidential"),

    # ── Shareable (mots composés d'abord) ─────────────────
    ("fiche commerciale",     "shareable"),
    ("flash commercial",      "shareable"),
    ("fiche produit",         "shareable"),
    ("roaming",               "shareable"),
    ("internet",              "shareable"),
    ("tarifaire",             "shareable"),
    ("validite",              "shareable"),
    ("activation",            "shareable"),
    ("promotion",             "shareable"),
    ("tourist",               "shareable"),
    ("netbox",                "shareable"),
    ("mobile",                "shareable"),
    ("offres",                "shareable"),
    ("forfaits",              "shareable"),
    ("packs",                 "shareable"),
    ("promo",                 "shareable"),
    ("forfait",               "shareable"),
    ("tarif",                 "shareable"),
    ("pack",                  "shareable"),
    ("offre",                 "shareable"),
    ("client",                "shareable"),
    ("clients",               "shareable"),
    ("prix",                  "shareable"),
    ("adsl",                  "shareable"),
    ("vdsl",                  "shareable"),
    ("gpon",                  "shareable"),
    ("esim",                  "shareable"),
    ("sim",                   "shareable"),
    ("4g",                    "shareable"),
]

# 3. Marqueurs dans le chemin du dossier (pas dans le nom de fichier)
PATH_MARKERS = ["clients", "client"]


def _normalize(text: str) -> str:
    """Remplace tirets/underscores par des espaces et met en minuscules."""
    return re.sub(r"[-_]", " ", text.lower())


def _classify_chunk(source_path: str, chunk_text: str) -> tuple[str, str]:
    """
    Retourne (confidentiality, confidentiality_reason).

    Priorité :
      1. hard_rule   : mot-clé confidentiel dans le nom de fichier ou le texte
      2. path_marker : marqueur dans le dossier parent
      3. keyword     : premier mot-clé matchant (nom de fichier, puis texte)
      4. default     : shareable
    """
    fname_norm  = _normalize(Path(source_path).name)
    text_norm   = _normalize(chunk_text)
    folder_norm = _normalize(source_path.replace(Path(source_path).name, ""))

    # 1. Hard rules
    for kw in HARD_RULES_CONFIDENTIAL:
        if kw in fname_norm or kw in text_norm:
            return "confidential", f"hard_rule:{kw}"

    # 2. Path markers (cherche dans le dossier, pas dans le nom du fichier)
    for marker in PATH_MARKERS:
        if marker in folder_norm:
            return "shareable", f"path_marker:{marker}"

    # 3. Keyword rules — nom de fichier en priorité, puis texte
    for kw, verdict in KEYWORD_RULES:
        if kw in fname_norm:
            return verdict, f"rule:{kw}"

    for kw, verdict in KEYWORD_RULES:
        if kw in text_norm:
            return verdict, f"rule:{kw}"

    # 4. Default
    return "shareable", "default:shareable"


def _generate_chunk_id(source_path: str, chunk_index: int, chunk_text: str) -> str:
    """SHA-1 reproductible : source | index | texte."""
    raw = f"{source_path}|{chunk_index}|{chunk_text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════
#  Transformation principale
# ══════════════════════════════════════════════════════════

def transform(input_path: str, output_path: str, base_path: str = "app/data/data") -> None:
    with open(input_path, encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []

    for doc in documents:
        year     = doc.get("year", "")
        month    = doc.get("month", "")
        filename = doc.get("filename", "")
        chunks   = doc.get("chunks", [])

        source       = f"{base_path}/{year}/{month}/{filename}"
        doc_type     = Path(filename).suffix.lstrip(".").lower() or "pdf"
        total_chunks = len(chunks)

        for chunk in chunks:
            chunk_index = chunk.get("chunk_index", 0)
            chunk_text  = chunk.get("chunk_text", "")

            confidentiality, reason = _classify_chunk(source, chunk_text)
            chunk_id = _generate_chunk_id(source, chunk_index, chunk_text)

            all_chunks.append({
                "chunk_id":               chunk_id,
                "source":                 source,
                "file_name":              filename,
                "year":                   year,
                "month":                  month,
                "document_type":          doc_type,
                "chunk_index":            chunk_index,
                "total_chunks":           total_chunks,
                "confidentiality":        confidentiality,
                "confidentiality_reason": reason,
                "shareable_with_client":  confidentiality == "shareable",
                "chunk_text":             chunk_text,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    n_share = sum(1 for c in all_chunks if c["confidentiality"] == "shareable")
    n_conf  = len(all_chunks) - n_share

    print(f"✓  {len(all_chunks)} chunks écrits dans « {output_path} »")
    print(f"   shareable    : {n_share}")
    print(f"   confidential : {n_conf}")


# ══════════════════════════════════════════════════════════
#  Point d'entrée CLI
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transforme Extract_Doc_TT_Biblio_clean.json → chunks.json"
    )
    parser.add_argument(
        "--input",  default="Extract_Doc_TT_Biblio_clean.json",
        help="Chemin du fichier source (défaut : Extract_Doc_TT_Biblio_clean.json)"
    )
    parser.add_argument(
        "--output", default="chunks.json",
        help="Chemin du fichier de sortie (défaut : chunks.json)"
    )
    parser.add_argument(
        "--base-path", default="app/data/data",
        help="Préfixe du chemin source dans le champ « source » (défaut : app/data/data)"
    )
    args = parser.parse_args()
    transform(args.input, args.output, args.base_path)