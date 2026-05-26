"""
split_chunks.py
===============
Transforme chunks.json → train.json / val.json / test.json

Pipeline :
  1. Charger chunks.json
  2. Garder uniquement les chunks « shareable »
  3. Dédupliquer par chunk_text (premier occurrence conservée)
  4. Exclure les chunks trop courts (min_length)
  5. Split aléatoire reproductible : 70 % train / 15 % val / 15 % test

Format de sortie (champs réduits) :
  { chunk_id, file_name, year, month, chunk_index, text }

Usage :
    python split_chunks.py \\
        --input   chunks.json \\
        --out-dir . \\
        [--min-length 50] \\
        [--train-ratio 0.70] \\
        [--val-ratio   0.15] \\
        [--test-ratio  0.15] \\
        [--seed 42]
"""

import json
import random
import argparse
from pathlib import Path


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def filter_and_dedup(chunks: list[dict], min_length: int) -> list[dict]:
    """
    Garde uniquement les chunks shareable, non-dupliqués, et assez longs.

    Ordre des filtres :
      1. confidentialité = 'shareable'
      2. len(chunk_text.strip()) >= min_length
      3. déduplication : si le même texte est déjà vu, on saute
    """
    seen_texts: set[str] = set()
    eligible: list[dict] = []

    for chunk in chunks:
        if chunk.get("confidentiality") != "shareable":
            continue

        text = chunk.get("chunk_text", "").strip()

        if len(text) < min_length:
            continue

        if text in seen_texts:
            continue

        seen_texts.add(text)
        eligible.append(chunk)

    return eligible


def to_output_format(chunk: dict) -> dict:
    """Sélectionne les champs présents dans train/val/test.json."""
    return {
        "chunk_id":    chunk["chunk_id"],
        "file_name":   chunk["file_name"],
        "year":        chunk["year"],
        "month":       chunk["month"],
        "chunk_index": chunk["chunk_index"],
        "text":        chunk["chunk_text"],
    }


def split(pool: list[dict],
          train_ratio: float,
          val_ratio: float,
          test_ratio: float,
          seed: int) -> tuple[list, list, list]:
    """
    Split aléatoire reproductible.
    Les ratios sont normalisés pour sommer à 1.
    """
    total = train_ratio + val_ratio + test_ratio
    train_r = train_ratio / total
    val_r   = val_ratio   / total

    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_r)
    n_val   = round(n * val_r)

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    return train, val, test


def save(data: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([to_output_format(c) for c in data], f,
                  ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère train.json / val.json / test.json depuis chunks.json"
    )
    parser.add_argument("--input",       default="chunks.json",
                        help="Chemin vers chunks.json")
    parser.add_argument("--out-dir",     default=".",
                        help="Dossier de sortie")
    parser.add_argument("--min-length",  type=int, default=50,
                        help="Longueur minimale du chunk_text (défaut : 50)")
    parser.add_argument("--train-ratio", type=float, default=0.70,
                        help="Part des données d'entraînement (défaut : 0.70)")
    parser.add_argument("--val-ratio",   type=float, default=0.15,
                        help="Part de validation (défaut : 0.15)")
    parser.add_argument("--test-ratio",  type=float, default=0.15,
                        help="Part de test (défaut : 0.15)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Graine aléatoire pour la reproductibilité (défaut : 42)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement de « {args.input} »…")
    chunks = load_chunks(args.input)
    print(f"  → {len(chunks)} chunks au total")

    eligible = filter_and_dedup(chunks, args.min_length)
    print(f"  → {len(eligible)} chunks éligibles "
          f"(shareable + len≥{args.min_length} + dédupliqués)")

    train, val, test = split(
        eligible,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.json"
        save(data, str(path))
        pct = len(data) / len(eligible) * 100
        print(f"  ✓ {name}.json : {len(data)} chunks ({pct:.1f} %)")

    print("Terminé.")


if __name__ == "__main__":
    main()