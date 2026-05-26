"""
build_chromadb_index.py
=======================
Indexation des chunks train / val / test dans ChromaDB.

Structure attendue du dossier :
    data/
    ├── train.json
    ├── val.json
    └── test.json

Chaque fichier est une liste de dicts avec au minimum :
    {
        "chunk_id"   : str,
        "text"       : str,
        "file_name"  : str,
        "year"       : str,
        "month"      : str,
        "chunk_index": int
    }

Collections créées dans ChromaDB :
    tt_train  ←  train.json
    tt_val    ←  val.json
    tt_test   ←  test.json

Modèle d'encodage :
    sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    (multilingue, bon pour le français, ~420 MB)

Lancement :
    pip install chromadb sentence-transformers
    python build_chromadb_index.py

    # Pour changer le dossier data :
    python build_chromadb_index.py --data_dir ./mon_dossier

    # Pour changer le dossier de la base ChromaDB :
    python build_chromadb_index.py --chroma_dir ./ma_chroma_db
"""

import os
import json
import time
import argparse
from pathlib import Path

# ── dépendances ──────────────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "\n[ERREUR] Installe d'abord les dépendances :\n"
        "  pip install chromadb sentence-transformers\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ENCODER_MODEL  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE     = 64        # chunks encodés en parallèle (adapter selon RAM/GPU)
COLLECTION_MAP = {
    "train.json" : "tt_train",
    "val.json"   : "tt_val",
    "test.json"  : "tt_test",
}

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Indexation ChromaDB — Tunisie Télécom")
parser.add_argument("--data_dir",   default="data",        help="Dossier contenant train/val/test.json")
parser.add_argument("--chroma_dir", default="chroma_tt_db",help="Dossier de persistance ChromaDB")
parser.add_argument("--reset",      action="store_true",   help="Supprimer les collections existantes avant indexation")
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
CHROMA_DIR = Path(args.chroma_dir)

# ─────────────────────────────────────────────────────────────────────────────
# VÉRIFICATION FICHIERS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Indexation ChromaDB — Tunisie Télécom")
print("=" * 60)
print(f"\nDossier data     : {DATA_DIR.resolve()}")
print(f"Dossier ChromaDB : {CHROMA_DIR.resolve()}")

missing = [f for f in COLLECTION_MAP if not (DATA_DIR / f).exists()]
if missing:
    raise SystemExit(
        f"\n[ERREUR] Fichiers manquants dans '{DATA_DIR}' :\n"
        + "\n".join(f"  - {f}" for f in missing)
        + f"\n\nAssure-toi que les fichiers sont dans : {DATA_DIR.resolve()}\n"
    )
print("\n✓ Fichiers JSON trouvés : train.json  val.json  test.json")

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE D'ENCODAGE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[1/3] Chargement du modèle d'encodage : {ENCODER_MODEL}")
t0 = time.time()
encoder = SentenceTransformer(ENCODER_MODEL)
print(f"      ✓ Modèle chargé en {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION CHROMADB
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2/3] Initialisation ChromaDB → '{CHROMA_DIR}'")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# FONCTION D'INDEXATION D'UNE COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
def index_collection(json_file: str, collection_name: str):
    """Charge un fichier JSON, encode les textes et les stocke dans ChromaDB."""

    filepath = DATA_DIR / json_file
    print(f"\n  ── {json_file}  →  collection '{collection_name}'")

    # Chargement des chunks
    with open(filepath, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"     Chunks chargés     : {len(chunks)}")

    # Reset collection si demandé
    if args.reset:
        try:
            client.delete_collection(collection_name)
            print(f"     Collection supprimée (--reset)")
        except Exception:
            pass

    # Création / récupération de la collection
    # on_bad_vectors="skip" évite les crashs sur embeddings nuls
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # distance cosinus pour la similarité sémantique
    )

    # Vérifier si déjà indexé (pour éviter les doublons)
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print(f"     ✓ Déjà indexé ({len(existing_ids)} chunks), rien à faire.")
        return

    print(f"     Déjà dans ChromaDB : {len(existing_ids)}")
    print(f"     Nouveaux à indexer : {len(new_chunks)}")

    # Encodage par batch
    texts = [c["text"] for c in new_chunks]
    print(f"     Encodage en cours  ...", end="", flush=True)
    t1 = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    print(f" ✓ ({time.time()-t1:.1f}s)")

    # Insertion dans ChromaDB par batch de 500 (limite ChromaDB)
    CHROMA_BATCH = 500
    inserted = 0
    for start in range(0, len(new_chunks), CHROMA_BATCH):
        batch_chunks = new_chunks[start : start + CHROMA_BATCH]
        batch_embs   = embeddings[start : start + CHROMA_BATCH]

        collection.add(
            ids        = [c["chunk_id"]    for c in batch_chunks],
            documents  = [c["text"]        for c in batch_chunks],
            embeddings = batch_embs.tolist(),
            metadatas  = [
                {
                    "file_name"  : c.get("file_name",   ""),
                    "year"       : c.get("year",        ""),
                    "month"      : c.get("month",       ""),
                    "chunk_index": str(c.get("chunk_index", "")),
                }
                for c in batch_chunks
            ],
        )
        inserted += len(batch_chunks)
        print(f"     Inséré {inserted}/{len(new_chunks)}...", end="\r")

    total = collection.count()
    print(f"     ✓ Collection '{collection_name}' : {total} chunks au total.          ")

# ─────────────────────────────────────────────────────────────────────────────
# INDEXATION DES 3 SPLITS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Indexation des collections")
t_total = time.time()

for json_file, col_name in COLLECTION_MAP.items():
    index_collection(json_file, col_name)

print(f"\n{'='*60}")
print(f"  ✓ Indexation terminée en {time.time()-t_total:.1f}s")
print(f"  Base ChromaDB sauvegardée dans : {CHROMA_DIR.resolve()}")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST RAPIDE DE RECHERCHE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TEST] Recherche sémantique sur tt_train ...")

test_query = "Quelles sont les offres roaming disponibles ?"
q_emb = encoder.encode([test_query]).tolist()

train_col = client.get_collection("tt_train")
results   = train_col.query(query_embeddings=q_emb, n_results=3)

print(f"  Requête : « {test_query} »")
print(f"  Résultats :")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
)):
    print(f"\n  [{i+1}] score={1-dist:.3f} | {meta['file_name']} ({meta['year']})")
    print(f"       {doc[:160]}...")

print("\n✓ Test OK — ChromaDB est prêt.\n")