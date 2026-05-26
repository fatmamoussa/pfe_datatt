#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_lora.py — Fusion LoRA Phi-3.5 corrigée
Fix : AttributeError 'list' object has no attribute 'keys'
      dans transformers >= 4.47 avec Phi-3.5 tied weights
"""

import os, gc, torch
from pathlib import Path

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
LORA_DIR   = Path("models/phi35-tt-lora")
MERGED_DIR = Path("models/phi35-tt-merged")

def main():
    print("=" * 55)
    print("  Fusion LoRA Phi-3.5 — version corrigée")
    print("=" * 55)

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if not LORA_DIR.exists():
        raise SystemExit(f"LoRA introuvable : {LORA_DIR.resolve()}")

    print(f"\n  LoRA     : {LORA_DIR.resolve()}")
    print(f"  Sortie   : {MERGED_DIR.resolve()}")

    # Tokenizer
    print("\n[1/4] Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(LORA_DIR), trust_remote_code=True
    )
    print(f"  OK — vocab={tokenizer.vocab_size:,}")

    # Modèle de base sur CPU
    print("\n[2/4] Chargement Phi-3.5 base sur CPU (~10 min)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    print("  Modèle de base chargé.")

    # Fusion
    print("\n[3/4] Fusion LoRA...")
    peft_model = PeftModel.from_pretrained(base, str(LORA_DIR))
    merged     = peft_model.merge_and_unload()
    print("  Fusion OK.")

    # ----------------------------------------------------------------
    # FIX : transformers >= 4.47 — _tied_weights_keys est une liste
    # pour Phi-3.5, ce qui casse remove_tied_weights_from_state_dict.
    # On la remplace par un dict vide (= aucun poids lié à traiter).
    # ----------------------------------------------------------------
    if isinstance(getattr(merged, "_tied_weights_keys", None), list):
        print("  Patch _tied_weights_keys (list → {}) appliqué.")
        merged._tied_weights_keys = {}

    # Sauvegarde
    print(f"\n[4/4] Sauvegarde → {MERGED_DIR}...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    # Vérification
    files          = list(MERGED_DIR.iterdir())
    safetensors    = [f for f in files if f.suffix == ".safetensors"]
    tokenizer_files = [f for f in files if "token" in f.name.lower()]

    print(f"\n  Fichiers dans {MERGED_DIR}:")
    for f in sorted(files):
        print(f"    {f.name:40s}  {f.stat().st_size/1e6:.1f} MB")

    if safetensors and tokenizer_files:
        print(f"\n  ✅ Fusion réussie — {len(safetensors)} fichier(s) .safetensors")
    else:
        print("\n  ⚠ Sauvegarde incomplète — vérifier les fichiers ci-dessus")

    print("\n  Lancer l'API :")
    print("    python api_v5.py")
    print("=" * 55)

if __name__ == "__main__":
    main()