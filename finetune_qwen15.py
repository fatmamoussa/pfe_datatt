#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
finetune_qwen15_1b8.py — Fine-tune + Fusion Qwen1.5-1.8B-Chat
Corpus : Tunisie Telecom  |  GPU : RTX 2070 8 GB
Prec.  : fp16 direct (1.8B tient largement dans 8 GB)
         QLoRA 4-bit optionnel (--qlora) pour économiser la VRAM
Data   : data/train.json + data/val.json

POURQUOI 1.8B au lieu de 7B ?
  - Qwen1.5-7B-Chat  : ~14 GB VRAM fp16, ~5 GB QLoRA
                        crash au chargement sur RTX 2070 (8.6 GB)
  - Qwen1.5-1.8B-Chat : ~4 GB VRAM fp16, ~1.5 GB QLoRA
                         entraînement rapide (~30-45 min sur RTX 2070)
                         qualité suffisante pour RAG (le contexte fait
                         la majorité du travail)

UTILISATION :
  # Mode recommandé : fp16 pur (rapide, stable)
  python finetune_qwen15_1b8.py

  # Mode QLoRA 4-bit (économise encore plus la VRAM)
  python finetune_qwen15_1b8.py --qlora

  # Fusion seulement (après crash)
  python finetune_qwen15_1b8.py --merge-only

SORTIES :
  models/qwen15-1b8-tt-lora/    -> adaptateur LoRA
  models/qwen15-1b8-tt-merged/  -> modèle fusionné prêt pour l'API

DIFFÉRENCES vs finetune_qwen15.py (7B) :
  - BASE_MODEL     : Qwen/Qwen1.5-1.8B-Chat  (1.8B vs 7B)
  - fp16 par défaut (pas besoin de QLoRA pour tenir dans 8 GB)
  - LORA_R         : 32  (1.8B = moins de paramètres)
  - BATCH_SIZE     : 2   GRAD_ACCUM : 4  (batch effectif = 8)
  - MAX_SEQ_LENGTH : 512
  - Durée          : ~30-45 min sur RTX 2070 8 GB
  - RAM fusion     : ~4-6 GB CPU (fp16, 1.8B) — très léger
  - Téléchargement : ~3.5 GB (vs 15 GB pour 7B)
=============================================================
"""

import os, gc, sys, json, torch, functools, argparse
from pathlib import Path
from datetime import datetime

# Désactiver torch.compile (Windows / PyTorch 2.6)
os.environ["TORCHDYNAMO_DISABLE"]   = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()
except Exception:
    pass

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# =============================================================
# CONFIG — Qwen1.5-1.8B-Chat
# =============================================================
BASE_MODEL  = "Qwen/Qwen1.5-1.8B-Chat"
DATA_DIR    = Path("data")
TRAIN_FILE  = DATA_DIR / "train.json"
VAL_FILE    = DATA_DIR / "val.json"
OUTPUT_DIR  = Path("models/qwen15-1b8-tt-lora")
MERGED_DIR  = Path("models/qwen15-1b8-tt-merged")

# LoRA — valeurs adaptées pour 1.8B
LORA_R         = 32          # 7B utilisait 64
LORA_ALPHA     = 64          # = 2 * LORA_R
LORA_DROPOUT   = 0.05
LORA_TARGET    = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

NUM_EPOCHS     = 3
BATCH_SIZE     = 2           # 7B utilisait 1 — 1.8B permet plus
GRAD_ACCUM     = 4           # batch effectif = 8
LEARNING_RATE  = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO   = 0.05
SAVE_STEPS     = 100
EVAL_STEPS     = 100
LOGGING_STEPS  = 10

# Tokens spéciaux ChatML Qwen1.5 (identiques toutes versions)
IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"

# =============================================================
# UTILITAIRES
# =============================================================

def check_gpu(use_qlora: bool):
    print("\n" + "=" * 60)
    print("  Vérification GPU")
    print("=" * 60)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA non disponible.")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU  : {name}")
    print(f"  VRAM : {vram:.1f} GB")
    if use_qlora:
        print(f"  Mode : QLoRA 4-bit — Qwen1.5-1.8B ~1.5 GB VRAM")
    else:
        print(f"  Mode : fp16 pur    — Qwen1.5-1.8B ~4 GB VRAM")
        if vram < 4:
            print(f"  ATTENTION : {vram:.1f} GB < 4 GB — utiliser --qlora")
    print(f"  Modèle : {BASE_MODEL} (1.8B params)")
    print(f"  Téléchargement : ~3.5 GB (si pas encore en cache)")


def detect_bnb() -> bool:
    try:
        import bitsandbytes as bnb
        version = getattr(bnb, "__version__", "?")
        print(f"  bitsandbytes {version} détecté — QLoRA disponible")
        return True
    except ImportError:
        print("  bitsandbytes absent — fp16 pur utilisé")
        return False


def format_chatml(record: dict) -> str:
    """
    Convertit un enregistrement en chaîne ChatML pour Qwen1.5.
    Supporte 3 formats :
      A) {"text": "..."}               → retourne tel quel
      B) {"messages": [...]}           → construit ChatML
      C) {"system":..,"user":..,"assistant":..} → construit ChatML
    """
    if "text" in record:
        return record["text"].strip()

    if "messages" in record:
        parts = []
        for msg in record["messages"]:
            role    = msg.get("role", "user")
            content = msg.get("content", "").strip()
            parts.append(f"{IM_START}{role}\n{content}\n{IM_END}")
        return "\n".join(parts)

    parts = []
    if "system" in record:
        parts.append(f"{IM_START}system\n{record['system'].strip()}\n{IM_END}")
    if "user" in record:
        parts.append(f"{IM_START}user\n{record['user'].strip()}\n{IM_END}")
    if "assistant" in record:
        parts.append(f"{IM_START}assistant\n{record['assistant'].strip()}\n{IM_END}")

    if parts:
        return "\n".join(parts)

    raise ValueError(f"Format de record non reconnu : {list(record.keys())}")


def load_data(path: Path) -> list:
    """Charge JSON array ou JSONL, filtre les textes trop courts."""
    records = []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        for obj in json.loads(content):
            try:
                text = format_chatml(obj)
                if len(text.strip()) > 20:
                    records.append({"text": text})
            except ValueError as e:
                print(f"   WARN : record ignoré — {e}")
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj  = json.loads(line)
                text = format_chatml(obj)
                if len(text.strip()) > 20:
                    records.append({"text": text})
            except (json.JSONDecodeError, ValueError) as e:
                print(f"   WARN : ligne ignorée — {e}")
    if not records:
        raise ValueError(f"Aucun exemple valide dans {path}")
    return records


class CLMDataset(torch.utils.data.Dataset):
    """Dataset Causal LM pour Qwen1.5-1.8B."""
    def __init__(self, records, tokenizer, max_length):
        self.examples = []
        eos = tokenizer.eos_token or "<|endoftext|>"
        skipped = 0
        for r in records:
            text = r["text"].strip() + eos
            enc  = tokenizer(
                text, truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )
            ids = enc["input_ids"]
            if len(ids) > 5:
                self.examples.append(ids)
            else:
                skipped += 1
        if skipped:
            print(f"   INFO : {skipped} exemples ignorés (trop courts après tokenisation)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return {
            "input_ids"     : torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels"        : torch.tensor(ids, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    """Padding dynamique à droite."""
    max_len        = max(x["input_ids"].size(0) for x in batch)
    input_ids      = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels         = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].size(0)
        input_ids[i, :n]      = x["input_ids"]
        attention_mask[i, :n] = x["attention_mask"]
        labels[i, :n]         = x["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# =============================================================
# ÉTAPE 1 : FINE-TUNE
# =============================================================

def run_finetune(use_qlora: bool = False):
    print("\n" + "=" * 60)
    print("  ÉTAPE 1 — FINE-TUNE Qwen1.5-1.8B-Chat")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    check_gpu(use_qlora)

    # 1. Tokenizer
    print(f"\n[1/6] Tokenizer : {BASE_MODEL}")
    print("      (téléchargement ~3.5 GB si pas encore en cache)")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"      EOS={tokenizer.eos_token!r}  PAD={tokenizer.pad_token!r}  "
          f"Vocab={tokenizer.vocab_size:,}")
    print(f"      Tokens ChatML : {IM_START!r} / {IM_END!r}")

    # 2. Modèle
    print(f"\n[2/6] Chargement Qwen1.5-1.8B-Chat "
          f"({'QLoRA 4-bit' if use_qlora else 'fp16 pur'})...")

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=torch.float16,          # nouveau nom (torch_dtype déprécié)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
        print("      Qwen1.5-1.8B chargé en 4-bit NF4.")
    else:
        # fp16 pur — stratégie CPU-first pour éviter le pic VRAM au chargement
        # 1) Charger sur CPU en fp16 (pas de pic GPU)
        # 2) Déplacer sur GPU après — ~4 GB VRAM stable
        print("      Chargement sur CPU d'abord (évite le pic VRAM)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,          # nouveau nom (torch_dtype déprécié)
            device_map="cpu",             # CPU d'abord — pas de pic GPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("      Déplacement vers GPU...")
        model = model.to("cuda:0")
        print("      Qwen1.5-1.8B chargé en fp16.")

    model.config.use_cache      = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_disable()

    vram_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"      VRAM occupée : ~{vram_used:.1f} GB")

    # 3. LoRA
    print(f"\n[3/6] LoRA : r={LORA_R}, alpha={LORA_ALPHA}")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model     = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"      Params entraînables : {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # 4. Datasets
    print(f"\n[4/6] Chargement données...")
    train_raw = load_data(TRAIN_FILE)
    val_raw   = load_data(VAL_FILE)
    print(f"      Train : {len(train_raw)} exemples")
    print(f"      Val   : {len(val_raw)} exemples")
    print(f"      Exemple (début) :")
    print(f"        {train_raw[0]['text'][:120]!r}...")

    train_dataset = CLMDataset(train_raw, tokenizer, MAX_SEQ_LENGTH)
    val_dataset   = CLMDataset(val_raw,   tokenizer, MAX_SEQ_LENGTH)
    print(f"      Train tokenisé : {len(train_dataset)}")
    print(f"      Val   tokenisé : {len(val_dataset)}")

    n_sample    = min(100, len(train_dataset))
    sample_lens = [len(train_dataset.examples[i]) for i in range(n_sample)]
    print(f"      Tokens — moy : {sum(sample_lens)/n_sample:.0f} | "
          f"max : {max(sample_lens)} | "
          f"tronqués : {sum(1 for l in sample_lens if l>=MAX_SEQ_LENGTH)/n_sample*100:.1f}%")

    collator = functools.partial(collate_fn, pad_id=tokenizer.pad_token_id)

    # 5. TrainingArguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM))

    print(f"\n[5/6] Configuration entraînement")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=int(WARMUP_RATIO * (NUM_EPOCHS * len(train_dataset)
                                         // (BATCH_SIZE * GRAD_ACCUM))),
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        # 1.8B fp16 → adamw_torch suffit (pas besoin de paged_adamw_8bit)
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        torch_compile=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # 6. Lancement
    print(f"\n[6/6] Lancement entraînement")
    print("=" * 60)
    print(f"  Modèle         : Qwen1.5-1.8B-Chat")
    print(f"  Mode           : {'QLoRA 4-bit NF4' if use_qlora else 'fp16 pur'}")
    print(f"  Epochs         : {NUM_EPOCHS}")
    print(f"  Batch effectif : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  LR             : {LEARNING_RATE}")
    print(f"  Optimiseur     : {'paged_adamw_8bit (QLoRA)' if use_qlora else 'adamw_torch'}")
    print(f"  Steps/epoch    : ~{steps_per_epoch}")
    print(f"  Durée estimée  : ~30-45 min sur RTX 2070 8 GB")
    print("-" * 60)

    trainer.train()

    # Sauvegarde LoRA
    print(f"\n  Sauvegarde LoRA -> {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("  LoRA sauvegardé.")

    # Libération GPU avant fusion
    print("\n  Libération GPU avant fusion...")
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("  GPU libéré.")

    return tokenizer


# =============================================================
# ÉTAPE 2 : FUSION LoRA -> MODÈLE COMPLET
# =============================================================

def run_merge(tokenizer=None):
    """
    Fusionne le LoRA avec Qwen1.5-1.8B de base.
    Très léger : ~4-6 GB RAM CPU seulement (vs 16-20 GB pour le 7B).
    """
    print("\n" + "=" * 60)
    print("  ÉTAPE 2 — FUSION LoRA + Qwen1.5-1.8B base")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not OUTPUT_DIR.exists():
        raise SystemExit(
            f"\n  LoRA introuvable : {OUTPUT_DIR.resolve()}\n"
            "  Lance d'abord : python finetune_qwen15_1b8.py\n"
        )

    if tokenizer is None:
        print(f"\n  Chargement tokenizer depuis {OUTPUT_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(OUTPUT_DIR), trust_remote_code=True
        )
        print("  OK.")

    print(f"\n  LoRA source  : {OUTPUT_DIR.resolve()}")
    print(f"  Destination  : {MERGED_DIR.resolve()}")
    print(f"\n  Chargement Qwen1.5-1.8B de base sur CPU...")
    print("  (~4-6 GB RAM, 1-3 minutes)")

    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("  Modèle de base chargé sur CPU.")

        print("\n  Fusion LoRA en cours...")
        merged = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
        merged = merged.merge_and_unload()
        print("  Fusion terminée.")

        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  Sauvegarde -> {MERGED_DIR}")
        merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
        tokenizer.save_pretrained(str(MERGED_DIR))

        if (MERGED_DIR / "config.json").exists():
            size_gb = sum(
                f.stat().st_size for f in MERGED_DIR.glob("*.safetensors")
            ) / 1e9
            print(f"  Modèle fusionné OK -> {MERGED_DIR.resolve()}")
            print(f"  Taille : {size_gb:.1f} GB")
            return True
        else:
            print("  ATTENTION : config.json absent — sauvegarde incomplète")
            return False

    except MemoryError:
        print("\n  ERREUR : RAM insuffisante (~4 GB requis pour Qwen1.5-1.8B)")
        print("  Ferme les autres programmes et relance :")
        print("    python finetune_qwen15_1b8.py --merge-only")
        return False

    except Exception as e:
        print(f"\n  Fusion échouée : {e}")
        print(f"  LoRA disponible dans : {OUTPUT_DIR.resolve()}")
        print("  Retenter avec : python finetune_qwen15_1b8.py --merge-only")
        return False


# =============================================================
# ÉTAPE 3 : TEST RAPIDE
# =============================================================

def run_test():
    """Test de génération sur le modèle fusionné — format ChatML."""
    print("\n" + "=" * 60)
    print("  ÉTAPE 3 — TEST DE GÉNÉRATION (ChatML)")
    print("=" * 60)

    if not MERGED_DIR.exists():
        print(f"  Modèle fusionné introuvable : {MERGED_DIR} — test ignoré.")
        return

    print(f"\n  Chargement depuis {MERGED_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MERGED_DIR), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(MERGED_DIR),
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        prompts = [
            "Quelles sont les offres Hayya de Tunisie Telecom ?",
            "Comment activer le roaming international ?",
            "Quels sont les avantages de l'abonnement fibra ?",
        ]

        system_msg = (
            "Tu es un assistant commercial expert de Tunisie Telecom. "
            "Réponds en français de manière concise et précise."
        )

        for question in prompts:
            chat_prompt = (
                f"{IM_START}system\n{system_msg}\n{IM_END}\n"
                f"{IM_START}user\n{question}\n{IM_END}\n"
                f"{IM_START}assistant\n"
            )
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            print(f"\n  Q : {question}")
            print(f"  R : {generated}")

        print("\n  Test OK.")

    except Exception as e:
        print(f"  Test échoué : {e}")
        print("  (Le modèle reste utilisable dans l'API)")


# =============================================================
# POINT D'ENTRÉE
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune + Fusion Qwen1.5-1.8B-Chat — Corpus Tunisie Telecom"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Sauter le fine-tune, fusionner uniquement le LoRA existant",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Activer QLoRA 4-bit (facultatif pour 1.8B, économise la VRAM)",
    )
    args = parser.parse_args()

    # QLoRA uniquement si demandé ET bitsandbytes disponible
    use_qlora = False
    if args.qlora:
        use_qlora = detect_bnb()
        if not use_qlora:
            print("  bitsandbytes absent — fp16 pur utilisé à la place")

    print("=" * 60)
    print("  finetune_qwen15_1b8.py — Qwen1.5-1.8B-Chat")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mode_str = "QLoRA 4-bit NF4" if use_qlora else "fp16 pur (recommandé)"
    print(f"  Mode précision : {mode_str}")
    print("=" * 60)

    tokenizer = None

    if args.merge_only:
        print("\n  Mode : --merge-only (fusion du LoRA existant)")
        merge_ok = run_merge(tokenizer=None)
    else:
        print("\n  Mode : complet (fine-tune + fusion)")
        tokenizer = run_finetune(use_qlora=use_qlora)
        merge_ok  = run_merge(tokenizer=tokenizer)

    if merge_ok:
        run_test()

    # Résumé
    print("\n" + "=" * 60)
    print("  RÉSUMÉ")
    print("=" * 60)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  LoRA     : {OUTPUT_DIR.resolve()}")
    if merge_ok:
        print(f"  Fusionné : {MERGED_DIR.resolve()}")
        print("  État     : SUCCÈS")
    else:
        print("  Fusion   : ÉCHEC — retenter avec --merge-only")
        print("  État     : LoRA disponible, fusion à retenter")

    print()
    print("  Pour utiliser dans l'API (api_qwen15.py) :")
    print("  → Modifier api_qwen15.py :")
    if merge_ok:
        print(f"      MODEL_DIR  = '{MERGED_DIR.resolve()}'")
    print(f"      BASE_MODEL = '{BASE_MODEL}'")
    print()
    print("  Prompt ChatML (identique au 7B) :")
    print(f"    {IM_START}system\\n<system_msg>\\n{IM_END}")
    print(f"    {IM_START}user\\n<question>\\n{IM_END}")
    print(f"    {IM_START}assistant\\n")
    print()
    print("  Comparaison modèles :")
    print("  ┌─────────────────────────┬──────────┬───────────┬─────────────┐")
    print("  │ Modèle                  │ VRAM fp16│ Durée FT  │ RAM fusion  │")
    print("  ├─────────────────────────┼──────────┼───────────┼─────────────┤")
    print("  │ Qwen1.5-1.8B-Chat  ✓   │  ~4 GB   │ ~30-45min │   ~4 GB     │")
    print("  │ Qwen1.5-7B-Chat    ✗   │  ~14 GB  │ ~2-4h     │  ~16-20 GB  │")
    print("  │ TinyLlama-1.1B         │  ~3 GB   │ ~20-30min │   ~3 GB     │")
    print("  └─────────────────────────┴──────────┴───────────┴─────────────┘")
    print("  ✓ = recommandé pour RTX 2070 8 GB")
    print("=" * 60)


if __name__ == "__main__":
    main()