#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
finetune_phi35.py — Fine-tune + Fusion Phi-3.5-mini-instruct
Corpus : Tunisie Telecom  |  GPU : RTX 2070 8 GB
Prec.  : fp16 direct — SANS bitsandbytes (CUDA 12.4 Windows)
Data   : data/train.json + data/val.json  (JSON array, cle "text")

UTILISATION :
  # Mode complet : fine-tune + fusion
  python finetune_phi35.py

  # Mode fusion seulement (si la fusion a echoue precedemment)
  python finetune_phi35.py --merge-only

SORTIES :
  models/phi35-tt-lora/    -> adaptateur LoRA
  models/phi35-tt-merged/  -> modele fusionne pret pour l'API

DIFFERENCES vs finetune_tt.py (TinyLlama) :
  - BASE_MODEL   : Phi-3.5-mini-instruct (3.8B vs 1.1B)
  - BATCH_SIZE   : 1  (VRAM plus chargee)
  - GRAD_ACCUM   : 8  (batch effectif = 8, identique)
  - LORA_R       : 16 (Phi-3.5 a plus de capacite de base)
  - attn_impl    : eager (flash_attention_2 non supporte RTX 2070)
  - Duree        : 1h30-3h (vs 20-40 min TinyLlama)
  - RAM fusion   : ~14-16 GB CPU (vs ~4 GB TinyLlama)
=============================================================
"""

import os, gc, sys, json, torch, functools, argparse
from pathlib import Path
from datetime import datetime

# Desactiver torch.compile (Windows / PyTorch 2.6)
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
)
from peft import LoraConfig, get_peft_model, PeftModel

# =============================================================
# CONFIG
# =============================================================
BASE_MODEL  = "microsoft/Phi-3.5-mini-instruct"
DATA_DIR    = Path("data")
TRAIN_FILE  = DATA_DIR / "train.json"
VAL_FILE    = DATA_DIR / "val.json"
OUTPUT_DIR  = Path("models/phi35-tt-lora")
MERGED_DIR  = Path("models/phi35-tt-merged")

LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGET    = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

NUM_EPOCHS     = 3
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LEARNING_RATE  = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO   = 0.05
SAVE_STEPS     = 100
EVAL_STEPS     = 100
LOGGING_STEPS  = 10

# =============================================================
# UTILITAIRES
# =============================================================

def check_gpu():
    print("\n" + "=" * 60)
    print("  Verification GPU")
    print("=" * 60)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA non disponible.")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU  : {name}")
    print(f"  VRAM : {vram:.1f} GB")
    if vram >= 8:
        print("  OK : 8 GB VRAM — Phi-3.5 fp16 fonctionnel (config serree)")
    else:
        print(f"  ATTENTION : {vram:.1f} GB < 8 GB — reduire MAX_SEQ_LENGTH=384 si OOM")
    print(f"  Modele : {BASE_MODEL} (3.8B params, ~7.6 GB fp16)")


def load_data(path: Path) -> list:
    """Charge JSON array ou JSONL, filtre les textes trop courts."""
    records = []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        for obj in json.loads(content):
            if len(obj.get("text", "").strip()) > 20:
                records.append(obj)
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if len(obj.get("text", "").strip()) > 20:
                    records.append(obj)
            except json.JSONDecodeError:
                pass
    if not records:
        raise ValueError(f"Aucun exemple valide dans {path}")
    return records


class CLMDataset(torch.utils.data.Dataset):
    """
    Dataset Causal LM : <bos> texte <eos>
    Labels = input_ids (prediction du token suivant).
    Note Phi-3.5 : EOS = '<|endoftext|>', BOS = '<s>'
    """
    def __init__(self, records, tokenizer, max_length):
        self.examples = []
        bos = tokenizer.bos_token or "<s>"
        eos = tokenizer.eos_token or "</s>"
        skipped = 0
        for r in records:
            text = f"{bos}{r['text'].strip()}{eos}"
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
            print(f"   INFO : {skipped} exemples ignores (trop courts)")

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
    """Padding dynamique a droite."""
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
# ETAPE 1 : FINE-TUNE
# =============================================================

def run_finetune():
    print("\n" + "=" * 60)
    print("  ETAPE 1 — FINE-TUNE Phi-3.5-mini-instruct")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    check_gpu()

    # 1. Tokenizer
    print(f"\n[1/6] Tokenizer : {BASE_MODEL}")
    print("      (premier lancement = telechargement ~8 GB)")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"      BOS={tokenizer.bos_token!r}  EOS={tokenizer.eos_token!r}  "
          f"PAD={tokenizer.pad_token!r}  Vocab={tokenizer.vocab_size:,}")

    # 2. Modele fp16
    # attn_implementation="eager" : obligatoire RTX 2070 (Turing)
    # flash_attention_2 requiert Ampere (RTX 3000+) ou plus recent
    print(f"\n[2/6] Chargement Phi-3.5 fp16 (~7.6 GB VRAM, 1-2 min)...")
    # FIX : device_map="auto" cause RuntimeError "meta vs cuda:0" avec LoRA
    # Solution : forcer tout le modele sur cuda:0, desactiver low_cpu_mem_usage
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},     # tout sur GPU, pas d'offload meta
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,       # False = pas de meta device partiel
    )
    model.config.use_cache      = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_disable()
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"      Charge. VRAM occupee : ~{vram_used:.1f} GB")

    # 3. LoRA
    print(f"\n[3/6] LoRA : r={LORA_R}, alpha={LORA_ALPHA}")
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, bias="none", task_type="CAUSAL_LM",
    )
    model     = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"      Params entrainables : {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # 4. Datasets
    print(f"\n[4/6] Chargement donnees...")
    train_raw = load_data(TRAIN_FILE)
    val_raw   = load_data(VAL_FILE)
    print(f"      Train : {len(train_raw)} exemples")
    print(f"      Val   : {len(val_raw)} exemples")
    print(f"      Exemple : {train_raw[0]['text'][:80]!r}...")

    train_dataset = CLMDataset(train_raw, tokenizer, MAX_SEQ_LENGTH)
    val_dataset   = CLMDataset(val_raw,   tokenizer, MAX_SEQ_LENGTH)
    print(f"      Train tokenise : {len(train_dataset)}")
    print(f"      Val   tokenise : {len(val_dataset)}")

    # Stats longueur
    n_sample    = min(100, len(train_dataset))
    sample_lens = [len(train_dataset.examples[i]) for i in range(n_sample)]
    print(f"      Tokens — moy : {sum(sample_lens)/n_sample:.0f} | "
          f"max : {max(sample_lens)} | "
          f"tronques : {sum(1 for l in sample_lens if l>=MAX_SEQ_LENGTH)/n_sample*100:.1f}%")

    collator = functools.partial(collate_fn, pad_id=tokenizer.pad_token_id)

    # 5. Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM))

    print(f"\n[5/6] Configuration entrainement")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=int(WARMUP_RATIO * (NUM_EPOCHS * 1462 // (BATCH_SIZE * GRAD_ACCUM))),
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        optim="adamw_torch",
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # 6. Lancement
    print(f"\n[6/6] Lancement entrainement")
    print("=" * 60)
    print(f"  Modele         : Phi-3.5-mini-instruct (3.8B)")
    print(f"  Epochs         : {NUM_EPOCHS}")
    print(f"  Batch effectif : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  LR             : {LEARNING_RATE}")
    print(f"  Steps/epoch    : ~{steps_per_epoch}")
    print(f"  Duree estimee  : 1h30 a 3h sur RTX 2070")
    print(f"  (finetune_tt.py TinyLlama prenait 20-40 min)")
    print("-" * 60)

    trainer.train()

    # Sauvegarde LoRA
    print(f"\n  Sauvegarde LoRA -> {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("  LoRA sauvegarde.")

    # Liberation GPU OBLIGATOIRE avant fusion Phi-3.5
    print("\n  Liberation GPU avant fusion (necessaire pour Phi-3.5)...")
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("  GPU libere.")

    return tokenizer


# =============================================================
# ETAPE 2 : FUSION LoRA -> MODELE COMPLET
# =============================================================

def run_merge(tokenizer=None):
    """
    Fusionne le LoRA avec le modele de base Phi-3.5.

    Appel automatique apres run_finetune(), ou manuel avec --merge-only.
    Necessite ~14-16 GB RAM CPU.
    """
    print("\n" + "=" * 60)
    print("  ETAPE 2 — FUSION LoRA + Phi-3.5 base")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not OUTPUT_DIR.exists():
        raise SystemExit(
            f"\n  LoRA introuvable : {OUTPUT_DIR.resolve()}\n"
            "  Lance d'abord : python finetune_phi35.py\n"
        )

    # Recharge tokenizer si mode standalone
    if tokenizer is None:
        print(f"\n  Chargement tokenizer depuis {OUTPUT_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(OUTPUT_DIR), trust_remote_code=True
        )
        print("  OK.")

    print(f"\n  LoRA source  : {OUTPUT_DIR.resolve()}")
    print(f"  Destination  : {MERGED_DIR.resolve()}")
    print(f"\n  Chargement Phi-3.5 de base sur CPU...")
    print("  (~14-16 GB RAM, 5-10 minutes)")

    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        print("  Modele de base charge.")
        print("\n  Fusion LoRA en cours...")
        merged = PeftModel.from_pretrained(base, str(OUTPUT_DIR)).merge_and_unload()
        print("  Fusion terminee.")

        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  Sauvegarde -> {MERGED_DIR}")
        merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
        tokenizer.save_pretrained(str(MERGED_DIR))

        if (MERGED_DIR / "config.json").exists():
            print(f"  Modele fusionne OK -> {MERGED_DIR.resolve()}")
            return True
        else:
            print("  ATTENTION : config.json absent — sauvegarde incomplete")
            return False

    except MemoryError:
        print("\n  ERREUR : RAM insuffisante pour Phi-3.5 (~14-16 GB requis)")
        print("  Solutions :")
        print("    1. Fermer les autres programmes puis :")
        print("       python finetune_phi35.py --merge-only")
        print("    2. Utiliser TinyLlama (finetune_tt.py, fusion ~4 GB RAM)")
        return False

    except Exception as e:
        print(f"\n  Fusion echouee : {e}")
        print(f"  LoRA disponible dans : {OUTPUT_DIR.resolve()}")
        print("  Retenter avec : python finetune_phi35.py --merge-only")
        return False


# =============================================================
# ETAPE 3 : TEST RAPIDE
# =============================================================

def run_test():
    """Test de generation sur le modele fusionne."""
    print("\n" + "=" * 60)
    print("  ETAPE 3 — TEST DE GENERATION")
    print("=" * 60)

    if not MERGED_DIR.exists():
        print(f"  Modele fusionne introuvable : {MERGED_DIR} — test ignore.")
        return

    print(f"\n  Chargement depuis {MERGED_DIR}...")
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=str(MERGED_DIR),
            tokenizer=str(MERGED_DIR),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        for prompt in [
            "Offre Hayya Tunisie Telecom :",
            "Les clients prepayés peuvent beneficier de",
            "Pour activer le roaming international, il faut",
        ]:
            out = pipe(
                prompt, max_new_tokens=60,
                do_sample=False, repetition_penalty=1.1,
            )[0]["generated_text"]
            print(f"\n  > {prompt}")
            print(f"    {out[len(prompt):].strip()}")
        print("\n  Test OK.")
    except Exception as e:
        print(f"  Test echoue : {e}")
        print("  (Le modele reste utilisable dans l'API)")


# =============================================================
# POINT D'ENTREE
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune + Fusion Phi-3.5-mini-instruct — Corpus Tunisie Telecom"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Sauter le fine-tune, fusionner uniquement le LoRA existant"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  finetune_phi35.py — Phi-3.5-mini-instruct")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  fp16 direct — sans bitsandbytes (CUDA 12.4 Windows)")
    print("=" * 60)

    tokenizer = None

    if args.merge_only:
        print("\n  Mode : --merge-only (fusion du LoRA existant)")
        merge_ok = run_merge(tokenizer=None)
    else:
        print("\n  Mode : complet (fine-tune + fusion)")
        tokenizer = run_finetune()
        merge_ok  = run_merge(tokenizer=tokenizer)

    if merge_ok:
        run_test()

    # Resume
    print("\n" + "=" * 60)
    print("  RESUME")
    print("=" * 60)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  LoRA     : {OUTPUT_DIR.resolve()}")
    if merge_ok:
        print(f"  Fusionne : {MERGED_DIR.resolve()}")
        print("  Etat     : SUCCES")
    else:
        print("  Fusion   : ECHEC — retenter avec --merge-only")
        print("  Etat     : LoRA disponible, fusion a retenter")
    print()
    print("  Pour utiliser dans api_tinyllama.py :")
    if merge_ok:
        print(f"    MODEL_DIR  = '{MERGED_DIR.resolve()}'")
    print(f"    BASE_MODEL = '{BASE_MODEL}'")
    print("=" * 60)


if __name__ == "__main__":
    main()