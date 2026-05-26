#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
Fine-tune TinyLlama-1.1B — Corpus Tunisie Télécom
Mode    : Causal Language Modeling (texte brut, champ "text")
GPU     : RTX 2070 8 GB
Trainer : HuggingFace Trainer natif
Préc.   : fp16 direct — SANS bitsandbytes (CUDA 12.4 Windows)
Data    : data/train.json + data/val.json  (JSON array, clé "text")
=============================================================
"""

import os, gc, json, torch, functools
from pathlib import Path
from datetime import datetime

# ── Désactiver torch.compile (Windows / PyTorch 2.6) ──────
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

# ── Imports HuggingFace au niveau module (erreurs visibles) ──
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
BASE_MODEL     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR       = Path("data")
TRAIN_FILE     = DATA_DIR / "train.json"
VAL_FILE       = DATA_DIR / "val.json"
OUTPUT_DIR     = Path("models/tinyllama-tt-lora")
MERGED_DIR     = Path("models/tinyllama-tt-merged")

LORA_R         = 32
LORA_ALPHA     = 64
LORA_DROPOUT   = 0.05
LORA_TARGET    = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

NUM_EPOCHS     = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4          # batch effectif = 2 x 4 = 8
LEARNING_RATE  = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO   = 0.05
SAVE_STEPS     = 100
EVAL_STEPS     = 100
LOGGING_STEPS  = 10
# =============================================================


def check_gpu():
    print("\n🔍 Vérification GPU...")
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA non disponible.")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU  : {name}")
    print(f"✅ VRAM : {vram:.1f} GB")
    if vram >= 8:
        print("✅ 8 GB VRAM — config optimale pour TinyLlama fp16")
    elif vram < 6:
        print("⚠️  < 6 GB VRAM — réduis BATCH_SIZE=1 ou MAX_SEQ_LENGTH=384 si OOM")


def load_data(path: Path) -> list:
    """
    Charge un fichier JSON array ou JSONL.
    Retourne une liste de dicts avec au moins la clé "text".
    """
    records = []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        # Format JSON array  [{...}, {...}]
        for obj in json.loads(content):
            if len(obj.get("text", "").strip()) > 20:
                records.append(obj)
    else:
        # Format JSONL  {...}\n{...}
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

    return records


class CLMDataset(torch.utils.data.Dataset):
    """
    Dataset Causal LM — chaque exemple est tokenise sous la forme :
        <bos> texte <eos>
    Labels = input_ids (on predit chaque token suivant).
    """
    def __init__(self, records, tokenizer, max_length):
        self.examples = []
        bos = tokenizer.bos_token or "<s>"
        eos = tokenizer.eos_token or "</s>"

        for r in records:
            text = f"{bos}{r['text'].strip()}{eos}"
            enc  = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            ids = enc["input_ids"]
            if len(ids) > 5:
                self.examples.append(ids)

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
    """Padding dynamique a droite (evite la VRAM gaspillee)."""
    max_len        = max(x["input_ids"].size(0) for x in batch)
    input_ids      = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels         = torch.full((len(batch), max_len), -100,   dtype=torch.long)

    for i, x in enumerate(batch):
        n = x["input_ids"].size(0)
        input_ids[i, :n]      = x["input_ids"]
        attention_mask[i, :n] = x["attention_mask"]
        labels[i, :n]         = x["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    print("=" * 60)
    print("🚀 Fine-tune TinyLlama — Corpus Tunisie Telecom")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   fp16 direct — sans bitsandbytes")
    print("=" * 60)

    check_gpu()

    # ── 1. Tokenizer ──────────────────────────────────────
    print(f"\n📥 Chargement tokenizer : {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"✅ EOS={tokenizer.eos_token!r}  PAD={tokenizer.pad_token!r}  Vocab={tokenizer.vocab_size:,}")

    # ── 2. Modele fp16 ────────────────────────────────────
    print(f"\n🦙 Chargement TinyLlama en fp16 (~2.2 GB VRAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache      = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_disable()
    print("✅ Modele charge.")

    # ── 3. LoRA ───────────────────────────────────────────
    print(f"\n⚙️  Config LoRA : r={LORA_R}, alpha={LORA_ALPHA}")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"✅ Parametres entrainables : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 4. Datasets ───────────────────────────────────────
    print(f"\n📂 Chargement donnees...")
    train_raw = load_data(TRAIN_FILE)
    val_raw   = load_data(VAL_FILE)
    print(f"   Train brut : {len(train_raw)} exemples  ({TRAIN_FILE})")
    print(f"   Val   brut : {len(val_raw)} exemples  ({VAL_FILE})")
    print(f"   Exemple    : {train_raw[0]['text'][:80]!r}...")

    train_dataset = CLMDataset(train_raw, tokenizer, MAX_SEQ_LENGTH)
    val_dataset   = CLMDataset(val_raw,   tokenizer, MAX_SEQ_LENGTH)
    print(f"   Train tokenise : {len(train_dataset)} exemples")
    print(f"   Val   tokenise : {len(val_dataset)} exemples")

    collator = functools.partial(collate_fn, pad_id=tokenizer.pad_token_id)

    # ── 5. Training arguments ─────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM))

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        optim="adamw_torch",           # pas de bitsandbytes
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
        dataloader_num_workers=0,      # Windows : obligatoire
        gradient_checkpointing=False,
        torch_compile=False,
        remove_unused_columns=False,
    )

    # ── 6. Trainer ────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # ── 7. Lancement ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("🚀 DEMARRAGE ENTRAINEMENT")
    print("=" * 60)
    print(f"   Modele         : TinyLlama-1.1B-Chat-v1.0")
    print(f"   Epochs         : {NUM_EPOCHS}")
    print(f"   Batch effectif : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   Steps/epoch    : ~{steps_per_epoch}")
    print(f"   Duree estimee  : 20-40 min sur RTX 2070")
    print("─" * 60)

    trainer.train()

    # ── 8. Sauvegarde LoRA ────────────────────────────────
    print(f"\n💾 Sauvegarde LoRA -> {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("✅ Adaptateur LoRA sauvegarde.")

    # ── 9. Liberation GPU ─────────────────────────────────
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ GPU libere.")

    # ── 10. Fusion LoRA -> modele complet ──────────────────
    print(f"\n🔀 Fusion LoRA + base (~4 GB RAM CPU)...")
    try:
        base   = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16,
            device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
        )
        merged = PeftModel.from_pretrained(base, str(OUTPUT_DIR)).merge_and_unload()
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
        tokenizer.save_pretrained(str(MERGED_DIR))
        print(f"✅ Modele fusionne -> {MERGED_DIR}")
    except Exception as e:
        print(f"⚠️  Fusion echouee : {e}")

    # ── 11. Test rapide ───────────────────────────────────
    print("\n🧪 Test de generation...")
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-generation", model=str(MERGED_DIR),
            tokenizer=str(MERGED_DIR), torch_dtype=torch.float16, device_map="auto",
        )
        for prompt in ["Offre Hayya Tunisie Telecom :",
                       "Les clients prepayés peuvent beneficier de"]:
            out = pipe(prompt, max_new_tokens=60, do_sample=False,
                       repetition_penalty=1.1)[0]["generated_text"]
            print(f"\n   > {prompt}")
            print(f"     {out[len(prompt):]}")
    except Exception as e:
        print(f"⚠️  Test echoue : {e}")

    # ── 12. Resume ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎉 FINE-TUNING TERMINE")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 LoRA     : {OUTPUT_DIR.resolve()}")
    print(f"📁 Fusionne : {MERGED_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()