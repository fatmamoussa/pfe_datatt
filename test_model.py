"""
Teste si le modèle fine-tuné génère correctement.
Lance : python test_model.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_FT   = "models/phi35-tt-merged"
MODEL_BASE = "microsoft/Phi-3.5-mini-instruct"

def test_model(model_path, label):
    print(f"\n{'='*60}")
    print(f"  TEST : {label}")
    print(f"  Path : {model_path}")
    print('='*60)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[OK] Tokenizer chargé | chat_template: {'OUI' if tokenizer.chat_template else 'NON'}")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()
        print(f"[OK] Modèle chargé")

        messages = [
            {"role": "system", "content": "Tu es l'assistant de Tunisie Telecom. Réponds en français."},
            {"role": "user",   "content": "Qu'est-ce que le service SOS Solde ?"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"\n[RÉPONSE] : {response[:300]}")

        # Diagnostic : est-ce du charabia ?
        import re
        alpha_ratio = len(re.findall(r'[a-zA-ZÀ-ÿ\u0600-\u06FF\s]', response)) / max(len(response), 1)
        print(f"\n[DIAGNOSTIC] ratio alphanumérique = {alpha_ratio:.2f}")
        if alpha_ratio > 0.6:
            print("[RÉSULTAT] ✓ Réponse LISIBLE — modèle fonctionnel")
        else:
            print("[RÉSULTAT] ✗ Réponse CHARABIA — modèle corrompu ou tokenizer incompatible")

    except Exception as e:
        print(f"[ERREUR] {e}")
    finally:
        try:
            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception:
            pass

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA non disponible")
        exit(1)

    import os
    if os.path.isdir(MODEL_FT):
        test_model(MODEL_FT, "Modèle FINE-TUNÉ (phi35-tt-merged)")
    else:
        print(f"[SKIP] Dossier {MODEL_FT} introuvable")

    print("\n" + "="*60)
    print("  Si le fine-tuné génère du charabia,")
    print("  utilise le modèle de base à la place :")
    print(f"  MODEL_DIR = '{MODEL_BASE}'")
    print("="*60)