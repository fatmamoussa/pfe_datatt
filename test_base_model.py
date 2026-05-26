"""
Teste le modèle de BASE Phi-3.5-mini-instruct avec un contexte RAG simulé.
Lance : python test_base_model.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_BASE = "microsoft/Phi-3.5-mini-instruct"

CONTEXT = (
    "Le service SOS Solde permet au client d'obtenir une avance de crédit "
    "lorsque son solde est insuffisant. "
    "Pour en bénéficier, composer le *150# gratuitement ou envoyer D au 85159. "
    "Le montant de l'avance est de 1 DT TTC. "
    "Le remboursement se fait automatiquement à la prochaine recharge."
)

SYSTEM = (
    "Tu es l'assistant officiel de Tunisie Telecom. "
    "Réponds en 2-3 phrases claires basées uniquement sur le contexte fourni."
)

def test_base():
    print("="*60)
    print("  TEST modèle de BASE : Phi-3.5-mini-instruct")
    print("="*60)

    print("[1] Chargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
    print(f"    chat_template: {'OUI' if tokenizer.chat_template else 'NON'}")

    print("[2] Chargement modèle 4-bit NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        quantization_config=bnb,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"    VRAM utilisée : {vram:.2f} GB")

    print("[3] Génération avec contexte RAG...")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": (
            f"Contexte Tunisie Telecom :\n{CONTEXT}\n\n"
            "Question : Comment fonctionne le service SOS Solde ?\n\n"
            "Réponds en 2-3 phrases claires avec tes propres mots."
        )},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda:0")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"\n[RÉPONSE] :\n{response}\n")

    import re
    gibberish = len(re.findall(r'\bgiven\b|;}\s*\w+|----+', response))
    if gibberish == 0 and len(response) > 20:
        print("[RÉSULTAT] ✓ Modèle de base FONCTIONNEL")
        print("\n→ Action : changer MODEL_DIR dans api_v5.py")
        print("  MODEL_DIR = os.environ.get('MODEL_DIR', 'microsoft/Phi-3.5-mini-instruct')")
        print("\n  OU lancer avec :")
        print("  $env:MODEL_DIR='microsoft/Phi-3.5-mini-instruct'; python api_v5.py")
    else:
        print("[RÉSULTAT] ✗ Problème détecté — vérifier la connexion internet/cache HuggingFace")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA non disponible")
        exit(1)
    test_base()