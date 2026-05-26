"""
Patch final — corrige api_v5.py pour le modèle de base Phi-3.5-mini-instruct.
Lance : python patch_final.py
"""
import re

with open("api_v5.py", "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ── 1. Nouveaux prompts système plus contraignants ──────────────────────────
OLD_FR = re.search(r'GENERATIVE_SYSTEM_PROMPT = \(.*?\n\)', content, re.DOTALL)
NEW_FR = '''GENERATIVE_SYSTEM_PROMPT = (
    "You are the official assistant of Tunisie Telecom. "
    "Answer ONLY using the CONTEXT provided below. "
    "Respond in French in 2-3 short sentences. "
    "Do NOT use your general knowledge. "
    "If the context does not contain the answer, say exactly: "
    "\\'Je n\\'ai pas trouvé d\\'information précise. Contactez le 1298.\\' "
    "Do NOT mention Ooredoo, Orange or Tunisiana. "
    "Do NOT repeat the question."
)'''

OLD_FR_ML = re.search(r'GENERATIVE_SYSTEM_PROMPT_MULTILINE = \(.*?\n\)', content, re.DOTALL)
NEW_FR_ML = '''GENERATIVE_SYSTEM_PROMPT_MULTILINE = (
    "You are the official assistant of Tunisie Telecom. "
    "Explain the steps from the CONTEXT in 3-5 French sentences. "
    "Use ONLY what is in the context. Do NOT invent steps. "
    "If not found, say: \\'Contactez le 1298.\\'"
)'''

OLD_AR = re.search(r'GENERATIVE_SYSTEM_PROMPT_AR = \(.*?\n\)', content, re.DOTALL)
NEW_AR = '''GENERATIVE_SYSTEM_PROMPT_AR = (
    "You are the official assistant of Tunisie Telecom. "
    "Answer ONLY using the CONTEXT below. Respond in Arabic in 2-3 sentences. "
    "Do NOT use general knowledge. "
    "If not found, say: \\'يرجى الاتصال بالرقم 1298.\\'"
)'''

OLD_AR_ML = re.search(r'GENERATIVE_SYSTEM_PROMPT_AR_MULTILINE = \(.*?\n\)', content, re.DOTALL)
NEW_AR_ML = '''GENERATIVE_SYSTEM_PROMPT_AR_MULTILINE = (
    "You are the official assistant of Tunisie Telecom. "
    "Explain the steps from the CONTEXT in 3-5 Arabic sentences. "
    "Use ONLY what is in the context. "
    "If not found, say: \\'يرجى الاتصال بالرقم 1298.\\'"
)'''

for old_match, new_str, label in [
    (OLD_AR_ML, NEW_AR_ML, "SYSTEM_PROMPT_AR_MULTILINE"),
    (OLD_AR,    NEW_AR,    "SYSTEM_PROMPT_AR"),
    (OLD_FR_ML, NEW_FR_ML, "SYSTEM_PROMPT_MULTILINE"),
    (OLD_FR,    NEW_FR,    "SYSTEM_PROMPT"),
]:
    if old_match:
        content = content[:old_match.start()] + new_str + content[old_match.end():]
        print(f"[OK] {label} remplacé")
        changes += 1
    else:
        print(f"[SKIP] {label} non trouvé")

# ── 2. Retirer "Response:" et "Question:" du output du modèle ──────────────
OLD_CLEAN = '_clean_phi_output'
OLD_FUNC = re.search(
    r'def _clean_phi_output\(text: str\) -> str:.*?return clean_jsonl_artifacts\(text\)',
    content, re.DOTALL
)
NEW_FUNC = '''def _clean_phi_output(text: str) -> str:
    text = re.sub(r"<\\|.*?\\|>", "", text).strip()
    text = re.sub(r"^(?:assistant|Assistant)\\s*:\\s*", "", text).strip()
    text = re.sub(r"^(?:Response|Réponse|Answer|Répondre)\\s*:\\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\\s*\\.\\.\\.\\s*$", ".", text).strip()
    # Supprime les fragments de question répétés avant la réponse
    text = re.sub(r"^.*?\\?\\s*(?:Response|Réponse)?\\s*:\\s*", "", text, flags=re.DOTALL).strip()
    return clean_jsonl_artifacts(text)'''

if OLD_FUNC:
    content = content[:OLD_FUNC.start()] + NEW_FUNC + content[OLD_FUNC.end():]
    print("[OK] _clean_phi_output amélioré")
    changes += 1
else:
    print("[SKIP] _clean_phi_output non trouvé")

# ── 3. Supprimer "connaissance_generale" du filtre PROMPT_LEAKS ────────────
# Le modèle de base dit "généralement" naturellement, ce n'est pas un leak
OLD_LEAKS = re.search(r'PROMPT_LEAKS_PHI = \[.*?\]', content, re.DOTALL)
NEW_LEAKS = '''PROMPT_LEAKS_PHI = [
    "وفقاً لمعرفتي","بحسب معلوماتي",
    "as an ai","as a language model",
    "je suis un assistant ia","en tant qu\\'ia",
]'''
if OLD_LEAKS:
    content = content[:OLD_LEAKS.start()] + NEW_LEAKS + content[OLD_LEAKS.end():]
    print("[OK] PROMPT_LEAKS_PHI allégé")
    changes += 1

# ── 4. Augmenter MAX_INPUT_LENGTH à 1200 pour le modèle de base ────────────
content = content.replace(
    'MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "1024"))',
    'MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "1200"))',
)
print("[OK] MAX_INPUT_LENGTH → 1200")
changes += 1

# ── 5. Instruction user plus directe ───────────────────────────────────────
OLD_INST = re.search(
    r'instruction = \("Résume en 3 à 5 phrases.*?uniquement sur le contexte\.\"\)',
    content, re.DOTALL
)
NEW_INST = '''instruction = ("Résume en 3 à 5 phrases claires les étapes disponibles en français."
                       if is_multiline
                       else "Réponds en 2-3 phrases en français en te basant UNIQUEMENT sur le contexte ci-dessus.")'''
if OLD_INST:
    content = content[:OLD_INST.start()] + NEW_INST + content[OLD_INST.end():]
    print("[OK] instruction user mise à jour")
    changes += 1

with open("api_v5.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"\n[DONE] {changes} modification(s) appliquées sur api_v5.py")
print("\nLance maintenant :")
print('  $env:MODEL_DIR="microsoft/Phi-3.5-mini-instruct"; python api_v5.py')