"""
Applique le patch de prompts renforcés sur api_v5.py
Lance : python patch_prompts.py
"""
import re

with open("api_v5.py", "r", encoding="utf-8") as f:
    content = f.read()

# Nouveau prompt FR simple
NEW_FR = '''GENERATIVE_SYSTEM_PROMPT = (
    "Tu es l'assistant officiel de Tunisie Telecom. "
    "Réponds UNIQUEMENT en utilisant les informations du CONTEXTE ci-dessous.\\n"
    "RÈGLES ABSOLUES :\\n"
    "1. Si l'information est dans le contexte → résume-la en 2-3 phrases simples.\\n"
    "2. Si l'information n'est PAS dans le contexte → réponds EXACTEMENT : "
    "'Je n\\'ai pas trouvé d\\'information précise. Contactez le 1298.'\\n"
    "3. N\\'invente RIEN. N\\'utilise PAS tes connaissances générales.\\n"
    "4. Ne mentionne jamais Ooredoo, Orange ou Tunisiana.\\n"
    "5. Réponds en français uniquement."
)'''

NEW_FR_MULTI = '''GENERATIVE_SYSTEM_PROMPT_MULTILINE = (
    "Tu es l'assistant officiel de Tunisie Telecom. "
    "Explique les étapes du CONTEXTE ci-dessous en 3-5 phrases.\\n"
    "RÈGLES ABSOLUES :\\n"
    "1. Utilise UNIQUEMENT ce qui est dans le contexte.\\n"
    "2. Si la réponse n\\'est pas dans le contexte : 'Contactez le 1298.'\\n"
    "3. N\\'invente aucune étape ni code absent du contexte.\\n"
    "4. Réponds en français uniquement."
)'''

NEW_AR = '''GENERATIVE_SYSTEM_PROMPT_AR = (
    "أنت المساعد الرسمي لـ Tunisie Telecom. "
    "أجب فقط باستخدام المعلومات الموجودة في السياق أدناه.\\n"
    "القواعد:\\n"
    "1. إذا كانت المعلومات في السياق → لخّصها في 2-3 جمل.\\n"
    "2. إذا لم تكن في السياق → قل: 'يرجى الاتصال بالرقم 1298.'\\n"
    "3. لا تخترع أي معلومات. لا تستخدم معرفتك العامة.\\n"
    "4. لا تذكر أي منافس."
)'''

NEW_AR_MULTI = '''GENERATIVE_SYSTEM_PROMPT_AR_MULTILINE = (
    "أنت المساعد الرسمي لـ Tunisie Telecom. "
    "اشرح الخطوات من السياق في 3-5 جمل.\\n"
    "القواعد:\\n"
    "1. استخدم فقط ما هو في السياق.\\n"
    "2. لا تخترع خطوات. إذا لم تجد الإجابة: 'يرجى الاتصال بالرقم 1298.'\\n"
    "3. لا تستخدم معرفتك العامة."
)'''

# Remplacements
replacements = [
    (r'GENERATIVE_SYSTEM_PROMPT_AR_MULTILINE = \(.*?\)', NEW_AR_MULTI, re.DOTALL),
    (r'GENERATIVE_SYSTEM_PROMPT_AR = \(.*?\)', NEW_AR, re.DOTALL),
    (r'GENERATIVE_SYSTEM_PROMPT_MULTILINE = \(.*?\)', NEW_FR_MULTI, re.DOTALL),
    (r'GENERATIVE_SYSTEM_PROMPT = \(.*?\)', NEW_FR, re.DOTALL),
]

for pattern, replacement, flags in replacements:
    content, n = re.subn(pattern, replacement, content, flags=flags)
    print(f"[{'OK' if n else 'SKIP'}] {replacement[:60].strip()[:50]}...")

# Sauvegarde
with open("api_v5.py", "w", encoding="utf-8") as f:
    f.write(content)

print("\n[DONE] api_v5.py patché avec les nouveaux prompts")
print("Lance maintenant :")
print("  $env:MODEL_DIR='microsoft/Phi-3.5-mini-instruct'; python api_v5.py")