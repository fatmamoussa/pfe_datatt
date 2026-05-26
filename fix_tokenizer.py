import json

with open('models/phi35-tt-merged/tokenizer_config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

cfg['chat_template'] = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}<|end|>\n"
    "{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}<|end|>\n"
    "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}<|end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)

with open('models/phi35-tt-merged/tokenizer_config.json', 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print("chat_template ajouté avec succès")
print("Aperçu:", cfg['chat_template'][:80])