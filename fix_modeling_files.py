# fix_modeling_files.py
import shutil
from pathlib import Path
from transformers.utils import cached_file

MERGED_DIR = Path("models/phi35-tt-merged")
BASE_MODEL  = "microsoft/Phi-3.5-mini-instruct"

REMOTE_FILES = [
    "modeling_phi3.py",
    "configuration_phi3.py",
]

print("Copie des fichiers de code distant dans le modèle fusionné...")
for fname in REMOTE_FILES:
    try:
        src = cached_file(BASE_MODEL, fname)
        dst = MERGED_DIR / fname
        shutil.copy2(src, dst)
        print(f"  ✅ {fname}")
    except Exception as e:
        print(f"  ⚠ {fname} introuvable : {e}")

print("\nTerminé. Relance l'API.")