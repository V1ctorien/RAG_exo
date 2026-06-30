import os, urllib.request

print("=== Variables offline ===")
print("TRANSFORMERS_OFFLINE :", os.environ.get("TRANSFORMERS_OFFLINE", "non defini OK"))
print("HF_HUB_OFFLINE       :", os.environ.get("HF_HUB_OFFLINE",       "non defini OK"))
print("HF_DATASETS_OFFLINE  :", os.environ.get("HF_DATASETS_OFFLINE",  "non defini OK"))

print("\n=== Connexion HuggingFace ===")
try:
    urllib.request.urlopen("https://huggingface.co", timeout=5)
    print("Connexion OK")
except Exception as e:
    print("Pas de connexion :", e)