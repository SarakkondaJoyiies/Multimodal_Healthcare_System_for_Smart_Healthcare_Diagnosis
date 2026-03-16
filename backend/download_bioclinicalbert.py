from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
SAVE_DIR = "./models/bioclinicalbert"

print("⬇️ Downloading Bio_ClinicalBERT...")

AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=SAVE_DIR)
AutoModel.from_pretrained(MODEL_NAME, cache_dir=SAVE_DIR)

print("✅ Download complete. Model saved locally.")
