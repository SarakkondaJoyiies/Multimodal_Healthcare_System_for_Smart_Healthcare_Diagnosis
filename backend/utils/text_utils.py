import pickle
import torch
from transformers import AutoModelForSequenceClassification

# -------------------------
# Load Pickle Bundle
# -------------------------
PKL_PATH = "models/pneumonia_text_model_new.pkl"
BASE_MODEL_PATH = "models/text_base_model"  # LOCAL ONLY

with open(PKL_PATH, "rb") as f:
    bundle = pickle.load(f)

tokenizer = bundle["tokenizer"]
NUM_LABELS = bundle["num_labels"]

# -------------------------
# Load Model OFFLINE
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=NUM_LABELS,
    local_files_only=True
)

model.load_state_dict(bundle["model_state_dict"])
model.eval()

# -------------------------
# Label Mapping
# -------------------------
LABELS = {
    0: "NORMAL",
    1: "BACTERIAL_PNEUMONIA",
    2: "VIRAL_PNEUMONIA"
}

# -------------------------
# Prediction Function
# -------------------------
def predict_text(text: str) -> str:
    if not text or not text.strip():
        return "NORMAL"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[pred]
