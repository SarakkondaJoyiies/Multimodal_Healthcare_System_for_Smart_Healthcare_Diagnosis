import tensorflow as tf
import numpy as np
from PIL import Image

# ============================
# Load Chest X-ray Validator
# ============================

MODEL_PATH = "models/image_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.trainable = False
except Exception as e:
    raise RuntimeError(f"❌ Failed to load chest X-ray validator model: {e}")

# ============================
# Configuration
# ============================

IMG_SIZE = 224
THRESHOLD = 0.5   # Adjust only if retraining

# ============================
# Image Preprocessing
# ============================

def preprocess_image(image_file):
    """
    Preprocess input image exactly as during training:
    - Convert to GRAYSCALE
    - Resize to 224x224
    - Normalize to [0,1]
    - Shape: (1, 224, 224, 1)
    """
    image = Image.open(image_file).convert("L")  # GRAYSCALE
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # channel dim
    img_array = np.expand_dims(img_array, axis=0)   # batch dim

    return img_array

# ============================
# Chest X-ray Validation
# ============================

def validate_chest_xray(image_file):
    """
    Returns:
        is_chest_xray (bool): Whether image is likely a chest X-ray
        confidence (float): Model confidence score
    """

    try:
        img = preprocess_image(image_file)
        prediction = float(model.predict(img, verbose=0)[0][0])

        # IMPORTANT:
        # Most datasets map:
        #   chest_xray -> 0
        #   non_chest  -> 1
        # So LOWER score means chest X-ray
        is_chest_xray = prediction < THRESHOLD

        return is_chest_xray, prediction

    except Exception as e:
        print(f"⚠️ Chest validation error: {e}")
        return False, 0.0
