import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Load model once
MODEL_PATH = "models/image_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

def is_chest_xray(file):
    """
    Returns:
      (True, confidence)  → Chest X-ray
      (False, confidence) → Non-chest (MRI / CT / other)
    """

    # Read bytes safely
    image_bytes = io.BytesIO(file.read())
    file.seek(0)

    # Preprocess
    image = Image.open(image_bytes).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    # Predict
    pred = float(model.predict(img_array, verbose=0)[0][0])

    # 🔴 CONFIDENCE-BASED REJECTION (THIS IS THE KEY)
    if pred > 0.50:
      return True, pred       # Chest X-ray
    else:
      return False, pred      # Non-chest
