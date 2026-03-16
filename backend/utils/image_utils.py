import pickle
import numpy as np
import tensorflow as tf
import cv2
import uuid
import os
import io

from tensorflow.keras.applications import EfficientNetB0, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image


# =========================================================
# LOAD IMAGE MODEL FROM PKL (ROBUST)
# =========================================================

PKL_PATH = "models/pneumonia_image_model_new.pkl"

with open(PKL_PATH, "rb") as f:
    bundle = pickle.load(f)

# Safe defaults
IMG_SIZE = 224
THRESHOLD = 0.5
model_weights = None
model = None

# Detect bundle type
if isinstance(bundle, dict):
    IMG_SIZE = bundle.get("input_size", IMG_SIZE)
    THRESHOLD = bundle.get("threshold", THRESHOLD)

    if "model_weights" in bundle:
        model_weights = bundle["model_weights"]
    elif "weights" in bundle:
        model_weights = bundle["weights"]
    elif "model" in bundle and hasattr(bundle["model"], "predict"):
        model = bundle["model"]

elif isinstance(bundle, list):
    model_weights = bundle

elif hasattr(bundle, "predict"):
    model = bundle

else:
    raise ValueError("Unsupported PKL format for pneumonia image model")


# =========================================================
# BUILD MODEL ARCHITECTURE (MATCH PKL WEIGHTS)
# =========================================================

def build_model(backbone_name, img_size):
    if backbone_name == "EfficientNetB0":
        base = EfficientNetB0(weights="imagenet", include_top=False,
                              input_shape=(img_size, img_size, 3))
        last_conv = "top_conv"

    elif backbone_name == "DenseNet121":
        base = DenseNet121(weights="imagenet", include_top=False,
                           input_shape=(img_size, img_size, 3))
        last_conv = "conv5_block16_concat"  # DenseNet121 last conv layer

    else:
        raise ValueError("Unsupported backbone")

    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)

    m = Model(inputs=base.input, outputs=out)
    return m, last_conv


if model is None:
    if model_weights is None:
        raise ValueError("Model weights not found in PKL")

    # ✅ AUTO-DETECT MODEL TYPE BY WEIGHT COUNT
    weight_len = len(model_weights)

    if weight_len == 316:
        model, LAST_CONV_LAYER = build_model("EfficientNetB0", IMG_SIZE)

    elif weight_len == 608:
        model, LAST_CONV_LAYER = build_model("DenseNet121", IMG_SIZE)

    else:
        raise ValueError(f"Unknown weight length {weight_len}. Cannot match model architecture.")

    # Load weights
    model.set_weights(model_weights)

else:
    # If PKL already contains a compiled Keras model
    # Try to select LAST_CONV_LAYER safely
    LAST_CONV_LAYER = "top_conv"
    try:
        model.get_layer(LAST_CONV_LAYER)
    except:
        # fallback for DenseNet
        LAST_CONV_LAYER = "conv5_block16_concat"


# =========================================================
# GRAD-CAM MODEL
# =========================================================

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
)


# =========================================================
# IMAGE PREDICTION FUNCTION
# =========================================================

def predict_image(image_file):
    """
    Predict NORMAL / PNEUMONIA from uploaded Chest X-ray
    """
  

    image_bytes = image_file.read()
    image_file.seek(0)

    img = keras_image.load_img(
        io.BytesIO(image_bytes),
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model(arr, training=False).numpy()[0][0])
    label = "PNEUMONIA" if prob >= THRESHOLD else "NORMAL"

    confidence = prob if label == "PNEUMONIA" else 1 - prob
    return label, round(confidence, 4)


# =========================================================
# GRAD-CAM GENERATION FUNCTION
# =========================================================

def generate_gradcam(image_file):
    """
    Generate and save Grad-CAM heatmap
    """

    image_bytes = image_file.read()
    image_file.seek(0)

    img = keras_image.load_img(
        io.BytesIO(image_bytes),
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    img_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_cv = cv2.resize(img_cv, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    filename = f"gradcam_{uuid.uuid4().hex}.png"
    save_path = os.path.join("static", "gradcam", filename)

    cv2.imwrite(save_path, overlay)
    return save_path
