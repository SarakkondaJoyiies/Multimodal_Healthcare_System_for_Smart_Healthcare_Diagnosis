from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv

from chatbot.chatbot_engine import chatbot_response
from utils.image_utils import predict_image, generate_gradcam
from utils.text_utils import predict_text
from utils.speech_utils import speech_to_text
from utils.report_utils import generate_patient_report
from utils.chest_utils import is_chest_xray

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__, static_folder="static")
CORS(app)

# -------------------------
# Ensure required directories exist
# -------------------------
os.makedirs("static/gradcam", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# -------------------------
# Health Check
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Backend running on localhost:5000"})

# -------------------------
# Multimodal Recommendation
# -------------------------
def generate_medical_recommendation(
    image_prediction,
    image_confidence,
    text_prediction=None
):
    if image_prediction == "NORMAL" and text_prediction and text_prediction != "NORMAL":
        return (
            "The chest X-ray does not show radiological evidence of pneumonia; "
            "however, the reported symptoms suggest a possible respiratory condition. "
            "Clinical consultation is recommended."
        )

    if image_prediction == "NORMAL":
        return (
            "No radiological evidence of pneumonia detected. "
            "Routine clinical observation is recommended."
        )

    if image_confidence >= 0.80:
        return (
            "High likelihood of pneumonia detected. "
            "Physician consultation is strongly recommended."
        )

    if image_confidence >= 0.60:
        return (
            "Moderate likelihood of pneumonia detected. "
            "Clinical correlation and follow-up assessment may be considered."
        )

    return (
        "Low confidence indication of pneumonia. "
        "Monitoring and clinical assessment are advised."
    )

# -------------------------
# Chest X-ray Validation API
# -------------------------
@app.route("/validate-image", methods=["POST"])
def validate_image():
    image = request.files.get("image")

    if image is None:
        return jsonify({
            "valid": False,
            "message": "No image received"
        }), 400

    is_valid, confidence = is_chest_xray(image)
    image.seek(0)

    if not is_valid:
        return jsonify({
            "valid": False,
            "confidence": round(float(confidence), 4),
            "message": (
                "Uploaded image is NOT a Chest X-ray❌"
            )
        }), 400

    return jsonify({
        "valid": True,
        "confidence": round(float(confidence), 4),
        "message": "Valid Chest X-ray detected"
    })

# -------------------------
# Main Diagnosis API
# -------------------------
@app.route("/diagnose", methods=["POST"])
def diagnose():
    image = request.files.get("image")
    audio = request.files.get("audio")
    text  = request.form.get("text")

    if image is None:
        return jsonify({
            "error": "Chest X-ray image is required"
        }), 400

    # -------------------------
    # 0️⃣ STRICT Chest X-ray Validation
    # -------------------------
    is_valid, validator_conf = is_chest_xray(image)
    image.seek(0)

    if not is_valid:
        return jsonify({
            "error": "Uploaded image is not a Chest X-ray.",
            "validator_confidence": round(float(validator_conf), 3),
            "message": (
                "The system is trained exclusively on Chest X-ray images. "
                "Brain MRI, CT scans, or unrelated images are not supported. "
                "Please re-upload a valid Chest X-ray."
            )
        }), 400

    # -------------------------
    # 1️⃣ Pneumonia Image Prediction
    # -------------------------
    image_prediction, image_confidence = predict_image(image)

    response = {
        "image_prediction": image_prediction,
        "image_confidence": round(float(image_confidence), 4)
    }

    # -------------------------
    # 2️⃣ Clinical Text / Speech
    # -------------------------
    clinical_text = None
    text_prediction = None

    if audio:
        try:
            clinical_text = speech_to_text(audio)
            response["transcription"] = clinical_text
        except Exception as e:
            response["transcription_error"] = str(e)
            clinical_text = None

    if clinical_text:
        text_prediction = predict_text(clinical_text)

    # -------------------------
    # 3️⃣ Grad-CAM (If Pneumonia)
    # -------------------------
    if image_prediction == "PNEUMONIA":
        image.seek(0)
        gradcam_path = generate_gradcam(image)
        response["gradcam_image"] = gradcam_path

        response["pneumonia_type"] = (
            text_prediction if text_prediction
            else "Undetermined (image-based)"
        )

    # -------------------------
    # 4️⃣ Recommendation
    # -------------------------
    response["recommendation"] = generate_medical_recommendation(
        image_prediction=image_prediction,
        image_confidence=image_confidence,
        text_prediction=text_prediction
    )

    # -------------------------
    # 5️⃣ Report Generation
    # -------------------------
    report_path = generate_patient_report(response)
    response["report_path"] = report_path

    return jsonify(response)

# -------------------------
# Serve Reports
# -------------------------
@app.route("/reports/<path:filename>")
def download_report(filename):
    return send_from_directory("reports", filename, as_attachment=True)

# -------------------------
# Chatbot API
# -------------------------
@app.route("/chatbot", methods=["POST"])
def chatbot_api():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please enter a valid question."})

        reply = chatbot_response(user_message)
        return jsonify({"reply": reply})

    except Exception as e:
        import traceback
        print("❌ Chatbot API error:", repr(e))
        traceback.print_exc()

        return jsonify({
            "reply": "Assistant temporarily unavailable."
        }), 500

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False, use_reloader=False)

