import whisper
import tempfile
import threading

speech_model = whisper.load_model("base")
whisper_lock = threading.Lock()

def speech_to_text(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)

        # Thread-safe whisper transcribe
        with whisper_lock:
            result = speech_model.transcribe(tmp.name, fp16=False)

    return result["text"]
