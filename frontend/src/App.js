import React, { useState, useRef } from "react";
import "./App.css";
import Chatbot from "./Chatbot";

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [text, setText] = useState("");
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioURL, setAudioURL] = useState(null);
  const [recording, setRecording] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // ✅ ADDED: controls hospital button visibility
  const [showHospitals, setShowHospitals] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  /* =============================
     CHATBOT STATE (ADDED)
  ============================= */
  const [chatMessages, setChatMessages] = useState([
    {
      role: "bot",
      text: "Hello! 👋 I can help you understand pneumonia, symptoms, and diagnosis results."
    }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  /* =============================
     CHATBOT API CALL (ADDED)
  ============================= */
  const sendChatMessage = async () => {
    if (!chatInput.trim()) return;

    const userText = chatInput.trim();

    setChatMessages((prev) => [...prev, { role: "user", text: userText }]);
    setChatInput("");
    setChatLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/chatbot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
      });

      const data = await res.json();

      setChatMessages((prev) => [
        ...prev,
        { role: "bot", text: data.reply || "No response received." }
      ]);
    } catch (err) {
      console.error("Chatbot error:", err);

      setChatMessages((prev) => [
        ...prev,
        { role: "bot", text: "⚠️ Chatbot service unavailable. Please try again." }
      ]);
    }

    setChatLoading(false);
  };

  /* =============================
     VOICE RECORDING
  ============================= */
  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);

    mediaRecorderRef.current = recorder;
    audioChunksRef.current = [];

    recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);

    recorder.onstop = () => {
      const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      setAudioBlob(blob);
      setAudioURL(url);
    };

    recorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const deleteAudio = () => {
    if (audioURL) URL.revokeObjectURL(audioURL);
    setAudioBlob(null);
    setAudioURL(null);
    setTranscription("");
  };

  /* =============================
     ✅ ADDED: GOOGLE MAPS (NO API, NO BACKEND)
  ============================= */
  const openNearbyHospitals = () => {
    if (!navigator.geolocation) {
      alert("Geolocation not supported");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const lat = position.coords.latitude;
        const lng = position.coords.longitude;

        const query = encodeURIComponent(
          "Pulmonologist OR Respiratory Specialist OR General Physician Hospital"
        );

        const mapsUrl = `https://www.google.com/maps/search/${query}/@${lat},${lng},12z`;
        window.open(mapsUrl, "_blank");
      },
      () => {
        alert("Please allow location access to find nearby hospitals.");
      }
    );
  };

  /* =============================
     RESET
  ============================= */
  const resetAll = () => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    if (audioURL) URL.revokeObjectURL(audioURL);

    setImage(null);
    setImagePreview(null);
    setText("");
    setAudioBlob(null);
    setAudioURL(null);
    setRecording(false);
    setTranscription("");
    setResult(null);
    setLoading(false);

    // ✅ ADDED
    setShowHospitals(false);

    setChatMessages([
      {
        role: "bot",
        text: "Hello! 👋 I can help you understand pneumonia, symptoms, and diagnosis results."
      }
    ]);
    setChatInput("");
    setChatLoading(false);

    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  /* =============================
     SUBMIT
  ============================= */
  const submitData = async () => {
    if (!image) {
      alert("Please upload a chest X-ray image.");
      return;
    }

    setLoading(true);
    setResult(null);

    // ✅ ADDED
    setShowHospitals(false);

    try {
      const validateForm = new FormData();
      validateForm.append("image", image);

      const validateRes = await fetch(
        "http://127.0.0.1:5000/validate-image",
        {
          method: "POST",
          body: validateForm
        }
      );

      const validateData = await validateRes.json();

      if (!validateData.valid) {
        alert(validateData.message || "Invalid image type uploaded.");
        setLoading(false);
        return;
      }

      const formData = new FormData();
      formData.append("image", image);
      if (audioBlob) formData.append("audio", audioBlob);
      if (text) formData.append("text", text);

      const res = await fetch(
        "http://127.0.0.1:5000/diagnose",
        {
          method: "POST",
          body: formData
        }
      );

      const data = await res.json();
      setResult(data);

      if (data.transcription) {
        setTranscription(data.transcription);
      }

      // ✅ ADDED: show button only if pneumonia
      if (
        data.image_prediction &&
        data.image_prediction.toLowerCase().includes("pneumonia")
      ) {
        setShowHospitals(true);
      }

      if (data.image_prediction) {
        setChatMessages((prev) => [
          ...prev,
          {
            role: "bot",
            text: `✅ Diagnosis Result: ${data.image_prediction} (confidence: ${data.image_confidence}).`
          }
        ]);
      }

    } catch (error) {
      console.error(error);
      alert("Backend service is not reachable.");
    }

    setLoading(false);
  };

  /* =============================
     UI
  ============================= */
  return (
    <div className="layout">

      {/* LEFT INFORMATION PANEL */}
      <div className="info-panel">
        <h2>Smart HealthAI</h2>
        <p className="subtitle">
          Multimodal AI System for Smart Healthcare Diagnosis
        </p>

        <div className="info-section highlight">
          <h3>What is Multimodal AI?</h3>
          <p>
            Multimodal AI combines medical imaging, clinical text, and speech
            inputs to enhance diagnostic reliability and reduce ambiguity.
          </p>
        </div>

        <div className="info-section highlight">
          <h3>Modalities Used</h3>
          <ul>
            <li><b>Image (Chest X-ray)</b> – Primary diagnostic evidence</li>
            <li><b>Text</b> – Patient-reported symptoms</li>
            <li><b>Speech</b> – Voice-based symptom input</li>
          </ul>
        </div>

        <div className="info-section highlight">
          <h3>Why Image Modality is Mandatory?</h3>
          <p>
            Pneumonia is fundamentally a radiological condition. Symptoms alone
            can overlap with asthma, bronchitis, or viral infections. Chest
            X-rays provide objective clinical validation and prevent false
            diagnosis.
          </p>
        </div>

        <div className="why-card">
          <i className="fas fa-brain why-icon"></i>
          <h3>AI-Powered Analysis</h3>
          <p>
            Advanced deep learning models analyze X-ray images,
            clinical text, and voice inputs for accurate diagnosis.
          </p>
        </div>

        <div className="why-card">
          <i className="fas fa-shield-heart why-icon"></i>
          <h3>Secure & Private</h3>
          <p>
            Patient data is processed securely and never stored permanently,
            ensuring privacy and ethical compliance.
          </p>
        </div>

        <div className="why-card">
          <i className="fas fa-bolt why-icon"></i>
          <h3>Instant Results</h3>
          <p>
            Get diagnostic insights, explanations, and confidence scores
            within seconds.
          </p>
        </div>

        {/* CHATBOT */}
        <div className="chatbot-container">
          <div className="chatbot-header">
            <i className="fas fa-robot"></i>
            <h3>Smart HealthAI Assistant</h3>
          </div>

          <div className="chatbot-messages">
            {chatMessages.map((msg, idx) => (
              <div
                key={idx}
                className={`chat-message ${msg.role === "bot" ? "bot" : "user"}`}
              >
                {msg.text}
              </div>
            ))}

            {chatLoading && (
              <div className="chat-message bot">
                <i className="fas fa-spinner fa-spin"></i> Thinking...
              </div>
            )}
          </div>

          <div className="chatbot-input">
            <input
              placeholder="Ask about your diagnosis..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") sendChatMessage();
              }}
            />
            <button onClick={sendChatMessage} disabled={chatLoading}>
              <i className="fas fa-paper-plane"></i>
            </button>
          </div>

          <div className="chatbot-footer">
            For informational purposes only. Not a medical diagnosis.
          </div>
        </div>

      </div>

      {/* RIGHT DIAGNOSIS PANEL */}
      <div className="page">
        <div className="card">

          <h1>
            <i className="fas fa-lungs"></i>{" "}
            Multimodal Pneumonia Diagnosis
          </h1>

          {/* IMAGE */}
          <div className="section">
            <h3><i className="fas fa-x-ray"></i> Chest X-ray Upload</h3>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files[0];
                if (!file) return;
                setImage(file);
                setImagePreview(URL.createObjectURL(file));
              }}
            />
            {imagePreview && <img src={imagePreview} alt="X-ray Preview" />}
          </div>

          {/* TEXT */}
          <div className="section">
            <h3><i className="fas fa-notes-medical"></i> Clinical Symptoms (Text)</h3>
            <textarea
              placeholder="Describe symptoms such as fever, cough, breathlessness..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          {/* VOICE */}
          <div className="section">
            <h3><i className="fas fa-microphone-alt"></i> Clinical Symptoms (Voice)</h3>

            {!recording && !audioBlob && (
              <button className="record" onClick={startRecording}>
                <i className="fas fa-microphone"></i> Start Recording
              </button>
            )}

            {recording && (
              <div className="recording-box">
                <i className="fas fa-circle"></i> Recording...
                <button className="stop" onClick={stopRecording}>
                  <i className="fas fa-stop"></i> Stop
                </button>
              </div>
            )}

            {audioURL && (
              <div className="audio-preview">
                <audio controls src={audioURL} />
                <div className="audio-actions">
                  <button onClick={submitData}>
                    <i className="fas fa-check"></i> Use Audio
                  </button>
                  <button className="delete" onClick={deleteAudio}>
                    <i className="fas fa-trash"></i> Delete
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* ACTIONS */}
          <div className="actions">
            <button className="diagnose" onClick={submitData}>
              <i className="fas fa-stethoscope"></i> Diagnose
            </button>
            <button className="reset" onClick={resetAll}>
              <i className="fas fa-rotate-left"></i> Reset
            </button>
          </div>

          {loading && (
            <p className="loading">
              <i className="fas fa-spinner fa-spin"></i> Processing...
            </p>
          )}

          {/* TRANSCRIPTION */}
          {transcription && (
            <div className="section">
              <h3><i className="fas fa-file-medical"></i> Transcribed Text</h3>
              <textarea value={transcription} readOnly />
            </div>
          )}

          {/* RESULT */}
          {result && (
            <div className="section result">
              <h3><i className="fas fa-chart-line"></i> Diagnostic Result</h3>

              <p><b>Image Prediction:</b> {result.image_prediction}</p>
              <p><b>Confidence:</b> {result.image_confidence}</p>

              {result.pneumonia_type && (
                <p><b>Pneumonia Type:</b> {result.pneumonia_type}</p>
              )}

              {result.recommendation && (
                <p><b>Clinical Recommendation:</b> {result.recommendation}</p>
              )}

              {/* ✅ ADDED BUTTON */}
              {showHospitals && (
                <button
                  className="diagnose"
                  style={{ marginTop: "15px", backgroundColor: "#d32f2f" }}
                  onClick={openNearbyHospitals}
                >
                  🏥 Show Nearby Hospitals
                </button>
              )}

              {result.gradcam_image && (
                <>
                  <h4><i className="fas fa-brain"></i> Grad-CAM Explainability</h4>
                  <img
                    src={`http://127.0.0.1:5000/${result.gradcam_image}`}
                    alt="Grad-CAM"
                  />
                </>
              )}

              {result.report_path && (
                <a
                  href={`http://127.0.0.1:5000/${result.report_path}`}
                  download="Diagnosis_Report.pdf"
                >
                  <button className="diagnose">
                    <i className="fas fa-file-download"></i> Download Report
                  </button>
                </a>
              )}
            </div>
          )}

        </div>
      </div>

    </div>
  );
}

export default App;
