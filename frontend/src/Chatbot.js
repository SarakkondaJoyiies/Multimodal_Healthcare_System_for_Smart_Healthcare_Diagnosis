import { useState } from "react";

function Chatbot() {
  const [msg, setMsg] = useState("");
  const [reply, setReply] = useState("");

  const sendMsg = async () => {
    const res = await fetch("http://127.0.0.1:5000/chatbot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    setReply(data.reply);
  };

  return (
    <div className="chatbot">
      <h3>SmartHealthAI Assistant</h3>
      <textarea onChange={(e) => setMsg(e.target.value)} />
      <button onClick={sendMsg}>Ask</button>
      <p>{reply}</p>
    </div>
  );
}

export default Chatbot;