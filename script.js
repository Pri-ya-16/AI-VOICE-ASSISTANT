let mediaRecorder, chunks = [];
let isRecording = false;
const chat = document.getElementById("chat");
const micBtn = document.getElementById("micBtn");

micBtn.onclick = async () => {
  if (!isRecording) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];
    mediaRecorder.start();
    micBtn.classList.add("recording");
    micBtn.innerHTML = '<i class="fa fa-stop-circle" aria-hidden="true"></i>';
    isRecording = true;
    mediaRecorder.ondataavailable = e => chunks.push(e.data);
  } else {
    mediaRecorder.stop();
    micBtn.classList.remove("recording");
    micBtn.innerHTML = '<i class="fa fa-microphone" aria-hidden="true"></i>';
    isRecording = false;

    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      chunks = [];
      addMessage("Processing Audio...", "user");

      const fd = new FormData();
      fd.append("file", blob, "speech.webm");
      fd.append("session_id", "demo-session");

      try {
        const res = await fetch("http://localhost:8000/assistant", {
          method: "POST",
          body: fd
        });
        const data = await res.json();

        // Replace placeholder bubble with user transcript
        replaceLastMessage(data.transcript, "user");

        // Add assistant text
        addMessage(data.response, "assistant");

        // Auto-play TTS
        const audioUrl = "data:audio/mpeg;base64," + data.audio_base64;
        const audio = new Audio(audioUrl);
        audio.play();

      } catch (err) {
        console.error(err);
        addMessage("❌ Error processing audio.", "assistant");
      }
    };
  }
};

function addMessage(text, role) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function replaceLastMessage(text, role) {
  const bubbles = chat.querySelectorAll(`.bubble.${role}`);
  if (bubbles.length > 0) {
    bubbles[bubbles.length - 1].textContent = text;
  } else {
    addMessage(text, role);
  }
}
