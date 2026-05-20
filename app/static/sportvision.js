let sessionId = localStorage.getItem("sportvision_session") || crypto.randomUUID();
let liveSocket = null;
let liveTimer = null;

localStorage.setItem("sportvision_session", sessionId);

function show(id) {
  document.querySelectorAll(".screen").forEach((screen) => screen.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  if (id === "dashboard") refreshState();
}

async function upload() {
  const file = document.getElementById("file").files[0];
  const msg = document.getElementById("uploadMsg");
  if (!file) { msg.textContent = "Choose a video first."; return; }
  if (!/\.(mp4|mov|avi|mkv)$/i.test(file.name)) { msg.textContent = "Unsupported video type."; return; }
  if (file.size > 4 * 1024 ** 3) { msg.textContent = "File exceeds 4GB."; return; }
  msg.textContent = "Uploading...";
  const form = new FormData();
  form.append("file", file);
  const uploaded = await fetch("/api/upload", { method: "POST", body: form }).then((r) => r.json());
  if (!uploaded.ok) { msg.textContent = uploaded.message || "Upload failed."; return; }
  sessionId = uploaded.session_id;
  localStorage.setItem("sportvision_session", sessionId);
  attachUploadStream();
  msg.textContent = "Analysis started.";
  show("upload");
  await fetch(`/api/analyze/${sessionId}`, { method: "POST" });
  pollState();
}

function attachUploadStream() {
  const streamUrl = `/api/stream/${sessionId}?t=${Date.now()}`;
  ["liveAnnotated", "uploadAnnotated"].forEach((id) => {
    const image = document.getElementById(id);
    if (image) image.src = streamUrl;
  });
}

async function startLive() {
  show("live");
  const video = document.getElementById("webcam");
  const annotated = document.getElementById("liveAnnotated");
  annotated.src = "";
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  video.srcObject = stream;
  liveSocket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live/${sessionId}`);
  liveSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.image) annotated.src = data.image;
    if (data.state) renderState(data.state);
  };
  liveSocket.onopen = () => {
    const canvas = document.getElementById("capture");
    const ctx = canvas.getContext("2d");
    liveTimer = setInterval(() => {
      if (liveSocket.readyState !== WebSocket.OPEN || video.readyState < 2) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      liveSocket.send(JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.72) }));
    }, 160);
  };
}

function stopLive() {
  if (liveTimer) clearInterval(liveTimer);
  if (liveSocket) liveSocket.close();
  const video = document.getElementById("webcam");
  if (video.srcObject) video.srcObject.getTracks().forEach((track) => track.stop());
  fetch(`/api/finalize/${sessionId}`, { method: "POST" }).then(() => refreshState());
}

function pollState() {
  refreshState();
  setTimeout(pollState, 1000);
}

async function refreshState() {
  try {
    const state = await fetch(`/api/state/${sessionId}`).then((r) => r.json());
    renderState(state);
  } catch (error) {
    console.warn("state unavailable", error);
  }
}

function renderState(state) {
  const players = state.players || [];
  const summary = state.summary || {};
  document.getElementById("kPlayers").textContent = summary.player_count ?? players.length;
  document.getElementById("kDistance").textContent = `${Number(summary.total_distance_km || 0).toFixed(2)} km`;
  document.getElementById("kTouches").textContent = summary.total_touches || 0;
  document.getElementById("kStatus").textContent = (state.status || "ready").toUpperCase();
  document.getElementById("players").innerHTML = players.map(playerRow).join("") || `<div class="row"><span>--</span><span>Awaiting data</span><span>--</span><span>0.00 km</span><span>0.0 km/h</span><span>0</span><span><div class="bar"><span style="width:0%"></span></div></span></div>`;
  document.getElementById("speedBars").innerHTML = players.slice().sort((a, b) => Number(b.avg_speed_kmh || 0) - Number(a.avg_speed_kmh || 0)).slice(0, 6).map((p) => {
    const speed = Math.max(Number(p.avg_speed_kmh || p.current_speed_kmh || 0), 0);
    return `<label><span>#${p.player_id}</span><div class="bar"><span style="width:${Math.min(speed * 4, 100)}%"></span></div><strong>${speed.toFixed(1)}</strong></label>`;
  }).join("") || "<p style='color:var(--muted)'>Speed data will appear after stable tracking.</p>";
  document.getElementById("liveEvents").innerHTML = (state.events || []).slice(-12).reverse().map((e) => `<div><span>${e.time}</span><p>${e.text}</p></div>`).join("");
  document.getElementById("timeline").innerHTML = (state.possession_timeline || []).slice(-24).map((p) => `<div title="Team ${p.team}" style="height:${p.team === "1" ? 70 : 42}%;background:${p.team === "1" ? "var(--neon)" : "var(--neon2)"}"></div>`).join("") || "<p style='color:var(--muted)'>Possession timeline appears after ball touches.</p>";
  document.getElementById("zones").innerHTML = (state.zone_occupancy || Array.from({length:9}, (_,i)=>({zone:i+1,percent:0}))).map((z) => `<div class="zone"><span>Z${z.zone}</span><strong>${Number(z.percent||0).toFixed(1)}%</strong></div>`).join("");
  document.getElementById("heatmaps").innerHTML = Object.entries(state.heatmaps || {}).slice(0, 8).map(([label, url]) => `<figure><img src="${url}?t=${Date.now()}" alt="${label} heatmap"><figcaption>${label}</figcaption></figure>`).join("") || "<p style='color:var(--muted)'>Heatmaps generate automatically when a session finalizes.</p>";
  document.getElementById("insights").innerHTML = (state.insights || []).map((item) => `<div>${item}</div>`).join("") || "<p style='color:var(--muted)'>Tactical insights will appear after enough stable tracking samples.</p>";
  ["json", "csv", "pdf"].forEach((type) => {
    document.getElementById(`${type}Report`).href = `/api/reports/${type}/${sessionId}`;
  });
}

function playerRow(player) {
  const distance = Math.max(Number(player.distance_km || 0), 0);
  const speed = Math.max(Number(player.current_speed_kmh || player.avg_speed_kmh || 0), 0);
  const touches = Math.max(Number(player.ball_touches || 0), 0);
  const activity = Math.min(Math.max(Number(player.activity_score || 0), 0), 100);
  return `<div class="row"><span>#${player.player_id}</span><strong>Player ${player.player_id}</strong><span>Team ${player.team ?? "N/A"}</span><span>${distance.toFixed(2)} km</span><span>${speed.toFixed(1)} km/h</span><span>${touches}</span><span><div class="bar"><span style="width:${activity}%"></span></div></span></div>`;
}

refreshState();
