import base64
import csv
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from analytics import PlayerAnalytics
from main import (
    bbox_center,
    build_tracks_for_frame,
    detections_for_frame,
    draw_frame,
    resolve_device,
    try_assign_teams,
    update_speed_and_distance,
)
from player_ball_assigner import PlayerBallAssigner
from team_assigner import TeamAssigner
from trackers import Tracker
from view_transformer import ViewTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
HEATMAP_DIR = REPORT_DIR / "heatmaps"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionState:
    session_id: str
    mode: str = "idle"
    status: str = "ready"
    progress: int = 0
    frame_number: int = 0
    latest_jpeg: Optional[bytes] = None
    players: Dict[str, dict] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    analytics: PlayerAnalytics = field(default_factory=lambda: PlayerAnalytics(sprint_speed_threshold_kmh=9999.0))
    previous_positions: Dict[int, tuple] = field(default_factory=dict)
    position_history: Dict[str, List[tuple]] = field(default_factory=dict)
    touch_positions: Dict[str, List[tuple]] = field(default_factory=dict)
    team_positions: Dict[str, List[tuple]] = field(default_factory=dict)
    possession_timeline: List[dict] = field(default_factory=list)
    zone_counts: List[int] = field(default_factory=lambda: [0] * 9)
    heatmaps: Dict[str, str] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    finalized: bool = False
    stop_requested: bool = False
    error: Optional[str] = None
    tracker: Optional[Tracker] = None
    team_assigner: TeamAssigner = field(default_factory=TeamAssigner)
    player_assigner: PlayerBallAssigner = field(default_factory=PlayerBallAssigner)
    view_transformer: ViewTransformer = field(default_factory=ViewTransformer)
    speed_state: Dict[int, dict] = field(default_factory=dict)
    team_ball_control: List[int] = field(default_factory=list)
    teams_ready: bool = False
    device: str = field(default_factory=lambda: resolve_device("auto"))
    lock: threading.RLock = field(default_factory=threading.RLock)

    def snapshot(self):
        with self.lock:
            return {
                "session_id": self.session_id,
                "mode": self.mode,
                "status": self.status,
                "progress": self.progress,
                "frame_number": self.frame_number,
                "players": list(self.players.values()),
                "events": self.events[-30:],
                "summary": self.summary,
                "heatmaps": self.heatmaps,
                "possession_timeline": self.possession_timeline[-120:],
                "zone_occupancy": self.zone_occupancy(),
                "insights": self.insights,
                "finalized": self.finalized,
                "error": self.error,
            }

    def zone_occupancy(self):
        total = max(sum(self.zone_counts), 1)
        return [{"zone": index + 1, "percent": round((count / total) * 100, 1)} for index, count in enumerate(self.zone_counts)]


class SportVisionEngine:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self.tracker_lock = threading.Lock()

    def verify(self):
        gpu = {"available": False, "name": "CPU"}
        try:
            import torch

            gpu["available"] = bool(torch.cuda.is_available())
            gpu["name"] = torch.cuda.get_device_name(0) if gpu["available"] else "CPU"
        except Exception as exc:
            gpu["error"] = str(exc)
        return {"model_exists": MODEL_PATH.exists(), "gpu": gpu}

    def get_tracker(self, session: SessionState):
        with self.tracker_lock:
            if session.tracker is None:
                session.tracker = Tracker(str(MODEL_PATH))
                try:
                    import torch

                    if torch.cuda.is_available():
                        session.tracker.model.to("cuda")
                except Exception:
                    pass
            return session.tracker

    def session(self, session_id=None):
        session_id = session_id or str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    def process_frame(self, frame, session: SessionState, fps=24):
        tracker = self.get_tracker(session)
        detections, cls_names, cls_names_inv = detections_for_frame(tracker, frame, 0.1, session.device)
        tracks = build_tracks_for_frame(tracker, detections, cls_names, cls_names_inv)
        players = tracks["players"]

        assigned_player = -1
        if tracks["ball"]:
            ball_bbox = tracks["ball"][1]["bbox"]
            assigned_player = session.player_assigner.assign_ball_to_player(players, ball_bbox)

        session.teams_ready = try_assign_teams(session.team_assigner, frame, players, session.teams_ready)
        update_speed_and_distance(tracks, session.speed_state, session.view_transformer, session.frame_number, fps)

        if assigned_player != -1 and assigned_player in players:
            players[assigned_player]["has_ball"] = True
            session.team_ball_control.append(players[assigned_player].get("team", 1))
        else:
            session.team_ball_control.append(session.team_ball_control[-1] if session.team_ball_control else 1)

        ball_position = bbox_center(tracks["ball"][1]["bbox"]) if tracks["ball"] else None
        for track_id, player in players.items():
            current_position = bbox_center(player["bbox"])
            previous_position = session.previous_positions.get(track_id)
            stats = session.analytics.update_player(
                player_id=track_id,
                current_position=current_position,
                previous_position=previous_position,
                frame_number=session.frame_number,
                fps=fps,
                ball_position=ball_position,
                team=player.get("team"),
            )
            if current_position is not None:
                session.previous_positions[track_id] = current_position
                self._record_position(session, str(track_id), str(player.get("team", 1)), current_position, frame.shape)
            player.update(stats)
            player["player_id"] = str(track_id)
            player["team"] = str(player.get("team", 1))

        if assigned_player != -1 and ball_position is not None:
            self._record_touch(session, str(assigned_player), str(players.get(assigned_player, {}).get("team", 1)), ball_position, frame.shape)

        annotated = draw_frame(tracker, frame.copy(), tracks, session.team_ball_control)

        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None

        with session.lock:
            session.frame_number += 1
            session.latest_jpeg = encoded.tobytes()
            session.players = {str(player_id): self._player_payload(player) for player_id, player in players.items()}
            if assigned_player != -1:
                session.events.append({"time": time.strftime("%H:%M:%S"), "text": f"Ball touched by #{assigned_player}", "team": "A"})
                session.possession_timeline.append({"frame": session.frame_number, "team": str(players.get(assigned_player, {}).get("team", 1))})
            session.summary = self._summary(session)
        return encoded.tobytes()

    def analyze_video(self, session_id: str, video_path: Path):
        session = self.session(session_id)
        with session.lock:
            session.mode = "upload"
            session.status = "processing"
            session.progress = 1
            session.error = None

        try:
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            index = 0
            while True:
                if session.stop_requested:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                self.process_frame(frame, session, fps=fps)
                index += 1
                with session.lock:
                    session.progress = min(100, int((index / total) * 100))
            cap.release()
            with session.lock:
                session.status = "complete"
                session.progress = 100
            self.finalize_session(session_id)
        except Exception as exc:
            with session.lock:
                session.status = "error"
                session.error = str(exc)

    def _summary(self, session: SessionState):
        players = list(session.players.values())
        total_distance = sum(float(player.get("distance_km") or 0) for player in players)
        total_touches = sum(int(player.get("touch_count") or player.get("ball_touches") or 0) for player in players)
        return {
            "player_count": len(players),
            "total_distance_km": round(max(total_distance, 0), 3),
            "total_touches": max(total_touches, 0),
            "possession_time_sec": session.frame_number / 24,
            "work_rate_km_per_min": round((max(total_distance, 0) / max(session.frame_number / 24 / 60, 0.001)), 3),
        }

    def finalize_session(self, session_id: str):
        session = self.session(session_id)
        with session.lock:
            if session.finalized:
                return session.snapshot()
            session.stop_requested = True
            session.heatmaps = self._generate_heatmaps(session)
            session.insights = self._generate_insights(session)
            session.summary = self._summary(session)
            session.finalized = True
            if session.status == "processing":
                session.status = "complete"
        self._write_json_report(session)
        self._write_csv_report(session)
        self._write_pdf_report(session)
        return session.snapshot()

    def _record_position(self, session, player_id, team, position, shape):
        session.position_history.setdefault(player_id, []).append(position)
        session.team_positions.setdefault(team, []).append(position)
        zone = self._zone(position, shape)
        if zone is not None:
            session.zone_counts[zone] += 1

    def _record_touch(self, session, player_id, team, position, shape):
        if position is None:
            return
        session.touch_positions.setdefault(player_id, []).append(position)
        session.team_positions.setdefault(team, []).append(position)
        zone = self._zone(position, shape)
        if zone is not None:
            session.zone_counts[zone] += 1

    def _zone(self, position, shape):
        try:
            height, width = shape[:2]
            x = max(0, min(width - 1, float(position[0])))
            y = max(0, min(height - 1, float(position[1])))
            col = min(2, int((x / max(width, 1)) * 3))
            row = min(2, int((y / max(height, 1)) * 3))
            return row * 3 + col
        except Exception:
            return None

    def _generate_heatmaps(self, session):
        heatmaps = {}
        for label, points in {**session.position_history, **{f"touches_{k}": v for k, v in session.touch_positions.items()}, **{f"team_{k}": v for k, v in session.team_positions.items()}}.items():
            path = HEATMAP_DIR / f"{session.session_id}_{label}.png"
            self._draw_heatmap(path, points)
            heatmaps[label] = f"/api/heatmaps/{session.session_id}/{path.name}"
        return heatmaps

    def _draw_heatmap(self, path, points):
        canvas = np.zeros((420, 760, 3), dtype=np.uint8)
        canvas[:] = (24, 13, 6)
        cv2.rectangle(canvas, (30, 28), (730, 392), (64, 180, 120), 2)
        cv2.line(canvas, (380, 28), (380, 392), (70, 120, 140), 1)
        cv2.circle(canvas, (380, 210), 56, (70, 120, 140), 1)
        for point in points[-1500:]:
            try:
                x = int(max(30, min(730, float(point[0]) / 1280 * 700 + 30)))
                y = int(max(28, min(392, float(point[1]) / 720 * 364 + 28)))
                cv2.circle(canvas, (x, y), 16, (0, 90, 180), -1)
                cv2.circle(canvas, (x, y), 8, (0, 229, 160), -1)
            except Exception:
                continue
        cv2.imwrite(str(path), canvas)

    def _generate_insights(self, session):
        players = list(session.players.values())
        insights = []
        if players:
            top_distance = max(players, key=lambda p: float(p.get("distance_km") or 0))
            top_work = max(players, key=lambda p: float(p.get("distance_km") or 0) / max(session.frame_number / 24 / 60, 0.001))
            insights.append(f"Player {top_distance['player_id']} covered the most ground.")
            insights.append(f"Player {top_work['player_id']} showed the highest work rate.")
        zones = session.zone_occupancy()
        if zones:
            top_zone = max(zones, key=lambda z: z["percent"])
            insights.append(f"Zone {top_zone['zone']} carried the highest occupancy at {top_zone['percent']}%.")
        if not insights:
            insights.append("More tracking samples are needed for tactical insights.")
        return insights

    def _write_json_report(self, session):
        path = REPORT_DIR / f"{session.session_id}_analytics_report.json"
        path.write_text(json.dumps(session.snapshot(), indent=2), encoding="utf-8")
        return path

    def _write_csv_report(self, session):
        path = REPORT_DIR / f"{session.session_id}_player_history.csv"
        rows = session.snapshot().get("players", [])
        with path.open("w", newline="", encoding="utf-8") as fh:
            fieldnames = ["player_id", "team", "distance_km", "current_speed_kmh", "avg_speed_kmh", "ball_touches", "activity_score"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_pdf_report(self, session):
        path = REPORT_DIR / f"{session.session_id}_match_report.pdf"
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(path), pagesize=letter)
            c.setTitle("SportVision Match Report")
            c.setFont("Helvetica-Bold", 18)
            c.drawString(72, 740, "SPORTVISION Match Report")
            c.setFont("Helvetica", 10)
            y = 710
            for key, value in session.summary.items():
                c.drawString(72, y, f"{key}: {value}")
                y -= 16
            y -= 10
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y, "Tactical Insights")
            c.setFont("Helvetica", 10)
            y -= 18
            for insight in session.insights[:8]:
                c.drawString(84, y, f"- {insight}")
                y -= 15
            for path_url in list(session.heatmaps.values())[:2]:
                image_name = path_url.rsplit("/", 1)[-1]
                image_path = HEATMAP_DIR / image_name
                if image_path.exists():
                    if y < 260:
                        c.showPage()
                        y = 720
                    c.drawImage(str(image_path), 72, y - 180, width=300, height=166)
                    y -= 190
            c.save()
        except Exception:
            path.write_text("SPORTVISION Match Report\n" + json.dumps(session.snapshot(), indent=2), encoding="utf-8")
        return path

    def _player_payload(self, player):
        return {
            "player_id": str(player.get("player_id")),
            "team": str(player.get("team", "N/A")),
            "distance_km": max(float(player.get("distance_km") or 0), 0),
            "current_speed_kmh": max(float(player.get("current_speed_kmh") or 0), 0),
            "avg_speed_kmh": max(float(player.get("avg_speed_kmh") or 0), 0),
            "ball_touches": int(player.get("touch_count") or player.get("ball_touches") or 0),
            "activity_score": max(float(player.get("confidence_score") or 0) * 100, 0),
        }

    def _bbox_center(self, bbox):
        if bbox is None:
            return None
        try:
            return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)
        except Exception:
            return None


engine = SportVisionEngine()
app = FastAPI(title="SportVision", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"ok": True, "product": "SPORTVISION", **engine.verify()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    session = engine.session()
    extension = Path(file.filename or "match.mp4").suffix.lower()
    if extension not in {".mp4", ".mov", ".avi", ".mkv"}:
        return JSONResponse({"ok": False, "message": "Unsupported video type"}, status_code=400)
    output = UPLOAD_DIR / f"{session.session_id}{extension}"
    with output.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            fh.write(chunk)
    with session.lock:
        session.mode = "upload"
        session.status = "uploaded"
        session.summary = {"filename": file.filename}
    return {"ok": True, "session_id": session.session_id, "filename": file.filename}


@app.post("/api/analyze/{session_id}")
def analyze(session_id: str):
    candidates = list(UPLOAD_DIR.glob(f"{session_id}.*"))
    if not candidates:
        return JSONResponse({"ok": False, "message": "Upload not found"}, status_code=404)
    thread = threading.Thread(target=engine.analyze_video, args=(session_id, candidates[0]), daemon=True)
    thread.start()
    return {"ok": True, "session_id": session_id}


@app.post("/api/finalize/{session_id}")
def finalize(session_id: str):
    return engine.finalize_session(session_id)


@app.get("/api/state/{session_id}")
def state(session_id: str):
    return engine.session(session_id).snapshot()


@app.get("/api/heatmaps/{session_id}/{filename}")
def heatmap_file(session_id: str, filename: str):
    path = HEATMAP_DIR / filename
    if path.exists() and filename.startswith(session_id):
        return FileResponse(path)
    return JSONResponse({"ok": False, "message": "heatmap unavailable"}, status_code=404)


@app.get("/api/stream/{session_id}")
def stream(session_id: str):
    session = engine.session(session_id)

    def frames():
        while True:
            jpeg = session.latest_jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/live/{session_id}")
async def live_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = engine.session(session_id)
    session.mode = "live"
    session.status = "processing"
    try:
        while True:
            message = await websocket.receive_json()
            image_data = message.get("image", "")
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            raw = base64.b64decode(image_data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            jpeg = engine.process_frame(frame, session, fps=24)
            if jpeg:
                await websocket.send_json({
                    "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
                    "state": session.snapshot(),
                })
    except WebSocketDisconnect:
        engine.finalize_session(session_id)


@app.get("/api/reports/json/{session_id}")
def report_json(session_id: str):
    session = engine.session(session_id)
    engine.finalize_session(session_id)
    return FileResponse(REPORT_DIR / f"{session.session_id}_analytics_report.json", media_type="application/json")


@app.get("/api/reports/csv/{session_id}")
def report_csv(session_id: str):
    session = engine.session(session_id)
    engine.finalize_session(session_id)
    return FileResponse(REPORT_DIR / f"{session.session_id}_player_history.csv", media_type="text/csv")


@app.get("/api/reports/pdf/{session_id}")
def report_pdf(session_id: str):
    path = REPORT_DIR / f"{session_id}_match_report.pdf"
    engine.finalize_session(session_id)
    return FileResponse(path, media_type="application/pdf")
