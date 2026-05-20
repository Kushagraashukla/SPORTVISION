import logging
from pathlib import Path

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .reports import ReportService
from .schemas import ApiResponse, StatsPayload
from .service import DashboardService
from .storage import AnalyticsRepository
from .websocket_manager import ConnectionManager


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BACKEND_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent
CACHE_DIR = DASHBOARD_DIR / "cache"
REPORTS_DIR = DASHBOARD_DIR / "reports"
FRONTEND_DIST = DASHBOARD_DIR / "frontend" / "dist"
FRONTEND_STATIC = DASHBOARD_DIR / "frontend" / "static"

repository = AnalyticsRepository(CACHE_DIR)
service = DashboardService(repository, PROJECT_ROOT)
reports = ReportService(repository, REPORTS_DIR)
websockets = ConnectionManager()

app = FastAPI(
    title="Football Intelligence Dashboard",
    version="1.0.0",
    description="Independent passive analytics dashboard for football CV systems.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
elif FRONTEND_STATIC.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_STATIC, html=True), name="frontend")


def get_service() -> DashboardService:
    return service


def payload_to_dict(payload: StatsPayload):
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    return payload.dict()


@app.get("/health")
def health():
    return {"ok": True, "service": "football-dashboard"}


@app.post("/stats", response_model=ApiResponse)
async def post_stats(payload: StatsPayload, dashboard: DashboardService = Depends(get_service)):
    try:
        result = dashboard.ingest(payload)
        await websockets.broadcast({"type": "stats", "payload": payload_to_dict(payload)})
        return ApiResponse(**result)
    except Exception as exc:
        LOGGER.error("POST /stats failed safely: %s", exc)
        return ApiResponse(ok=False, message="stats rejected safely")


@app.get("/players")
def get_players(match_id: str = "default_match", dashboard: DashboardService = Depends(get_service)):
    try:
        return {"players": dashboard.players(match_id)}
    except Exception as exc:
        LOGGER.error("GET /players failed safely: %s", exc)
        return {"players": []}


@app.get("/player/{player_id}")
def get_player(player_id: str, match_id: str = "default_match", dashboard: DashboardService = Depends(get_service)):
    try:
        return dashboard.player(player_id, match_id)
    except Exception as exc:
        LOGGER.error("GET /player/%s failed safely: %s", player_id, exc)
        return {"player_id": player_id, "missing": True}


@app.get("/heatmap/{player_id}")
def get_heatmap(player_id: str, match_id: str = "default_match", dashboard: DashboardService = Depends(get_service)):
    try:
        path = dashboard.heatmap_path(player_id, match_id)
        if path:
            return FileResponse(path)
        return JSONResponse({"ok": False, "message": "heatmap unavailable"}, status_code=404)
    except Exception as exc:
        LOGGER.error("GET /heatmap/%s failed safely: %s", player_id, exc)
        return JSONResponse({"ok": False, "message": "heatmap unavailable"}, status_code=404)


@app.get("/team/{team_id}")
def get_team(team_id: str, match_id: str = "default_match", dashboard: DashboardService = Depends(get_service)):
    try:
        return dashboard.team(team_id, match_id)
    except Exception as exc:
        LOGGER.error("GET /team/%s failed safely: %s", team_id, exc)
        return {"team": team_id, "players": []}


@app.get("/match_summary")
def get_match_summary(match_id: str = "default_match", dashboard: DashboardService = Depends(get_service)):
    try:
        return dashboard.match_summary(match_id)
    except Exception as exc:
        LOGGER.error("GET /match_summary failed safely: %s", exc)
        return {"match_id": match_id, "player_count": 0, "teams": []}


@app.get("/reports/json")
def export_json(match_id: str = "default_match"):
    return FileResponse(reports.json_export(match_id), media_type="application/json")


@app.get("/reports/csv")
def export_csv(match_id: str = "default_match"):
    return FileResponse(reports.csv_export(match_id), media_type="text/csv")


@app.get("/reports/pdf")
def export_pdf(match_id: str = "default_match"):
    return FileResponse(reports.pdf_report(match_id), media_type="application/pdf")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websockets.connect(websocket)
    try:
        await websocket.send_json({"type": "snapshot", "payload": service.match_summary()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websockets.disconnect(websocket)
    except Exception as exc:
        LOGGER.error("WebSocket recovered after failure: %s", exc)
        websockets.disconnect(websocket)
