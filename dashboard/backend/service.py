import json
import logging
from pathlib import Path
from typing import Any, Dict

from .schemas import StatsPayload
from .storage import AnalyticsRepository


LOGGER = logging.getLogger(__name__)


class DashboardService:
    def __init__(self, repository: AnalyticsRepository, project_root: Path):
        self.repository = repository
        self.project_root = project_root

    def ingest(self, payload: StatsPayload) -> Dict[str, Any]:
        try:
            self.repository.save_payload(payload)
            return {"ok": True, "message": "stats accepted"}
        except Exception as exc:
            LOGGER.error("Stats ingest failed: %s", exc)
            return {"ok": False, "message": "stats ingest failed"}

    def players(self, match_id: str = "default_match"):
        return self.repository.list_players(match_id)

    def player(self, player_id: str, match_id: str = "default_match"):
        return self.repository.get_player(player_id, match_id) or {"player_id": player_id, "missing": True}

    def team(self, team_id: str, match_id: str = "default_match"):
        return self.repository.get_team(team_id, match_id)

    def match_summary(self, match_id: str = "default_match") -> Dict[str, Any]:
        players = self.repository.list_players(match_id)
        history = self.repository.history(match_id, limit=1500)
        teams = sorted({str(player.get("team")) for player in players if player.get("team") is not None})
        total_distance = sum(float(player.get("distance_km") or 0) for player in players)
        total_touches = sum(int(player.get("ball_touches") or 0) for player in players)
        total_possession = sum(float(player.get("possession_time_sec") or 0) for player in players)
        team_possession = self._team_totals(players, "possession_time_sec")
        team_distance = self._team_totals(players, "distance_km")
        team_touches = self._team_totals(players, "ball_touches")
        speed_bands = self._speed_bands(players)

        top_distance = sorted(players, key=lambda p: float(p.get("distance_km") or 0), reverse=True)[:5]
        top_activity = sorted(players, key=lambda p: float(p.get("activity_score") or 0), reverse=True)[:5]
        top_touches = sorted(players, key=lambda p: int(p.get("ball_touches") or 0), reverse=True)[:5]

        distance_trends = self._trend(history, "distance_km")
        touch_trends = self._trend(history, "ball_touches")
        possession_trends = self._trend(history, "possession_time_sec")

        return {
            "match_id": match_id,
            "player_count": len(players),
            "team_count": len(teams),
            "teams": teams,
            "total_distance_km": round(total_distance, 3),
            "total_touches": total_touches,
            "total_possession_sec": round(total_possession, 2),
            "top_distance": top_distance,
            "top_activity": top_activity,
            "top_touches": top_touches,
            "distance_trends": distance_trends,
            "touch_trends": touch_trends,
            "possession_trends": possession_trends,
            "team_possession": team_possession,
            "team_distance": team_distance,
            "team_touches": team_touches,
            "speed_bands": speed_bands,
            "latest_event": self.repository.latest_event(match_id),
        }

    def heatmap_path(self, player_id: str, match_id: str = "default_match"):
        player = self.repository.get_player(player_id, match_id)
        candidates = []
        if player and player.get("heatmap_path"):
            candidates.append(Path(str(player["heatmap_path"])))
        candidates.extend(
            [
                self.project_root / "heatmaps" / f"player_{player_id}.png",
                self.project_root / "dashboard" / "assets" / "heatmaps" / f"player_{player_id}.png",
            ]
        )
        for candidate in candidates:
            try:
                path = candidate if candidate.is_absolute() else self.project_root / candidate
                if path.exists():
                    return path
            except Exception as exc:
                LOGGER.error("Heatmap lookup failed for player %s: %s", player_id, exc)
        return None

    def _trend(self, rows, key: str):
        points = []
        for row in rows[-300:]:
            try:
                payload = json.loads(row.get("payload_json") or "{}")
                points.append(
                    {
                        "timestamp": row.get("timestamp"),
                        "player_id": row.get("player_id"),
                        "value": float(payload.get(key, row.get(key, 0)) or 0),
                    }
                )
            except Exception:
                continue
        return points

    def _team_totals(self, players, key: str):
        totals = {}
        for player in players:
            team = str(player.get("team") or "Unassigned")
            try:
                value = float(player.get(key) or 0)
            except (TypeError, ValueError):
                value = 0.0
            totals[team] = round(totals.get(team, 0.0) + max(value, 0.0), 3)
        return totals

    def _speed_bands(self, players):
        values = []
        for player in players:
            try:
                values.append(max(float(player.get("avg_speed_kmh") or player.get("current_speed_kmh") or 0), 0.0))
            except (TypeError, ValueError):
                values.append(0.0)
        if not values:
            return {"avg": 0.0, "max": 0.0, "sample_size": 0}
        return {
            "avg": round(sum(values) / len(values), 2),
            "max": round(max(values), 2),
            "sample_size": len(values),
        }
