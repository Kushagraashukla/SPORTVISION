import csv
import json
import logging
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from .schemas import PlayerStat, StatsPayload


LOGGER = logging.getLogger(__name__)


def _model_json(model) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return model.json()


class AnalyticsRepository:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "analytics.db"
        self.jsonl_path = self.cache_dir / "stats_events.jsonl"
        self.lock = RLock()
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self.lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS player_latest (
                    match_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    team TEXT,
                    distance_km REAL,
                    current_speed_kmh REAL,
                    avg_speed_kmh REAL,
                    ball_touches INTEGER,
                    possession_time_sec REAL,
                    activity_score REAL,
                    confidence_score REAL,
                    heatmap_path TEXT,
                    timestamp TEXT,
                    payload_json TEXT,
                    PRIMARY KEY (match_id, player_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS player_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    team TEXT,
                    distance_km REAL,
                    ball_touches INTEGER,
                    possession_time_sec REAL,
                    activity_score REAL,
                    timestamp TEXT,
                    frame_number INTEGER,
                    payload_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS match_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT,
                    frame_number INTEGER,
                    payload_json TEXT
                )
                """
            )

    def save_payload(self, payload: StatsPayload) -> None:
        try:
            event_json = _model_json(payload)
            with self.lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO match_events(match_id, camera_id, timestamp, frame_number, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        payload.match_id,
                        payload.camera_id,
                        payload.timestamp.isoformat(),
                        payload.frame_number,
                        event_json,
                    ),
                )

                for player in payload.players:
                    self._save_player(conn, payload, player)

            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(event_json + "\n")
        except Exception as exc:
            LOGGER.error("Failed to persist analytics payload: %s", exc)

    def _save_player(self, conn, payload: StatsPayload, player: PlayerStat) -> None:
        touches = player.normalized_touches()
        player_json = _model_json(player)
        values = (
            payload.match_id,
            payload.camera_id,
            player.player_id,
            player.team,
            float(player.distance_km or 0),
            float(player.current_speed_kmh or 0),
            float(player.avg_speed_kmh or 0),
            touches,
            float(player.possession_time_sec or 0),
            float(player.activity_score or 0),
            float(player.confidence_score or 0),
            player.heatmap_path,
            payload.timestamp.isoformat(),
            player_json,
        )
        conn.execute(
            """
            INSERT INTO player_latest (
                match_id, camera_id, player_id, team, distance_km, current_speed_kmh,
                avg_speed_kmh, ball_touches, possession_time_sec, activity_score,
                confidence_score, heatmap_path, timestamp, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id, player_id) DO UPDATE SET
                camera_id=excluded.camera_id,
                team=excluded.team,
                distance_km=excluded.distance_km,
                current_speed_kmh=excluded.current_speed_kmh,
                avg_speed_kmh=excluded.avg_speed_kmh,
                ball_touches=excluded.ball_touches,
                possession_time_sec=excluded.possession_time_sec,
                activity_score=excluded.activity_score,
                confidence_score=excluded.confidence_score,
                heatmap_path=excluded.heatmap_path,
                timestamp=excluded.timestamp,
                payload_json=excluded.payload_json
            """,
            values,
        )
        conn.execute(
            """
            INSERT INTO player_history (
                match_id, camera_id, player_id, team, distance_km, ball_touches,
                possession_time_sec, activity_score, timestamp, frame_number, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.match_id,
                payload.camera_id,
                player.player_id,
                player.team,
                float(player.distance_km or 0),
                touches,
                float(player.possession_time_sec or 0),
                float(player.activity_score or 0),
                payload.timestamp.isoformat(),
                payload.frame_number,
                player_json,
            ),
        )

    def list_players(self, match_id: str = "default_match") -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM player_latest WHERE match_id=? ORDER BY team, player_id", (match_id,))

    def get_player(self, player_id: str, match_id: str = "default_match") -> Optional[Dict[str, Any]]:
        rows = self._fetch_all("SELECT * FROM player_latest WHERE match_id=? AND player_id=?", (match_id, player_id))
        return rows[0] if rows else None

    def get_team(self, team_id: str, match_id: str = "default_match") -> Dict[str, Any]:
        players = self._fetch_all(
            "SELECT * FROM player_latest WHERE match_id=? AND COALESCE(team, '')=? ORDER BY player_id",
            (match_id, team_id),
        )
        return {"team": team_id, "players": players}

    def history(self, match_id: str = "default_match", limit: int = 1200) -> List[Dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM player_history
            WHERE match_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (match_id, limit),
        )[::-1]

    def latest_event(self, match_id: str = "default_match") -> Optional[Dict[str, Any]]:
        rows = self._fetch_all(
            "SELECT * FROM match_events WHERE match_id=? ORDER BY id DESC LIMIT 1",
            (match_id,),
        )
        if not rows:
            return None
        try:
            return json.loads(rows[0]["payload_json"])
        except Exception:
            return rows[0]

    def export_json(self, output_path: Path, match_id: str = "default_match") -> Path:
        data = {
            "players": self.list_players(match_id),
            "history": self.history(match_id, limit=100000),
            "latest_event": self.latest_event(match_id),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return output_path

    def export_csv(self, output_path: Path, match_id: str = "default_match") -> Path:
        rows = self.history(match_id, limit=100000)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["match_id", "player_id"]
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    def _fetch_all(self, query: str, params=()) -> List[Dict[str, Any]]:
        try:
            with self.lock, self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query, params).fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:
            LOGGER.error("Database read failed: %s", exc)
            return []
