from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlayerStat(BaseModel):
    player_id: str = Field(..., description="Tracker/player identifier.")
    team: Optional[str] = None
    distance_km: float = 0.0
    current_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    ball_touches: int = 0
    touch_count: Optional[int] = None
    possession_time_sec: float = 0.0
    activity_score: float = 0.0
    confidence_score: float = 0.0
    heatmap_path: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    def normalized_touches(self) -> int:
        return int(self.touch_count if self.touch_count is not None else self.ball_touches)


class StatsPayload(BaseModel):
    match_id: str = "default_match"
    camera_id: str = "default_camera"
    frame_number: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    players: List[PlayerStat] = Field(default_factory=list)
    teams: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    match: Dict[str, Any] = Field(default_factory=dict)


class ApiResponse(BaseModel):
    ok: bool
    message: str = ""
