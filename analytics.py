from collections import deque
from dataclasses import dataclass, field
from math import isfinite, sqrt


@dataclass
class _PlayerState:
    player_id: object
    team: object = None
    distance_px: float = 0.0
    speed_history: deque = field(default_factory=lambda: deque(maxlen=10))
    heatmap_positions: list = field(default_factory=list)
    sprint_count: int = 0
    touch_count: int = 0
    sprint_frames: int = 0
    in_sprint: bool = False
    last_touch_frame: int = -10_000
    valid_updates: int = 0
    ignored_updates: int = 0
    last_stable_position: tuple = None
    smoothed_position: tuple = None


class PlayerAnalytics:
    """Stable per-player analytics for demo-friendly football insights."""

    def __init__(
        self,
        sprint_speed_threshold_kmh=20.0,
        sprint_min_frames=10,
        speed_window=10,
        max_speed_kmh=38.0,
        touch_distance_px=30.0,
        touch_cooldown_frames=15,
        tracker_jump_px=100.0,
        pixels_per_meter=12.0,
        position_smoothing=0.35,
    ):
        self.players = {}
        self.sprint_speed_threshold_kmh = float(sprint_speed_threshold_kmh)
        self.sprint_min_frames = int(sprint_min_frames)
        self.speed_window = int(speed_window)
        self.max_speed_kmh = float(max_speed_kmh)
        self.touch_distance_px = float(touch_distance_px)
        self.touch_cooldown_frames = int(touch_cooldown_frames)
        self.tracker_jump_px = float(tracker_jump_px)
        self.pixels_per_meter = pixels_per_meter
        self.position_smoothing = float(position_smoothing)

    def update_player(
        self,
        player_id,
        current_position=None,
        previous_position=None,
        frame_number=0,
        fps=0,
        ball_position=None,
        team=None,
    ):
        state = self._get_state(player_id, team)
        if team is not None:
            state.team = team

        current = self._safe_point(current_position)
        previous = self._safe_point(previous_position) or state.last_stable_position

        if current is not None:
            state.heatmap_positions.append(current)

        if current is None:
            state.ignored_updates += 1
            self._update_touch(state, current, ball_position, frame_number)
            return self.get_player_stats(player_id)

        if previous is None:
            state.last_stable_position = current
            state.smoothed_position = current
            state.valid_updates += 1
            self._update_touch(state, current, ball_position, frame_number)
            return self.get_player_stats(player_id)

        jump_px = self._distance(current, previous)
        if jump_px > self.tracker_jump_px:
            state.ignored_updates += 1
            self._update_touch(state, current, ball_position, frame_number)
            return self.get_player_stats(player_id)

        smoothed = self._smooth_position(state.smoothed_position or previous, current)
        distance_px = self._distance(smoothed, state.smoothed_position or previous)
        raw_speed_kmh = self._speed_kmh(distance_px, fps)
        if raw_speed_kmh > self.max_speed_kmh:
            state.ignored_updates += 1
            speed_kmh = self._current_speed(state)
            self._update_sprint(state, speed_kmh)
            self._update_touch(state, current, ball_position, frame_number)
            return self.get_player_stats(player_id)

        safe_speed = self._safe_metric(raw_speed_kmh)
        safe_distance = self._safe_metric(distance_px)
        state.distance_px += safe_distance
        state.speed_history.append(safe_speed)
        state.last_stable_position = current
        state.smoothed_position = smoothed
        state.valid_updates += 1

        speed_kmh = self._current_speed(state)
        self._update_sprint(state, speed_kmh)
        self._update_touch(state, current, ball_position, frame_number)
        return self.get_player_stats(player_id)

    def update(self, *args, **kwargs):
        return self.update_player(*args, **kwargs)

    def get_player_stats(self, player_id):
        state = self.players.get(player_id)
        if state is None:
            return self._empty_stats(player_id)

        return {
            "player_id": state.player_id,
            "distance_km": round(self._distance_km(state.distance_px), 3),
            "current_speed_kmh": round(self._current_speed(state), 2),
            "avg_speed_kmh": round(self._avg_speed(state), 2),
            "sprint_count": state.sprint_count,
            "touch_count": state.touch_count,
            "team": state.team,
            "confidence_score": round(self._confidence_score(state), 2),
        }

    def get_all_stats(self):
        return {player_id: self.get_player_stats(player_id) for player_id in self.players}

    def get_heatmap_history(self, player_id=None):
        if player_id is not None:
            state = self.players.get(player_id)
            return [] if state is None else list(state.heatmap_positions)
        return {pid: list(state.heatmap_positions) for pid, state in self.players.items()}

    def reset(self):
        self.players.clear()

    def _get_state(self, player_id, team=None):
        if player_id not in self.players:
            self.players[player_id] = _PlayerState(player_id=player_id, team=team)
            self.players[player_id].speed_history = deque(maxlen=self.speed_window)
        return self.players[player_id]

    def _safe_point(self, point):
        try:
            if point is None or len(point) < 2:
                return None
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        if not isfinite(x) or not isfinite(y):
            return None
        return (x, y)

    def _safe_frame(self, frame_number):
        try:
            return int(frame_number)
        except (TypeError, ValueError):
            return 0

    def _distance(self, first, second):
        if first is None or second is None:
            return 0.0
        value = sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)
        return self._safe_metric(value)

    def _smooth_position(self, previous, current):
        if previous is None:
            return current
        alpha = max(0.0, min(1.0, self.position_smoothing))
        return (
            (previous[0] * (1.0 - alpha)) + (current[0] * alpha),
            (previous[1] * (1.0 - alpha)) + (current[1] * alpha),
        )

    def _speed_kmh(self, distance_px, fps):
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return 0.0

        if fps <= 0:
            return self._safe_metric(0.0)

        distance_m = self._distance_meters(distance_px)
        return self._safe_metric(distance_m * fps * 3.6)

    def _distance_meters(self, distance_px):
        if self.pixels_per_meter:
            try:
                pixels_per_meter = float(self.pixels_per_meter)
                if pixels_per_meter > 0:
                    return distance_px / pixels_per_meter
            except (TypeError, ValueError):
                pass
        return self._safe_metric(distance_px)

    def _distance_km(self, distance_px):
        return self._safe_metric(self._distance_meters(distance_px) / 1000.0)

    def _current_speed(self, state):
        if not state.speed_history:
            return 0.0
        return self._safe_metric(sum(state.speed_history) / len(state.speed_history))

    def _avg_speed(self, state):
        return self._current_speed(state)

    def _update_sprint(self, state, speed_kmh):
        if speed_kmh > self.sprint_speed_threshold_kmh:
            state.sprint_frames += 1
            if state.sprint_frames >= self.sprint_min_frames and not state.in_sprint:
                state.sprint_count += 1
                state.in_sprint = True
        else:
            state.sprint_frames = 0
            state.in_sprint = False

    def _update_touch(self, state, player_position, ball_position, frame_number):
        ball = self._safe_point(ball_position)
        frame = self._safe_frame(frame_number)

        if player_position is None or ball is None:
            return

        if frame - state.last_touch_frame < self.touch_cooldown_frames:
            return

        if self._distance(player_position, ball) < self.touch_distance_px:
            state.touch_count += 1
            state.last_touch_frame = frame

    def _confidence_score(self, state):
        total_updates = state.valid_updates + state.ignored_updates
        if total_updates <= 0:
            return 0.0

        valid_ratio = state.valid_updates / total_updates
        history_bonus = min(len(state.heatmap_positions) / 50.0, 1.0)
        return max(0.0, min(1.0, (valid_ratio * 0.8) + (history_bonus * 0.2)))

    def _empty_stats(self, player_id):
        return {
            "player_id": player_id,
            "distance_km": 0.0,
            "current_speed_kmh": 0.0,
            "avg_speed_kmh": 0.0,
            "sprint_count": 0,
            "touch_count": 0,
            "team": None,
            "confidence_score": 0.0,
        }

    def _safe_metric(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not isfinite(value) or value < 0:
            return 0.0
        return value
