import json
import logging
from typing import Any, Dict, Set

from fastapi import WebSocket


LOGGER = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        stale = []
        encoded = json.dumps(message, default=str)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(encoded)
            except Exception as exc:
                LOGGER.error("WebSocket broadcast failed: %s", exc)
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)
