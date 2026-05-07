from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import websockets


class WebSocketTopicBroadcaster:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_clients: int = 8,
        max_message_bytes: int = 8192,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.max_clients = int(max_clients)
        self.max_message_bytes = int(max_message_bytes)
        self._clients: set[Any] = set()
        self._lock = asyncio.Lock()
        self._server: Any = None

    async def __aenter__(self) -> "WebSocketTopicBroadcaster":
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
            max_size=self.max_message_bytes,
            max_queue=4,
        )
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        async with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            with contextlib.suppress(Exception):
                await client.close()

    async def publish(self, message: str) -> None:
        if len(message.encode("utf-8")) > self.max_message_bytes:
            raise ValueError("topic message exceeds max_message_bytes")
        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        results = await asyncio.gather(
            *(client.send(message) for client in clients),
            return_exceptions=True,
        )
        stale_clients = [
            client
            for client, result in zip(clients, results, strict=True)
            if isinstance(result, Exception)
        ]
        if stale_clients:
            async with self._lock:
                for client in stale_clients:
                    self._clients.discard(client)

    async def _handler(self, websocket: Any) -> None:
        async with self._lock:
            if len(self._clients) >= self.max_clients:
                await websocket.close(code=1013, reason="too many clients")
                return
            self._clients.add(websocket)
        try:
            await _wait_closed(websocket)
        finally:
            async with self._lock:
                self._clients.discard(websocket)


async def _wait_closed(websocket: Any) -> None:
    wait_closed: Callable[[], Awaitable[None]] | None = getattr(websocket, "wait_closed", None)
    if callable(wait_closed):
        await wait_closed()
        return
    async for _message in websocket:
        pass
