"""ComfyUI's live progress over its websocket (/ws?clientId=...), stdlib only (no Qt, no
downloads): a daemon thread per ComfyClient keeps the latest sampler step of each prompt.

ComfyUI sends {"type": "progress", "data": {"value", "max", "prompt_id", "node"}} (and newer
versions "progress_state" with every running node) to the client id that queued the prompt, so
the socket is opened before the first /prompt. Anything unexpected (an old/fake server, a proxy,
https without a certificate) just leaves `state` empty: the progress bars fall back to time.
PIXEL_COMFY_WS=0 turns it off.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import struct
import threading
import time
import urllib.parse

ENV = "PIXEL_COMFY_WS"


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "off", "false", "no")


class ProgressSocket(threading.Thread):
    """Reads ComfyUI's websocket until closed; `progress(prompt_id)` -> (value, max) or None."""

    def __init__(self, base_url: str, client_id: str, connect_timeout: float = 2.0):
        super().__init__(daemon=True, name="comfyui-progress")
        self.base_url = base_url
        self.client_id = client_id
        self.connect_timeout = connect_timeout
        self.connected = threading.Event()
        self.finished = threading.Event()
        self.error: str = ""
        self._lock = threading.Lock()
        self._state: dict[str, tuple[int, int, float, str]] = {}   # prompt -> value, max, when, node
        self._done: set[str] = set()
        self._sock: socket.socket | None = None
        self.messages = 0

    # ---- reading the state
    def progress(self, prompt_id: str) -> tuple[int, int] | None:
        with self._lock:
            s = self._state.get(prompt_id)
        return None if s is None else (s[0], s[1])

    def finished_prompt(self, prompt_id: str) -> bool:
        with self._lock:
            return prompt_id in self._done

    @property
    def alive(self) -> bool:
        return self.is_alive() and not self.finished.is_set()

    def close(self):
        s = self._sock
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    # ---- the thread
    def run(self):
        try:
            self._connect()
            self.connected.set()
            self._loop()
        except Exception as e:  # noqa: BLE001 - no progress is fine
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.finished.set()
            self.connected.set()          # nobody waits forever
            self.close()

    def _connect(self):
        u = urllib.parse.urlsplit(self.base_url)
        secure = u.scheme == "https"
        host = u.hostname or "127.0.0.1"
        port = u.port or (443 if secure else 80)
        raw = socket.create_connection((host, port), timeout=self.connect_timeout)
        sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host) if secure else raw
        self._sock = sock
        key = base64.b64encode(os.urandom(16)).decode()
        path = (u.path.rstrip("/") or "") + "/ws?clientId=" + urllib.parse.quote(self.client_id)
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
               "User-Agent: pixel-pipeline\r\n\r\n")
        sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("closed during the handshake")
            head += chunk
            if len(head) > 65536:
                raise ConnectionError("handshake too long")
        status, _, rest = head.partition(b"\r\n\r\n")
        if b" 101 " not in status.split(b"\r\n", 1)[0] + b" ":
            raise ConnectionError(status.split(b"\r\n", 1)[0].decode("latin-1", "replace"))
        self._buf = rest
        sock.settimeout(None)             # wait for messages as long as the client lives

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send(self, opcode: int, payload: bytes = b""):
        mask = os.urandom(4)
        n = len(payload)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(head + mask + body)

    def _loop(self):
        parts: list[bytes] = []
        op0 = 1
        while True:
            b0, b1 = self._read(2)
            fin, op = b0 & 0x80, b0 & 0x0F
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            payload = self._read(n)
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if op == 8:                    # close
                return
            if op == 9:                    # ping -> pong
                self._send(10, payload)
                continue
            if op == 10:
                continue
            if op in (1, 2):
                op0, parts = op, [payload]
            elif op == 0:
                parts.append(payload)
            if not fin:
                continue
            data = b"".join(parts)
            parts = []
            if op0 == 1:                   # text = JSON; binary = preview images (ignored)
                try:
                    self._message(json.loads(data.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    pass

    def _message(self, msg):
        if not isinstance(msg, dict):
            return
        self.messages += 1
        kind, data = msg.get("type"), msg.get("data") or {}
        if not isinstance(data, dict):
            return
        now = time.monotonic()
        if kind == "progress":
            pid = data.get("prompt_id")
            try:
                v, m = int(data.get("value", 0)), int(data.get("max", 0))
            except (TypeError, ValueError):
                return
            if pid and m > 0:
                with self._lock:
                    self._state[pid] = (v, m, now, str(data.get("node", "")))
        elif kind == "progress_state":
            pid = data.get("prompt_id")
            nodes = data.get("nodes") or {}
            running = [n for n in nodes.values() if isinstance(n, dict) and n.get("state") == "running"
                       and (n.get("max") or 0) > 1]
            if pid and running:
                n = running[-1]
                with self._lock:
                    self._state[pid] = (int(n.get("value", 0)), int(n.get("max", 1)), now,
                                        str(n.get("node_id", "")))
        elif kind in ("executing", "execution_success", "execution_error", "execution_interrupted"):
            pid = data.get("prompt_id")
            if pid and (kind != "executing" or data.get("node") is None):
                with self._lock:
                    self._done.add(pid)


_sockets: dict[tuple[str, str], ProgressSocket] = {}
_tried: dict[tuple[str, str], float] = {}
_lock = threading.Lock()
RETRY_S = 20.0                             # a failed connect is tried again after this long


def watch(base_url: str, client_id: str, wait: float = 1.0) -> ProgressSocket | None:
    """The live socket for this client (opened now if needed, waiting up to `wait` s for the
    handshake so the first prompt's steps arrive). None = no websocket (off, failed lately)."""
    if not enabled():
        return None
    key = (base_url, client_id)
    with _lock:
        ws = _sockets.get(key)
        if ws is not None and ws.alive:
            return ws
        if time.monotonic() - _tried.get(key, -1e9) < RETRY_S and ws is not None:
            return None
        _tried[key] = time.monotonic()
        ws = ProgressSocket(base_url, client_id)
        _sockets[key] = ws
        ws.start()
    ws.connected.wait(wait)
    return ws if ws.alive else None


def current(base_url: str, client_id: str) -> ProgressSocket | None:
    ws = _sockets.get((base_url, client_id))
    return ws if ws is not None and ws.alive else None
