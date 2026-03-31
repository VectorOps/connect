from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import secrets
import typing
import urllib.parse


def create_oauth_state() -> str:
    return secrets.token_hex(16)


def base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def generate_pkce_pair() -> tuple[str, str]:
    verifier = base64url_encode(secrets.token_bytes(32))
    challenge = base64url_encode(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def parse_authorization_input(value: str) -> dict[str, str]:
    raw = value.strip()
    if not raw:
        return {}

    try:
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme and parsed.netloc:
            params = urllib.parse.parse_qs(parsed.query)
            result: dict[str, str] = {}
            for key in ("code", "state", "error", "error_description"):
                current = params.get(key)
                if current and current[0]:
                    result[key] = current[0]
            return result
    except ValueError:
        pass

    if "code=" in raw:
        params = urllib.parse.parse_qs(raw, keep_blank_values=False)
        result: dict[str, str] = {}
        for key in ("code", "state", "error", "error_description"):
            current = params.get(key)
            if current and current[0]:
                result[key] = current[0]
        return result

    if "#" in raw:
        code, state = raw.split("#", 1)
        result = {}
        if code:
            result["code"] = code
        if state:
            result["state"] = state
        return result

    return {"code": raw}


def oauth_success_html(message: str) -> str:
    text = html.escape(message)
    return (
        "<html><body><h1>Authentication complete</h1>"
        f"<p>{text}</p></body></html>"
    )


def oauth_error_html(message: str) -> str:
    text = html.escape(message)
    return (
        "<html><body><h1>Authentication failed</h1>"
        f"<p>{text}</p></body></html>"
    )


class LocalOAuthCallbackServer:
    def __init__(
        self,
        *,
        server: asyncio.AbstractServer,
        host: str,
        port: int,
        callback_path: str,
        state: str,
    ) -> None:
        self._server = server
        self.host = host
        self.port = port
        self.callback_path = callback_path
        self.state = state
        self._result: asyncio.Future[dict[str, str] | None] = asyncio.get_running_loop().create_future()

    @property
    def callback_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.callback_path}"

    @classmethod
    async def start(
        cls,
        *,
        host: str = "127.0.0.1",
        port: int = 1455,
        callback_path: str = "/auth/callback",
        state: str,
    ) -> LocalOAuthCallbackServer:
        holder: dict[str, LocalOAuthCallbackServer] = {}

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            server = holder["server"]
            try:
                request_line = await reader.readline()
                while True:
                    line = await reader.readline()
                    if line in {b"", b"\r\n", b"\n"}:
                        break

                response_status = 200
                response_body = oauth_success_html("Authentication completed. You can close this window.")
                result: dict[str, str] | None = None

                try:
                    method, target, _ = request_line.decode("utf-8", errors="replace").strip().split(" ", 2)
                except ValueError:
                    method = ""
                    target = ""

                if method != "GET":
                    response_status = 405
                    response_body = oauth_error_html("Unsupported request method.")
                else:
                    parsed = urllib.parse.urlparse(target)
                    params = urllib.parse.parse_qs(parsed.query)
                    if parsed.path != callback_path:
                        response_status = 404
                        response_body = oauth_error_html("Callback route not found.")
                    elif params.get("state", [None])[0] != server.state:
                        response_status = 400
                        response_body = oauth_error_html("State mismatch.")
                    elif params.get("error", [None])[0]:
                        description = params.get("error_description", [params["error"][0]])[0]
                        response_status = 400
                        response_body = oauth_error_html(description or "OAuth provider returned an error.")
                        result = {
                            "error": params["error"][0],
                            "error_description": description or params["error"][0],
                        }
                    elif not params.get("code", [None])[0]:
                        response_status = 400
                        response_body = oauth_error_html("Missing authorization code.")
                    else:
                        result = {
                            "code": params["code"][0],
                            "state": params["state"][0],
                        }

                body_bytes = response_body.encode("utf-8")
                writer.write(
                    (
                        f"HTTP/1.1 {response_status} {'OK' if response_status == 200 else 'Error'}\r\n"
                        "Content-Type: text/html; charset=utf-8\r\n"
                        f"Content-Length: {len(body_bytes)}\r\n"
                        "Connection: close\r\n\r\n"
                    ).encode("utf-8")
                )
                writer.write(body_bytes)
                await writer.drain()

                if result is not None and not server._result.done():
                    server._result.set_result(result)
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

        import contextlib

        server = await asyncio.start_server(handle, host=host, port=port)
        socket = next(iter(server.sockets or []), None)
        if socket is None:
            server.close()
            await server.wait_closed()
            raise RuntimeError("Failed to bind OAuth callback server")
        bound_port = int(socket.getsockname()[1])
        instance = cls(server=server, host=host, port=bound_port, callback_path=callback_path, state=state)
        holder["server"] = instance
        return instance

    async def wait_for_callback(self) -> dict[str, str] | None:
        return await self._result

    def cancel_wait(self) -> None:
        if not self._result.done():
            self._result.set_result(None)

    async def close(self) -> None:
        self._server.close()
        await self._server.wait_closed()