"""Bounded local power RPC with a separate, explicitly paired credential."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout

from .tv import TvAuthorizationIncomplete, TvConnectionError, TvError, TvPairingRejected

_MAX_BODY = 65_536
_SESSION_TIMEOUT = ClientTimeout(total=None)


class TvPowerPairingRejected(TvPairingRejected):
    """The power interface explicitly rejected authorization."""


class TvPowerAuthorizationIncomplete(TvAuthorizationIncomplete):
    """The power interface did not complete authorization."""


class TvPowerClient:
    """Use a shared session without owning its lifetime or retrying writes."""

    def __init__(self, session: ClientSession, host: str, token: str | None) -> None:
        self._session = session
        self._url = f"https://{host}:1516/"
        self._token = token
        self._authorization_error: (
            type[TvPowerPairingRejected | TvPowerAuthorizationIncomplete] | None
        ) = None

    @property
    def token(self) -> str | None:
        """The saved or successfully issued power credential."""
        return self._token

    def _ensure_authorized(self, *, pairing: bool = False) -> None:
        if self._authorization_error is not None:
            raise self._authorization_error("TV power authorization is suspended") from None
        if not pairing and not self._token:
            self._authorization_error = TvPowerAuthorizationIncomplete
            raise TvPowerAuthorizationIncomplete("TV power authorization is required") from None

    async def async_pair(self) -> str:
        """Perform one explicit pairing exchange without replacing a token on failure."""
        self._ensure_authorized(pairing=True)
        try:
            result = await self._request("createAccessToken", seconds=120)
            token = result.get("AccessToken")
            if not isinstance(token, str) or not token.strip():
                raise TvPowerAuthorizationIncomplete("TV power authorization is incomplete")
        except asyncio.CancelledError:
            self._authorization_error = TvPowerAuthorizationIncomplete
            raise
        except TvPowerPairingRejected as error:
            failure = error
        except TvError:
            self._authorization_error = TvPowerAuthorizationIncomplete
            failure = TvPowerAuthorizationIncomplete("TV power authorization is incomplete")
        else:
            self._token = token
            return token
        raise failure from None

    async def async_read_power(self) -> bool:
        """Return only explicit on or off evidence; missing evidence is an error."""
        self._ensure_authorized()
        result = await self._request("powerControl", params={"AccessToken": self._token})
        return self._power(result)

    async def async_read_art_mode(self) -> bool:
        """Return only an explicit Art Mode flag."""
        self._ensure_authorized()
        result = await self._request("artModeControl", params={"AccessToken": self._token})
        art_mode = result.get("artMode")
        if art_mode == "artModeOn":
            return True
        if art_mode == "artModeOff":
            return False
        raise TvError("TV Art Mode response is incomplete") from None

    async def async_read_picture_mode(self) -> str:
        """Return the panel mode without changing its semantic value."""
        self._ensure_authorized()
        result = await self._request("getTVStates", params={"AccessToken": self._token})
        picture_mode = result.get("pictureMode")
        if not isinstance(picture_mode, str) or not picture_mode.strip():
            raise TvError("TV picture mode response is incomplete") from None
        return picture_mode

    async def async_set_art_mode(self, enabled: bool) -> None:
        """Send one mode setter; the caller observes the later transition."""
        self._ensure_authorized()
        await self._request(
            "artModeControl",
            params={
                "AccessToken": self._token,
                "artMode": "artModeOn" if enabled else "artModeOff",
            },
        )

    async def async_turn_off(self) -> None:
        """Send one full-off setter; the caller observes the later transition."""
        self._ensure_authorized()
        result = await self._request(
            "powerControl", params={"AccessToken": self._token, "power": "powerOff"}
        )
        if self._power(result):
            raise TvError("TV power-off request was not acknowledged") from None

    @staticmethod
    def _power(result: dict[str, Any]) -> bool:
        power = result.get("power")
        if power == "powerOn":
            return True
        if power == "powerOff":
            return False
        raise TvError("TV power response is incomplete") from None

    async def _request(
        self, method: str, *, params: dict[str, Any] | None = None, seconds: int = 10
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"jsonrpc": "2.0", "id": "1", "method": method}
        if params is not None:
            request["params"] = params
        try:
            # One deadline includes the request and every decoded body chunk.
            async with asyncio.timeout(seconds):
                async with self._session.post(
                    self._url,
                    json=request,
                    headers={"Accept": "application/json"},
                    ssl=False,
                    allow_redirects=False,
                    auto_decompress=True,
                    timeout=_SESSION_TIMEOUT,
                ) as response:
                    if not 200 <= response.status < 300:
                        raise TvError("TV power HTTP request was refused")
                    body = bytearray()
                    while chunk := await response.content.read(_MAX_BODY + 1 - len(body)):
                        body.extend(chunk)
                        if len(body) > _MAX_BODY:
                            raise TvError("TV power response is too large")
                    return self._parse(json.loads(body))
        except TvPowerPairingRejected as error:
            self._authorization_error = TvPowerPairingRejected
            failure = error
        except TvError as error:
            failure = error
        except (ClientError, OSError, TimeoutError):
            failure = TvConnectionError("TV power connection failed")
        except (ValueError, RecursionError):
            failure = TvError("TV power response is invalid")
        # Raising outside handlers removes raw transport and decoder context.
        raise failure from None

    @staticmethod
    def _parse(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TvError("TV power response is invalid") from None
        flat_error = "code" in payload and not any(
            field in payload for field in ("jsonrpc", "id", "result", "error", "method", "params")
        )
        # Only the bare flat error form lacks correlation fields on this API.
        if not flat_error and (
            payload.get("jsonrpc") != "2.0"
            or payload.get("id") != "1"
            or "code" in payload
            or ("error" in payload and "result" in payload)
        ):
            raise TvError("TV power response is invalid") from None
        if "error" in payload or "code" in payload:
            error = payload.get("error", payload)
            if isinstance(error, dict) and error.get("code") == -32010:
                raise TvPowerPairingRejected("TV power authorization was rejected") from None
            raise TvError("TV power command was refused") from None
        if not isinstance(payload.get("result"), dict):
            raise TvError("TV power response is invalid") from None
        return payload["result"]
