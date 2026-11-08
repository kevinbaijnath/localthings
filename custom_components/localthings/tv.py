"""Local Samsung Frame protocol facade. SDK methods are blocking."""

from __future__ import annotations

import logging
import socket
import ssl
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Any, Literal
from xml.etree import ElementTree

import websocket
from aiohttp import ClientError, ClientSession, ClientTimeout
from samsungtvws import SamsungTVWS
from samsungtvws import helper as sdk_helper
from samsungtvws.art import SamsungTVArt
from samsungtvws.command import SamsungTVCommand
from samsungtvws.connection import SamsungTVWSConnection
from samsungtvws.event import (
    IGNORE_EVENTS_AT_STARTUP,
    MS_CHANNEL_CONNECT_EVENT,
    MS_CHANNEL_READY_EVENT,
    MS_CHANNEL_UNAUTHORIZED,
)
from samsungtvws.exceptions import ConnectionFailure, UnauthorizedError
from websocket import (
    WebSocket,
    WebSocketConnectionClosedException,
    WebSocketException,
    WebSocketTimeoutException,
)

_HTTP_TIMEOUT = ClientTimeout(total=5)
_RENDERING_CONTROL = "urn:schemas-upnp-org:service:RenderingControl:1"
_LOGGER = logging.getLogger(__name__)


class FrameMode(StrEnum):
    """The modes supported by a Frame TV."""

    TV = "tv"
    ART = "art"
    OFF = "off"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TvIdentity:
    """Identity and power evidence from the local HTTP endpoint."""

    device_id: str
    model: str
    mac: str | None
    power_state: str | None = None


@dataclass(frozen=True)
class TvState:
    """Independent observations; unavailable evidence stays unknown."""

    mode: FrameMode
    art_mode: bool | None
    muted: bool | None


class TvError(Exception):
    """A sanitized local TV protocol failure."""


class TvConnectionError(TvError):
    """The TV connection failed."""


class TvPairingRejected(TvError):
    """The TV rejected authorization."""


class TvAuthorizationIncomplete(TvConnectionError):
    """The WebSocket opened but channel authorization did not complete."""


class TvPairingTimeout(TvAuthorizationIncomplete):
    """The TV did not answer the pairing request in time."""


@dataclass(frozen=True)
class TvFailure:
    """Fixed labels only; never retain a protocol reply or exception."""

    channel: Literal["art", "remote"]
    stage: Literal["transport", "upgrade", "channel_connect", "art_ready", "art_reply"]
    error: Literal[
        "timeout", "closed", "connection_error", "unexpected_event", "rejected", "protocol_error"
    ]


@dataclass
class _ArtReadTrace:
    startup_entered: bool = False
    token_supplied: bool = False
    channel_entered: bool = False
    channel_active: bool = False
    channel_ms: int = 0
    channel_recv_entered: int = 0
    channel_recv_returned: int = 0
    channel_recv_bytes: int = 0
    channel_messages: int = 0
    channel_events: int = 0
    channel_last_event: Literal["none", "connect", "unauthorized", "ready", "ignored", "other"] = (
        "none"
    )
    ready_started: float | None = None
    ready_ms: int = 0
    ready_active: bool = False
    ready_accepted: bool = False
    recv_entered: int = 0
    recv_returned: int = 0
    recv_bytes: int = 0


def _failure(
    channel: Literal["art", "remote"],
    stage: Literal["transport", "upgrade", "channel_connect", "art_ready", "art_reply"],
    error: BaseException,
) -> TvFailure:
    if isinstance(error, ConnectionFailure) and isinstance(
        error.__cause__, (TimeoutError, WebSocketTimeoutException)
    ):
        error = error.__cause__
    if isinstance(error, (UnauthorizedError, TvPairingRejected)):
        category = "rejected"
    elif isinstance(error, (TimeoutError, WebSocketTimeoutException)):
        category = "timeout"
    elif isinstance(
        error, (WebSocketConnectionClosedException, ConnectionResetError, BrokenPipeError, EOFError)
    ):
        category = "closed"
    elif isinstance(error, OSError):
        category = "connection_error"
    elif isinstance(error, ConnectionFailure):
        category = "unexpected_event"
    else:
        category = "protocol_error"
    return TvFailure(channel, stage, category)


class _RejectSdkLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return False


_SDK_LOG_FILTER = _RejectSdkLogs()


def _suppress_sdk_logs() -> None:
    # The SDK logs issued tokens at INFO and raw responses at WARNING. Parent
    # logger filters do not cover descendants, so guard each SDK module logger.
    logging.getLogger("samsungtvws").addFilter(_SDK_LOG_FILTER)
    for name, logger in logging.Logger.manager.loggerDict.copy().items():
        if name.startswith("samsungtvws.") and isinstance(logger, logging.Logger):
            logger.addFilter(_SDK_LOG_FILTER)


def _call_sdk[T](operation: Callable[[], T], *, pairing: bool = False) -> T:
    _suppress_sdk_logs()
    try:
        return operation()
    except Exception as error:
        if isinstance(error, TvError):
            failure = error
        elif isinstance(error, UnauthorizedError):
            failure = TvPairingRejected("TV authorization was rejected")
        elif isinstance(error, (TimeoutError, WebSocketTimeoutException)) and pairing:
            failure = TvPairingTimeout("TV pairing timed out")
        elif isinstance(error, (ConnectionFailure, WebSocketException, OSError)):
            failure = TvConnectionError("TV connection failed")
        else:
            failure = TvError("TV protocol request failed")
    # Raise outside the handler so even __context__ contains no raw SDK reply.
    raise failure from None


def resolve_frame_mode(power_state: str | None, art_mode: bool | None) -> FrameMode:
    """Resolve explicit evidence without treating missing replies as off."""
    power = power_state.casefold() if power_state is not None else None
    if power in {"off", "standby"}:
        return FrameMode.UNKNOWN if art_mode is True else FrameMode.OFF
    if art_mode is True:
        return FrameMode.ART
    if power == "on" and art_mode is False:
        return FrameMode.TV
    return FrameMode.UNKNOWN


async def async_read_tv_identity(session: ClientSession, host: str) -> TvIdentity:
    """Read identity without keeping the full HTTP response."""
    try:
        async with session.get(f"http://{host}:8001/api/v2/", timeout=_HTTP_TIMEOUT) as response:
            response.raise_for_status()
            payload = await response.json()
    except (ClientError, TimeoutError):
        failure = TvConnectionError("TV identity request failed")
    except ValueError:
        failure = TvError("TV identity response is invalid")
    else:
        device = payload.get("device") if isinstance(payload, dict) else None
        if isinstance(device, dict):
            device_id = device.get("id")
            model = device.get("modelName")
            mac = device.get("wifiMac")
            power = device.get("PowerState")
            if isinstance(device_id, str) and device_id and isinstance(model, str) and model:
                return TvIdentity(
                    device_id,
                    model,
                    mac if isinstance(mac, str) and mac else None,
                    power if isinstance(power, str) else None,
                )
        failure = TvError("TV identity response is incomplete")
    raise failure from None


class _Deadline:
    """One absolute budget shared by a facade's serialized SDK connections."""

    def __init__(self) -> None:
        self._expires_at = 0.0

    def remaining(self) -> float:
        remaining = self._expires_at - monotonic()
        if remaining <= 0:
            raise TimeoutError("TV operation deadline expired")
        return remaining

    def run[T](self, operation: Callable[[], T], seconds: float = 10) -> T:
        self._expires_at = monotonic() + seconds
        try:
            return operation()
        finally:
            self._expires_at = 0.0


class _DeadlineSocket:
    """Keep SDK-internal frame and handshake loops within the same budget."""

    def __init__(
        self,
        sock: socket.socket,
        deadline: _Deadline,
        read_trace: Callable[[], _ArtReadTrace | None] | None = None,
    ) -> None:
        self._sock = sock
        self._deadline = deadline
        self._read_trace = read_trace

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)

    def gettimeout(self) -> float:
        return self._deadline.remaining()

    def settimeout(self, timeout: float) -> None:
        self._sock.settimeout(min(timeout, self._deadline.remaining()))

    def recv(self, size: int) -> bytes:
        self._sock.settimeout(self._deadline.remaining())
        trace = self._read_trace() if self._read_trace is not None else None
        if trace is not None and trace.ready_active:
            trace.recv_entered += 1
        if trace is not None and trace.channel_active:
            trace.channel_recv_entered += 1
        data = self._sock.recv(size)
        if trace is not None and trace.ready_active:
            trace.recv_returned += 1
            trace.recv_bytes += len(data)
        if trace is not None and trace.channel_active:
            trace.channel_recv_returned += 1
            trace.channel_recv_bytes += len(data)
        self._deadline.remaining()
        return data

    def send(self, data: bytes) -> int:
        # websocket-client's HTTP Upgrade writer does not retry partial sends.
        with memoryview(data) as pending:
            sent = 0
            while sent < len(pending):
                self._sock.settimeout(self._deadline.remaining())
                sent += self._sock.send(pending[sent:])
                self._deadline.remaining()
        return sent

    def sendall(self, data: bytes) -> None:
        self.send(data)


class _DeadlineConnection(SamsungTVWSConnection):
    """Bound each complete SDK event and each socket wait to one deadline."""

    _deadline: _Deadline
    _pairing = False
    _authorization_check: Callable[[], None]
    _authorization_failed: Callable[[type[TvPairingRejected | TvAuthorizationIncomplete]], None]
    _token_updated: Callable[[str], None]
    _diagnostic_failure: Callable[[TvFailure], None]
    _channel: Literal["art", "remote"]
    _channel_connected = False
    _read_trace: _ArtReadTrace | None = None

    def _report_pairing_failure(self, stage: str, error: BaseException) -> None:
        if self._pairing:
            status = getattr(error, "status_code", None)
            if type(status) is not int or not 100 <= status <= 599:
                status = None
            _LOGGER.warning(
                "TV pairing failed at %s (%s, HTTP status %s)",
                stage,
                type(error).__name__,
                status,
            )

    def _check_for_token(self, response: dict[str, Any]) -> None:
        super()._check_for_token(response)
        if isinstance(self.token, str) and self.token:
            self._token_updated(self.token)

    def _connect_socket(self) -> _DeadlineSocket:
        # DNS resolution is synchronous; the I/O budget starts constraining
        # socket operations after it returns. Numeric hosts avoid that gap.
        addresses = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        for family, kind, protocol, _, address in addresses:
            raw = socket.socket(family, kind, protocol)
            try:
                raw.settimeout(self._deadline.remaining())
                raw.connect(address)
            except OSError:
                raw.close()
                self._deadline.remaining()
            else:
                break
        else:
            raise OSError("TV socket connection failed")
        try:
            if self._is_ssl_connection():
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                raw.settimeout(self._deadline.remaining())
                raw = context.wrap_socket(raw, server_hostname=self.host)
            self._deadline.remaining()
            return _DeadlineSocket(raw, self._deadline, lambda: self._read_trace)
        except BaseException:
            raw.close()
            raise

    def open(self) -> WebSocket:
        self._authorization_check()
        if self.connection is not None:
            self._channel_connected = True
            return self.connection
        # SDK 3.0.7 leaves this socket local during startup; own it so every
        # failure closes it and ignored events cannot renew the time budget.
        try:
            sock = self._connect_socket()
        except BaseException as error:
            self._diagnostic_failure(_failure(self._channel, "transport", error))
            self._report_pairing_failure("transport", error)
            raise
        try:
            connection = websocket.create_connection(
                self._format_websocket_url(self.endpoint),
                self._deadline.remaining(),
                socket=sock,
                redirect_limit=0,
                connection="Connection: Upgrade",
            )
        except BaseException as error:
            self._diagnostic_failure(_failure(self._channel, "upgrade", error))
            self._report_pairing_failure("upgrade", error)
            sock.close()
            raise
        trace = self._read_trace
        channel_started = monotonic() if trace is not None else 0.0
        if trace is not None:
            trace.channel_entered = True
            trace.channel_active = True
        try:
            event: str | None = None
            while event is None or event in IGNORE_EVENTS_AT_STARTUP:
                connection.settimeout(self._deadline.remaining())
                message = connection.recv()
                if trace is not None:
                    trace.channel_messages += 1
                response = sdk_helper.process_api_response(message)
                event = response.get("event", "*")
                if trace is not None:
                    trace.channel_events += 1
                    if event == MS_CHANNEL_CONNECT_EVENT:
                        trace.channel_last_event = "connect"
                    elif event == MS_CHANNEL_UNAUTHORIZED:
                        trace.channel_last_event = "unauthorized"
                    elif event == MS_CHANNEL_READY_EVENT:
                        trace.channel_last_event = "ready"
                    elif event in IGNORE_EVENTS_AT_STARTUP:
                        trace.channel_last_event = "ignored"
                    else:
                        trace.channel_last_event = "other"
                self._websocket_event(event, response)
            if event != MS_CHANNEL_CONNECT_EVENT:
                raise ConnectionFailure(response)
            self._check_for_token(response)
            self.connection = connection
            self._channel_connected = True
            return connection
        except Exception as error:
            self._diagnostic_failure(_failure(self._channel, "channel_connect", error))
            self._report_pairing_failure("authorization", error)
            if isinstance(error, UnauthorizedError):
                failure = TvPairingRejected("TV authorization was rejected")
            elif self._pairing and isinstance(error, (TimeoutError, WebSocketTimeoutException)):
                failure = TvPairingTimeout("TV pairing timed out")
            else:
                failure = TvAuthorizationIncomplete("TV channel authorization did not complete")
            self._authorization_failed(type(failure))
            with suppress(Exception):
                connection.shutdown()
        except BaseException:
            connection.shutdown()
            raise
        finally:
            if trace is not None:
                trace.channel_active = False
                trace.channel_ms = max(0, int((monotonic() - channel_started) * 1000))
        raise failure from None

    def _websocket_event(self, event: str, response: dict[str, Any]) -> None:
        if event == MS_CHANNEL_UNAUTHORIZED:
            self._authorization_failed(TvPairingRejected)
            raise UnauthorizedError(response)
        self._deadline.remaining()
        super()._websocket_event(event, response)

    def _send_command(
        self,
        connection: WebSocket,
        command: SamsungTVCommand | dict[str, Any],
        delay: float,
    ) -> None:
        connection.settimeout(self._deadline.remaining())
        super()._send_command(connection, command, delay)
        self._deadline.remaining()

    def close(self) -> None:
        # The facade never starts SDK listener threads. Socket shutdown avoids
        # adding a graceful-close receive loop to an expired operation.
        if self.connection is not None:
            self.connection.shutdown()
            self.connection = None


class _DeadlineArt(SamsungTVArt, _DeadlineConnection):
    """Art API using the facade's deadline-owned connections."""

    _channel = "art"
    _reading_ready = False

    def open(self) -> WebSocket:
        self._channel_connected = False
        self._reading_ready = True
        if self._read_trace is not None:
            self._read_trace.startup_entered = True
            self._read_trace.token_supplied = isinstance(self.token, str) and bool(self.token)
        try:
            return super().open()
        except Exception as error:
            if self._channel_connected:
                self._diagnostic_failure(_failure("art", "art_ready", error))
            raise
        finally:
            self._reading_ready = False

    def _recv_frame(self) -> tuple[str, Any]:
        trace = self._read_trace
        if trace is None or not self._reading_ready:
            return super()._recv_frame()
        trace.ready_started = monotonic()
        trace.ready_active = True
        try:
            event, frame = super()._recv_frame()
            trace.ready_accepted = event == MS_CHANNEL_READY_EVENT
            return event, frame
        finally:
            trace.ready_ms = max(0, int((monotonic() - trace.ready_started) * 1000))
            trace.ready_active = False


class _DeadlineRemote(SamsungTVWS, _DeadlineConnection):
    _channel = "remote"

    def art(self) -> _DeadlineArt:
        art = _DeadlineArt(
            self.host,
            token=self.token,
            port=self.port,
            timeout=self.timeout,
            key_press_delay=self.key_press_delay,
            name=self.name,
        )
        art._deadline = self._deadline
        return art


class TvClient:
    """Blocking SDK owner; callers must use an executor and serialize calls."""

    def __init__(self, host: str, token: str | None, name: str = "LocalThings") -> None:
        self._deadline = _Deadline()
        self._remote = _call_sdk(
            lambda: _DeadlineRemote(
                host=host,
                token=token,
                port=8002,
                timeout=10,
                key_press_delay=0,
                name=name,
            )
        )
        self._remote._deadline = self._deadline
        self._remote._authorization_check = self._ensure_authorized
        self._remote._authorization_failed = self._suspend_authorization
        self._remote._token_updated = self._update_token
        self._remote._diagnostic_failure = self._record_failure
        self._art: _DeadlineArt | None = None
        self._token = token
        self._authorization_error: type[TvPairingRejected | TvAuthorizationIncomplete] | None = None
        self._last_failure: TvFailure | None = None
        self._art_read_trace_attempt = 0
        self._closed = False

    @property
    def token(self) -> str | None:
        """The latest token issued by either authorized channel."""
        return self._token

    @property
    def authorization_error(self) -> type[TvPairingRejected | TvAuthorizationIncomplete] | None:
        """A latched channel failure, read only after the SDK operation completes."""
        return self._authorization_error

    @property
    def last_failure(self) -> TvFailure | None:
        """The last actual connection or Art-read failure, without another request."""
        return self._last_failure

    def _record_failure(self, failure: TvFailure) -> None:
        self._last_failure = failure

    def _update_token(self, token: str) -> None:
        self._token = self._remote.token = token
        if self._art is not None:
            self._art.token = token

    def _suspend_authorization(
        self, error: type[TvPairingRejected | TvAuthorizationIncomplete]
    ) -> None:
        self._authorization_error = (
            TvAuthorizationIncomplete if issubclass(error, TvAuthorizationIncomplete) else error
        )

    def _ensure_authorized(self) -> None:
        if self._authorization_error is not None:
            raise self._authorization_error("TV WebSocket authorization is suspended") from None

    def _authorized_sdk[T](self, operation: Callable[[], T], *, pairing: bool = False) -> T:
        self._ensure_authorized()
        try:
            return _call_sdk(operation, pairing=pairing)
        except (TvPairingRejected, TvAuthorizationIncomplete) as error:
            self._suspend_authorization(type(error))
            raise

    def _run_sdk[T](
        self, operation: Callable[[], T], *, seconds: float = 10, pairing: bool = False
    ) -> T:
        return self._authorized_sdk(lambda: self._deadline.run(operation, seconds), pairing=pairing)

    def pair(self) -> str:
        """Wait for authorization and return the token held only in memory."""
        self._remote.timeout = 40
        self._remote._pairing = True
        try:
            self._run_sdk(self._remote.open, seconds=40, pairing=True)
        finally:
            self._remote.timeout = 10
            self._remote._pairing = False
        token = self._remote.token
        if not isinstance(token, str) or not token:
            _LOGGER.warning("TV pairing failed at token (authorization token absent)")
            self._suspend_authorization(TvAuthorizationIncomplete)
            raise TvAuthorizationIncomplete("TV did not issue an authorization token") from None
        self._update_token(token)
        return token

    def _art_client(self) -> _DeadlineArt:
        if self._art is None:
            self._art = self._remote.art()
            self._art._authorization_check = self._ensure_authorized
            self._art._authorization_failed = self._suspend_authorization
            self._art._token_updated = self._update_token
            self._art._diagnostic_failure = self._record_failure
        return self._art

    def read_art_mode(self) -> bool:
        """Read explicit on/off; invalid or missing evidence is an error."""

        def read() -> Any:
            previous = self._last_failure
            attempt, self._art_read_trace_attempt = self._art_read_trace_attempt, 0
            trace = _ArtReadTrace() if attempt else None
            started = monotonic()
            try:
                art = self._art_client()
                art._read_trace = trace
                return art.get_artmode()
            except Exception as error:
                if self._last_failure is previous:
                    self._record_failure(_failure("art", "art_reply", error))
                raise
            finally:
                if self._art is not None:
                    self._art._read_trace = None
                if trace is not None:
                    _LOGGER.warning(
                        "TV Art receive trace: attempt=%s elapsed_ms=%s startup_entered=%s "
                        "token_supplied=%s channel_entered=%s channel_ms=%s "
                        "channel_recv_entered=%s channel_recv_returned=%s channel_recv_bytes=%s "
                        "channel_messages=%s channel_events=%s channel_last_event=%s "
                        "ready_entered=%s ready_accepted=%s ready_ms=%s "
                        "recv_entered=%s recv_returned=%s recv_bytes=%s",
                        attempt,
                        max(0, int((monotonic() - started) * 1000)),
                        trace.startup_entered,
                        trace.token_supplied,
                        trace.channel_entered,
                        trace.channel_ms,
                        trace.channel_recv_entered,
                        trace.channel_recv_returned,
                        trace.channel_recv_bytes,
                        trace.channel_messages,
                        trace.channel_events,
                        trace.channel_last_event,
                        trace.ready_started is not None,
                        trace.ready_accepted,
                        trace.ready_ms,
                        trace.recv_entered,
                        trace.recv_returned,
                        trace.recv_bytes,
                    )

        reply = self._run_sdk(read)
        if reply == "on":
            return True
        if reply == "off":
            return False
        self._record_failure(_failure("art", "art_reply", TvError()))
        raise TvError("TV Art Mode response is incomplete") from None

    def set_art_mode(self, enabled: bool) -> None:
        """Set the requested Art Mode state."""
        self._run_sdk(lambda: self._art_client().set_artmode(enabled))

    def send_key(self, key: str) -> None:
        """Send once; a lost reply never triggers a command retry."""

        def send() -> None:
            try:
                self._remote.send_key(key)
            finally:
                self._remote.close()

        self._run_sdk(send)

    def hold_power(self, seconds: float) -> None:
        """Always attempt Release, even if Press or the hold fails."""

        def hold() -> None:
            try:
                try:
                    self._deadline.run(lambda: self._remote.send_key("KEY_POWER", cmd="Press"))
                    time.sleep(seconds)
                finally:
                    self._deadline.run(lambda: self._remote.send_key("KEY_POWER", cmd="Release"))
            finally:
                self._remote.close()

        self._authorized_sdk(hold)

    def close(self) -> None:
        """Close both connections once, including after a partial failure."""
        if self._closed:
            return
        self._closed = True

        def close_connections() -> None:
            try:
                if self._art is not None:
                    self._art.close()
            finally:
                self._remote.close()

        _call_sdk(close_connections)


async def _async_mute_request(
    session: ClientSession, host: str, action: str, arguments: str = ""
) -> ElementTree.Element:
    envelope = (
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        f'<s:Body><u:{action} xmlns:u="{_RENDERING_CONTROL}">'
        f"<InstanceID>0</InstanceID><Channel>Master</Channel>{arguments}"
        f"</u:{action}></s:Body></s:Envelope>"
    )
    try:
        async with session.post(
            f"http://{host}:9197/upnp/control/RenderingControl1",
            data=envelope,
            headers={
                "Content-Type": 'text/xml; charset="utf-8"',
                "SOAPAction": f'"{_RENDERING_CONTROL}#{action}"',
            },
            timeout=_HTTP_TIMEOUT,
        ) as response:
            response.raise_for_status()
            body = await response.text()
    except (ClientError, TimeoutError):
        failure = TvConnectionError("TV mute request failed")
    else:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError:
            failure = TvError("TV mute response is invalid")
        else:
            reply = root.find(f".//{{{_RENDERING_CONTROL}}}{action}Response")
            if reply is not None and not any(
                element.tag.rsplit("}", 1)[-1] == "Fault" for element in root.iter()
            ):
                return reply
            failure = TvError("TV mute response is incomplete")
    raise failure from None


async def async_read_mute(session: ClientSession, host: str) -> bool:
    """Read explicit mute state; absent or failed replies remain errors."""
    reply = await _async_mute_request(session, host, "GetMute")
    for element in reply.iter():
        if element.tag.rsplit("}", 1)[-1] == "CurrentMute":
            value = (element.text or "").strip()
            if value in {"0", "1"}:
                return value == "1"
    raise TvError("TV mute state is incomplete") from None


async def async_set_mute(session: ClientSession, host: str, muted: bool) -> None:
    """Set a desired mute state without a remote-key toggle."""
    await _async_mute_request(session, host, "SetMute", f"<DesiredMute>{int(muted)}</DesiredMute>")
