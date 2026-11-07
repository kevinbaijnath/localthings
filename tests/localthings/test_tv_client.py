"""TV facade boundary tests. All replies and identity values are synthetic."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import socket
import traceback
from dataclasses import FrozenInstanceError, asdict
from unittest.mock import MagicMock, Mock

import pytest
from aiohttp import ClientConnectionError, ClientResponseError, ClientSession
from samsungtvws import connection as sdk_connection
from samsungtvws.exceptions import ConnectionFailure, ResponseError, UnauthorizedError
from websocket import ABNF, WebSocketConnectionClosedException, WebSocketTimeoutException

from custom_components.localthings import tv

SYNTHETIC_HOST = "192.0.2.10"
SYNTHETIC_TOKEN = "synthetic-pairing-token"
SYNTHETIC_RAW_REPLY = {"data": {"token": SYNTHETIC_TOKEN}, "synthetic": "raw-reply"}


class SyntheticResponse:
    """Synthetic HTTP response; no sockets are opened."""

    def __init__(self, body: str, status: int = 200) -> None:
        self.body = body
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise ClientResponseError(Mock(), (), status=self.status, message=self.body)

    async def text(self) -> str:
        return self.body

    async def json(self):
        return json.loads(self.body)


def synthetic_session(body: str, *, status: int = 200) -> MagicMock:
    session = MagicMock(spec=ClientSession)
    session.get.return_value = SyntheticResponse(body, status)
    session.post.return_value = SyntheticResponse(body, status)
    return session


def synthetic_soap(action: str, state: str = "") -> str:
    return (
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
        f'<s:Body><u:{action}Response xmlns:u="urn:schemas-upnp-org:service:RenderingControl:1">'
        f"{state}</u:{action}Response></s:Body></s:Envelope>"
    )


def assert_sanitized(error: BaseException) -> None:
    rendered = "".join(traceback.format_exception(error))
    assert SYNTHETIC_TOKEN not in rendered
    assert "raw-reply" not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__


def test_art_on_with_power_on_is_art() -> None:
    assert tv.resolve_frame_mode("on", True) is tv.FrameMode.ART


@pytest.mark.parametrize(
    ("power_state", "art_mode", "expected"),
    [
        ("on", False, "tv"),
        ("on", None, "unknown"),
        ("off", False, "off"),
        ("off", None, "off"),
        ("standby", False, "off"),
        ("standby", None, "off"),
        ("off", True, "unknown"),
        ("standby", True, "unknown"),
        (None, True, "art"),
        (None, False, "unknown"),
        (None, None, "unknown"),
        ("unexpected", False, "unknown"),
        ("unexpected", True, "art"),
    ],
)
def test_synthetic_state_evidence(power_state, art_mode, expected) -> None:
    assert tv.resolve_frame_mode(power_state, art_mode).value == expected


def test_state_and_identity_are_immutable() -> None:
    identity = tv.TvIdentity("synthetic-device", "SYNTHETIC-MODEL", None)
    assert identity.power_state is None
    with pytest.raises(FrozenInstanceError):
        identity.model = "changed"  # ty: ignore[invalid-assignment]
    state = tv.TvState(tv.FrameMode.UNKNOWN, None, None)
    with pytest.raises(FrozenInstanceError):
        state.muted = False  # ty: ignore[invalid-assignment]


async def test_identity_retains_power_and_optional_mac() -> None:
    session = synthetic_session(
        json.dumps(
            {
                "device": {
                    "id": "synthetic-device",
                    "modelName": "SYNTHETIC-MODEL",
                    "wifiMac": "02:00:00:00:00:01",
                    "PowerState": "on",
                }
            }
        )
    )
    identity = await tv.async_read_tv_identity(session, SYNTHETIC_HOST)
    assert identity == tv.TvIdentity(
        "synthetic-device", "SYNTHETIC-MODEL", "02:00:00:00:00:01", "on"
    )
    assert session.get.call_args.args == (f"http://{SYNTHETIC_HOST}:8001/api/v2/",)


async def test_identity_accepts_absent_optional_values() -> None:
    session = synthetic_session(
        json.dumps(
            {
                "device": {
                    "id": "synthetic-device",
                    "modelName": "SYNTHETIC-MODEL",
                }
            }
        )
    )
    assert await tv.async_read_tv_identity(session, SYNTHETIC_HOST) == tv.TvIdentity(
        "synthetic-device", "SYNTHETIC-MODEL", None
    )


@pytest.mark.parametrize("body", ["not json", "{}", '{"device": {}}', '{"device": []}'])
async def test_identity_missing_required_evidence_fails(body) -> None:
    with pytest.raises(tv.TvError):
        await tv.async_read_tv_identity(synthetic_session(body), SYNTHETIC_HOST)


async def test_identity_http_failure_is_sanitized() -> None:
    session = synthetic_session(json.dumps(SYNTHETIC_RAW_REPLY), status=500)
    with pytest.raises(tv.TvConnectionError) as caught:
        await tv.async_read_tv_identity(session, SYNTHETIC_HOST)
    assert_sanitized(caught.value)


@pytest.fixture
def sdk(monkeypatch):
    remote = MagicMock()
    remote.token = SYNTHETIC_TOKEN
    art = remote.art.return_value
    factory = Mock(return_value=remote)
    monkeypatch.setattr(tv, "_DeadlineRemote", factory)
    return factory, remote, art


def test_pair_accepted_uses_memory_token_and_secure_port(sdk) -> None:
    factory, remote, _ = sdk
    client = tv.TvClient(SYNTHETIC_HOST, None)
    assert client.pair() == SYNTHETIC_TOKEN
    remote.open.assert_called_once_with()
    assert factory.call_args.kwargs["port"] == 8002
    assert factory.call_args.kwargs.get("token_file") is None


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (UnauthorizedError(SYNTHETIC_RAW_REPLY), "TvPairingRejected"),
        (WebSocketTimeoutException(SYNTHETIC_RAW_REPLY), "TvPairingTimeout"),
        (TimeoutError(SYNTHETIC_RAW_REPLY), "TvPairingTimeout"),
        (ConnectionFailure(SYNTHETIC_RAW_REPLY), "TvConnectionError"),
    ],
)
def test_pair_errors_are_sanitized(sdk, error, expected) -> None:
    _, remote, _ = sdk
    remote.open.side_effect = error
    with pytest.raises(getattr(tv, expected)) as caught:
        tv.TvClient(SYNTHETIC_HOST, None).pair()
    assert_sanitized(caught.value)


def test_pair_without_issued_token_fails(sdk) -> None:
    _, remote, _ = sdk
    remote.token = None
    client = tv.TvClient(SYNTHETIC_HOST, None)
    with pytest.raises(tv.TvAuthorizationIncomplete) as caught:
        client.pair()
    assert type(caught.value) is tv.TvAuthorizationIncomplete
    with pytest.raises(tv.TvAuthorizationIncomplete):
        client.pair()
    remote.open.assert_called_once_with()


@pytest.mark.parametrize(("reply", "expected"), [("on", True), ("off", False)])
def test_read_art_mode_normalizes_explicit_sdk_values(sdk, reply, expected) -> None:
    _, _, art = sdk
    art.get_artmode.return_value = reply
    assert tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN).read_art_mode() is expected


@pytest.mark.parametrize("reply", [None, "", "unexpected", True, {"value": "on"}])
def test_read_art_mode_missing_or_invalid_reply_is_unknown_error(sdk, reply) -> None:
    _, _, art = sdk
    art.get_artmode.return_value = reply
    with pytest.raises(tv.TvError):
        tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN).read_art_mode()


def test_set_art_mode_uses_explicit_boolean(sdk) -> None:
    _, _, art = sdk
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    client.set_art_mode(False)
    art.set_artmode.assert_called_once_with(False)


def test_art_sdk_failure_is_sanitized(sdk) -> None:
    _, _, art = sdk
    art.get_artmode.side_effect = ResponseError(SYNTHETIC_RAW_REPLY)
    with pytest.raises(tv.TvError) as caught:
        tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN).read_art_mode()
    assert_sanitized(caught.value)


def test_lost_key_response_sends_only_one_attempt(sdk) -> None:
    _, remote, _ = sdk
    remote.send_key.side_effect = WebSocketTimeoutException(SYNTHETIC_RAW_REPLY)
    with pytest.raises(tv.TvConnectionError) as caught:
        tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN).send_key("KEY_HOME")
    remote.send_key.assert_called_once_with("KEY_HOME")
    assert_sanitized(caught.value)


@pytest.mark.parametrize("failure", [None, "press", "sleep"])
def test_hold_power_always_attempts_release(sdk, monkeypatch, failure) -> None:
    _, remote, _ = sdk
    calls = []

    def send(key, **kwargs):
        calls.append((key, kwargs))
        if failure == "press" and kwargs["cmd"] == "Press":
            raise ConnectionFailure(SYNTHETIC_RAW_REPLY)

    def delay(seconds):
        assert seconds == 2.5
        if failure == "sleep":
            raise OSError(SYNTHETIC_RAW_REPLY)

    remote.send_key.side_effect = send
    monkeypatch.setattr(tv.time, "sleep", delay)
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    if failure:
        with pytest.raises(tv.TvError) as caught:
            client.hold_power(2.5)
        assert_sanitized(caught.value)
    else:
        client.hold_power(2.5)
    assert calls == [
        ("KEY_POWER", {"cmd": "Press"}),
        ("KEY_POWER", {"cmd": "Release"}),
    ]


def test_close_releases_both_sdk_connections_and_is_idempotent(sdk) -> None:
    _, remote, art = sdk
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    art.get_artmode.return_value = "on"
    client.read_art_mode()
    client.close()
    client.close()
    remote.close.assert_called_once_with()
    art.close.assert_called_once_with()


def test_close_attempts_other_connection_after_failure(sdk) -> None:
    _, remote, art = sdk
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    art.get_artmode.return_value = "on"
    client.read_art_mode()
    art.close.side_effect = ConnectionFailure(SYNTHETIC_RAW_REPLY)
    with pytest.raises(tv.TvError) as caught:
        client.close()
    remote.close.assert_called_once_with()
    assert_sanitized(caught.value)


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
async def test_get_mute_parses_explicit_state(value, expected) -> None:
    session = synthetic_session(synthetic_soap("GetMute", f"<CurrentMute>{value}</CurrentMute>"))
    assert await tv.async_read_mute(session, SYNTHETIC_HOST) is expected
    assert session.post.call_args.args == (
        f"http://{SYNTHETIC_HOST}:9197/upnp/control/RenderingControl1",
    )
    assert "#GetMute" in session.post.call_args.kwargs["headers"]["SOAPAction"]


async def test_get_mute_parses_namespaced_current_mute() -> None:
    session = synthetic_session(
        synthetic_soap("GetMute", '<u:CurrentMute xmlns:u="urn:synthetic">1</u:CurrentMute>')
    )
    assert await tv.async_read_mute(session, SYNTHETIC_HOST) is True


@pytest.mark.parametrize(
    "body",
    [
        synthetic_soap("GetMute"),
        synthetic_soap("GetMute", "<CurrentMute>2</CurrentMute>"),
        "not XML",
        '<s:Fault xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">synthetic</s:Fault>',
    ],
)
async def test_get_mute_missing_or_invalid_state_fails(body) -> None:
    with pytest.raises(tv.TvError):
        await tv.async_read_mute(synthetic_session(body), SYNTHETIC_HOST)


@pytest.mark.parametrize("muted", [False, True])
async def test_set_mute_sends_explicit_desired_boolean(muted) -> None:
    session = synthetic_session(synthetic_soap("SetMute"))
    await tv.async_set_mute(session, SYNTHETIC_HOST, muted)
    request = session.post.call_args.kwargs
    assert f"<DesiredMute>{int(muted)}</DesiredMute>" in request["data"]
    assert "<InstanceID>0</InstanceID>" in request["data"]
    assert "<Channel>Master</Channel>" in request["data"]
    assert "KEY_MUTE" not in request["data"]
    assert "#SetMute" in request["headers"]["SOAPAction"]


@pytest.mark.parametrize("operation", ["async_read_mute", "async_set_mute"])
async def test_mute_http_failure_is_sanitized(operation) -> None:
    session = synthetic_session(json.dumps(SYNTHETIC_RAW_REPLY), status=500)
    args = [session, SYNTHETIC_HOST]
    if operation == "async_set_mute":
        args.append(True)
    with pytest.raises(tv.TvConnectionError) as caught:
        await getattr(tv, operation)(*args)
    assert_sanitized(caught.value)


async def test_set_mute_soap_failure_does_not_report_success() -> None:
    session = synthetic_session(
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"><s:Body>'
        "<s:Fault><faultstring>synthetic</faultstring></s:Fault></s:Body></s:Envelope>"
    )
    with pytest.raises(tv.TvError):
        await tv.async_set_mute(session, SYNTHETIC_HOST, True)


async def test_mute_transport_failure_is_sanitized() -> None:
    session = synthetic_session("")
    session.post.side_effect = ClientConnectionError(SYNTHETIC_RAW_REPLY)
    with pytest.raises(tv.TvConnectionError) as caught:
        await tv.async_read_mute(session, SYNTHETIC_HOST)
    assert_sanitized(caught.value)


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING])
def test_sdk_child_logs_cannot_expose_credentials(sdk, caplog, level) -> None:
    _, remote, _ = sdk
    client = tv.TvClient(SYNTHETIC_HOST, None)
    logger = logging.getLogger(f"samsungtvws.synthetic_{level}")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    caplog.set_level(logging.INFO, logger="synthetic_unrelated")
    caplog.set_level(logging.DEBUG)

    def emit_credentials():
        logger.log(level, "synthetic credential %s", SYNTHETIC_TOKEN)
        logging.getLogger("synthetic_unrelated").info("synthetic unrelated marker")

    remote.open.side_effect = emit_credentials
    client.pair()
    logger.setLevel(logging.DEBUG)
    logger.log(level, "synthetic credential after debug enabled %s", SYNTHETIC_TOKEN)
    assert SYNTHETIC_TOKEN not in caplog.text
    assert "synthetic unrelated marker" in caplog.text


def test_hold_power_release_failure_is_sanitized(sdk, monkeypatch) -> None:
    _, remote, _ = sdk
    remote.send_key.side_effect = [None, ConnectionFailure(SYNTHETIC_RAW_REPLY)]
    monkeypatch.setattr(tv.time, "sleep", lambda _: None)
    with pytest.raises(tv.TvConnectionError) as caught:
        tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN).hold_power(1)
    assert remote.send_key.call_count == 2
    assert_sanitized(caught.value)


async def test_identity_transport_timeout_is_sanitized() -> None:
    session = synthetic_session("")
    session.get.side_effect = TimeoutError(SYNTHETIC_RAW_REPLY)
    with pytest.raises(tv.TvConnectionError) as caught:
        await tv.async_read_tv_identity(session, SYNTHETIC_HOST)
    assert_sanitized(caught.value)


class SyntheticClock:
    """Advance only when a fake socket consumes a synthetic complete frame."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SyntheticSocket:
    """Drive inherited SDK stream loops without opening a connection."""

    def __init__(self, clock: SyntheticClock, *, startup_noise: bool = False) -> None:
        self.clock = clock
        self.startup_noise = startup_noise
        self.initial = [
            {"event": "ms.channel.connect", "data": {"token": SYNTHETIC_TOKEN}},
            {"event": "ms.channel.ready"},
        ]
        self.recv_count = 0
        self.sent: list[dict] = []
        self.timeouts: list[float] = []
        self.shutdown_count = 0
        self.match = False
        self.on_send = None

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def recv(self) -> str:
        self.recv_count += 1
        self.clock.now += 1
        if self.recv_count > 100:
            raise ConnectionFailure("synthetic stream safety stop")
        if self.startup_noise:
            return json.dumps({"event": "ed.edenTV.update"})
        if self.initial:
            return json.dumps(self.initial.pop(0))
        request_id = (
            json.loads(self.sent[-1]["params"]["data"])["id"] if self.match else "unrelated"
        )
        return json.dumps(
            {
                "event": "d2d_service_message",
                "data": json.dumps({"id": request_id, "value": "on"}),
            }
        )

    def send(self, payload: str) -> None:
        command = json.loads(payload)
        self.sent.append(command)
        if self.on_send is not None:
            self.on_send(command)

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture
def sdk_stream(monkeypatch):
    clock = SyntheticClock()
    connection = SyntheticSocket(clock)
    factory = Mock(return_value=connection)
    monkeypatch.setattr(tv, "monotonic", clock, raising=False)
    monkeypatch.setattr(tv.time, "sleep", lambda _: None)
    monkeypatch.setattr(tv._DeadlineConnection, "_connect_socket", lambda _: connection)
    monkeypatch.setattr(sdk_connection.websocket, "create_connection", factory)
    return clock, connection, factory


@pytest.mark.parametrize("stage", ["transport", "upgrade", "authorization", "token"])
def test_pairing_failure_identifies_stage_without_sensitive_data(
    sdk_stream, monkeypatch, caplog, stage
):
    _, connection, factory = sdk_stream
    failure = OSError(SYNTHETIC_RAW_REPLY)
    if stage == "transport":
        monkeypatch.setattr(tv._DeadlineConnection, "_connect_socket", Mock(side_effect=failure))
    elif stage == "upgrade":
        connection.close = Mock()
        factory.side_effect = failure
    elif stage == "authorization":
        connection.recv = Mock(side_effect=failure)
    else:
        connection.initial = [{"event": "ms.channel.connect", "data": {}}]
    with caplog.at_level(logging.WARNING), pytest.raises(tv.TvError) as caught:
        tv.TvClient(SYNTHETIC_HOST, None).pair()
    assert_sanitized(caught.value)
    records = [r for r in caplog.records if r.name == "custom_components.localthings.tv"]
    assert len(records) == 1
    assert stage in records[0].getMessage()
    assert records[0].exc_info is None
    assert SYNTHETIC_TOKEN not in caplog.text
    assert "raw-reply" not in caplog.text
    assert SYNTHETIC_HOST not in caplog.text


def test_ordinary_connection_failure_does_not_log_pairing_diagnostic(sdk_stream, caplog):
    _, connection, _ = sdk_stream
    connection.recv = Mock(side_effect=OSError(SYNTHETIC_RAW_REPLY))
    with caplog.at_level(logging.WARNING), pytest.raises(tv.TvError):
        tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN).send_key("KEY_HOME")
    assert not [r for r in caplog.records if r.name == "custom_components.localthings.tv"]


@pytest.mark.parametrize(
    ("stage", "category"),
    [
        ("transport", "timeout"),
        ("upgrade", "connection_error"),
        ("channel_connect", "unexpected_event"),
        ("art_ready", "unexpected_event"),
        ("art_reply", "timeout"),
    ],
)
def test_runtime_failure_snapshot_identifies_stage_without_retaining_data(
    sdk_stream, monkeypatch, stage, category
):
    _, connection, factory = sdk_stream
    if stage == "transport":
        monkeypatch.setattr(
            tv._DeadlineConnection,
            "_connect_socket",
            Mock(side_effect=TimeoutError(SYNTHETIC_RAW_REPLY)),
        )
    elif stage == "upgrade":
        connection.close = Mock()
        factory.side_effect = OSError(SYNTHETIC_RAW_REPLY)
    elif stage == "channel_connect":
        connection.initial[0] = {"event": "unexpected", "data": SYNTHETIC_RAW_REPLY}
    elif stage == "art_ready":
        connection.initial[1] = {"event": "unexpected", "data": SYNTHETIC_RAW_REPLY}
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)

    with pytest.raises(tv.TvError):
        client.read_art_mode()

    failure = client.last_failure
    assert failure is not None
    assert asdict(failure) == {"channel": "art", "stage": stage, "error": category}
    encoded = json.dumps(asdict(failure))
    assert SYNTHETIC_HOST not in encoded
    assert SYNTHETIC_TOKEN not in encoded
    assert "raw-reply" not in encoded


@pytest.mark.parametrize(("stage", "startup_frames"), [("art_ready", 1), ("art_reply", 2)])
def test_sdk_wrapped_socket_timeout_keeps_timeout_snapshot(sdk_stream, stage, startup_frames):
    _, connection, _ = sdk_stream
    connection.recv = Mock(
        side_effect=[
            *(json.dumps(frame) for frame in connection.initial[:startup_frames]),
            WebSocketTimeoutException(SYNTHETIC_RAW_REPLY),
        ]
    )
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)

    with pytest.raises(tv.TvError) as caught:
        client.read_art_mode()

    assert client.last_failure == tv.TvFailure("art", stage, "timeout")
    assert_sanitized(caught.value)


@pytest.mark.parametrize("rejected", [False, True])
def test_suppressed_attempt_keeps_first_failure_snapshot_and_opens_no_socket(sdk_stream, rejected):
    _, connection, factory = sdk_stream
    connection.initial[0] = {
        "event": "ms.channel.unauthorized" if rejected else "unexpected",
        "data": SYNTHETIC_RAW_REPLY,
    }
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvError):
        client.read_art_mode()
    first = client.last_failure
    assert first is not None
    with pytest.raises(tv.TvError):
        client.send_key("KEY_HOME")
    assert client.last_failure is first
    assert asdict(first) == {
        "channel": "art",
        "stage": "channel_connect",
        "error": "rejected" if rejected else "unexpected_event",
    }
    factory.assert_called_once()
    assert connection.sent == []


def test_invalid_art_reply_has_safe_protocol_failure_snapshot(sdk):
    _, _, art = sdk
    art.get_artmode.return_value = SYNTHETIC_RAW_REPLY
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvError):
        client.read_art_mode()
    failure = client.last_failure
    assert failure is not None
    assert asdict(failure) == {
        "channel": "art",
        "stage": "art_reply",
        "error": "protocol_error",
    }
    art.get_artmode.assert_called_once()


def test_remote_failure_snapshot_keeps_remote_channel(sdk_stream):
    _, connection, _ = sdk_stream
    connection.initial[0] = {"event": "unexpected", "data": SYNTHETIC_RAW_REPLY}
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvError):
        client.send_key("KEY_HOME")
    assert client.last_failure == tv.TvFailure("remote", "channel_connect", "unexpected_event")
    assert connection.sent == []


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (WebSocketConnectionClosedException(SYNTHETIC_RAW_REPLY), "closed"),
        (ConnectionRefusedError(SYNTHETIC_RAW_REPLY), "connection_error"),
        (ConnectionFailure(f"Websocket Time out: {SYNTHETIC_RAW_REPLY}"), "unexpected_event"),
        (ValueError(SYNTHETIC_RAW_REPLY), "protocol_error"),
    ],
)
def test_runtime_failure_classification_ignores_exception_data(sdk_stream, error, category):
    _, connection, _ = sdk_stream
    connection.recv = Mock(side_effect=error)
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvError):
        client.read_art_mode()
    failure = client.last_failure
    assert failure is not None
    assert asdict(failure) == {"channel": "art", "stage": "channel_connect", "error": category}


@pytest.mark.parametrize("status", [403, SYNTHETIC_TOKEN])
def test_pairing_diagnostic_only_logs_numeric_http_status(sdk_stream, caplog, status):
    _, connection, factory = sdk_stream
    connection.close = Mock()
    factory.side_effect = tv.websocket.WebSocketBadStatusException(str(SYNTHETIC_RAW_REPLY), status)
    with caplog.at_level(logging.WARNING), pytest.raises(tv.TvError):
        tv.TvClient(SYNTHETIC_HOST, None).pair()
    assert ("HTTP status 403" if status == 403 else "HTTP status None") in caplog.text
    assert SYNTHETIC_TOKEN not in caplog.text
    assert "raw-reply" not in caplog.text


def test_remote_operations_reopen_with_token_and_preserve_art(monkeypatch):
    clock = SyntheticClock()
    connections = []

    def connect(_):
        connection = SyntheticSocket(clock)
        connection.match = True
        connections.append(connection)
        return connection

    factory = Mock(side_effect=lambda *_args, socket, **_kwargs: socket)
    monkeypatch.setattr(tv, "monotonic", clock)
    monkeypatch.setattr(tv.time, "sleep", lambda _: None)
    monkeypatch.setattr(tv._DeadlineConnection, "_connect_socket", connect)
    monkeypatch.setattr(sdk_connection.websocket, "create_connection", factory)
    client = tv.TvClient(SYNTHETIC_HOST, None)
    remote = client._remote
    assert client.read_art_mode() is True
    art = client._art
    assert art is not None
    art_connection = connections[0]

    for held in (False, True, False):
        clock.now += 3600
        before = len(connections)
        if held:
            client.hold_power(3)
        else:
            client.send_key("KEY_HOME")
        assert len(connections) == before + 1
        connection = connections[-1]
        assert [command["params"]["Cmd"] for command in connection.sent] == (
            ["Press", "Release"] if held else ["Click"]
        )
        assert connection.shutdown_count == 1
        assert remote.connection is None
        assert client._remote is remote
        assert remote.token == SYNTHETIC_TOKEN
        assert client._art is art
        assert art.connection is art_connection
        assert art_connection.shutdown_count == 0
        assert not client._closed
        assert f"token={SYNTHETIC_TOKEN}" in factory.call_args.args[0]

    assert client.read_art_mode() is True
    assert len(connections) == 4
    client.close()
    assert art_connection.shutdown_count == 1


@pytest.mark.parametrize("failure", ["key", "press", "hold", "release"])
def test_remote_operation_failure_closes_socket_after_release(sdk_stream, monkeypatch, failure):
    _, connection, _ = sdk_stream

    def fail_send(command):
        assert connection.shutdown_count == 0
        cmd = command["params"]["Cmd"]
        if (failure, cmd) in {("key", "Click"), ("press", "Press"), ("release", "Release")}:
            raise WebSocketTimeoutException(SYNTHETIC_RAW_REPLY)

    def delay(seconds):
        if failure == "hold" and seconds == 3:
            raise OSError(SYNTHETIC_RAW_REPLY)

    connection.on_send = fail_send
    monkeypatch.setattr(tv.time, "sleep", delay)
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvConnectionError) as caught:
        if failure == "key":
            client.send_key("KEY_HOME")
        else:
            client.hold_power(3)

    assert [command["params"]["Cmd"] for command in connection.sent] == (
        ["Click"] if failure == "key" else ["Press", "Release"]
    )
    assert connection.shutdown_count == 1
    assert client._remote.connection is None
    assert not client._closed
    assert_sanitized(caught.value)


@pytest.mark.parametrize("held", [False, True])
def test_remote_cleanup_failure_remains_sanitized_error(sdk_stream, monkeypatch, held):
    _, connection, _ = sdk_stream

    def fail_shutdown():
        raise OSError(SYNTHETIC_RAW_REPLY)

    monkeypatch.setattr(connection, "shutdown", fail_shutdown)
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvConnectionError) as caught:
        if held:
            client.hold_power(3)
        else:
            client.send_key("KEY_HOME")

    assert [command["params"]["Cmd"] for command in connection.sent] == (
        ["Press", "Release"] if held else ["Click"]
    )
    assert_sanitized(caught.value)


@pytest.mark.parametrize(("operation", "deadline"), [("pair", 40), ("send_key", 10)])
def test_sdk_startup_event_stream_has_absolute_deadline(sdk_stream, operation, deadline):
    clock, connection, factory = sdk_stream
    connection.startup_noise = True
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    expected = tv.TvPairingTimeout if operation == "pair" else tv.TvAuthorizationIncomplete

    with pytest.raises(expected) as caught:
        if operation == "pair":
            client.pair()
        else:
            client.send_key("KEY_HOME")

    assert clock.now == deadline
    assert connection.recv_count == deadline
    assert connection.shutdown_count == 1
    assert connection.sent == []
    factory.assert_called_once()
    assert_sanitized(caught.value)


@pytest.mark.parametrize("operation", ["read_art_mode", "set_art_mode"])
def test_inherited_art_event_stream_has_absolute_deadline(sdk_stream, operation):
    clock, connection, factory = sdk_stream
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)

    with pytest.raises(tv.TvConnectionError) as caught:
        if operation == "read_art_mode":
            client.read_art_mode()
        else:
            client.set_art_mode(True)

    assert clock.now == 10
    assert connection.recv_count == 10
    assert len(connection.sent) == 1
    factory.assert_called_once()
    client.close()
    assert connection.shutdown_count == 1
    assert_sanitized(caught.value)


def test_inherited_art_reply_accepts_matching_event_and_resets_deadline(sdk_stream):
    clock, connection, _ = sdk_stream
    connection.match = True
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)

    assert client.read_art_mode() is True
    clock.now += 20
    assert client.read_art_mode() is True

    assert len(connection.sent) == 2
    assert connection.timeouts[-1] == 10
    client.close()


def test_startup_failure_closes_temporary_socket(sdk_stream):
    _, connection, _ = sdk_stream
    connection.initial = [{"event": "ms.channel.unauthorized", "data": SYNTHETIC_RAW_REPLY}]

    with pytest.raises(tv.TvPairingRejected) as caught:
        tv.TvClient(SYNTHETIC_HOST, None).pair()

    assert connection.shutdown_count == 1
    assert_sanitized(caught.value)


def test_expired_press_budget_still_sends_one_release(sdk_stream):
    clock, connection, _ = sdk_stream

    def expire_press(command):
        if command["params"]["Cmd"] == "Press":
            clock.now = 10
            raise WebSocketTimeoutException(SYNTHETIC_RAW_REPLY)

    connection.on_send = expire_press
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)

    with pytest.raises(tv.TvConnectionError) as caught:
        client.hold_power(3)

    assert [command["params"]["Cmd"] for command in connection.sent] == ["Press", "Release"]
    assert connection.timeouts[-1] == 10
    assert_sanitized(caught.value)
    client.close()


def wire_frame(payload: str | bytes, opcode: int = ABNF.OPCODE_TEXT, *, fin: int = 1):
    frame = ABNF.create_frame(payload, opcode, fin)
    frame.mask_value = 0
    return frame.format()


class SyntheticWireSocket:
    """Supply raw bytes to the real HTTP Upgrade and WebSocket frame loops."""

    def __init__(self, clock: SyntheticClock) -> None:
        self.clock = clock
        self.timeout = 0.0
        self.timeouts: list[float] = []
        self._closed = False
        self.close_count = 0
        self.connect_seconds = 0.0
        self.connect_failure = False
        self.tls_seconds = 0.0
        self.http_read_seconds = 0.0
        self.http_write_seconds = 0.0
        self.read_seconds = 0.0
        self.write_seconds = 0.0
        self.read_width = 65536
        self.free_read_bytes = 0
        self.write_width = 65536
        self.http_request = bytearray()
        self.http_response = bytearray()
        self.incoming = bytearray()
        self.outgoing = bytearray()
        self.upgraded = False
        self.recv_count = 0
        self.send_count = 0

    def wait(self, seconds: float) -> None:
        self.clock.now += min(seconds, self.timeout)
        if seconds >= self.timeout:
            raise TimeoutError("synthetic socket wait expired")

    def settimeout(self, seconds: float) -> None:
        self.timeout = seconds
        self.timeouts.append(seconds)

    def gettimeout(self) -> float:
        return self.timeout

    def connect(self, address) -> None:
        assert address[0] == SYNTHETIC_HOST
        self.wait(self.connect_seconds)
        if self.connect_failure:
            raise OSError("synthetic refused connection")

    def recv(self, size: int) -> bytes:
        self.recv_count += 1
        assert self.recv_count < 2000, "synthetic stream safety stop"
        pending = self.http_response if self.http_response else self.incoming
        size = min(size, self.read_width, len(pending))
        if self.http_response:
            self.wait(self.http_read_seconds)
        elif self.free_read_bytes:
            self.free_read_bytes -= size
        else:
            self.wait(self.read_seconds)
        result = bytes(pending[:size])
        del pending[:size]
        return result

    def send(self, data) -> int:
        self.send_count += 1
        assert self.send_count < 2000, "synthetic write safety stop"
        self.wait(self.write_seconds if self.upgraded else self.http_write_seconds)
        size = min(len(data), self.write_width)
        if self.upgraded:
            self.outgoing.extend(data[:size])
        else:
            self.http_request.extend(data[:size])
            if self.http_request.endswith(b"\r\n\r\n"):
                key = next(
                    line.split(b": ", 1)[1]
                    for line in self.http_request.split(b"\r\n")
                    if line.startswith(b"Sec-WebSocket-Key:")
                )
                accept = base64.b64encode(
                    hashlib.sha1(key + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest()
                )
                self.http_response.extend(
                    b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                    b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n"
                )
                self.upgraded = True
        return size

    def close(self) -> None:
        self._closed = True
        self.close_count += 1


@pytest.fixture
def sdk_wire(monkeypatch):
    clock = SyntheticClock()
    raw = SyntheticWireSocket(clock)
    factory = Mock(return_value=raw)
    monkeypatch.setattr(tv, "monotonic", clock)
    monkeypatch.setattr(tv.time, "sleep", lambda _: None)
    monkeypatch.setattr(tv.socket, "socket", factory)
    monkeypatch.setattr(
        tv.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (SYNTHETIC_HOST, 8002))
        ],
    )

    def wrap_socket(sock, *, server_hostname):
        assert sock is raw and server_hostname == SYNTHETIC_HOST
        raw.wait(raw.tls_seconds)
        return raw

    context = Mock()
    context.wrap_socket.side_effect = wrap_socket
    monkeypatch.setattr(tv.ssl, "SSLContext", Mock(return_value=context))
    monkeypatch.setattr(tv._DeadlineArt, "_new_request_uuid", lambda _: "synthetic-request")
    return clock, raw, factory


def startup_frame() -> bytes:
    return wire_frame(json.dumps({"event": "ms.channel.connect", "data": {}}))


@pytest.mark.parametrize(
    ("traffic", "token"),
    [
        ("silent", SYNTHETIC_TOKEN),
        ("partial", SYNTHETIC_TOKEN),
        ("ping", SYNTHETIC_TOKEN),
        ("unexpected", SYNTHETIC_TOKEN),
        ("ready", SYNTHETIC_TOKEN),
        ("ready", None),
    ],
)
def test_art_receive_trace_counts_only_readiness_bytes(sdk_wire, caplog, traffic, token):
    _, raw, _ = sdk_wire
    raw.connect_seconds = 1
    ready = wire_frame('{"event":"ms.channel.ready"}')
    data = {
        "silent": b"",
        "partial": b"\x81",
        "ping": wire_frame(b"", ABNF.OPCODE_PING),
        "unexpected": wire_frame(
            json.dumps({"event": SYNTHETIC_TOKEN, "data": SYNTHETIC_RAW_REPLY})
        ),
        "ready": ready,
    }[traffic]
    raw.incoming.extend(startup_frame() + data)
    if traffic == "ready":
        raw.incoming.extend(
            wire_frame(
                json.dumps(
                    {
                        "event": "d2d_service_message",
                        "data": json.dumps({"id": "synthetic-request", "value": "on"}),
                    }
                )
            )
        )
    raw.free_read_bytes = len(raw.incoming)
    raw.read_seconds = 10
    client = tv.TvClient(SYNTHETIC_HOST, token)
    client._art_read_trace_attempt = 1

    if traffic == "ready":
        assert client.read_art_mode() is True
    else:
        with pytest.raises(tv.TvConnectionError) as caught:
            client.read_art_mode()
        assert_sanitized(caught.value)

    records = [r for r in caplog.records if r.getMessage().startswith("TV Art receive trace:")]
    assert len(records) == 1
    fields = dict(part.split("=") for part in records[0].getMessage().split(": ", 1)[1].split())
    assert fields["attempt"] == "1"
    assert fields["startup_entered"] == "True"
    assert fields["token_supplied"] == str(token is not None)
    assert fields["channel_last_event"] == "connect"
    assert int(fields["channel_recv_bytes"]) == len(startup_frame())
    assert fields["channel_messages"] == fields["channel_events"] == "1"
    assert fields["ready_entered"] == "True"
    assert fields["ready_accepted"] == str(traffic == "ready")
    assert int(fields["recv_bytes"]) == len(data)
    pending_receive = int(fields["recv_entered"]) - int(fields["recv_returned"])
    assert pending_receive == (1 if traffic in {"silent", "partial", "ping"} else 0)
    assert int(fields["ready_ms"]) == (9000 if traffic in {"silent", "partial", "ping"} else 0)
    assert int(fields["elapsed_ms"]) == 1000 + int(fields["ready_ms"])
    assert records[0].exc_info is None
    assert SYNTHETIC_TOKEN not in caplog.text
    assert "raw-reply" not in caplog.text
    assert SYNTHETIC_HOST not in caplog.text
    client.close()


def test_art_receive_trace_is_one_read_only_on_reused_socket(sdk_wire, caplog):
    _, raw, factory = sdk_wire
    reply = wire_frame(
        json.dumps(
            {
                "event": "d2d_service_message",
                "data": json.dumps({"id": "synthetic-request", "value": "on"}),
            }
        )
    )
    raw.incoming.extend(startup_frame() + wire_frame('{"event":"ms.channel.ready"}') + reply * 3)
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    for attempt in (1, 2, 0):
        client._art_read_trace_attempt = attempt
        assert client.read_art_mode() is True

    records = [r for r in caplog.records if r.getMessage().startswith("TV Art receive trace:")]
    assert len(records) == 2
    assert records[1].getMessage() == (
        "TV Art receive trace: attempt=2 elapsed_ms=0 startup_entered=False token_supplied=False "
        "channel_entered=False channel_ms=0 channel_recv_entered=0 channel_recv_returned=0 "
        "channel_recv_bytes=0 channel_messages=0 channel_events=0 channel_last_event=none "
        "ready_entered=False ready_accepted=False ready_ms=0 "
        "recv_entered=0 recv_returned=0 recv_bytes=0"
    )
    factory.assert_called_once()
    client.close()


@pytest.mark.parametrize(
    ("traffic", "messages", "events", "last_event", "error"),
    [
        ("silent", 0, 0, "none", "timeout"),
        ("partial", 0, 0, "none", "timeout"),
        ("ping", 0, 0, "none", "timeout"),
        ("ignored", 1, 1, "ignored", "timeout"),
        ("malformed", 1, 0, "none", "protocol_error"),
        ("unauthorized", 1, 1, "unauthorized", "rejected"),
        ("unknown", 1, 1, "other", "unexpected_event"),
        ("non_string", 1, 1, "other", "unexpected_event"),
        ("ready", 1, 1, "ready", "unexpected_event"),
        ("connected", 1, 1, "connect", None),
    ],
)
def test_art_channel_trace_records_handshake_boundary(
    sdk_wire, caplog, monkeypatch, traffic, messages, events, last_event, error
):
    _, raw, _ = sdk_wire
    raw.connect_seconds = 1
    data = {
        "silent": b"",
        "partial": b"\x81",
        "ping": wire_frame(b"", ABNF.OPCODE_PING),
        "ignored": wire_frame(json.dumps({"event": tv.IGNORE_EVENTS_AT_STARTUP[0]})),
        "malformed": wire_frame("{" + SYNTHETIC_TOKEN),
        "unauthorized": wire_frame('{"event":"ms.channel.unauthorized"}'),
        "unknown": wire_frame(json.dumps({"event": SYNTHETIC_TOKEN})),
        "non_string": wire_frame(json.dumps({"event": [SYNTHETIC_TOKEN]})),
        "ready": wire_frame('{"event":"ms.channel.ready"}'),
        "connected": startup_frame(),
    }[traffic]
    raw.incoming.extend(data)
    ready = wire_frame('{"event":"ms.channel.ready"}')
    if traffic == "connected":
        raw.incoming.extend(
            ready
            + wire_frame(
                json.dumps(
                    {
                        "event": "d2d_service_message",
                        "data": json.dumps({"id": "synthetic-request", "value": "off"}),
                    }
                )
            )
        )
    raw.free_read_bytes = len(raw.incoming)
    raw.read_seconds = 10
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    client._art_read_trace_attempt = 1
    art = client._art_client()
    trace = tv._ArtReadTrace()
    monkeypatch.setattr(tv, "_ArtReadTrace", Mock(return_value=trace))
    if error is None:
        assert client.read_art_mode() is False
    else:
        with pytest.raises(tv.TvError) as caught:
            client.read_art_mode()
        assert_sanitized(caught.value)
        assert client.last_failure == tv.TvFailure("art", "channel_connect", error)
        expected = (
            tv.TvPairingRejected if traffic == "unauthorized" else tv.TvAuthorizationIncomplete
        )
        assert client.authorization_error is expected

    records = [r for r in caplog.records if r.getMessage().startswith("TV Art receive trace:")]
    assert len(records) == 1
    fields = dict(part.split("=") for part in records[0].getMessage().split(": ", 1)[1].split())
    assert fields["channel_entered"] == "True"
    assert fields["channel_messages"] == str(messages)
    assert fields["channel_events"] == str(events)
    assert fields["channel_last_event"] == last_event
    assert int(fields["channel_recv_bytes"]) == len(data)
    timed_out = error == "timeout"
    assert int(fields["channel_recv_entered"]) - int(fields["channel_recv_returned"]) == timed_out
    assert int(fields["channel_ms"]) == (9000 if timed_out else 0)
    assert fields["ready_entered"] == str(error is None)
    assert int(fields["recv_bytes"]) == (len(ready) if error is None else 0)
    assert art._read_trace is None
    assert not trace.channel_active
    assert not trace.ready_active
    assert records[0].exc_info is None
    assert SYNTHETIC_TOKEN not in caplog.text
    assert SYNTHETIC_HOST not in caplog.text
    client.close()


@pytest.mark.parametrize("stage", ["startup", "pair", "art"])
@pytest.mark.parametrize("traffic", ["ping", "pong", "fragmented", "partial"])
@pytest.mark.parametrize("expires", [False, True])
def test_real_websocket_receive_loops_share_absolute_budget(sdk_wire, stage, traffic, expires):
    clock, raw, factory = sdk_wire
    budget = 40 if stage == "pair" else 10
    payload = (
        json.dumps(
            {
                "event": "d2d_service_message",
                "data": json.dumps({"id": "synthetic-request", "value": "on"}),
            }
        )
        if stage == "art"
        else json.dumps({"event": "ms.channel.connect", "data": {}})
    )
    if traffic in {"ping", "pong"}:
        opcode = ABNF.OPCODE_PING if traffic == "ping" else ABNF.OPCODE_PONG
        incoming = wire_frame(b"", opcode) * (100 if expires else 2) + wire_frame(payload)
    elif traffic == "fragmented":
        incoming = wire_frame(payload[:1], fin=0)
        if expires:
            incoming += wire_frame(" ", ABNF.OPCODE_CONT, fin=0) * 100
        else:
            incoming += wire_frame(payload[1:-1], ABNF.OPCODE_CONT, fin=0)
            incoming += wire_frame(payload[-1:], ABNF.OPCODE_CONT)
    else:
        raw.read_width = 1
        incoming = wire_frame(payload)
    if stage == "art":
        raw.incoming.extend(startup_frame() + wire_frame('{"event":"ms.channel.ready"}'))
        raw.free_read_bytes = len(raw.incoming)
    raw.incoming.extend(incoming)
    raw.read_seconds = 1 if expires else 0.001
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)

    def operate():
        if stage == "art":
            assert client.read_art_mode() is True
        elif stage == "pair":
            assert client.pair() == SYNTHETIC_TOKEN
        else:
            client.send_key("KEY_HOME")

    if expires:
        expected = tv.TvPairingTimeout if stage == "pair" else tv.TvConnectionError
        with pytest.raises(expected) as caught:
            operate()
        assert clock.now == budget
        assert_sanitized(caught.value)
    else:
        operate()
        assert clock.now < budget
        if stage != "pair":
            assert raw.outgoing
    factory.assert_called_once()
    assert all(0 < timeout <= budget for timeout in raw.timeouts)
    client.close()
    assert raw._closed


@pytest.mark.parametrize("direction", ["read", "write"])
@pytest.mark.parametrize("expires", [False, True])
def test_real_http_upgrade_io_shares_absolute_budget(sdk_wire, direction, expires):
    clock, raw, _ = sdk_wire
    raw.incoming.extend(startup_frame())
    if direction == "read":
        raw.http_read_seconds = 1 if expires else 0.001
    else:
        raw.write_width = 1
        raw.http_write_seconds = 1 if expires else 0.001
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    if expires:
        with pytest.raises(tv.TvConnectionError) as caught:
            client.send_key("KEY_HOME")
        assert clock.now == 10
        assert raw.outgoing == b""
        assert_sanitized(caught.value)
    else:
        client.send_key("KEY_HOME")
        assert clock.now < 10
        assert raw.http_request.endswith(b"\r\n\r\n")
    client.close()
    assert raw._closed


@pytest.mark.parametrize("expires", [False, True])
def test_real_websocket_partial_writes_share_absolute_budget(sdk_wire, expires):
    clock, raw, _ = sdk_wire
    raw.incoming.extend(startup_frame())
    raw.write_width = 1
    raw.write_seconds = 1 if expires else 0.001
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    if expires:
        with pytest.raises(tv.TvConnectionError) as caught:
            client.send_key("KEY_HOME")
        assert clock.now == 10
        assert len(raw.outgoing) == 9
        assert_sanitized(caught.value)
    else:
        client.send_key("KEY_HOME")
        assert clock.now < 10
        assert len(raw.outgoing) > 10
    client.close()
    assert raw._closed


def test_real_expired_press_write_keeps_one_fresh_release_attempt(sdk_wire, monkeypatch):
    clock, raw, _ = sdk_wire
    raw.incoming.extend(startup_frame())
    original_send = raw.send
    command_attempts = 0

    def send(data):
        nonlocal command_attempts
        if raw.upgraded:
            command_attempts += 1
            if command_attempts == 1:
                raw.wait(10)
        return original_send(data)

    monkeypatch.setattr(raw, "send", send)
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvConnectionError) as caught:
        client.hold_power(3)
    assert clock.now == 10
    assert command_attempts == 2
    assert raw.outgoing
    assert raw.timeouts[-1] == 10
    assert_sanitized(caught.value)
    client.close()


def test_tcp_address_attempts_share_absolute_budget(sdk_wire, monkeypatch):
    clock, raw, factory = sdk_wire
    raw.connect_seconds = 6
    raw.connect_failure = True
    monkeypatch.setattr(
        tv.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: (
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (SYNTHETIC_HOST, 8002))] * 2
        ),
    )
    with pytest.raises(tv.TvConnectionError) as caught:
        tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN).send_key("KEY_HOME")
    assert clock.now == 10
    assert raw.timeouts == [10, 4]
    assert raw.close_count == 2
    assert factory.call_count == 2
    assert_sanitized(caught.value)


@pytest.mark.parametrize("expires", [False, True])
def test_tls_handshake_uses_budget_remaining_after_tcp(sdk_wire, expires):
    clock, raw, _ = sdk_wire
    raw.connect_seconds = 6
    raw.tls_seconds = 6 if expires else 2
    raw.incoming.extend(startup_frame())
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    if expires:
        with pytest.raises(tv.TvConnectionError) as caught:
            client.send_key("KEY_HOME")
        assert clock.now == 10
        assert raw.timeouts == [10, 4]
        assert raw.http_request == b""
        assert_sanitized(caught.value)
    else:
        client.send_key("KEY_HOME")
        assert clock.now == 8
        assert raw.timeouts[:3] == [10, 4, 2]
    client.close()
    assert raw._closed


@pytest.mark.parametrize("startup", ["timeout", "unexpected", "closed", "rejected"])
def test_incomplete_authorization_suspends_both_channels(sdk_stream, startup):
    _, connection, factory = sdk_stream
    if startup == "timeout":
        connection.startup_noise = True
    elif startup == "closed":
        connection.recv = Mock(side_effect=OSError(SYNTHETIC_RAW_REPLY))
    else:
        connection.initial = [
            {
                "event": "ms.channel.unauthorized" if startup == "rejected" else "unexpected",
                "data": SYNTHETIC_RAW_REPLY,
            }
        ]
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    expected = "TvPairingRejected" if startup == "rejected" else "TvAuthorizationIncomplete"
    for operation in (
        client.read_art_mode,
        lambda: client.send_key("KEY_HOME"),
        lambda: client.hold_power(3),
        client.read_art_mode,
    ):
        with pytest.raises(tv.TvError) as caught:
            operation()
        assert type(caught.value).__name__ == expected
        assert_sanitized(caught.value)
    assert factory.call_count == 1
    assert connection.shutdown_count == 1
    assert connection.sent == []


@pytest.mark.parametrize("stage", ["tcp", "upgrade"])
def test_failure_before_upgrade_keeps_authorization_retryable(sdk_stream, monkeypatch, stage):
    _, connection, factory = sdk_stream
    if stage == "tcp":
        original = tv._DeadlineConnection._connect_socket
        attempts = []

        def connect(client):
            attempts.append(None)
            if len(attempts) == 1:
                raise TimeoutError(SYNTHETIC_RAW_REPLY)
            return original(client)

        monkeypatch.setattr(tv._DeadlineConnection, "_connect_socket", connect)
    else:
        connection.close = Mock()
        factory.side_effect = [WebSocketTimeoutException(SYNTHETIC_RAW_REPLY), connection]
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvConnectionError) as caught:
        client.send_key("KEY_HOME")
    assert type(caught.value) is tv.TvConnectionError
    client.send_key("KEY_HOME")
    assert len(connection.sent) == 1


def test_remote_issued_token_updates_already_created_art(sdk_stream):
    _, connection, factory = sdk_stream
    connection.match = True
    client = tv.TvClient(SYNTHETIC_HOST, "synthetic-old-token")
    assert client.read_art_mode() is True
    assert client._art is not None
    client._art.close()
    connection.initial = [{"event": "ms.channel.connect", "data": {"token": "synthetic-new-token"}}]
    client.send_key("KEY_HOME")
    connection.initial = [
        {"event": "ms.channel.connect", "data": {}},
        {"event": "ms.channel.ready"},
    ]
    assert client.read_art_mode() is True
    assert "token=synthetic-new-token" in factory.call_args.args[0]
    assert client.token == "synthetic-new-token"


def test_art_issued_token_survives_failure_before_ready(sdk_stream):
    _, connection, _ = sdk_stream
    connection.initial[1] = {"event": "unexpected", "data": SYNTHETIC_RAW_REPLY}
    client = tv.TvClient(SYNTHETIC_HOST, "synthetic-old-token")
    with pytest.raises(tv.TvConnectionError) as caught:
        client.read_art_mode()
    assert type(caught.value) is tv.TvConnectionError
    assert client.token == SYNTHETIC_TOKEN
    assert client._remote.token == SYNTHETIC_TOKEN


def test_established_art_timeout_keeps_authorization_retryable(sdk_stream):
    _, connection, factory = sdk_stream
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvConnectionError) as caught:
        client.read_art_mode()
    assert type(caught.value) is tv.TvConnectionError
    connection.match = True
    assert client.read_art_mode() is True
    assert factory.call_count == 1


@pytest.mark.parametrize("startup", ["rejected", "incomplete"])
def test_startup_cleanup_failure_preserves_authorization_suspension(
    sdk_stream, monkeypatch, startup
):
    _, connection, factory = sdk_stream
    connection.initial = [
        {
            "event": "ms.channel.unauthorized" if startup == "rejected" else "unexpected",
            "data": SYNTHETIC_RAW_REPLY,
        }
    ]
    monkeypatch.setattr(connection, "shutdown", Mock(side_effect=OSError(SYNTHETIC_RAW_REPLY)))
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    for _ in range(2):
        with pytest.raises(tv.TvError) as caught:
            client.send_key("KEY_HOME")
        assert type(caught.value).__name__ == (
            "TvPairingRejected" if startup == "rejected" else "TvAuthorizationIncomplete"
        )
        assert_sanitized(caught.value)
    assert factory.call_count == 1


def test_hold_authorization_failure_never_opens_socket_for_release(sdk_stream):
    _, connection, factory = sdk_stream
    connection.startup_noise = True
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    with pytest.raises(tv.TvError) as caught:
        client.hold_power(3)
    assert type(caught.value).__name__ == "TvAuthorizationIncomplete"
    assert connection.shutdown_count == 1
    assert factory.call_count == 1
    assert connection.sent == []


@pytest.mark.parametrize("stage", ["art_ready", "art_read", "art_write"])
def test_explicit_rejection_in_art_stream_suspends_every_websocket(sdk_stream, stage):
    _, connection, factory = sdk_stream
    connection.match = True
    client = tv.TvClient(SYNTHETIC_HOST, SYNTHETIC_TOKEN)
    rejected = {"event": "ms.channel.unauthorized", "data": SYNTHETIC_RAW_REPLY}
    if stage == "art_ready":
        connection.initial[1] = rejected
    else:
        assert client.read_art_mode() is True
        connection.initial = [rejected]
        connection.match = False
    with pytest.raises(tv.TvPairingRejected) as caught:
        if stage == "art_write":
            client.set_art_mode(False)
        else:
            client.read_art_mode()
    assert client.authorization_error is tv.TvPairingRejected
    assert_sanitized(caught.value)
    for operation in (
        lambda: client.send_key("KEY_HOME"),
        lambda: client.hold_power(3),
        client.read_art_mode,
    ):
        with pytest.raises(tv.TvPairingRejected):
            operation()
    assert factory.call_count == 1
    assert all(command["method"] != "ms.remote.control" for command in connection.sent)
    client.close()
    assert connection.shutdown_count == 1
