"""Bounded power RPC tests. All credentials and streams are synthetic."""

from __future__ import annotations

import asyncio
import json
import socket
import traceback
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientConnectionError, ClientPayloadError, ClientRequest, ClientSession

from custom_components.localthings import tv_power as power
from custom_components.localthings.tv import (
    TvAuthorizationIncomplete,
    TvConnectionError,
    TvError,
    TvPairingRejected,
)

HOST = "192.0.2.10"
TOKEN = "synthetic-power-token"
RAW = f"https://user:password@{HOST}/ raw-response {TOKEN}"
AUTHENTICATED_OPERATIONS = [
    ("async_read_power", ()),
    ("async_turn_off", ()),
    ("async_read_art_mode", ()),
    ("async_read_picture_mode", ()),
    ("async_set_art_mode", (True,)),
    ("async_set_art_mode", (False,)),
]


def encoded(result: Any) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": "1", "result": result}).encode()


class Response:
    """A chunked, decoded HTTP body without a network connection."""

    def __init__(
        self,
        body: bytes,
        *,
        chunk_size: int = 7,
        status: int = 200,
        body_error: BaseException | None = None,
        before_read: Callable[[], Any] | None = None,
    ) -> None:
        self.body = body
        self.chunk_size = chunk_size
        self.status = status
        self.body_error = body_error
        self.before_read = before_read
        self.content = self
        self.closed = False
        self.sizes: list[int] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    async def read(self, size: int) -> bytes:
        self.sizes.append(size)
        if self.before_read is not None:
            await self.before_read()
        if not self.body and self.body_error is not None:
            raise self.body_error
        size = min(size, self.chunk_size)
        chunk, self.body = self.body[:size], self.body[size:]
        return chunk


def session_for(*responses: Response | BaseException) -> MagicMock:
    session = MagicMock(spec=ClientSession)
    session.post.side_effect = responses
    return session


def assert_sanitized(error: BaseException) -> None:
    rendered = "".join(traceback.format_exception(error))
    for secret in (HOST, TOKEN, "password", "raw-response"):
        assert secret not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__


@pytest.mark.parametrize(("operation", "args"), [("async_pair", ()), *AUTHENTICATED_OPERATIONS])
@pytest.mark.parametrize("shared_accept", [None, "text/plain"])
async def test_power_rpc_meets_ipcontrol_http_header_contract(
    monkeypatch, operation, args, shared_accept
) -> None:
    class RequestCaptured(Exception):
        pass

    requests = []

    class CaptureRequest(ClientRequest):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            requests.append(self)
            raise RequestCaptured

    network = MagicMock(side_effect=AssertionError("Unexpected network access"))
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(socket, "socket", network)
    monkeypatch.setattr(socket, "getaddrinfo", network)
    monkeypatch.setattr(loop, "create_connection", network)
    monkeypatch.setattr(loop, "getaddrinfo", network)
    headers = {"User-Agent": "synthetic shared session"}
    if shared_accept is not None:
        headers["Accept"] = shared_accept
    async with ClientSession(headers=headers, request_class=CaptureRequest) as session:
        client = power.TvPowerClient(session, HOST, TOKEN)
        with pytest.raises(RequestCaptured):
            await getattr(client, operation)(*args)
        assert dict(session.headers) == headers
    network.assert_not_called()
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert "application/json" in request.headers["Accept"]
    assert "application/json" in request.headers["Content-Type"]
    assert request.headers["Host"] == f"{HOST}:1516"
    assert int(request.headers["Content-Length"]) > 0
    assert "Transfer-Encoding" not in request.headers


async def test_pair_sends_fixed_rpc_and_reuses_issued_token() -> None:
    pair = Response(encoded({"AccessToken": TOKEN}))
    status = Response(encoded({"power": "powerOn"}))
    session = session_for(pair, status)
    client = power.TvPowerClient(session, HOST, None)
    assert await client.async_pair() == TOKEN
    assert client.token == TOKEN
    assert await client.async_read_power() is True
    assert session.post.call_args_list[0].args == (f"https://{HOST}:1516/",)
    assert session.post.call_args_list[0].kwargs["json"] == {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "createAccessToken",
    }
    assert session.post.call_args_list[1].kwargs["json"] == {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "powerControl",
        "params": {"AccessToken": TOKEN},
    }
    for call in session.post.call_args_list:
        assert call.kwargs["ssl"] is False
        assert call.kwargs["allow_redirects"] is False
    assert pair.closed and status.closed
    session.close.assert_not_called()


async def test_body_limit_uses_decoded_content_even_if_shared_session_disables_it() -> None:
    session = session_for(Response(encoded({"power": "powerOff"})))
    assert await power.TvPowerClient(session, HOST, TOKEN).async_read_power() is False
    assert session.post.call_args.kwargs["auto_decompress"] is True


@pytest.mark.parametrize(
    "body",
    [
        b'{"jsonrpc":"2.0","id":"1","result":{"power":"powerOff"},"error":null}',
        b'{"jsonrpc":"2.0","id":"1","result":{"power":"powerOff"},"error":[]}',
        b'{"jsonrpc":"2.0","id":"1","result":{"power":"powerOff"},"code":-32002}',
        b"[" * 2000 + b"]" * 2000,
    ],
)
async def test_error_or_deeply_nested_json_is_never_success(body) -> None:
    with pytest.raises(TvError) as caught:
        await power.TvPowerClient(session_for(Response(body)), HOST, TOKEN).async_read_power()
    assert_sanitized(caught.value)


@pytest.mark.parametrize(
    "response",
    [
        Response(b"not-json"),
        Response(encoded({"AccessToken": TOKEN}), status=403),
        Response(encoded({"AccessToken": TOKEN}), body_error=ClientPayloadError(RAW)),
        Response(b" " * 65537, chunk_size=4096),
        Response(json.dumps({"code": -32002, "message": RAW}).encode()),
    ],
)
async def test_pairing_protocol_failures_latch_as_incomplete(response) -> None:
    session = session_for(response)
    client = power.TvPowerClient(session, HOST, TOKEN)
    with pytest.raises(power.TvPowerAuthorizationIncomplete) as caught:
        await client.async_pair()
    assert_sanitized(caught.value)
    with pytest.raises(power.TvPowerAuthorizationIncomplete):
        await client.async_read_power()
    assert session.post.call_count == 1
    assert response.closed
    assert client.token == TOKEN


@pytest.mark.parametrize(("value", "expected"), [("powerOn", True), ("powerOff", False)])
async def test_status_requires_explicit_power(value, expected) -> None:
    response = Response(encoded({"power": value}))
    client = power.TvPowerClient(session_for(response), HOST, TOKEN)
    assert await client.async_read_power() is expected
    assert response.closed


async def test_off_sends_one_explicit_setter() -> None:
    response = Response(encoded({"power": "powerOff"}))
    session = session_for(response)
    client = power.TvPowerClient(session, HOST, TOKEN)
    assert await client.async_turn_off() is None
    assert session.post.call_count == 1
    assert session.post.call_args.kwargs["json"] == {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "powerControl",
        "params": {"AccessToken": TOKEN, "power": "powerOff"},
    }


@pytest.mark.parametrize(("value", "expected"), [("artModeOn", True), ("artModeOff", False)])
async def test_art_getter_uses_saved_token_and_explicit_flag(value, expected) -> None:
    response = Response(encoded({"artMode": value}))
    session = session_for(response)
    client = power.TvPowerClient(session, HOST, TOKEN)
    assert await client.async_read_art_mode() is expected
    assert session.post.call_count == 1
    assert session.post.call_args.kwargs["json"] == {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "artModeControl",
        "params": {"AccessToken": TOKEN},
    }
    assert client.token == TOKEN
    assert response.closed


@pytest.mark.parametrize(
    "value",
    [None, False, True, 0, 1, [], {}, "", "on", "ARTMODEON", " artModeOn", "artModeOff ", RAW],
)
async def test_art_getter_rejects_malformed_flags(value) -> None:
    session = session_for(Response(encoded({"artMode": value})))
    with pytest.raises(TvError) as caught:
        await power.TvPowerClient(session, HOST, TOKEN).async_read_art_mode()
    assert_sanitized(caught.value)
    assert session.post.call_count == 1


async def test_art_getter_rejects_missing_flag() -> None:
    with pytest.raises(TvError) as caught:
        await power.TvPowerClient(
            session_for(Response(encoded({}))), HOST, TOKEN
        ).async_read_art_mode()
    assert_sanitized(caught.value)


@pytest.mark.parametrize(
    "value", ["Ambient", "Standard", "ambient", " Ambient ", "Movie", "  Custom  "]
)
async def test_picture_getter_preserves_nonempty_semantic_value(value) -> None:
    response = Response(encoded({"pictureMode": value}))
    session = session_for(response)
    client = power.TvPowerClient(session, HOST, TOKEN)
    assert await client.async_read_picture_mode() == value
    assert session.post.call_count == 1
    assert session.post.call_args.kwargs["json"] == {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "getTVStates",
        "params": {"AccessToken": TOKEN},
    }
    assert client.token == TOKEN
    assert response.closed


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"pictureMode": None},
        {"pictureMode": False},
        {"pictureMode": 0},
        {"pictureMode": []},
        {"pictureMode": {}},
        {"pictureMode": ""},
        {"pictureMode": " "},
        {"pictureMode": "\t\r\n"},
        {"pictureMode": "\u2003"},
    ],
)
async def test_picture_getter_rejects_missing_malformed_or_whitespace_value(result) -> None:
    with pytest.raises(TvError) as caught:
        await power.TvPowerClient(
            session_for(Response(encoded(result))), HOST, TOKEN
        ).async_read_picture_mode()
    assert_sanitized(caught.value)


@pytest.mark.parametrize(("enabled", "value"), [(True, "artModeOn"), (False, "artModeOff")])
@pytest.mark.parametrize("ack", [{}, {"artMode": "artModeOn"}, {"artMode": "artModeOff"}])
async def test_art_setter_sends_once_without_claiming_observed_state(enabled, value, ack) -> None:
    response = Response(encoded(ack))
    session = session_for(response)
    client = power.TvPowerClient(session, HOST, TOKEN)
    assert await client.async_set_art_mode(enabled) is None
    assert session.post.call_count == 1
    assert session.post.call_args.kwargs["json"] == {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "artModeControl",
        "params": {"AccessToken": TOKEN, "artMode": value},
    }
    assert response.closed
    assert client.token == TOKEN


@pytest.mark.parametrize(
    "result",
    [{}, {"power": None}, {"power": False}, {"power": "off"}, {"power": "POWERON"}, {"power": 0}],
)
async def test_missing_or_invalid_power_never_becomes_off(result) -> None:
    client = power.TvPowerClient(session_for(Response(encoded(result))), HOST, TOKEN)
    with pytest.raises(TvError) as caught:
        await client.async_read_power()
    assert_sanitized(caught.value)


@pytest.mark.parametrize("result", [{}, {"power": "powerOn"}, {"power": None}])
async def test_off_requires_explicit_off_acknowledgement(result) -> None:
    session = session_for(Response(encoded(result)))
    with pytest.raises(TvError):
        await power.TvPowerClient(session, HOST, TOKEN).async_turn_off()
    assert session.post.call_count == 1


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b"null",
        b'"raw-response"',
        b'{"jsonrpc":"1.0","id":"1","result":{"power":"powerOff"}}',
        b'{"jsonrpc":"2.0","id":1,"result":{"power":"powerOff"}}',
        b'{"jsonrpc":"2.0","id":"2","result":{"power":"powerOff"}}',
        b'{"id":"1","result":{"power":"powerOff"}}',
        b'{"jsonrpc":"2.0","result":{"power":"powerOff"}}',
        b'{"jsonrpc":"2.0","id":"1","result":[]}',
        b'{"jsonrpc":"2.0","id":"1","result":null}',
        b'{"jsonrpc":"2.0","id":"1"}',
        b"\xff",
    ],
)
@pytest.mark.parametrize(("operation", "args"), AUTHENTICATED_OPERATIONS)
async def test_invalid_success_envelopes_are_sanitized(body, operation, args) -> None:
    response = Response(body)
    client = power.TvPowerClient(session_for(response), HOST, TOKEN)
    with pytest.raises(TvError) as caught:
        await getattr(client, operation)(*args)
    assert_sanitized(caught.value)
    assert response.closed


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize(("operation", "args"), [("async_pair", ()), *AUTHENTICATED_OPERATIONS])
async def test_explicit_rejection_latches_authenticated_requests(flat, operation, args) -> None:
    error = {"code": -32010, "message": RAW}
    body = error if flat else {"jsonrpc": "2.0", "id": "1", "error": error}
    session = session_for(Response(json.dumps(body).encode()))
    client = power.TvPowerClient(session, HOST, TOKEN)
    assert issubclass(power.TvPowerPairingRejected, TvPairingRejected)
    for attempt, arguments in [(operation, args), *AUTHENTICATED_OPERATIONS]:
        with pytest.raises(power.TvPowerPairingRejected) as caught:
            await getattr(client, attempt)(*arguments)
        assert_sanitized(caught.value)
    assert session.post.call_count == 1
    assert client.token == TOKEN


@pytest.mark.parametrize(
    "envelope",
    [
        {"jsonrpc": "2.0", "id": "2"},
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "1.0", "id": "1"},
        {"jsonrpc": "2.0"},
        {"id": "1"},
        {},
        {"jsonrpc": "2.0", "id": "1", "result": {"power": "powerOff"}},
    ],
)
async def test_malformed_nested_errors_do_not_revoke_saved_permission(envelope) -> None:
    body = {**envelope, "error": {"code": -32010, "message": RAW}}
    response = Response(json.dumps(body).encode())
    session = session_for(response, Response(encoded({"power": "powerOn"})))
    client = power.TvPowerClient(session, HOST, TOKEN)
    with pytest.raises(TvError) as caught:
        await client.async_read_power()
    assert not isinstance(caught.value, (TvPairingRejected, TvAuthorizationIncomplete))
    assert_sanitized(caught.value)
    assert response.closed
    assert client.token == TOKEN
    assert await client.async_read_power() is True
    assert session.post.call_count == 2


@pytest.mark.parametrize(
    "fields",
    [
        {"jsonrpc": "1.0"},
        {"jsonrpc": "2.0"},
        {"id": "2"},
        {"id": 1},
        {"id": "1"},
        {"jsonrpc": "2.0", "id": "1"},
        {"result": {"power": "powerOff"}},
        {"error": {"code": -32010}},
        {"method": "powerControl"},
        {"params": {}},
    ],
)
async def test_ambiguous_flat_errors_do_not_revoke_saved_permission(fields) -> None:
    body = {"code": -32010, "message": RAW, **fields}
    session = session_for(
        Response(json.dumps(body).encode()), Response(encoded({"power": "powerOn"}))
    )
    client = power.TvPowerClient(session, HOST, TOKEN)
    with pytest.raises(TvError) as caught:
        await client.async_read_power()
    assert not isinstance(caught.value, (TvPairingRejected, TvAuthorizationIncomplete))
    assert_sanitized(caught.value)
    assert client.token == TOKEN
    assert await client.async_read_power() is True
    assert session.post.call_count == 2


@pytest.mark.parametrize(
    "envelope",
    [
        {"jsonrpc": "2.0", "id": "2"},
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "1.0", "id": "1"},
    ],
)
async def test_malformed_pairing_error_latches_incomplete_instead_of_rejected(envelope) -> None:
    body = {**envelope, "error": {"code": -32010, "message": RAW}}
    session = session_for(Response(json.dumps(body).encode()))
    client = power.TvPowerClient(session, HOST, TOKEN)
    with pytest.raises(power.TvPowerAuthorizationIncomplete) as caught:
        await client.async_pair()
    assert_sanitized(caught.value)
    with pytest.raises(power.TvPowerAuthorizationIncomplete):
        await client.async_read_power()
    assert client.token == TOKEN
    assert session.post.call_count == 1


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("code", [-32002, -32601, "-32010", None])
@pytest.mark.parametrize(("operation", "args"), AUTHENTICATED_OPERATIONS)
async def test_command_refusal_keeps_authorization_retryable(flat, code, operation, args) -> None:
    error = {"code": code, "message": RAW}
    body = error if flat else {"jsonrpc": "2.0", "id": "1", "error": error}
    session = session_for(
        Response(json.dumps(body).encode()), Response(encoded({"power": "powerOn"}))
    )
    client = power.TvPowerClient(session, HOST, TOKEN)
    with pytest.raises(TvError) as caught:
        await getattr(client, operation)(*args)
    assert not isinstance(caught.value, TvPairingRejected)
    assert not isinstance(caught.value, TvAuthorizationIncomplete)
    assert_sanitized(caught.value)
    assert session.post.call_count == 1
    assert await client.async_read_power() is True
    assert client.token == TOKEN


@pytest.mark.parametrize(
    "result",
    [{}, {"AccessToken": None}, {"AccessToken": ""}, {"AccessToken": "  "}, {"AccessToken": 1}],
)
async def test_incomplete_pairing_latches_without_replacing_saved_token(result) -> None:
    session = session_for(Response(encoded(result)))
    client = power.TvPowerClient(session, HOST, TOKEN)
    assert issubclass(power.TvPowerAuthorizationIncomplete, TvAuthorizationIncomplete)
    for operation, args in [("async_pair", ()), *AUTHENTICATED_OPERATIONS]:
        with pytest.raises(power.TvPowerAuthorizationIncomplete) as caught:
            await getattr(client, operation)(*args)
        assert_sanitized(caught.value)
    assert client.token == TOKEN
    assert session.post.call_count == 1


async def test_missing_token_blocks_authenticated_requests_without_pairing() -> None:
    session = session_for()
    client = power.TvPowerClient(session, HOST, None)
    for operation, args in AUTHENTICATED_OPERATIONS:
        with pytest.raises(power.TvPowerAuthorizationIncomplete):
            await getattr(client, operation)(*args)
    session.post.assert_not_called()


@pytest.mark.parametrize(("operation", "args"), AUTHENTICATED_OPERATIONS)
@pytest.mark.parametrize("failure", [ClientConnectionError(RAW), TimeoutError(RAW), OSError(RAW)])
async def test_paired_network_failure_does_not_retry_pair_or_discard_token(
    operation, args, failure
) -> None:
    session = session_for(failure, Response(encoded({"power": "powerOn"})))
    client = power.TvPowerClient(session, HOST, TOKEN)
    with pytest.raises(TvConnectionError) as caught:
        await getattr(client, operation)(*args)
    assert not isinstance(caught.value, TvAuthorizationIncomplete)
    assert_sanitized(caught.value)
    assert session.post.call_count == 1
    assert client.token == TOKEN
    assert await client.async_read_power() is True


@pytest.mark.parametrize("failure", [ClientConnectionError(RAW), TimeoutError(RAW), OSError(RAW)])
async def test_failed_pairing_latches_without_retry(failure) -> None:
    session = session_for(failure)
    client = power.TvPowerClient(session, HOST, None)
    for operation, args in [("async_pair", ()), *AUTHENTICATED_OPERATIONS]:
        with pytest.raises(power.TvPowerAuthorizationIncomplete) as caught:
            await getattr(client, operation)(*args)
        assert_sanitized(caught.value)
    assert session.post.call_count == 1


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 500])
@pytest.mark.parametrize(("operation", "args"), AUTHENTICATED_OPERATIONS)
async def test_non_success_http_is_sanitized_without_following_redirects(
    status, operation, args
) -> None:
    response = Response(encoded({"power": "powerOn"}), status=status)
    session = session_for(response)
    client = power.TvPowerClient(session, HOST, TOKEN)
    with pytest.raises(TvError) as caught:
        await getattr(client, operation)(*args)
    assert_sanitized(caught.value)
    assert response.closed
    assert session.post.call_args.kwargs["allow_redirects"] is False
    assert session.post.call_count == 1


@pytest.mark.parametrize(("size", "accepted"), [(65535, True), (65536, True), (65537, False)])
@pytest.mark.parametrize(
    ("operation", "args", "expected"),
    [
        ("async_read_power", (), False),
        ("async_turn_off", (), None),
        ("async_read_art_mode", (), False),
        ("async_read_picture_mode", (), "Ambient"),
        ("async_set_art_mode", (True,), None),
        ("async_set_art_mode", (False,), None),
    ],
)
async def test_decoded_stream_body_limit(size, accepted, operation, args, expected) -> None:
    body = encoded({"power": "powerOff", "artMode": "artModeOff", "pictureMode": "Ambient"})
    response = Response(body + b" " * (size - len(body)), chunk_size=4096)
    client = power.TvPowerClient(session_for(response), HOST, TOKEN)
    if accepted:
        assert await getattr(client, operation)(*args) == expected
    else:
        with pytest.raises(TvError) as caught:
            await getattr(client, operation)(*args)
        assert_sanitized(caught.value)
    assert response.closed
    assert all(size <= 65537 for size in response.sizes)


@pytest.mark.parametrize(("operation", "args"), AUTHENTICATED_OPERATIONS)
async def test_truncated_body_is_not_a_success_even_after_valid_json(operation, args) -> None:
    response = Response(encoded({"power": "powerOff"}), body_error=ClientPayloadError(RAW))
    client = power.TvPowerClient(session_for(response), HOST, TOKEN)
    with pytest.raises(TvConnectionError) as caught:
        await getattr(client, operation)(*args)
    assert_sanitized(caught.value)
    assert response.closed


@pytest.mark.parametrize(
    ("operation", "args", "budget"),
    [("async_pair", (), 120), *((method, args, 10) for method, args in AUTHENTICATED_OPERATIONS)],
)
@pytest.mark.parametrize("phase", ["request", "body"])
async def test_one_absolute_deadline_covers_request_and_body(
    monkeypatch, operation, args, budget, phase
):
    real_timeout = asyncio.timeout
    deadlines = []

    def bounded_timeout(seconds):
        assert seconds == budget
        context = real_timeout(seconds)
        deadlines.append(context)
        return context

    async def expire():
        deadlines[0].reschedule(asyncio.get_running_loop().time())
        await asyncio.Future()

    class SlowResponse(Response):
        async def __aenter__(self):
            if phase == "request":
                await expire()
            return self

    response = SlowResponse(encoded({"power": "powerOff"}), before_read=expire)
    session = session_for(response)
    client = power.TvPowerClient(session, HOST, TOKEN)
    monkeypatch.setattr(power.asyncio, "timeout", bounded_timeout)
    expected = (
        power.TvPowerAuthorizationIncomplete if operation == "async_pair" else TvConnectionError
    )
    with pytest.raises(expected) as caught:
        await getattr(client, operation)(*args)
    assert_sanitized(caught.value)
    assert len(deadlines) == 1
    assert session.post.call_count == 1
    if phase == "body":
        assert response.closed


@pytest.mark.parametrize(("operation", "args"), [("async_pair", ()), *AUTHENTICATED_OPERATIONS])
async def test_cancellation_propagates_and_closes_response(operation, args) -> None:
    response = Response(b"", body_error=asyncio.CancelledError())
    session = session_for(response, Response(encoded({"power": "powerOn"})))
    client = power.TvPowerClient(session, HOST, TOKEN)
    with pytest.raises(asyncio.CancelledError):
        await getattr(client, operation)(*args)
    assert response.closed
    session.close.assert_not_called()
    if operation == "async_pair":
        with pytest.raises(power.TvPowerAuthorizationIncomplete):
            await client.async_read_power()
        assert session.post.call_count == 1
    else:
        assert await client.async_read_power() is True
    assert client.token == TOKEN
