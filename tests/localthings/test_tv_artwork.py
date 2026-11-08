"""Independent artwork transport tests using synthetic replies and sockets."""

from __future__ import annotations

import json
import socket
import ssl
import threading
import traceback
from importlib.metadata import PackageNotFoundError
from typing import Any
from unittest.mock import Mock
from unittest.mock import call as mock_call

import pytest
from samsungtvws.event import D2D_SERVICE_MESSAGE_EVENT
from samsungtvws.exceptions import ConnectionFailure, UnauthorizedError

from custom_components.localthings import tv, tv_artwork
from custom_components.localthings.tv import TvFailure

HOST = "192.0.2.10"
CONTENT_ID = "SYNTHETIC-1"
JPEG = b"\xff\xd8\xffsynthetic-jpeg\xff\xd9"
PNG = b"\x89PNG\r\n\x1a\nsynthetic-png"
RAW_SECRET = "synthetic-token-raw-reply"


@pytest.fixture
def sdk(monkeypatch):
    factory = Mock()
    art = factory.return_value
    art._deadline = tv_artwork._Deadline()
    art._failure_stage = "request"
    art._failure_category = None
    art._handshake_ms = None
    art.available.return_value = []
    art.get_current.return_value = {"content_id": CONTENT_ID}
    art.get_thumbnail_list.return_value = {f"{CONTENT_ID}.jpg": bytearray(JPEG)}
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", factory)
    return factory, art


def test_catalog_deduplicates_and_uses_metadata_fallbacks(sdk) -> None:
    _, art = sdk
    art.available.return_value = [
        {"content_id": CONTENT_ID, "title": "Synthetic title", "image_date": "2026-01-01"},
        {"content_id": CONTENT_ID, "category_id": "SYNTHETIC-FAVOURITES"},
        {"content_id": "SYNTHETIC-2"},
    ]

    assert tv_artwork.TvArtworkClient(HOST).list_artwork() == [
        tv_artwork.Artwork(CONTENT_ID, "Synthetic title", "2026-01-01"),
        tv_artwork.Artwork("SYNTHETIC-2", "SYNTHETIC-2", ""),
    ]
    art.close.assert_called_once_with()


def test_current_reads_once_without_selecting(sdk) -> None:
    _, art = sdk
    assert tv_artwork.TvArtworkClient(HOST).current() == CONTENT_ID
    art.get_current.assert_called_once_with()
    art.select_image.assert_not_called()
    art.available.assert_not_called()
    art.close.assert_called_once_with()


@pytest.mark.parametrize("payload", [None, [], {}, {"content_id": "../image"}])
def test_invalid_current_response_fails_closed(sdk, payload) -> None:
    _, art = sdk
    art.get_current.return_value = payload
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).current()
    art.close.assert_called_once_with()


def test_each_operation_uses_fresh_tokenless_connection(sdk) -> None:
    factory, art = sdk
    client = tv_artwork.TvArtworkClient(HOST)
    client.list_artwork()
    client.thumbnail(CONTENT_ID)
    client.select(CONTENT_ID)

    assert factory.call_count == 3
    for call in factory.call_args_list:
        assert call.args == (HOST,)
        assert call.kwargs["port"] == 8002
        assert call.kwargs["name"] == "LocalThings"
        assert call.kwargs.get("token") is None
        assert call.kwargs.get("token_file") is None
    assert art.close.call_count == 3


@pytest.mark.parametrize("payload", [None, {}, [None], [{"content_id": "../image"}]])
def test_invalid_catalog_fails_closed_and_closes(sdk, payload) -> None:
    _, art = sdk
    art.available.return_value = payload
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).list_artwork()
    art.close.assert_called_once_with()


@pytest.mark.parametrize("content_id", ["", "../image", "/image", "https://example.test/i", "x?y"])
def test_external_identifiers_rejected_before_connection(sdk, content_id) -> None:
    factory, _ = sdk
    client = tv_artwork.TvArtworkClient(HOST)
    with pytest.raises(tv_artwork.ArtworkError):
        client.thumbnail(content_id)
    with pytest.raises(tv_artwork.ArtworkError):
        client.select(content_id)
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("image", "extension", "mime"), [(JPEG, "jpg", "image/jpeg"), (PNG, "png", "image/png")]
)
def test_one_id_batch_thumbnail(sdk, image, extension, mime) -> None:
    _, art = sdk
    art.get_thumbnail_list.return_value = {f"{CONTENT_ID}.{extension}": bytearray(image)}
    assert tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID) == (image, mime)
    art.get_thumbnail_list.assert_called_once_with([CONTENT_ID])


@pytest.mark.parametrize(
    ("image", "extension", "mime"),
    [(PNG, "jpg", "image/png"), (PNG, "jpeg", "image/png"), (JPEG, "png", "image/jpeg")],
)
def test_thumbnail_mime_follows_supported_bytes_when_transfer_extension_is_wrong(
    monkeypatch, image, extension, mime
):
    art = make_art()
    art._deadline = tv_artwork._Deadline()
    art._send_art_request = Mock(return_value={"conn_info": {}})
    payload = d2d_bytes(fileType=extension, fileLength=len(image)).replace(JPEG, image)
    raw = fake_receiving_socket(payload)
    art._open_d2d_socket = Mock(return_value=raw)
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(return_value=art))

    assert tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID) == (image, mime)

    art._send_art_request.assert_called_once()
    raw.close.assert_called_once_with()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {f"{CONTENT_ID}.svg": b"<svg/>"},
        {f"{CONTENT_ID}.svg": JPEG},
        {f"{CONTENT_ID}.png": b"wrong"},
        {f"{CONTENT_ID}.jpg": b"wrong"},
        {"other.jpg": JPEG},
        {f"{CONTENT_ID}.jpg": JPEG, "other.jpg": JPEG},
    ],
)
def test_thumbnail_rejects_invalid_image_result(sdk, payload) -> None:
    _, art = sdk
    art.get_thumbnail_list.return_value = payload
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)
    art.close.assert_called_once_with()


def test_thumbnail_size_boundary(sdk) -> None:
    _, art = sdk
    image = JPEG + bytes(8 * 1024 * 1024 - len(JPEG))
    art.get_thumbnail_list.return_value = {f"{CONTENT_ID}.jpg": image}
    assert tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID) == (image, "image/jpeg")


def test_thumbnail_rejects_one_byte_over_size_limit(sdk) -> None:
    _, art = sdk
    image = JPEG + bytes(8 * 1024 * 1024 - len(JPEG) + 1)
    art.get_thumbnail_list.return_value = {f"{CONTENT_ID}.jpg": image}
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)


def test_selection_confirms_with_read_and_never_sets_category(sdk) -> None:
    _, art = sdk
    art.get_current.side_effect = [{"content_id": "SYNTHETIC-OTHER"}, {"content_id": CONTENT_ID}]
    assert tv_artwork.TvArtworkClient(HOST).select(CONTENT_ID) == CONTENT_ID
    art.select_image.assert_called_once_with(CONTENT_ID, show=True)
    assert art.get_current.call_count == 2


def test_failed_confirmation_does_not_repeat_setter(sdk) -> None:
    _, art = sdk
    art.get_current.return_value = {"content_id": "SYNTHETIC-OTHER"}
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).select(CONTENT_ID)
    art.select_image.assert_called_once_with(CONTENT_ID, show=True)
    art.close.assert_called_once_with()


def test_failed_setter_does_not_repeat_or_confirm(sdk) -> None:
    _, art = sdk
    art.select_image.side_effect = TimeoutError(RAW_SECRET)
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).select(CONTENT_ID)
    art.select_image.assert_called_once_with(CONTENT_ID, show=True)
    art.get_current.assert_not_called()


@pytest.mark.parametrize("failure", [UnauthorizedError(RAW_SECRET), OSError(RAW_SECRET)])
def test_raw_failures_are_sanitized_without_context(sdk, failure) -> None:
    _, art = sdk
    art.available.side_effect = failure
    error_type = (
        tv_artwork.ArtworkAuthorizationError
        if isinstance(failure, UnauthorizedError)
        else tv_artwork.ArtworkError
    )
    with pytest.raises(error_type) as caught:
        tv_artwork.TvArtworkClient(HOST).list_artwork()
    assert RAW_SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    art.close.assert_called_once_with()


def make_art():
    art = tv_artwork._ArtworkArt(HOST, port=8002, name="LocalThings")
    art._deadline = Mock()
    art._deadline.remaining.return_value = 4.0
    art.connection = Mock()
    art.connection.sock.getpeername.return_value = (HOST, 8002)
    return art


@pytest.mark.parametrize("secure", [False, True])
def test_d2d_connect_and_tls_share_deadline_and_close(monkeypatch, secure) -> None:
    art = make_art()
    raw = Mock(spec=socket.socket)
    tls = Mock(spec=socket.socket)
    context = Mock()
    context.wrap_socket.return_value = tls
    sequence = Mock()
    sequence.attach_mock(raw, "raw")
    sequence.attach_mock(tls, "tls")
    sequence.attach_mock(Mock(return_value=context), "context")
    sequence.attach_mock(context.wrap_socket, "wrap")
    monkeypatch.setattr(tv_artwork.socket, "socket", Mock(return_value=raw))
    monkeypatch.setattr(tv_artwork.ssl, "SSLContext", sequence.context)
    monkeypatch.setattr(tv_artwork, "monotonic", Mock(side_effect=[10.0, 10.125]), raising=False)

    result = art._open_d2d_socket({"ip": HOST, "port": 9123, "secured": secure})
    raw.connect.assert_called_once_with((HOST, 9123))
    raw.settimeout.assert_called_with(4.0)
    if secure:
        context.wrap_socket.assert_called_once_with(raw, server_hostname=HOST)
        tls.settimeout.assert_called_with(4.0)
        assert context.check_hostname is False
        assert context.verify_mode == ssl.CERT_NONE
        assert sequence.mock_calls == [
            mock_call.raw.settimeout(4.0),
            mock_call.raw.connect((HOST, 9123)),
            mock_call.context(ssl.PROTOCOL_TLS_CLIENT),
            mock_call.raw.settimeout(4.0),
            mock_call.wrap(raw, server_hostname=HOST),
            mock_call.tls.settimeout(4.0),
        ]
        assert art._failure_stage == "transfer_post_handshake"
        assert art._handshake_ms == 125
    else:
        sequence.context.assert_not_called()
        assert art._handshake_ms is None
    result.close()
    (tls if secure else raw).close.assert_called_once_with()


@pytest.mark.parametrize("stage", ["connect", "tls"])
def test_d2d_failure_closes_owned_socket(monkeypatch, stage) -> None:
    art = make_art()
    raw = Mock(spec=socket.socket)
    context = Mock()
    monkeypatch.setattr(tv_artwork.socket, "socket", Mock(return_value=raw))
    monkeypatch.setattr(tv_artwork.ssl, "SSLContext", Mock(return_value=context))
    if stage == "connect":
        raw.connect.side_effect = OSError(RAW_SECRET)
    else:
        context.wrap_socket.side_effect = OSError(RAW_SECRET)
    with pytest.raises(OSError):
        art._open_d2d_socket({"ip": HOST, "port": 9123, "secured": True})
    raw.close.assert_called_once_with()


@pytest.mark.parametrize(
    "changes",
    [
        {"ip": "192.0.2.99"},
        {"ip": "example.test"},
        {"port": 0},
        {"port": 65536},
        {"secured": "false"},
    ],
)
def test_d2d_rejects_untrusted_endpoint_before_socket(monkeypatch, changes) -> None:
    art = make_art()
    factory = Mock()
    monkeypatch.setattr(tv_artwork.socket, "socket", factory)
    with pytest.raises(tv_artwork.ArtworkError):
        art._open_d2d_socket({"ip": HOST, "port": 9123, "secured": True, **changes})
    factory.assert_not_called()


def test_d2d_hostname_uses_the_connected_websocket_peer(monkeypatch) -> None:
    art = make_art()
    art.host = "synthetic-frame.local"
    raw = Mock(spec=socket.socket)
    monkeypatch.setattr(tv_artwork.socket, "socket", Mock(return_value=raw))
    transfer = art._open_d2d_socket({"ip": HOST, "port": 9123, "secured": False})
    raw.connect.assert_called_once_with((HOST, 9123))
    transfer.close()
    raw.close.assert_called_once_with()


def test_d2d_rejects_configured_host_when_it_differs_from_peer(monkeypatch) -> None:
    art = make_art()
    art.connection.sock.getpeername.return_value = ("192.0.2.99", 8002)
    factory = Mock()
    monkeypatch.setattr(tv_artwork.socket, "socket", factory)
    with pytest.raises(tv_artwork.ArtworkError):
        art._open_d2d_socket({"ip": HOST, "port": 9123, "secured": False})
    factory.assert_not_called()


def d2d_bytes(**changes) -> bytes:
    header = json.dumps(
        {
            "fileID": CONTENT_ID,
            "fileType": "jpg",
            "fileLength": len(JPEG),
            "num": 0,
            "total": 1,
            **changes,
        }
    ).encode()
    return len(header).to_bytes(4, "big") + header + JPEG


def fake_receiving_socket(payload: bytes):
    sock = Mock()
    pending = bytearray(payload)

    def recv(size):
        data = bytes(pending[:size])
        del pending[:size]
        return data

    sock.recv.side_effect = recv
    return sock


def test_d2d_read_is_capped_and_each_recv_has_same_deadline() -> None:
    art = make_art()
    raw = fake_receiving_socket(d2d_bytes())
    bounded = tv_artwork._DeadlineSocket(raw, art._deadline)
    assert art._recv_d2d_file(bounded) == (f"{CONTENT_ID}.jpg", bytearray(JPEG), 0, 1)
    assert raw.settimeout.call_count == 3
    assert all(call.args == (4.0,) for call in raw.settimeout.call_args_list)


@pytest.mark.parametrize(
    "changes",
    [
        {"fileLength": str(len(JPEG))},
        {"num": "0"},
        {"total": "1"},
        {"fileLength": str(len(JPEG)), "num": "0", "total": "1"},
        {"fileLength": str(len(JPEG)), "num": 0, "total": "1"},
        {"fileLength": f"000{len(JPEG)}", "num": "000", "total": "001"},
    ],
)
def test_d2d_accepts_decimal_string_header_numbers(changes) -> None:
    art = make_art()
    raw = fake_receiving_socket(d2d_bytes(**changes))
    assert art._recv_d2d_file(raw) == (f"{CONTENT_ID}.jpg", bytearray(JPEG), 0, 1)


@pytest.mark.parametrize("field", ["fileLength", "num", "total"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        1.0,
        None,
        "",
        "-1",
        "+1",
        "1.0",
        " 1",
        "1 ",
        "\uff11",
        "\u0661",
        "1e0",
        "1_0",
    ],
)
def test_d2d_rejects_non_decimal_header_numbers_before_body_read(field, value) -> None:
    art = make_art()
    raw = fake_receiving_socket(d2d_bytes(**{field: value}))
    with pytest.raises(tv_artwork.ArtworkError):
        art._recv_d2d_file(raw)
    assert raw.recv.call_count == 2


@pytest.mark.parametrize("value", [str(8 * 1024 * 1024 + 1), "9" * 5000])
def test_d2d_rejects_oversized_decimal_string_before_body_read(value) -> None:
    art = make_art()
    raw = fake_receiving_socket(d2d_bytes(fileLength=value))
    with pytest.raises(tv_artwork.ArtworkError):
        art._recv_d2d_file(raw)
    assert raw.recv.call_count == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"fileLength": 8 * 1024 * 1024 + 1},
        {"fileLength": -1},
        {"fileLength": "0"},
        {"fileID": "../image"},
        {"fileType": "svg"},
        {"num": 1},
        {"num": "1"},
        {"total": 2},
        {"total": "2"},
    ],
)
def test_d2d_rejects_invalid_header_before_body_read(changes) -> None:
    art = make_art()
    raw = fake_receiving_socket(d2d_bytes(**changes))
    with pytest.raises(tv_artwork.ArtworkError):
        art._recv_d2d_file(raw)
    assert raw.recv.call_count == 2


def test_d2d_header_length_is_capped() -> None:
    art = make_art()
    raw = fake_receiving_socket((65537).to_bytes(4, "big"))
    with pytest.raises(tv_artwork.ArtworkError):
        art._recv_d2d_file(raw)
    raw.recv.assert_called_once_with(4)


def test_sdk_request_correlation_remains_exact() -> None:
    art = make_art()
    art._recv_frame = Mock(
        side_effect=[
            (
                D2D_SERVICE_MESSAGE_EVENT,
                {"data": json.dumps({"request_id": "wrong", "event": "get_thumbnail_list"})},
            ),
            (
                D2D_SERVICE_MESSAGE_EVENT,
                {"data": json.dumps({"request_id": "right", "event": "get_thumbnail_list"})},
            ),
        ]
    )
    assert art._wait_for_d2d(request_uuid="right", wait_for_sub_event=None)["request_id"] == "right"
    assert art._recv_frame.call_count == 2


@pytest.mark.parametrize("size", [8 * 1024 * 1024 + 1, str(8 * 1024 * 1024 + 1)])
def test_d2d_failure_inside_sdk_always_closes_transfer_socket(size) -> None:
    art = make_art()
    art._send_art_request = Mock(return_value={"conn_info": {}})
    raw = fake_receiving_socket(d2d_bytes(fileLength=size))
    art._open_d2d_socket = Mock(return_value=raw)
    with pytest.raises(tv_artwork.ArtworkError):
        art.get_thumbnail_list([CONTENT_ID])
    raw.close.assert_called_once_with()


def test_d2d_eof_closes_transfer_socket_without_retry() -> None:
    art = make_art()
    art._send_art_request = Mock(return_value={"conn_info": {}})
    raw = fake_receiving_socket(b"\x00\x00")
    art._open_d2d_socket = Mock(return_value=raw)
    with pytest.raises(ConnectionFailure):
        art.get_thumbnail_list([CONTENT_ID])
    assert raw.recv.call_count == 2
    art._open_d2d_socket.assert_called_once_with({})
    raw.close.assert_called_once_with()


def test_d2d_deadline_expiry_closes_transfer_socket() -> None:
    art = make_art()
    art._send_art_request = Mock(return_value={"conn_info": {}})
    raw = fake_receiving_socket(d2d_bytes())
    bounded = tv_artwork._DeadlineSocket(raw, art._deadline)
    art._open_d2d_socket = Mock(return_value=bounded)
    art._deadline.remaining.side_effect = [1.0, TimeoutError(RAW_SECRET)]
    with pytest.raises(TimeoutError):
        art.get_thumbnail_list([CONTENT_ID])
    raw.recv.assert_called_once_with(4)
    raw.close.assert_called_once_with()


def test_authorization_callbacks_cannot_change_control_credentials() -> None:
    art = make_art()
    assert art.token is None
    assert art.token_file is None
    art._token_updated(RAW_SECRET)
    art._authorization_failed(tv_artwork.TvPairingRejected)
    art._authorization_check()
    assert art.token is None
    assert art.token_file is None


@pytest.mark.parametrize(
    ("boundary", "stage", "category", "kind"),
    [
        ("header", "transfer_header", "protocol_error", "artwork_validation"),
        ("body", "transfer_body", "timeout", "socket_error"),
        ("image", "thumbnail_validation", "protocol_error", "artwork_validation"),
    ],
)
def test_failure_diagnostic_preserves_stage_without_private_data(
    monkeypatch, caplog, boundary, stage, category, kind
) -> None:
    art = make_art()
    art._deadline = tv_artwork._Deadline()
    payload = d2d_bytes(fileLength=-1) if boundary == "header" else d2d_bytes()
    if boundary == "image":
        payload = payload.replace(JPEG, bytes(len(JPEG)))
    raw = fake_receiving_socket(payload)
    if boundary == "body":
        receive = raw.recv.side_effect
        reads = 0

        def receive_until_body(size):
            nonlocal reads
            reads += 1
            if reads == 3:
                raise TimeoutError(RAW_SECRET)
            return receive(size)

        raw.recv.side_effect = receive_until_body
    art._send_art_request = Mock(return_value={"conn_info": {}})
    art._open_d2d_socket = Mock(return_value=raw)
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(return_value=art))
    monkeypatch.setattr(tv_artwork, "version", lambda _: "3.0.7", raising=False)

    with pytest.raises(tv_artwork.ArtworkError) as caught:
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)

    records = [row for row in caplog.records if row.name.endswith("tv_artwork")]
    assert len(records) == 1
    assert records[0].args == (stage, category, kind, "none", "none", "none", "none", "3.0.7")
    assert records[0].exc_info is None
    for private in (HOST, CONTENT_ID, RAW_SECRET, "synthetic-jpeg"):
        assert private not in caplog.text
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    raw.close.assert_called_once_with()
    art._send_art_request.assert_called_once()


def test_websocket_failure_diagnostic_uses_original_safe_callback(monkeypatch, caplog) -> None:
    art = make_art()
    art._deadline = tv_artwork._Deadline()
    art._diagnostic_failure(TvFailure("art", "upgrade", "timeout"))
    art.get_thumbnail_list = Mock(side_effect=tv_artwork.TvAuthorizationIncomplete(RAW_SECRET))
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(return_value=art))
    monkeypatch.setattr(tv_artwork, "version", lambda _: "3.0.7", raising=False)
    with pytest.raises(tv_artwork.ArtworkAuthorizationError):
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)
    records = [row for row in caplog.records if row.name.endswith("tv_artwork")]
    assert len(records) == 1
    assert records[0].args == (
        "upgrade",
        "timeout",
        "other",
        "none",
        "none",
        "none",
        "none",
        "3.0.7",
    )
    assert RAW_SECRET not in caplog.text


async def test_library_version_diagnostic_is_resolved_in_worker(hass, sdk, monkeypatch, caplog):
    _, art = sdk
    art.available.side_effect = OSError(RAW_SECRET)
    main_thread = threading.get_ident()
    metadata_threads = []

    def installed_version(package):
        assert package == "samsungtvws"
        metadata_threads.append(threading.get_ident())
        return "3.0.7"

    monkeypatch.setattr(tv_artwork, "version", installed_version, raising=False)
    with pytest.raises(tv_artwork.ArtworkError):
        await hass.async_add_executor_job(tv_artwork.TvArtworkClient(HOST).list_artwork)
    assert len(metadata_threads) == 1
    assert metadata_threads[0] != main_thread
    assert RAW_SECRET not in caplog.text


@pytest.mark.parametrize("missing", [False, True])
def test_library_metadata_cannot_leak_or_replace_request_failure(sdk, monkeypatch, caplog, missing):
    _, art = sdk
    art.available.side_effect = OSError(RAW_SECRET)

    def installed_version(_package):
        if missing:
            raise PackageNotFoundError(RAW_SECRET)
        return RAW_SECRET

    monkeypatch.setattr(tv_artwork, "version", installed_version, raising=False)
    with pytest.raises(tv_artwork.ArtworkError) as caught:
        tv_artwork.TvArtworkClient(HOST).list_artwork()
    records = [row for row in caplog.records if row.name.endswith("tv_artwork")]
    assert len(records) == 1
    assert records[0].args[-1] == "unknown"
    assert RAW_SECRET not in caplog.text
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("original", "kind"),
    [
        (AttributeError(RAW_SECRET), "attribute_error"),
        (TypeError(RAW_SECRET), "type_error"),
        (KeyError(RAW_SECRET), "key_error"),
        (tv_artwork.ArtworkError(RAW_SECRET), "artwork_validation"),
        (json.JSONDecodeError(RAW_SECRET, RAW_SECRET, 0), "json_error"),
        (ValueError(RAW_SECRET), "other"),
        (type(RAW_SECRET, (AttributeError,), {})(RAW_SECRET), "attribute_error"),
        (type(RAW_SECRET, (Exception,), {})(RAW_SECRET), "other"),
    ],
)
def test_failure_diagnostic_distinguishes_only_whitelisted_exception_kinds(
    sdk, monkeypatch, caplog, original, kind
) -> None:
    _, art = sdk
    art.available.side_effect = original
    monkeypatch.setattr(tv_artwork, "version", lambda _: "3.0.7")

    with pytest.raises(tv_artwork.ArtworkError) as caught:
        tv_artwork.TvArtworkClient(HOST).list_artwork()

    records = [row for row in caplog.records if row.name.endswith("tv_artwork")]
    assert len(records) == 1
    assert f"kind={kind}" in records[0].getMessage()
    assert RAW_SECRET not in caplog.text
    assert records[0].exc_info is None
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("error_type", "kind"),
    [
        (ssl.SSLEOFError, "ssl_eof"),
        (ssl.SSLZeroReturnError, "ssl_zero_return"),
        (ssl.SSLError, "ssl_error"),
        (ssl.SSLCertVerificationError, "ssl_error"),
        (OSError, "socket_error"),
        (TimeoutError, "socket_error"),
        (type(RAW_SECRET, (ssl.SSLEOFError,), {}), "ssl_eof"),
    ],
)
def test_ssl_diagnostic_orders_subclasses_before_socket_errors(sdk, caplog, error_type, kind):
    _, art = sdk
    art.available.side_effect = error_type(1, RAW_SECRET)

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).list_artwork()

    assert f"kind={kind}" in caplog.text
    assert RAW_SECRET not in caplog.text


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("UNEXPECTED_EOF_WHILE_READING", "UNEXPECTED_EOF_WHILE_READING"),
        ("SSLV3_ALERT_HANDSHAKE_FAILURE", "SSLV3_ALERT_HANDSHAKE_FAILURE"),
        ("TLSV1_ALERT_INTERNAL_ERROR", "TLSV1_ALERT_INTERNAL_ERROR"),
        ("WRONG_VERSION_NUMBER", "WRONG_VERSION_NUMBER"),
        ("NO_SHARED_CIPHER", "NO_SHARED_CIPHER"),
        (RAW_SECRET, "other"),
        ("UNEXPECTED_EOF_WHILE_READING\n" + RAW_SECRET, "other"),
        ([RAW_SECRET], "other"),
        (None, "other"),
    ],
)
def test_ssl_reason_is_allowlisted_without_raw_error_data(sdk, caplog, reason, expected):
    _, art = sdk
    error = ssl.SSLError(1, RAW_SECRET)
    error.reason = reason
    error.library = RAW_SECRET
    art.available.side_effect = error

    with pytest.raises(tv_artwork.ArtworkError) as caught:
        tv_artwork.TvArtworkClient(HOST).list_artwork()

    assert f"ssl_reason={expected}" in caplog.text
    assert RAW_SECRET not in caplog.text
    record = next(row for row in caplog.records if row.name.endswith("tv_artwork"))
    assert record.exc_info is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("error_type", "code", "ssl_code", "os_errno"),
    [
        (ssl.SSLError, 1, 1, "none"),
        (ssl.SSLError, 255, 255, "none"),
        (ssl.SSLError, 256, "none", "none"),
        (OSError, 104, "none", 104),
        (OSError, 4095, "none", 4095),
        (OSError, 4096, "none", "none"),
        (ssl.SSLError, -1, "none", "none"),
        (OSError, -1, "none", "none"),
        (ssl.SSLError, True, "none", "none"),
        (OSError, True, "none", "none"),
        (ssl.SSLError, RAW_SECRET, "none", "none"),
        (OSError, RAW_SECRET, "none", "none"),
    ],
)
def test_ssl_and_socket_codes_are_separate_bounded_integers(
    sdk, caplog, error_type, code, ssl_code, os_errno
):
    _, art = sdk
    error = error_type(RAW_SECRET)
    error.errno = code
    art.available.side_effect = error

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).list_artwork()

    assert f"ssl_code={ssl_code} os_errno={os_errno}" in caplog.text
    assert RAW_SECRET not in caplog.text


@pytest.mark.parametrize(
    ("boundary", "expected_stage", "handshake_ms"),
    [
        ("context", "transfer_tls_context", "none"),
        ("handshake", "transfer_tls_handshake", 125),
        ("post_handshake", "transfer_post_handshake", 125),
    ],
)
def test_tls_failure_attributes_only_its_stage_and_handshake_time(
    monkeypatch, caplog, boundary, expected_stage, handshake_ms
):
    art = make_art()
    art._deadline = tv_artwork._Deadline()
    art._send_art_request = Mock(
        return_value={"conn_info": {"ip": HOST, "port": 9123, "secured": True}}
    )
    raw = Mock(spec=socket.socket)
    tls = Mock(spec=socket.socket)
    context = Mock()
    context.wrap_socket.return_value = tls
    context_factory = Mock(return_value=context)
    failure = ssl.SSLError(8, RAW_SECRET)
    failure.reason = "UNEXPECTED_EOF_WHILE_READING"
    if boundary == "context":
        context_factory.side_effect = failure
    elif boundary == "handshake":
        context.wrap_socket.side_effect = failure
    else:
        tls.settimeout.side_effect = failure
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(return_value=art))
    monkeypatch.setattr(tv_artwork.socket, "socket", Mock(return_value=raw))
    monkeypatch.setattr(tv_artwork.ssl, "SSLContext", context_factory)
    clock = Mock(side_effect=[10.0, 10.125])
    monkeypatch.setattr(tv_artwork, "monotonic", clock, raising=False)

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)

    assert f"stage={expected_stage}" in caplog.text
    assert f"handshake_ms={handshake_ms}" in caplog.text
    assert "ssl_code=8 os_errno=none" in caplog.text
    for private in (HOST, CONTENT_ID, RAW_SECRET):
        assert private not in caplog.text
    assert clock.call_count == (0 if boundary == "context" else 2)
    raw.connect.assert_called_once_with((HOST, 9123))
    art._send_art_request.assert_called_once()
    (tls if boundary == "post_handshake" else raw).close.assert_called_once_with()


def test_connection_failure_does_not_inherit_previous_operation_timing(monkeypatch, caplog):
    first, second = make_art(), make_art()
    for art in (first, second):
        art._deadline = tv_artwork._Deadline()
        art._send_art_request = Mock(
            return_value={"conn_info": {"ip": HOST, "port": 9123, "secured": True}}
        )
    raw = Mock(spec=socket.socket)
    raw.connect.side_effect = [None, OSError(5, RAW_SECRET)]
    context = Mock()
    context.wrap_socket.side_effect = ssl.SSLEOFError(8, RAW_SECRET)
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(side_effect=[first, second]))
    monkeypatch.setattr(tv_artwork.socket, "socket", Mock(return_value=raw))
    monkeypatch.setattr(tv_artwork.ssl, "SSLContext", Mock(return_value=context))
    clock = Mock(side_effect=[10.0, 10.125])
    monkeypatch.setattr(tv_artwork, "monotonic", clock, raising=False)

    for _ in range(2):
        with pytest.raises(tv_artwork.ArtworkError):
            tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)

    records = [row.getMessage() for row in caplog.records if row.name.endswith("tv_artwork")]
    assert len(records) == 2
    assert "handshake_ms=125" in records[0]
    assert "stage=transfer_connect" in records[1]
    assert "handshake_ms=none" in records[1]
    assert clock.call_count == 2


@pytest.fixture
def thumbnail_transfer(monkeypatch):
    art = make_art()
    art._deadline = tv_artwork._Deadline()
    art.connection = None
    websocket = Mock()
    websocket.sock.getpeername.return_value = (HOST, 8002)
    art.open = Mock(return_value=websocket)
    art._new_request_uuid = Mock(side_effect=["SYNTHETIC-FIRST", "SYNTHETIC-SECOND"])
    monkeypatch.setattr("samsungtvws.art.art.generate_connection_id", Mock(side_effect=[101, 102]))
    first, second = Mock(spec=socket.socket), Mock(spec=socket.socket)
    secured = fake_receiving_socket(d2d_bytes())
    context = Mock()
    factory = Mock(return_value=art)
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", factory)
    monkeypatch.setattr(tv_artwork.socket, "socket", Mock(side_effect=[first, second]))
    monkeypatch.setattr(tv_artwork.ssl, "SSLContext", Mock(return_value=context))
    clock = [100.0]
    monkeypatch.setattr(tv, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tv_artwork, "monotonic", lambda: clock[0])
    state: dict[str, Any] = {
        "art": art,
        "websocket": websocket,
        "first": first,
        "second": second,
        "secured": secured,
        "context": context,
        "factory": factory,
        "clock": clock,
        "events": [],
        "requests": [],
        "snapshots": [],
        "first_elapsed": 0.014,
        "second_error": None,
    }

    def send(payload):
        request = json.loads(json.loads(payload)["params"]["data"])
        state["events"].append("request")
        state["requests"].append(request)
        state["snapshots"].append(
            (
                art._failure_stage,
                art._failure_category,
                art._handshake_ms,
                art._deadline._expires_at,
            )
        )

    def receive():
        request = state["requests"][-1]
        return json.dumps(
            {
                "event": D2D_SERVICE_MESSAGE_EVENT,
                "data": json.dumps(
                    {
                        "request_id": request["request_id"],
                        "event": "get_thumbnail_list",
                        "conn_info": {
                            "ip": HOST,
                            "port": 9122 + len(state["requests"]),
                            "secured": True,
                        },
                    }
                ),
            }
        )

    def handshake(raw, *, server_hostname):
        assert server_hostname == HOST
        if raw is first:
            clock[0] += state["first_elapsed"]
            art._failure_category = "connection_error"
            error = ssl.SSLEOFError(8, RAW_SECRET)
            error.reason = "UNEXPECTED_EOF_WHILE_READING"
            raise error
        if state["second_error"] is not None:
            raise state["second_error"]
        return secured

    websocket.send.side_effect = send
    websocket.recv.side_effect = receive
    context.wrap_socket.side_effect = handshake
    first.close.side_effect = lambda: state["events"].append("first_closed")
    return state


def test_thumbnail_recovers_one_handshake_eof_with_new_transfer_on_same_websocket(
    thumbnail_transfer, caplog
):
    state = thumbnail_transfer

    assert tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID) == (JPEG, "image/jpeg")

    assert state["events"] == ["request", "first_closed", "request"]
    assert [row["request"] for row in state["requests"]] == [
        "get_thumbnail_list",
        "get_thumbnail_list",
    ]
    assert [row["request_id"] for row in state["requests"]] == [
        "SYNTHETIC-FIRST",
        "SYNTHETIC-SECOND",
    ]
    assert [row["conn_info"]["connection_id"] for row in state["requests"]] == [101, 102]
    assert all(row["content_id_list"] == [{"content_id": CONTENT_ID}] for row in state["requests"])
    assert state["snapshots"] == [("request", None, None, 110.0), ("request", None, None, 110.0)]
    state["first"].connect.assert_called_once_with((HOST, 9123))
    state["second"].connect.assert_called_once_with((HOST, 9124))
    assert state["second"].settimeout.call_args_list[0].args[0] == pytest.approx(9.986)
    state["art"].open.assert_called_once_with()
    state["factory"].assert_called_once_with(HOST, port=8002, name="LocalThings")
    state["secured"].close.assert_called_once_with()
    state["websocket"].shutdown.assert_called_once_with()
    assert not [row for row in caplog.records if row.name.endswith("tv_artwork")]


def test_second_handshake_eof_ends_thumbnail_recovery_without_third_request(
    thumbnail_transfer, caplog
):
    state = thumbnail_transfer
    failure = ssl.SSLEOFError(8, RAW_SECRET)
    failure.reason = "UNEXPECTED_EOF_WHILE_READING"
    state["second_error"] = failure

    with pytest.raises(tv_artwork.ArtworkError) as caught:
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)

    assert len(state["requests"]) == 2
    assert state["context"].wrap_socket.call_count == 2
    state["first"].close.assert_called_once_with()
    state["second"].close.assert_called_once_with()
    state["art"].open.assert_called_once_with()
    assert "stage=transfer_tls_handshake" in caplog.text
    assert RAW_SECRET not in caplog.text
    assert caught.value.__context__ is None


def test_thumbnail_recovery_expired_budget_stops_before_second_request(thumbnail_transfer, caplog):
    state = thumbnail_transfer
    state["first_elapsed"] = 10.001

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)

    assert len(state["requests"]) == 1
    state["first"].close.assert_called_once_with()
    state["second"].connect.assert_not_called()
    assert "stage=request category=timeout" in caplog.text
    assert "handshake_ms=none" in caplog.text


def test_second_request_failure_does_not_keep_previous_handshake_diagnostics(
    thumbnail_transfer, caplog
):
    state = thumbnail_transfer
    send = state["websocket"].send.side_effect

    def fail_second_send(payload):
        send(payload)
        if len(state["requests"]) == 2:
            raise TimeoutError(RAW_SECRET)

    state["websocket"].send.side_effect = fail_second_send

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)

    assert len(state["requests"]) == 2
    assert "stage=request category=timeout" in caplog.text
    assert "handshake_ms=none" in caplog.text
    assert "ssl_reason=none ssl_code=none" in caplog.text


@pytest.mark.parametrize(
    ("stage", "error_type", "reason"),
    [
        ("request", ssl.SSLEOFError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transport", ssl.SSLEOFError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_tls_context", ssl.SSLEOFError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_post_handshake", ssl.SSLEOFError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_header", ssl.SSLEOFError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_body", ssl.SSLEOFError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_tls_handshake", ssl.SSLEOFError, RAW_SECRET),
        ("transfer_tls_handshake", ssl.SSLEOFError, None),
        ("transfer_tls_handshake", ssl.SSLError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_tls_handshake", ssl.SSLZeroReturnError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_tls_handshake", OSError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_tls_handshake", TimeoutError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_tls_handshake", UnauthorizedError, "UNEXPECTED_EOF_WHILE_READING"),
        ("transfer_tls_handshake", tv_artwork.TvPairingRejected, "UNEXPECTED_EOF_WHILE_READING"),
        (
            "transfer_tls_handshake",
            tv_artwork.TvAuthorizationIncomplete,
            "UNEXPECTED_EOF_WHILE_READING",
        ),
    ],
)
def test_thumbnail_recovery_rejects_other_failure_stages_types_and_reasons(
    sdk, stage, error_type, reason
):
    _, art = sdk
    error = error_type(RAW_SECRET)
    error.reason = reason
    art._failure_stage = stage
    art.get_thumbnail_list.side_effect = error

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).thumbnail(CONTENT_ID)

    art.get_thumbnail_list.assert_called_once_with([CONTENT_ID])
    art.close.assert_called_once_with()


@pytest.mark.parametrize("operation", ["catalog", "selection"])
def test_thumbnail_recovery_never_replays_catalog_or_picture_setter(sdk, operation):
    _, art = sdk
    error = ssl.SSLEOFError(8, RAW_SECRET)
    error.reason = "UNEXPECTED_EOF_WHILE_READING"
    art._failure_stage = "transfer_tls_handshake"
    target = art.available if operation == "catalog" else art.select_image
    target.side_effect = error
    client = tv_artwork.TvArtworkClient(HOST)

    with pytest.raises(tv_artwork.ArtworkError):
        if operation == "catalog":
            client.list_artwork()
        else:
            client.select(CONTENT_ID)

    assert target.call_count == 1
    art.get_current.assert_not_called()
    art.close.assert_called_once_with()


def test_delete_one_id_uses_strict_correlated_sdk_request(sdk):
    _, art = sdk

    def reply(payload, *, request_uuid):
        assert payload == {
            "request": "delete_image_list",
            "content_id_list": [{"content_id": CONTENT_ID}],
        }
        assert request_uuid
        return {
            "event": "delete_image_list",
            "id": request_uuid,
            "request_id": request_uuid,
            "content_id_list": json.dumps([{"content_id": CONTENT_ID}]),
        }

    art._send_art_request.side_effect = reply
    tv_artwork.TvArtworkClient(HOST).delete(CONTENT_ID)
    art._send_art_request.assert_called_once()
    art.delete.assert_not_called()
    art.delete_list.assert_not_called()
    art.close.assert_called_once()


@pytest.mark.parametrize(
    "failure",
    ["event", "missing_id", "id", "request_id", "list", "empty", "extra", "malformed", "none"],
)
def test_delete_reply_failure_never_resends(sdk, caplog, failure):
    _, art = sdk

    def reply(_payload, *, request_uuid):
        response = {
            "event": "delete_image_list",
            "id": request_uuid,
            "request_id": request_uuid,
            "content_id_list": [{"content_id": CONTENT_ID}],
        }
        if failure == "event":
            response["event"] = "image_deleted"
        elif failure == "missing_id":
            response.pop("id")
            response.pop("request_id")
        elif failure in {"id", "request_id"}:
            response[failure] = RAW_SECRET
        elif failure == "list":
            response["content_id_list"] = [{"content_id": "wrong"}]
        elif failure == "empty":
            response["content_id_list"] = []
        elif failure == "extra":
            response["content_id_list"].append({"content_id": "other"})
        elif failure == "malformed":
            response["content_id_list"] = RAW_SECRET
        else:
            return None
        return response

    art._send_art_request.side_effect = reply
    with pytest.raises(tv_artwork.ArtworkError, match="Artwork request failed") as caught:
        tv_artwork.TvArtworkClient(HOST).delete(CONTENT_ID)
    art._send_art_request.assert_called_once()
    assert RAW_SECRET not in str(caught.value)
    assert RAW_SECRET not in caplog.text


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"event": "get_slideshow_status", "value": "off"}, False),
        (
            {
                "event": "get_slideshow_status",
                "value": "3",
                "category_id": "MY",
                "type": "slideshow",
            },
            True,
        ),
    ],
)
def test_slideshow_guard_reads_existing_sdk_getter(sdk, payload, expected):
    _, art = sdk
    art.get_slideshow_status.return_value = payload
    assert tv_artwork.TvArtworkClient(HOST).slideshow_active() is expected
    art.get_slideshow_status.assert_called_once()
    art.set_slideshow_status.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"value": "off"},
        {"event": "get_slideshow_status", "value": 0},
        {"event": "get_slideshow_status", "value": "0"},
        {"event": "get_slideshow_status", "value": "3"},
    ],
)
def test_unknown_slideshow_fails_closed(sdk, payload):
    _, art = sdk
    art.get_slideshow_status.return_value = payload
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).slideshow_active()


def test_native_slideshow_stop_sends_once_and_reads_off(sdk):
    _, art = sdk
    art.set_slideshow_status.return_value = {"event": "slideshow_changed"}
    art.get_slideshow_status.return_value = {"event": "get_slideshow_status", "value": "off"}
    tv_artwork.TvArtworkClient(HOST).stop_slideshow()
    art.set_slideshow_status.assert_called_once_with(duration=0)
    art.get_slideshow_status.assert_called_once_with()
    art.select_image.assert_not_called()
    art.close.assert_called_once_with()


@pytest.mark.parametrize("stage", ["readback", "malformed_readback", "authorization"])
def test_native_stop_never_retries_failed_or_unconfirmed_write(sdk, stage):
    _, art = sdk
    art.set_slideshow_status.return_value = {"event": "set_slideshow_status", "value": "off"}
    art.get_slideshow_status.return_value = {"event": "get_slideshow_status", "value": "off"}
    if stage == "readback":
        art.get_slideshow_status.return_value = {"event": "get_slideshow_status", "value": "3"}
    elif stage == "malformed_readback":
        art.get_slideshow_status.return_value = {"value": "off"}
    else:
        art.set_slideshow_status.side_effect = UnauthorizedError(RAW_SECRET)
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).stop_slideshow()
    art.set_slideshow_status.assert_called_once_with(duration=0)
    art.close.assert_called_once_with()


def test_delete_invalid_id_never_opens_connection(sdk):
    factory, _ = sdk
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).delete("../source.jpg")
    factory.assert_not_called()


def test_delete_authorization_reply_is_sanitized_and_not_retried(sdk, caplog):
    _, art = sdk
    art._send_art_request.side_effect = UnauthorizedError(RAW_SECRET)
    with pytest.raises(tv_artwork.ArtworkAuthorizationError) as caught:
        tv_artwork.TvArtworkClient(HOST).delete(CONTENT_ID)
    art._send_art_request.assert_called_once()
    assert RAW_SECRET not in str(caught.value)
    assert RAW_SECRET not in caplog.text
