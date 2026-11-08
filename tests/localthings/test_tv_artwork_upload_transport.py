"""Single-send artwork uploads with bounded synthetic SDK connections."""

from __future__ import annotations

import io
import json
import socket
import traceback
from unittest.mock import Mock, call

import pytest
from PIL import Image
from samsungtvws.event import D2D_SERVICE_MESSAGE_EVENT
from samsungtvws.exceptions import UnauthorizedError

from custom_components.localthings import tv, tv_artwork

HOST = "192.0.2.10"
CONTENT_ID = "MY-SYNTHETIC-1"
JPEG = b"\xff\xd8\xffsynthetic-jpeg\xff\xd9"
RAW_SECRET = "synthetic-private-sdk-error"


@pytest.fixture
def sdk(monkeypatch):
    factory = Mock()
    art = factory.return_value
    art._deadline = tv._Deadline()
    art._failure_stage = "request"
    art._failure_category = None
    art._handshake_ms = None
    art.available.return_value = []
    art.upload.return_value = CONTENT_ID
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", factory)
    return factory, art


def test_upload_sends_once_without_matte_or_control_commands(sdk):
    factory, art = sdk
    assert tv_artwork.TvArtworkClient(HOST).upload(JPEG) == CONTENT_ID
    factory.assert_called_once_with(HOST, port=8002, name="LocalThings")
    art.upload.assert_called_once_with(JPEG, matte="none", portrait_matte="none", file_type="jpg")
    art.close.assert_called_once_with()
    art.select_image.assert_not_called()
    art.set_artmode.assert_not_called()
    art.delete.assert_not_called()


@pytest.mark.parametrize("data", [b"", b"wrong", b"\x89PNG\r\n\x1a\n", JPEG[:-2]])
def test_upload_rejects_invalid_jpeg_before_connecting(sdk, data):
    factory, _ = sdk
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).upload(data)
    factory.assert_not_called()


@pytest.mark.parametrize("extra", [0, 1])
def test_upload_uses_twenty_mib_boundary_separate_from_thumbnail_limit(sdk, extra):
    factory, art = sdk
    data = JPEG[:-2] + bytes(20 * 1024 * 1024 - len(JPEG) + extra) + JPEG[-2:]
    if extra:
        with pytest.raises(tv_artwork.ArtworkError):
            tv_artwork.TvArtworkClient(HOST).upload(data)
        factory.assert_not_called()
    else:
        assert tv_artwork.TvArtworkClient(HOST).upload(data) == CONTENT_ID
        art.upload.assert_called_once()


@pytest.mark.parametrize("content_id", [None, "", "../picture", "picture?secret", 123])
def test_upload_rejects_invalid_accepted_id_without_resending(sdk, content_id):
    _, art = sdk
    art.upload.return_value = content_id
    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).upload(JPEG)
    art.upload.assert_called_once()
    art.close.assert_called_once_with()


@pytest.mark.parametrize("error", [TimeoutError(RAW_SECRET), UnauthorizedError(RAW_SECRET)])
def test_uncertain_upload_never_retries_and_sanitizes_errors(sdk, error):
    _, art = sdk
    art.upload.side_effect = error
    expected = (
        tv_artwork.ArtworkAuthorizationError
        if isinstance(error, UnauthorizedError)
        else tv_artwork.ArtworkError
    )
    with pytest.raises(expected) as caught:
        tv_artwork.TvArtworkClient(HOST).upload(JPEG)
    assert RAW_SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    art.upload.assert_called_once()
    art.close.assert_called_once_with()


def test_sendall_uses_partial_writes_and_one_shrinking_deadline():
    raw = Mock(spec=socket.socket)
    raw.send.side_effect = [2, 2, 3]
    deadline = Mock()
    deadline.remaining.side_effect = [6, 5, 4, 3, 2, 1]
    bounded = tv._DeadlineSocket(raw, deadline)
    assert bounded.sendall(b"head") is None
    assert bounded.sendall(b"jpg") is None
    assert [bytes(value.args[0]) for value in raw.send.call_args_list] == [b"head", b"ad", b"jpg"]
    assert raw.settimeout.call_args_list == [call(6), call(4), call(2)]
    raw.sendall.assert_not_called()


def test_sdk_upload_expiry_closes_transfer_socket_without_resending():
    art = tv_artwork._ArtworkArt(HOST, port=8002, name="LocalThings")
    art.get_api_version = Mock(return_value="1.0")
    art._send_art_request = Mock(return_value={"conn_info": {"key": "synthetic"}})
    art._wait_for_d2d = Mock(return_value={"content_id": CONTENT_ID})
    deadline = Mock()
    deadline.remaining.side_effect = [2, 1, TimeoutError(RAW_SECRET)]
    raw = Mock(spec=socket.socket)
    raw.send.side_effect = lambda data: len(data)
    art._open_d2d_socket = Mock(return_value=tv._DeadlineSocket(raw, deadline))
    with pytest.raises(TimeoutError):
        art.upload(JPEG, matte="none", portrait_matte="none", file_type="jpg")
    raw.send.assert_called_once()
    raw.sendall.assert_not_called()
    raw.close.assert_called_once_with()
    art._send_art_request.assert_called_once()
    art._wait_for_d2d.assert_not_called()


def test_sdk_legacy_upload_stops_before_binary_send_or_fallback(monkeypatch):
    art = tv_artwork._ArtworkArt(HOST, port=8002, name="LocalThings")
    art.get_api_version = Mock(return_value="0.97")
    art.available = Mock(return_value=[])
    art.connection = Mock()
    art._send_art_request = Mock(return_value={"conn_info": {"key": "synthetic"}})
    art._wait_for_d2d = Mock(return_value={"content_id": CONTENT_ID})
    raw = Mock(spec=socket.socket)
    art._open_d2d_socket = Mock(return_value=raw)
    art.close = Mock()
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(return_value=art))

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).upload(JPEG)

    art.connection.send_binary.assert_not_called()
    art._send_art_request.assert_not_called()
    art._open_d2d_socket.assert_not_called()
    art._wait_for_d2d.assert_not_called()
    art.close.assert_called_once_with()


def test_real_sdk_d2d_upload_writes_one_jpeg_and_closes_both_connections(monkeypatch):
    art = tv_artwork._ArtworkArt(HOST, port=8002, name="LocalThings")
    art.get_api_version = Mock(return_value="1.0")
    art.available = Mock(return_value=[])
    art._send_art_request = Mock(return_value={"conn_info": {"key": "synthetic"}})
    art._wait_for_d2d = Mock(return_value={"content_id": CONTENT_ID})
    raw = Mock(spec=socket.socket)
    sent = bytearray()

    def send(data):
        size = min(len(data), 7)
        sent.extend(data[:size])
        return size

    raw.send.side_effect = send
    art._open_d2d_socket = Mock(return_value=tv._DeadlineSocket(raw, art._deadline))
    art.close = Mock()
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(return_value=art))

    assert tv_artwork.TvArtworkClient(HOST).upload(JPEG) == CONTENT_ID

    header_length = int.from_bytes(sent[:4], "big")
    header = json.loads(sent[4 : 4 + header_length])
    assert header["fileLength"] == len(JPEG)
    assert header["fileType"] == "jpg"
    assert sent[4 + header_length :] == JPEG
    assert art._send_art_request.call_args.args[0]["matte_id"] == "none"
    assert art._send_art_request.call_args.args[0]["portrait_matte_id"] == "none"
    art._send_art_request.assert_called_once()
    art._wait_for_d2d.assert_called_once()
    raw.sendall.assert_not_called()
    raw.close.assert_called_once_with()
    art.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("completion_ids", "correlated"),
    [
        ({"request_id": "active-upload"}, True),
        ({"id": "active-upload"}, True),
        ({"id": "active-upload", "request_id": "active-upload"}, True),
        ({"request_id": "foreign-upload"}, False),
        ({"id": "foreign-upload"}, False),
        ({}, False),
        ({"request_id": None}, False),
        ({"request_id": "active-upload", "id": "foreign-upload"}, False),
        ({"id": "active-upload", "request_id": "foreign-upload"}, False),
    ],
)
def test_real_sdk_upload_requires_correlated_image_added(monkeypatch, completion_ids, correlated):
    art = tv_artwork._ArtworkArt(HOST, port=8002, name="LocalThings")
    art.get_api_version = Mock(return_value="1.0")
    art.available = Mock(return_value=[])
    art._new_request_uuid = Mock(return_value="active-upload")
    art.send_command = Mock()
    art._recv_frame = Mock(
        side_effect=[
            (
                D2D_SERVICE_MESSAGE_EVENT,
                {
                    "data": json.dumps(
                        {
                            "event": "ready_to_use",
                            "request_id": "active-upload",
                            "conn_info": {"key": "synthetic"},
                        }
                    )
                },
            ),
            (
                D2D_SERVICE_MESSAGE_EVENT,
                {
                    "data": json.dumps(
                        {"event": "image_added", "content_id": CONTENT_ID, **completion_ids}
                    )
                },
            ),
        ]
    )
    raw = Mock(spec=socket.socket)
    raw.send.side_effect = lambda data: len(data)
    art._open_d2d_socket = Mock(return_value=tv._DeadlineSocket(raw, art._deadline))
    art.close = Mock()
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(return_value=art))

    if correlated:
        assert tv_artwork.TvArtworkClient(HOST).upload(JPEG) == CONTENT_ID
    else:
        with pytest.raises(tv_artwork.ArtworkError):
            tv_artwork.TvArtworkClient(HOST).upload(JPEG)

    art.send_command.assert_called_once()
    assert art._recv_frame.call_count == 2
    art._open_d2d_socket.assert_called_once()
    raw.close.assert_called_once_with()
    assert art.close.call_count == (1 if correlated else 2)
    assert art.available.call_count == (1 if correlated else 2)


def _jpeg(size, color):
    with Image.new("RGB", size, color) as image, io.BytesIO() as output:
        image.save(output, format="JPEG", quality=95, subsampling=0)
        return output.getvalue()


@pytest.fixture
def uncorrelated_upload(monkeypatch):
    art = tv_artwork._ArtworkArt(HOST, port=8002, name="LocalThings")
    art.get_api_version = Mock(return_value="1.0")
    art._new_request_uuid = Mock(return_value="active-upload")
    art.send_command = Mock()
    art.available = Mock(
        side_effect=[
            [{"content_id": "MY_EXISTING"}],
            [{"content_id": "MY_EXISTING"}, {"content_id": "MY_NEW"}],
        ]
    )
    raw = Mock(spec=socket.socket)
    raw.send.side_effect = lambda data: len(data)
    art._open_d2d_socket = Mock(return_value=tv._DeadlineSocket(raw, art._deadline))
    art.get_thumbnail_list = Mock(
        return_value={"MY_NEW.jpg": bytearray(_jpeg((384, 216), (42, 95, 163)))}
    )
    art.close = Mock()
    monkeypatch.setattr(tv_artwork, "_ArtworkArt", Mock(return_value=art))

    def completion(identifiers):
        art._recv_frame = Mock(
            side_effect=[
                (
                    D2D_SERVICE_MESSAGE_EVENT,
                    {
                        "data": json.dumps(
                            {
                                "event": "ready_to_use",
                                "request_id": "active-upload",
                                "conn_info": {"key": "synthetic"},
                            }
                        )
                    },
                ),
                (
                    D2D_SERVICE_MESSAGE_EVENT,
                    {
                        "data": json.dumps(
                            {
                                "event": "image_added",
                                "content_id": "FOREIGN_EVENT_ID",
                                **identifiers,
                            }
                        )
                    },
                ),
            ]
        )

    completion({})
    return art, raw, _jpeg((3840, 2160), (42, 95, 163)), completion


@pytest.mark.parametrize(
    "identifiers",
    [
        {},
        {"request_id": "foreign-upload"},
        {"request_id": "active-upload", "id": "foreign-upload"},
    ],
)
def test_uncorrelated_completion_recovers_only_from_new_matching_stored_picture(
    uncorrelated_upload, identifiers, caplog
):
    art, raw, data, completion = uncorrelated_upload
    completion(identifiers)

    assert tv_artwork.TvArtworkClient(HOST).upload(data) == "MY_NEW"

    assert art.available.call_count == 2
    art.get_thumbnail_list.assert_called_once_with(["MY_NEW"])
    art.send_command.assert_called_once()
    assert raw.send.call_count == 3
    assert art._recv_frame.call_count == 2
    assert art.close.call_count == 3
    assert not any(record.levelno >= 30 for record in caplog.records)


@pytest.mark.parametrize(
    "failure",
    ["missing", "multiple", "not_custom", "wrong_image", "damaged", "wrong_size", "oversized"],
)
def test_uncorrelated_completion_stays_uncertain_when_storage_does_not_prove_acceptance(
    uncorrelated_upload, failure
):
    art, raw, data, _ = uncorrelated_upload
    if failure == "missing":
        art.available.side_effect = [
            [{"content_id": "MY_EXISTING"}],
            [{"content_id": "MY_EXISTING"}],
        ]
    elif failure == "multiple":
        art.available.side_effect = [[], [{"content_id": "MY_NEW"}, {"content_id": "MY_OTHER"}]]
    elif failure == "not_custom":
        art.available.side_effect = [[], [{"content_id": "STORE_IMAGE"}]]
    elif failure == "wrong_image":
        art.get_thumbnail_list.return_value = {
            "MY_NEW.jpg": bytearray(_jpeg((384, 216), (255, 0, 0)))
        }
    elif failure == "damaged":
        art.get_thumbnail_list.return_value = {"MY_NEW.jpg": bytearray(JPEG)}
    elif failure == "wrong_size":
        art.get_thumbnail_list.return_value = {
            "MY_NEW.jpg": bytearray(_jpeg((1920, 1080), (42, 95, 163)))
        }
    else:
        image = bytearray(_jpeg((384, 216), (42, 95, 163)))
        position = image.index(b"\xff\xc0")
        image[position + 5 : position + 9] = b"\xff\xff\xff\xff"
        art.get_thumbnail_list.return_value = {"MY_NEW.jpg": image}

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).upload(data)

    art.send_command.assert_called_once()
    assert raw.send.call_count == 3
    assert art.available.call_count == 2


@pytest.mark.parametrize(
    "error", [TimeoutError(RAW_SECRET), OSError(RAW_SECRET), UnauthorizedError(RAW_SECRET)]
)
def test_noncompletion_upload_failure_does_not_read_catalog_again_or_thumbnail(sdk, error):
    _, art = sdk
    art.upload.side_effect = error

    with pytest.raises(tv_artwork.ArtworkError) as caught:
        tv_artwork.TvArtworkClient(HOST).upload(JPEG)

    assert RAW_SECRET not in "".join(traceback.format_exception(caught.value))
    art.upload.assert_called_once()
    art.available.assert_called_once()
    art.get_thumbnail_list.assert_not_called()


@pytest.mark.parametrize("catalog", [None, {}, [None], [{"content_id": "../invalid"}]])
def test_upload_requires_valid_fresh_catalog_before_sending(sdk, catalog):
    _, art = sdk
    art.available.return_value = catalog

    with pytest.raises(tv_artwork.ArtworkError):
        tv_artwork.TvArtworkClient(HOST).upload(JPEG)

    art.upload.assert_not_called()
    art.close.assert_called_once()


@pytest.mark.parametrize(
    "failure",
    [
        "catalog_invalid",
        "catalog_timeout",
        "catalog_authorization",
        "thumbnail_timeout",
        "thumbnail_authorization",
    ],
)
def test_failed_recovery_read_stays_uncertain_and_does_not_resend(uncorrelated_upload, failure):
    art, raw, data, _ = uncorrelated_upload
    if failure == "catalog_invalid":
        art.available.side_effect = [[], [{"content_id": "../invalid"}]]
    elif failure == "catalog_timeout":
        art.available.side_effect = [[], TimeoutError(RAW_SECRET)]
    elif failure == "catalog_authorization":
        art.available.side_effect = [[], UnauthorizedError(RAW_SECRET)]
    else:
        error_type = TimeoutError if failure == "thumbnail_timeout" else UnauthorizedError
        art.get_thumbnail_list.side_effect = error_type(RAW_SECRET)

    with pytest.raises(tv_artwork.ArtworkError) as caught:
        tv_artwork.TvArtworkClient(HOST).upload(data)

    assert RAW_SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ or caught.value.__context__ is None
    art.send_command.assert_called_once()
    assert raw.send.call_count == 3
    assert art.available.call_count == 2
