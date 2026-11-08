"""Independent, bounded artwork requests without remote-control credentials."""

from __future__ import annotations

import io
import json
import logging
import re
import socket
import ssl
import warnings
import zlib
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from samsungtvws.exceptions import UnauthorizedError

from .tv import (
    TvAuthorizationIncomplete,
    TvFailure,
    TvPairingRejected,
    _Deadline,
    _DeadlineArt,
    _DeadlineSocket,
    _failure,
    _suppress_sdk_logs,
)

_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024
_CONTENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,255}")
_SSL_REASONS = frozenset(
    {
        "UNEXPECTED_EOF_WHILE_READING",
        "SSLV3_ALERT_HANDSHAKE_FAILURE",
        "TLSV1_ALERT_INTERNAL_ERROR",
        "TLSV1_ALERT_PROTOCOL_VERSION",
        "WRONG_VERSION_NUMBER",
        "NO_SHARED_CIPHER",
        "UNSUPPORTED_PROTOCOL",
        "CERTIFICATE_VERIFY_FAILED",
    }
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Artwork:
    """Catalog metadata; a missing TV title falls back to its content ID."""

    content_id: str
    title: str
    revision: str


class ArtworkError(Exception):
    """A sanitized artwork request failure."""


class ArtworkAuthorizationError(ArtworkError):
    """Artwork authorization failed without changing control permissions."""


class _UploadCompletionUncorrelated(ArtworkError):
    """Image bytes were sent, but the completion event does not identify that send."""


def _thumbnail_matches_upload(data: bytes, thumbnail: bytes) -> bool:
    from PIL import Image, ImageChops, ImageStat

    from .tv_artwork_image import _load

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(thumbnail), formats=("JPEG", "PNG")) as header:
                if header.size != (384, 216):
                    return False
            with _load(data) as prepared, _load(thumbnail) as actual:
                if prepared.size != (3840, 2160):
                    return False
                expected = prepared.convert("RGB").resize((384, 216), Image.Resampling.LANCZOS)
                difference = ImageStat.Stat(ImageChops.difference(actual.convert("RGB"), expected))
                return all(value < 15 for value in difference.mean)
    except (
        OSError,
        ValueError,
        SyntaxError,
        UserWarning,
        Image.DecompressionBombError,
        zlib.error,
    ):
        return False


def _validate_content_id(value: Any) -> str:
    if not isinstance(value, str) or _CONTENT_ID.fullmatch(value) is None:
        raise ArtworkError("Artwork content ID is invalid")
    return value


def _header_number(value: Any) -> int:
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        digits = value.lstrip("0") or "0"
        if len(digits) <= len(str(_MAX_IMAGE_BYTES)):
            value = int(digits)
    if type(value) is not int:
        raise ArtworkError("Artwork transfer header is invalid")
    return value


def _discard(*_: Any) -> None:
    pass


def _exception_kind(error: Exception) -> str:
    for error_type, label in (
        (AttributeError, "attribute_error"),
        (TypeError, "type_error"),
        (KeyError, "key_error"),
        (ArtworkError, "artwork_validation"),
        (json.JSONDecodeError, "json_error"),
        (ssl.SSLEOFError, "ssl_eof"),
        (ssl.SSLZeroReturnError, "ssl_zero_return"),
        (ssl.SSLError, "ssl_error"),
        (OSError, "socket_error"),
    ):
        if isinstance(error, error_type):
            return label
    return "other"


def _bounded_code(value: Any, maximum: int) -> int | str:
    return value if type(value) is int and 0 <= value <= maximum else "none"


def _ssl_detail(error: Exception) -> tuple[str, int | str, int | str]:
    if isinstance(error, ssl.SSLError):
        reason = getattr(error, "reason", None)
        reason = reason if type(reason) is str and reason in _SSL_REASONS else "other"
        return reason, _bounded_code(error.errno, 255), "none"
    if isinstance(error, OSError):
        return "none", "none", _bounded_code(error.errno, 4095)
    return "none", "none", "none"


def _installed_library_version() -> str:
    try:
        installed = version("samsungtvws")
    except (PackageNotFoundError, OSError, ValueError):
        return "unknown"
    return installed if re.fullmatch(r"[0-9][A-Za-z0-9.+!-]{0,63}", installed) else "unknown"


class _ArtworkArt(_DeadlineArt):
    """Keep the SDK's request correlation and bound its separate D2D transfer."""

    def __init__(self, host: str, *, port: int, name: str) -> None:
        super().__init__(host, port=port, name=name, timeout=10, key_press_delay=0)
        self._deadline = _Deadline()
        self._authorization_check = _discard
        self._authorization_failed = _discard
        self._token_updated = _discard
        self._failure_stage = "request"
        self._failure_category: str | None = None
        self._handshake_ms: int | None = None
        self._upload_request_uuid: str | None = None
        self._diagnostic_failure = self._capture_failure

    def _capture_failure(self, detail: TvFailure) -> None:
        self._failure_stage = detail.stage
        self._failure_category = detail.error

    def _upload_ws_binary_send_image(
        self, *, upload_id: str, data: bytes, matte: str, file_type: str
    ) -> None:
        # The SDK can repeat a legacy send after an uncertain ResponseError.
        raise ArtworkError("Legacy artwork upload is unsupported")

    def _wait_for_d2d(self, *, request_uuid: str | None, wait_for_sub_event: str | None) -> Any:
        result = super()._wait_for_d2d(
            request_uuid=request_uuid, wait_for_sub_event=wait_for_sub_event
        )
        if wait_for_sub_event == "ready_to_use":
            self._upload_request_uuid = request_uuid
        elif wait_for_sub_event == "image_added":
            # The SDK does not correlate its final upload completion.
            identifiers = [result[key] for key in ("request_id", "id") if key in result]
            if (
                not self._upload_request_uuid
                or not identifiers
                or any(value != self._upload_request_uuid for value in identifiers)
            ):
                raise _UploadCompletionUncorrelated("Artwork upload completion is uncorrelated")
        return result

    def _open_d2d_socket(self, conn_info: dict[str, Any]) -> socket.socket:
        self._failure_stage = "transfer_endpoint"
        # Use the established peer so configured hostnames remain valid without
        # allowing the TV to redirect a transfer to another host.
        assert self.connection is not None and self.connection.sock is not None
        peer_host = self.connection.sock.getpeername()[0]
        host = conn_info.get("ip")
        port = conn_info.get("port")
        secured = conn_info.get("secured", False)
        if isinstance(port, str) and port.isascii() and port.isdecimal():
            port = int(port)
        if (
            host != peer_host
            or type(port) is not int
            or not 1 <= port <= 65535
            or type(secured) is not bool
        ):
            raise ArtworkError("Artwork transfer endpoint is invalid")
        self._failure_stage = "transfer_connect"
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            raw.settimeout(self._deadline.remaining())
            raw.connect((host, port))
            if secured:
                self._failure_stage = "transfer_tls_context"
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                self._failure_stage = "transfer_tls_handshake"
                raw.settimeout(self._deadline.remaining())
                started = monotonic()
                try:
                    raw = context.wrap_socket(raw, server_hostname=self.host)
                finally:
                    self._handshake_ms = max(0, int((monotonic() - started) * 1000))
                self._failure_stage = "transfer_post_handshake"
            raw.settimeout(self._deadline.remaining())
            return cast(socket.socket, _DeadlineSocket(raw, self._deadline))
        except BaseException:
            raw.close()
            raise

    def _recv_d2d_file(self, sock: socket.socket) -> tuple[str, bytearray, int, int]:
        self._failure_stage = "transfer_header"
        header_length = int.from_bytes(self._recv_exact(sock, 4), "big")
        if not 0 < header_length <= _MAX_HEADER_BYTES:
            raise ArtworkError("Artwork transfer header is invalid")
        header = json.loads(self._recv_exact(sock, header_length))
        if not isinstance(header, dict):
            raise ArtworkError("Artwork transfer header is invalid")
        size = _header_number(header.get("fileLength"))
        current = _header_number(header.get("num"))
        total = _header_number(header.get("total"))
        file_id = _validate_content_id(header.get("fileID"))
        file_type = header.get("fileType")
        if (
            not 0 < size <= _MAX_IMAGE_BYTES
            or current != 0
            or total != 1
            or not isinstance(file_type, str)
            or file_type.lower() not in {"jpg", "jpeg", "png"}
        ):
            raise ArtworkError("Artwork transfer header is invalid")
        self._failure_stage = "transfer_body"
        data = self._recv_exact(sock, size)
        return f"{file_id}.{file_type.lower()}", bytearray(data), current, total


class TvArtworkClient:
    """Blocking artwork requests, each with one tokenless connection and budget."""

    def __init__(self, host: str) -> None:
        self._host = host

    def _run[T](self, operation: Callable[[_ArtworkArt], T]) -> T:
        _suppress_sdk_logs()
        art: _ArtworkArt | None = None
        try:
            art = _ArtworkArt(self._host, port=8002, name="LocalThings")
            try:
                return art._deadline.run(lambda: operation(art))
            finally:
                with suppress(Exception):
                    art.close()
        except Exception as error:
            stage = "initialization"
            category: str = _failure("art", "art_reply", error).error
            handshake_ms: int | str = "none"
            if art is not None:
                stage = art._failure_stage
                category = art._failure_category or category
                if art._handshake_ms is not None:
                    handshake_ms = art._handshake_ms
            reason, ssl_code, os_errno = _ssl_detail(error)
            log_failure = (
                _LOGGER.debug
                if isinstance(error, _UploadCompletionUncorrelated)
                else _LOGGER.warning
            )
            log_failure(
                "Artwork request failed: stage=%s category=%s kind=%s ssl_reason=%s "
                "ssl_code=%s os_errno=%s handshake_ms=%s samsungtvws=%s",
                stage,
                category,
                _exception_kind(error),
                reason,
                ssl_code,
                os_errno,
                handshake_ms,
                _installed_library_version(),
            )
            if isinstance(error, _UploadCompletionUncorrelated):
                failure = _UploadCompletionUncorrelated("Artwork upload completion is uncorrelated")
            elif isinstance(
                error, (UnauthorizedError, TvPairingRejected, TvAuthorizationIncomplete)
            ):
                failure = ArtworkAuthorizationError("Artwork authorization failed")
            else:
                failure = ArtworkError("Artwork request failed")
        # An exception context can contain raw SDK responses and credentials.
        raise failure from None

    def list_artwork(self) -> list[Artwork]:
        """Read and deduplicate available catalog metadata."""

        def read(art: _ArtworkArt) -> list[Artwork]:
            rows = art.available()
            if not isinstance(rows, list):
                raise ArtworkError("Artwork catalog is invalid")
            result: dict[str, Artwork] = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ArtworkError("Artwork catalog is invalid")
                content_id = _validate_content_id(row.get("content_id"))
                title = row.get("title")
                revision = row.get("image_date")
                result.setdefault(
                    content_id,
                    Artwork(
                        content_id,
                        title if isinstance(title, str) and title else content_id,
                        revision if isinstance(revision, str) else "",
                    ),
                )
            return list(result.values())

        return self._run(read)

    def thumbnail(self, content_id: str) -> tuple[bytes, str]:
        """Download one bounded JPEG or PNG through the tested batch request."""
        _validate_content_id(content_id)

        def read(art: _ArtworkArt) -> tuple[bytes, str]:
            try:
                images = art.get_thumbnail_list([content_id])
            except ssl.SSLEOFError as error:
                if (
                    art._failure_stage != "transfer_tls_handshake"
                    or getattr(error, "reason", None) != "UNEXPECTED_EOF_WHILE_READING"
                ):
                    raise
                art._failure_stage = "request"
                art._failure_category = None
                art._handshake_ms = None
                art._deadline.remaining()
                images = art.get_thumbnail_list([content_id])
            art._failure_stage = "thumbnail_validation"
            if not isinstance(images, dict) or len(images) != 1:
                raise ArtworkError("Artwork thumbnail is invalid")
            filename, data = next(iter(images.items()))
            if (
                not isinstance(filename, str)
                or not isinstance(data, (bytes, bytearray))
                or not 0 < len(data) <= _MAX_IMAGE_BYTES
            ):
                raise ArtworkError("Artwork thumbnail is invalid")
            stem, _, extension = filename.rpartition(".")
            if stem != content_id:
                raise ArtworkError("Artwork thumbnail content ID does not match")
            if extension.lower() not in {"jpg", "jpeg", "png"}:
                raise ArtworkError("Artwork thumbnail is invalid")
            # The TV can label PNG bytes as JPG, so the signature determines MIME.
            if data.startswith(b"\xff\xd8\xff"):
                return bytes(data), "image/jpeg"
            if data.startswith(b"\x89PNG\r\n\x1a\n"):
                return bytes(data), "image/png"
            raise ArtworkError("Artwork thumbnail is invalid")

        return self._run(read)

    def current(self) -> str:
        """Read one current artwork identifier without changing the screen."""

        def read(art: _ArtworkArt) -> str:
            current = art.get_current()
            if not isinstance(current, dict):
                raise ArtworkError("Current artwork is invalid")
            return _validate_content_id(current.get("content_id"))

        return self._run(read)

    def slideshow_active(self) -> bool:
        """Require an explicit, validated native slideshow state."""

        def read(art: _ArtworkArt) -> bool:
            status = art.get_slideshow_status()
            if not isinstance(status, dict) or status.get("event") != "get_slideshow_status":
                raise ArtworkError("Artwork slideshow state is invalid")
            value = status.get("value")
            if value == "off":
                return False
            if (
                not isinstance(value, str)
                or not value.isascii()
                or not value.isdecimal()
                or not value.strip("0")
                or not isinstance(status.get("category_id"), str)
                or not status["category_id"]
                or status.get("type") not in {"slideshow", "shuffleslideshow"}
            ):
                raise ArtworkError("Artwork slideshow state is invalid")
            return True

        return self._run(read)

    def stop_slideshow(self) -> None:
        """Send one native stop and require a fresh, validated off state."""

        def stop(art: _ArtworkArt) -> None:
            art.set_slideshow_status(duration=0)
            status = art.get_slideshow_status()
            if (
                not isinstance(status, dict)
                or status.get("event") != "get_slideshow_status"
                or status.get("value") != "off"
            ):
                raise ArtworkError("Native slideshow stop was not confirmed")

        self._run(stop)

    def delete(self, content_id: str) -> None:
        """Send one exact ID once; a reply is not proof of catalog absence."""
        _validate_content_id(content_id)

        def send(art: _ArtworkArt) -> None:
            request_uuid = str(uuid4())
            expected = [{"content_id": content_id}]
            response = art._send_art_request(
                {"request": "delete_image_list", "content_id_list": expected},
                request_uuid=request_uuid,
            )
            if not isinstance(response, dict) or response.get("event") != "delete_image_list":
                raise ArtworkError("Artwork deletion reply is invalid")
            identifiers = [response[key] for key in ("request_id", "id") if key in response]
            if not identifiers or any(value != request_uuid for value in identifiers):
                raise ArtworkError("Artwork deletion reply is uncorrelated")
            returned = response.get("content_id_list")
            if isinstance(returned, str):
                returned = json.loads(returned)
            if returned != expected:
                raise ArtworkError("Artwork deletion reply does not match")

        self._run(send)

    def select(self, content_id: str) -> str:
        """Send one selection, then confirm it with at most three reads."""
        _validate_content_id(content_id)

        def select_and_confirm(art: _ArtworkArt) -> str:
            # This Frame rejects a category override for some listed pictures.
            art.select_image(content_id, show=True)
            for _ in range(3):
                current = art.get_current()
                if isinstance(current, dict) and current.get("content_id") == content_id:
                    return content_id
            raise ArtworkError("Artwork selection was not confirmed")

        return self._run(select_and_confirm)

    def upload(self, data: bytes) -> str:
        """Send one prepared JPEG without selecting it or retrying an uncertain send."""
        if (
            not 0 < len(data) <= _MAX_UPLOAD_BYTES
            or not data.startswith(b"\xff\xd8\xff")
            or not data.endswith(b"\xff\xd9")
        ):
            raise ArtworkError("Artwork upload must be a bounded JPEG")

        previous_ids: set[str] = set()

        def send(art: _ArtworkArt) -> str:
            rows = art.available()
            if not isinstance(rows, list):
                raise ArtworkError("Artwork catalog is invalid")
            for row in rows:
                if not isinstance(row, dict):
                    raise ArtworkError("Artwork catalog is invalid")
                previous_ids.add(_validate_content_id(row.get("content_id")))
            return _validate_content_id(
                art.upload(data, matte="none", portrait_matte="none", file_type="jpg")
            )

        try:
            return self._run(send)
        except _UploadCompletionUncorrelated:
            pass
        # Do not trust the rejected event or repeat a send that the TV may have stored.
        added = {row.content_id for row in self.list_artwork()} - previous_ids
        if len(added) != 1:
            raise ArtworkError("Artwork upload could not be confirmed")
        content_id = added.pop()
        if not content_id.startswith("MY_"):
            raise ArtworkError("Artwork upload could not be confirmed")
        thumbnail, _mime_type = self.thumbnail(content_id)
        if not _thumbnail_matches_upload(data, thumbnail):
            raise ArtworkError("Artwork upload could not be confirmed")
        return content_id
