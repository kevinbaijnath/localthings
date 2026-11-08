"""Persistent Frame artwork data, with no reads while the TV is asleep."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from homeassistant.core import CoreState, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util.file import WriteError
from homeassistant.util.json import SerializationError

from .const import DOMAIN
from .tv_artwork import (
    Artwork,
    ArtworkAuthorizationError,
    ArtworkError,
    TvArtworkClient,
    _thumbnail_matches_upload,
)
from .tv_artwork_delete import (
    ArtworkDeleteNotSentError as ArtworkDeleteNotSentError,
)
from .tv_artwork_delete import (
    ArtworkDeletePreview as ArtworkDeletePreview,
)
from .tv_artwork_delete import (
    ArtworkDeleteResult as ArtworkDeleteResult,
)

_LOGGER = logging.getLogger(__name__)
_CATALOG_LIFETIME = timedelta(days=7).total_seconds()
_THUMBNAIL_LIFETIME = timedelta(days=30).total_seconds()
_CURRENT_CHECK_INTERVAL = 30
_MAX_THUMBNAIL_BYTES = 8 * 1024 * 1024
MAX_CACHE_BYTES = 32 * 1024 * 1024


class _ArtworkStore(Store[dict[str, Any]]):
    """A destructive operation must observe a failed or deferred write."""

    async def async_save(self, data: dict[str, Any]) -> None:
        if self.hass.state is CoreState.stopping:
            raise OSError("Artwork storage is unavailable while Home Assistant is stopping")
        await super().async_save(data)

    async def _async_write_data(self, data: dict[str, Any]) -> None:
        try:
            await super()._async_write_data(data)
        except (WriteError, SerializationError):
            # Store logs and swallows these types; admission needs a failed await.
            raise OSError("Artwork storage could not be saved") from None


def artwork_store(hass: HomeAssistant, entry_id: str) -> Store[dict[str, Any]]:
    """Return this entry's artwork store without creating a client."""
    return _ArtworkStore(hass, 1, f"{DOMAIN}.{entry_id}.artwork")


@dataclass(frozen=True)
class ArtworkUploadResult:
    """Keep TV acceptance separate from a later catalog confirmation."""

    content_id: str
    catalog_confirmed: bool


class ArtworkUploadNotSentError(HomeAssistantError):
    """Admission failed before the upload callback began."""


@dataclass(frozen=True)
class ArtworkObservation:
    """A transient observation, not a promise of live slideshow state."""

    content_id: str
    checked_at: datetime


@dataclass(frozen=True)
class _Thumbnail:
    data: bytes
    mime_type: str
    fetched_at: float
    revision: str


class ArtworkCache:
    """Keep artwork reads separate from the TV's control authorization."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        client: TvArtworkClient,
        run_operation: Callable[[Callable[[], Any]], Awaitable[Any]],
        is_awake: Callable[[], bool],
        *,
        device_id: str,
        run_verified_operation: Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]],
        slideshow_running: Callable[[], bool] = lambda: False,
    ) -> None:
        self._hass = hass
        self._store = artwork_store(hass, entry_id)
        self._client = client
        self._run_operation = run_operation
        self._is_awake = is_awake
        self._device_id = device_id
        self._run_verified_operation = run_verified_operation
        self._slideshow_running = slideshow_running
        self._uploads: dict[str, dict[str, Any]] = {}
        self._deletions: dict[str, dict[str, Any]] = {}
        self._catalog: dict[str, Artwork] = {}
        self._thumbnails: dict[str, _Thumbnail] = {}
        self._last_attempt: float | None = None
        self._last_success: float | None = None
        self.current: ArtworkObservation | None = None
        self._last_current_attempt: float | None = None
        self._current_generation = 0
        self._authorization_suspended = False
        self._failed_thumbnails: set[str] = set()
        self._refresh_lock = asyncio.Lock()
        self._thumbnail_lock = asyncio.Lock()
        self._selection_lock = asyncio.Lock()
        self._save_lock = asyncio.Lock()
        self._closed = False
        self._unsubscribe: Callable[[], None] | None = None
        self._background_task: asyncio.Task[None] | None = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise HomeAssistantError("Artwork cache is closed")

    def _catalog_due(self) -> bool:
        return (
            self._last_attempt is None
            or dt_util.utcnow().timestamp() - self._last_attempt >= _CATALOG_LIFETIME
        )

    async def async_load(self) -> None:
        """Load cached data and register the due check without contacting the TV."""
        self._ensure_open()
        payload = await self._store.async_load()
        if payload is not None:
            self._uploads = payload.get("uploads", {})
            self._deletions = payload.get("deletions", {})
            for operation in self._deletions.values():
                if operation["outcome"] == "pending":
                    operation["outcome"] = "uncertain"
            self._uploads = {
                key: value
                for key, value in self._uploads.items()
                if value["device_id"] == self._device_id
            }
            self._catalog = {value["content_id"]: Artwork(**value) for value in payload["catalog"]}
            self._last_attempt = payload["last_attempt"]
            self._last_success = payload["last_success"]
            for content_id, value in payload["thumbnails"].items():
                artwork = self._catalog.get(content_id)
                if artwork is None or artwork.revision != value["revision"]:
                    continue
                data = base64.b64decode(value["data"], validate=True)
                if len(data) <= _MAX_THUMBNAIL_BYTES:
                    self._thumbnails[content_id] = _Thumbnail(
                        data, value["mime_type"], value["fetched_at"], value["revision"]
                    )
            self._evict_thumbnails()
        self._ensure_open()
        self._unsubscribe = async_track_time_interval(
            self._hass, self._async_check_due, timedelta(hours=1)
        )

    @callback
    def _async_check_due(self, _now: datetime) -> None:
        if (
            not self._closed
            and not self._authorization_suspended
            and self._is_awake()
            and self._catalog_due()
            and (self._background_task is None or self._background_task.done())
        ):
            self._background_task = self._hass.async_create_task(self._async_due_refresh())

    async def _async_due_refresh(self) -> None:
        with suppress(HomeAssistantError):
            await self._async_refresh(automatic=True)

    async def async_catalog(self) -> list[Artwork]:
        """Return the catalog, keeping stale entries usable while offline."""
        self._ensure_open()
        await self._async_refresh(automatic=True)
        return list(self._catalog.values())

    async def async_refresh(self) -> None:
        """Allow one new catalog attempt, including after an authorization failure."""
        self._ensure_open()
        self.clear_current()
        await self._async_refresh(automatic=False)

    async def _async_refresh(self, *, automatic: bool) -> None:
        async with self._refresh_lock:
            self._ensure_open()
            if automatic:
                if self._authorization_suspended or not self._catalog_due() or not self._is_awake():
                    return
            else:
                if not self._is_awake():
                    raise HomeAssistantError("TV must be on to refresh artwork")
                self._authorization_suspended = False
                self._failed_thumbnails.clear()
            self._last_attempt = dt_util.utcnow().timestamp()
            # Persist the attempt before a failed read can be retried after reload.
            await self._async_save()
            self._ensure_open()
            try:
                catalog = await self._run_operation(self._client.list_artwork)
            except (ArtworkError, HomeAssistantError) as error:
                if isinstance(error, ArtworkAuthorizationError):
                    self._authorization_suspended = True
                _LOGGER.warning("Artwork catalog refresh failed")
                if not automatic or not self._catalog:
                    raise HomeAssistantError("Artwork catalog could not be refreshed") from None
                return
            self._set_catalog(catalog)
            await self._async_save()

    def _set_catalog(self, catalog: list[Artwork]) -> None:
        self._catalog = {artwork.content_id: artwork for artwork in catalog}
        self._thumbnails = {
            content_id: thumbnail
            for content_id, thumbnail in self._thumbnails.items()
            if content_id in self._catalog
            and thumbnail.revision == self._catalog[content_id].revision
        }
        self._uploads = {
            content_id: record
            for content_id, record in self._uploads.items()
            if content_id in self._catalog
            and record["metadata"] == asdict(self._catalog[content_id])
        }
        self._last_success = dt_util.utcnow().timestamp()

    def clear_current(self) -> None:
        """Invalidate earlier and in-flight observations when the screen can change."""
        self.current = None
        self._last_current_attempt = None
        self._current_generation += 1

    def current_title(self) -> str | None:
        """Use cached metadata without another TV request."""
        if self.current is None:
            return None
        artwork = self._catalog.get(self.current.content_id)
        return artwork.title if artwork is not None else self.current.content_id

    async def async_check_current(self) -> None:
        """Check on gallery access, sharing attempts within thirty seconds."""
        self._ensure_open()
        generation = self._current_generation
        async with self._selection_lock:
            self._ensure_open()
            if generation != self._current_generation:
                return
            if self._authorization_suspended or not self._is_awake():
                self.current = None
                return
            now = dt_util.utcnow().timestamp()
            if (
                self._last_current_attempt is not None
                and now - self._last_current_attempt < _CURRENT_CHECK_INTERVAL
            ):
                return
            self.current = None
            self._last_current_attempt = now

            def read() -> str:
                if self._closed or generation != self._current_generation or not self._is_awake():
                    raise HomeAssistantError("Current artwork check is no longer applicable")
                return self._client.current()

            try:
                content_id = await self._run_operation(read)
            except (ArtworkError, HomeAssistantError) as error:
                if isinstance(error, ArtworkAuthorizationError):
                    self._authorization_suspended = True
                _LOGGER.warning("Current artwork check failed")
                return
            if not self._closed and generation == self._current_generation and self._is_awake():
                self.current = ArtworkObservation(content_id, dt_util.utcnow())

    async def async_thumbnail(self, content_id: str) -> tuple[bytes | None, str | None]:
        """Return cached bytes or admit one bounded fetch before the TV lock."""
        self._ensure_open()
        async with self._thumbnail_lock:
            self._ensure_open()
            artwork = self._catalog.get(content_id)
            if artwork is None:
                return None, None
            cached = self._thumbnails.get(content_id)
            if cached is not None and (
                dt_util.utcnow().timestamp() - cached.fetched_at < _THUMBNAIL_LIFETIME
            ):
                return cached.data, cached.mime_type
            if (
                self._authorization_suspended
                or content_id in self._failed_thumbnails
                or not self._is_awake()
            ):
                return (cached.data, cached.mime_type) if cached is not None else (None, None)
            try:
                data, mime_type = await self._run_operation(
                    lambda: self._client.thumbnail(content_id)
                )
                if len(data) > _MAX_THUMBNAIL_BYTES:
                    raise ArtworkError("Artwork thumbnail exceeds the cache limit")
            except (ArtworkError, HomeAssistantError) as error:
                branch = "admission"
                if isinstance(error, ArtworkError):
                    self._failed_thumbnails.add(content_id)
                    branch = "artwork_request"
                if isinstance(error, ArtworkAuthorizationError):
                    self._authorization_suspended = True
                    branch = "artwork_authorization"
                _LOGGER.warning("Artwork thumbnail fetch failed: branch=%s", branch)
                cached = self._thumbnails.get(content_id)
                return (cached.data, cached.mime_type) if cached is not None else (None, None)
            # A catalog refresh can invalidate this image while its fetch is in flight.
            current = self._catalog.get(content_id)
            if current is None or current.revision != artwork.revision:
                return None, None
            self._thumbnails[content_id] = _Thumbnail(
                data, mime_type, dt_util.utcnow().timestamp(), artwork.revision
            )
            self._evict_thumbnails()
            await self._async_save()
            return data, mime_type

    async def async_select(self, content_id: str) -> str:
        """Select one known picture once, using transport confirmation."""
        self._ensure_open()
        async with self._selection_lock:
            self._ensure_open()
            if content_id not in self._catalog:
                raise HomeAssistantError("Unknown artwork")
            if self._authorization_suspended:
                raise HomeAssistantError("Artwork authorization is suspended; refresh to retry")
            if not self._is_awake():
                raise HomeAssistantError("TV must be on to select artwork")
            self.clear_current()
            generation = self._current_generation
            try:
                selected = await self._run_operation(lambda: self._client.select(content_id))
            except ArtworkError as error:
                if isinstance(error, ArtworkAuthorizationError):
                    self._authorization_suspended = True
                raise HomeAssistantError("Artwork could not be selected") from None
            if not self._closed and generation == self._current_generation and self._is_awake():
                self.current = ArtworkObservation(selected, dt_util.utcnow())
                self._last_current_attempt = self.current.checked_at.timestamp()
            return selected

    async def async_upload(self, data: bytes, thumbnail: bytes) -> ArtworkUploadResult:
        """Upload once, then confirm the returned ID with one catalog read."""
        if self._closed:
            raise ArtworkUploadNotSentError("Artwork upload was not sent")
        async with self._refresh_lock, self._selection_lock:
            if self._closed or self._authorization_suspended or not self._is_awake():
                raise ArtworkUploadNotSentError("Artwork upload was not sent")
            self.clear_current()
            started = False

            def send() -> str:
                nonlocal started
                started = True
                return self._client.upload(data)

            try:
                content_id = await self._run_operation(send)
            except ArtworkError as error:
                if isinstance(error, ArtworkAuthorizationError):
                    self._authorization_suspended = True
                raise HomeAssistantError("Artwork could not be uploaded") from None
            except HomeAssistantError:
                if not started:
                    raise ArtworkUploadNotSentError("Artwork upload was not sent") from None
                raise HomeAssistantError("Artwork could not be uploaded") from None
            if self._closed:
                return ArtworkUploadResult(content_id, False)
            self._uploads.pop(content_id, None)
            self._last_attempt = dt_util.utcnow().timestamp()
            try:
                catalog = await self._run_operation(self._client.list_artwork)
                self._set_catalog(catalog)
                artwork = self._catalog.get(content_id)
                if artwork is not None:
                    self._thumbnails[content_id] = _Thumbnail(
                        thumbnail, "image/jpeg", dt_util.utcnow().timestamp(), artwork.revision
                    )
                    self._failed_thumbnails.discard(content_id)
                    self._evict_thumbnails()
                await self._async_save()
            except (ArtworkError, HomeAssistantError, OSError) as error:
                if isinstance(error, ArtworkAuthorizationError):
                    self._authorization_suspended = True
                _LOGGER.warning("Artwork upload accepted; catalog confirmation failed")
                return ArtworkUploadResult(content_id, False)
            if artwork is not None:

                def fingerprint() -> tuple[str, str]:
                    remote, _mime_type = self._client.thumbnail(content_id)
                    if not _thumbnail_matches_upload(data, remote):
                        raise ArtworkError("Uploaded artwork fingerprint does not match")
                    return hashlib.sha256(data).hexdigest(), hashlib.sha256(remote).hexdigest()

                async def record_upload() -> None:
                    source_hash, thumbnail_hash = await self._async_executor(fingerprint)
                    self._uploads[content_id] = {
                        "device_id": self._device_id,
                        "source_sha256": source_hash,
                        "metadata": asdict(artwork),
                        "thumbnail_sha256": thumbnail_hash,
                        "uploaded_at": dt_util.utcnow().timestamp(),
                        "record_token": uuid4().hex,
                    }
                    try:
                        await self._async_save()
                    except BaseException:
                        self._uploads.pop(content_id, None)
                        raise

                try:
                    await self._run_verified_operation(record_upload)
                except (ArtworkError, HomeAssistantError, OSError):
                    self._uploads.pop(content_id, None)
                    _LOGGER.warning("Artwork upload accepted; deletion ownership unavailable")
            return ArtworkUploadResult(content_id, artwork is not None)

    def validate_slideshow_ids(self, content_ids: list[str]) -> None:
        """Reject invalid selections before making any TV request."""
        self._ensure_open()
        if (
            not isinstance(content_ids, list)
            or len(content_ids) < 2
            or any(
                not isinstance(value, str)
                or not value.startswith("MY_")
                or value not in self._catalog
                for value in content_ids
            )
            or len(set(content_ids)) != len(content_ids)
        ):
            raise HomeAssistantError("Choose at least two distinct known personal photos")
        if self._authorization_suspended:
            raise HomeAssistantError("Artwork authorization is suspended; refresh to retry")

    async def async_slideshow_select(
        self,
        content_id: str,
        content_ids: list[str],
        *,
        expected_current: str | None,
        start: bool,
        admitted: Callable[[], bool],
        run_operation: Callable[..., Awaitable[Any]],
    ) -> str | None:
        """Keep the cache's selection-before-command lock order for each change."""
        async with self._selection_lock:
            self.validate_slideshow_ids(content_ids)

            async def select() -> str | None:
                self.validate_slideshow_ids(content_ids)
                current = await self._async_executor(self._client.current)
                self.current = ArtworkObservation(current, dt_util.utcnow())
                if expected_current is not None and current != expected_current:
                    raise HomeAssistantError(
                        "Home Assistant playlist stopped: another photo was selected."
                    )
                native = await self._async_executor(self._client.slideshow_active)
                if native and not start:
                    raise HomeAssistantError(
                        "Home Assistant playlist stopped: native rotation was enabled."
                    )
                if not admitted():
                    return None
                if native:
                    await self._async_executor(self._client.stop_slideshow)
                self._ensure_open()
                if not admitted():
                    return None
                self.clear_current()
                selected = await self._async_executor(lambda: self._client.select(content_id))
                if not self._closed:
                    self.current = ArtworkObservation(selected, dt_util.utcnow())
                    self._last_current_attempt = self.current.checked_at.timestamp()
                return selected

            try:
                return await run_operation(select)
            except ArtworkError:
                raise HomeAssistantError(
                    "Home Assistant playlist stopped: artwork request failed; it was not retried."
                ) from None

    async def async_stop_native_slideshow(
        self, run_operation: Callable[..., Awaitable[Any]]
    ) -> None:
        """Share selection serialization with explicit Stop, without waking."""
        async with self._selection_lock:
            self._ensure_open()
            if self._authorization_suspended:
                raise HomeAssistantError("Artwork authorization is suspended")

            async def stop() -> None:
                if await self._async_executor(self._client.slideshow_active):
                    await self._async_executor(self._client.stop_slideshow)

            try:
                await run_operation(stop, require_art=False)
            except ArtworkError:
                raise HomeAssistantError("Native rotation could not be stopped") from None

    async def _async_executor[T](self, operation: Callable[[], T]) -> T:
        try:
            future = self._hass.async_add_executor_job(operation)
            return await asyncio.shield(future)
        except ArtworkAuthorizationError:
            self._authorization_suspended = True
            raise
        except asyncio.CancelledError:
            # A cancelled browser waiter must not release the TV lock over a live send.
            while not future.done():
                with suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(future)
            if not future.cancelled():
                future.exception()
            raise

    def _delete_record(self, content_id: str, record_token: str | None = None) -> dict[str, Any]:
        record = self._uploads.get(content_id)
        if (
            record is None
            or record["device_id"] != self._device_id
            or (record_token is not None and record["record_token"] != record_token)
        ):
            raise ArtworkDeleteNotSentError("This photo has no matching recorded upload")
        return record

    def _ensure_delete_available(self) -> None:
        if self._slideshow_running():
            raise ArtworkDeleteNotSentError("Stop the Home Assistant playlist before deleting")
        if (
            self._closed
            or self._authorization_suspended
            or not self._is_awake()
            or self._hass.state is CoreState.stopping
        ):
            raise ArtworkDeleteNotSentError("TV must be awake and available for deletion")

    async def _async_delete_guard(self, content_id: str, record_token: str) -> ArtworkDeletePreview:
        self._ensure_delete_available()
        record = self._delete_record(content_id, record_token)
        catalog = await self._async_executor(self._client.list_artwork)
        self._set_catalog(catalog)
        if content_id not in self._uploads:
            await self._async_save()
            raise ArtworkDeleteNotSentError("The recorded photo is absent or changed")
        try:
            thumbnail, mime_type = await self._async_executor(
                lambda: self._client.thumbnail(content_id)
            )
        except ArtworkError:
            self._uploads.pop(content_id, None)
            self._thumbnails.pop(content_id, None)
            await self._async_save()
            raise ArtworkDeleteNotSentError("The recorded photo could not be verified") from None
        if hashlib.sha256(thumbnail).hexdigest() != record["thumbnail_sha256"]:
            self._uploads.pop(content_id, None)
            self._thumbnails.pop(content_id, None)
            await self._async_save()
            raise ArtworkDeleteNotSentError("The recorded photo has changed")
        current = await self._async_executor(self._client.current)
        if current == content_id:
            raise ArtworkDeleteNotSentError("Choose another picture before deleting this photo")
        if await self._async_executor(self._client.slideshow_active):
            raise ArtworkDeleteNotSentError("Stop native slideshow rotation before deleting")
        self._ensure_delete_available()
        self._delete_record(content_id, record_token)
        return ArtworkDeletePreview(
            content_id, self._catalog[content_id].title, record_token, thumbnail, mime_type
        )

    async def async_prepare_delete(self, content_id: str) -> ArtworkDeletePreview:
        """Check recorded ownership and fresh evidence without sending a mutation."""
        async with self._refresh_lock, self._selection_lock:
            self._ensure_delete_available()
            record = self._delete_record(content_id)
            if record["record_token"] in self._deletions:
                raise ArtworkDeleteNotSentError("This photo already has a deletion result")
            try:
                return await self._run_verified_operation(
                    lambda: self._async_delete_guard(content_id, record["record_token"])
                )
            except ArtworkDeleteNotSentError:
                raise
            except (ArtworkError, HomeAssistantError, OSError):
                raise ArtworkDeleteNotSentError(
                    "The TV photo state could not be verified"
                ) from None

    async def _async_reconcile_delete(self, operation: dict[str, Any]) -> ArtworkDeleteResult:
        content_id = operation["content_id"]
        try:
            self._ensure_delete_available()
            catalog = await self._async_executor(self._client.list_artwork)
            self._set_catalog(catalog)
            outcome = "present" if content_id in self._catalog else "absent"
        except (ArtworkError, HomeAssistantError):
            outcome = "uncertain"
        operation["outcome"] = outcome
        self.clear_current()
        self._thumbnails.pop(content_id, None)
        if outcome == "absent":
            self._uploads.pop(content_id, None)
        try:
            await self._async_save()
        except OSError:
            # The durable pending record still prevents a write replay after restart.
            _LOGGER.warning("Artwork deletion result could not be saved")
        return ArtworkDeleteResult(content_id, outcome)

    async def async_delete(self, preview: ArtworkDeletePreview) -> ArtworkDeleteResult:
        """Admit one durable operation, send once, then read the full catalog."""
        async with self._refresh_lock, self._selection_lock:
            existing = self._deletions.get(preview.record_token)
            if existing is not None:
                if existing["content_id"] != preview.content_id:
                    raise ArtworkDeleteNotSentError(
                        "The deletion preview does not match its record"
                    )
                return await self._async_delete_status_locked(existing)
            self._ensure_delete_available()
            self._delete_record(preview.content_id, preview.record_token)

            async def confirmed() -> ArtworkDeleteResult:
                await self._async_delete_guard(preview.content_id, preview.record_token)
                operation = {
                    "content_id": preview.content_id,
                    "device_id": self._device_id,
                    "record_token": preview.record_token,
                    "created_at": dt_util.utcnow().timestamp(),
                    "outcome": "pending",
                }
                self._deletions[preview.record_token] = operation
                try:
                    await self._async_save()
                except OSError:
                    self._deletions.pop(preview.record_token, None)
                    raise ArtworkDeleteNotSentError(
                        "Deletion was not sent because its record could not be saved"
                    ) from None
                self._ensure_delete_available()
                self.clear_current()
                with suppress(ArtworkError):
                    await self._async_executor(lambda: self._client.delete(preview.content_id))
                return await self._async_reconcile_delete(operation)

            try:
                return await self._run_verified_operation(confirmed)
            except ArtworkDeleteNotSentError:
                raise
            except (ArtworkError, HomeAssistantError, OSError):
                operation = self._deletions.get(preview.record_token)
                if operation is not None:
                    return await self._async_delete_status_locked(operation)
                raise ArtworkDeleteNotSentError(
                    "The TV photo state could not be verified"
                ) from None

    async def _async_delete_status_locked(self, operation: dict[str, Any]) -> ArtworkDeleteResult:
        upload = self._uploads.get(operation["content_id"])
        superseded = False
        for candidate in reversed(self._deletions.values()):
            if candidate is operation:
                break
            if candidate["content_id"] == operation["content_id"]:
                superseded = True
                break
        if (
            operation.get("device_id") != self._device_id
            or superseded
            or (upload is not None and upload["record_token"] != operation["record_token"])
            or self._closed
            or self._authorization_suspended
            or not self._is_awake()
        ):
            operation["outcome"] = "uncertain"
            return ArtworkDeleteResult(operation["content_id"], "uncertain")
        try:
            return await self._run_verified_operation(
                lambda: self._async_reconcile_delete(operation)
            )
        except (ArtworkError, HomeAssistantError):
            operation["outcome"] = "uncertain"
            with suppress(OSError):
                await self._async_save()
            return ArtworkDeleteResult(operation["content_id"], "uncertain")

    async def async_delete_status(
        self, content_id: str, *, record_token: str | None = None
    ) -> ArtworkDeleteResult | None:
        """Reconcile existing durable work using reads only, including after restart."""
        async with self._refresh_lock, self._selection_lock:
            if record_token is not None:
                operation = self._deletions.get(record_token)
                if operation is None or operation["content_id"] != content_id:
                    return None
                return await self._async_delete_status_locked(operation)
            operation = max(
                (
                    value
                    for value in reversed(self._deletions.values())
                    if value["content_id"] == content_id
                ),
                key=lambda value: value["created_at"],
                default=None,
            )
            if operation is None:
                return None
            return await self._async_delete_status_locked(operation)

    def thumbnail_failed(self, content_id: str) -> bool:
        """Identify failed previews without hiding a usable cached image."""
        self._ensure_open()
        return content_id in self._failed_thumbnails and content_id not in self._thumbnails

    def image_revision(self, content_id: str) -> str:
        """Return an opaque image-cache key for the current picture revision."""
        self._ensure_open()
        artwork = self._catalog.get(content_id)
        if artwork is None:
            return ""
        thumbnail = self._thumbnails.get(content_id)
        timestamp = thumbnail.fetched_at if thumbnail is not None else ""
        return hashlib.sha256(f"{artwork.revision}:{timestamp}".encode()).hexdigest()[:16]

    def _evict_thumbnails(self) -> None:
        total = sum(len(thumbnail.data) for thumbnail in self._thumbnails.values())
        while total > MAX_CACHE_BYTES:
            oldest = min(self._thumbnails, key=lambda key: self._thumbnails[key].fetched_at)
            total -= len(self._thumbnails.pop(oldest).data)

    async def _async_save(self) -> None:
        async with self._save_lock:
            await self._store.async_save(
                {
                    "uploads": self._uploads,
                    "deletions": self._deletions,
                    "catalog": [asdict(artwork) for artwork in self._catalog.values()],
                    "last_attempt": self._last_attempt,
                    "last_success": self._last_success,
                    "thumbnails": {
                        content_id: {
                            "data": base64.b64encode(thumbnail.data).decode("ascii"),
                            "mime_type": thumbnail.mime_type,
                            "fetched_at": thumbnail.fetched_at,
                            "revision": thumbnail.revision,
                        }
                        for content_id, thumbnail in self._thumbnails.items()
                    },
                }
            )

    async def async_close(self) -> None:
        """Stop scheduled work and wait for every admitted operation to finish."""
        self._closed = True
        self.clear_current()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()
            await asyncio.gather(self._background_task, return_exceptions=True)
        async with (
            self._refresh_lock,
            self._thumbnail_lock,
            self._selection_lock,
            self._save_lock,
        ):
            pass

    async def async_remove(self) -> None:
        """Remove storage only after all possible writers have stopped."""
        await self.async_close()
        await self._store.async_remove()
