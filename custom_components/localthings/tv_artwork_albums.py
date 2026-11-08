"""Durable, versioned Frame albums; metadata never requests TV state."""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import Any
from uuid import uuid4

from homeassistant.core import CoreState, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util.file import WriteError
from homeassistant.util.json import SerializationError

from .const import DOMAIN
from .tv_artwork_cache import ArtworkCache
from .tv_artwork_slideshow import INTERVALS

_LOGGER = logging.getLogger(__name__)


class _AlbumStore(Store[dict[str, Any]]):
    """A successful save must mean the write reached storage."""

    async def async_save(self, data: dict[str, Any]) -> None:
        if self.hass.state is CoreState.stopping:
            raise OSError("Album storage is unavailable while Home Assistant is stopping")
        if self._read_only:
            raise OSError("Album storage is read-only")
        await super().async_save(data)

    async def _async_write_data(self, data: dict[str, Any]) -> None:
        try:
            await super()._async_write_data(data)
        except (WriteError, SerializationError):
            # Store otherwise swallows write failures and reports success.
            raise OSError("Album storage could not be saved") from None


def album_store(hass: HomeAssistant, entry_id: str) -> Store[dict[str, Any]]:
    """Keep definitions separate from artwork cache and other configured TVs."""
    return _AlbumStore(hass, 1, f"{DOMAIN}.{entry_id}.albums", atomic_writes=True)


def _validate_album(album: dict[str, Any]) -> None:
    if (
        set(album) != {"album_id", "name", "content_ids", "interval_minutes", "shuffle"}
        or not isinstance(album["album_id"], str)
        or not album["album_id"]
    ):
        raise HomeAssistantError("Invalid album ID")
    name = album["name"]
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
        raise HomeAssistantError("Album name must contain 1 to 60 characters")
    content_ids = album["content_ids"]
    if (
        not isinstance(content_ids, list)
        or not 1 <= len(content_ids) <= 100
        or any(not isinstance(value, str) or not value.startswith("MY_") for value in content_ids)
        or len(set(content_ids)) != len(content_ids)
    ):
        raise HomeAssistantError("Choose 1 to 100 distinct personal photos")
    if type(album["interval_minutes"]) is not int or album["interval_minutes"] not in INTERVALS:
        raise HomeAssistantError("Choose a supported album interval")
    if type(album["shuffle"]) is not bool:
        raise HomeAssistantError("Album shuffle must be true or false")


class ArtworkAlbums:
    """Serialize disk writes and publish only definitions known to be saved."""

    def __init__(
        self, hass: HomeAssistant, entry_id: str, device_id: str, artwork: ArtworkCache
    ) -> None:
        self._store = album_store(hass, entry_id)
        self._device_id = device_id
        self._artwork = artwork
        self._lock = asyncio.Lock()
        self._closed = False
        self._loaded = False
        self._document: dict[str, Any] = {"revision": 0, "albums": []}

    def _ensure_open(self) -> None:
        if self._closed:
            raise HomeAssistantError("Frame albums are closed")
        if not self._loaded:
            raise HomeAssistantError("Frame album storage is unavailable; reload the integration")

    async def async_load(self) -> None:
        """Isolate damaged album storage from normal appliance operation."""
        async with self._lock:
            try:
                payload = await self._store.async_load()
                if payload is not None:
                    if not isinstance(payload["device_id"], str):
                        raise ValueError("Invalid configured device identity")
                    if payload["device_id"] == self._device_id:
                        revision, albums = payload["revision"], payload["albums"]
                        if type(revision) is not int or revision < 0:
                            raise ValueError("Invalid album revision")
                        if not isinstance(albums, list) or len(albums) > 50:
                            raise ValueError("Invalid album list")
                        for album in albums:
                            _validate_album(album)
                            if album["name"] != album["name"].strip():
                                raise ValueError("Untrimmed album name")
                        if len({a["album_id"] for a in albums}) != len(albums) or len(
                            {a["name"].casefold() for a in albums}
                        ) != len(albums):
                            raise ValueError("Duplicate album")
                        self._document = {"revision": revision, "albums": deepcopy(albums)}
            except Exception:
                _LOGGER.warning(
                    "Frame album storage could not be loaded; album controls are unavailable"
                )
                return
            self._loaded = True

    async def async_list(self) -> dict[str, Any]:
        """Return only local metadata and independent collections."""
        async with self._lock:
            self._ensure_open()
            return deepcopy(self._document)

    def _check_revision(self, expected_revision: int) -> None:
        if type(expected_revision) is not int or expected_revision < 0:
            raise HomeAssistantError("Invalid album revision")
        if expected_revision != self._document["revision"]:
            raise HomeAssistantError("Albums changed; refresh and try again")

    def _album(self, album_id: str) -> dict[str, Any]:
        for album in self._document["albums"]:
            if album["album_id"] == album_id:
                return album
        raise HomeAssistantError("Unknown album; refresh and try again")

    def _check_personal(self, content_id: str) -> None:
        if (
            not isinstance(content_id, str)
            or not content_id.startswith("MY_")
            or not self._artwork.image_revision(content_id)
        ):
            raise HomeAssistantError("Choose a cached personal photo")

    async def _async_commit(self, document: dict[str, Any]) -> dict[str, Any]:
        write = asyncio.create_task(
            self._store.async_save({"device_id": self._device_id, **deepcopy(document)})
        )
        cancelled = False
        # A cancelled executor await can leave a writer running. Keep the lock
        # until it settles so that an older disk write cannot replace a new one.
        try:
            while not write.done():
                try:
                    await asyncio.shield(write)
                except asyncio.CancelledError:
                    cancelled = True
            write.result()
        except Exception:
            raise HomeAssistantError("Frame albums could not be saved") from None
        self._document = document
        if cancelled:
            raise asyncio.CancelledError
        return deepcopy(document)

    async def async_save(
        self,
        *,
        expected_revision: int,
        name: str,
        content_ids: list[str],
        interval_minutes: int,
        shuffle: bool,
        album_id: str | None = None,
    ) -> dict[str, Any]:
        """Reject stale edits and retain missing members only in their own album."""
        async with self._lock:
            self._ensure_open()
            self._check_revision(expected_revision)
            prior = self._album(album_id) if album_id is not None else None
            album = {
                "album_id": album_id if prior is not None else uuid4().hex,
                "name": name,
                "content_ids": content_ids,
                "interval_minutes": interval_minutes,
                "shuffle": shuffle,
            }
            _validate_album(album)
            album["name"] = name.strip()
            if any(
                other["album_id"] != album["album_id"]
                and other["name"].casefold() == album["name"].casefold()
                for other in self._document["albums"]
            ):
                raise HomeAssistantError("An album with this name already exists")
            if prior is None and len(self._document["albums"]) >= 50:
                raise HomeAssistantError("At most 50 albums can be saved")
            for content_id in content_ids:
                if prior is None or content_id not in prior["content_ids"]:
                    self._check_personal(content_id)
            document = deepcopy(self._document)
            if prior is None:
                document["albums"].append(deepcopy(album))
            else:
                index = self._document["albums"].index(prior)
                document["albums"][index] = deepcopy(album)
            document["revision"] += 1
            return await self._async_commit(document)

    async def async_append(self, album_id: str, content_id: str) -> dict[str, Any]:
        """Merge a confirmed cached upload without overwriting concurrent edits."""
        async with self._lock:
            self._ensure_open()
            album = self._album(album_id)
            if content_id in album["content_ids"]:
                return deepcopy(self._document)
            self._check_personal(content_id)
            if len(album["content_ids"]) >= 100:
                raise HomeAssistantError("At most 100 photos can be saved in one album")
            document = deepcopy(self._document)
            index = self._document["albums"].index(album)
            document["albums"][index]["content_ids"].append(content_id)
            document["revision"] += 1
            return await self._async_commit(document)

    async def async_snapshot(self, album_id: str, expected_revision: int) -> dict[str, Any]:
        """Freeze a complete available selection before the slideshow boundary."""
        async with self._lock:
            self._ensure_open()
            self._check_revision(expected_revision)
            album = self._album(album_id)
            self._artwork.validate_slideshow_ids(album["content_ids"])
            return deepcopy(album)

    async def async_close(self) -> None:
        """Reject queued requests and drain admitted storage writes."""
        self._closed = True
        async with self._lock:
            pass
