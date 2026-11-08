"""Ephemeral, owner-bound prepared artwork previews."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.http.auth import async_sign_path
from homeassistant.core import HomeAssistant, callback

if TYPE_CHECKING:
    from .tv_artwork_image import PreparedArtwork

_STORE = "localthings_artwork_previews"
PREPARATION_LIFETIME = 600
PREVIEW_PATH = "/api/localthings/artwork/preview/{token}"


@dataclass(eq=False)
class Preparation:
    """One Frame's admission slot, including any running image processing."""

    entry_id: str
    flow_id: str
    owner_id: str
    refresh_token_id: str
    expires_at: float
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    source: bytes = field(default=b"", repr=False)
    image: PreparedArtwork | None = field(default=None, repr=False)
    thumbnail: bytes = field(default=b"", repr=False)
    thumbnail_mime_type: str = "image/jpeg"
    processing: bool = False
    disposed: bool = False
    timer: asyncio.TimerHandle | None = field(default=None, repr=False)


class PreviewStore:
    """Hold no public files and never contact the TV from a preview request."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._entries: dict[str, Preparation] = {}
        self.registered = False

    @callback
    def reserve(
        self, entry_id: str, flow_id: str, owner_id: str, refresh_token_id: str
    ) -> Preparation:
        """Reserve before admitting an executor job."""
        if previous := self._entries.get(entry_id):
            self.get(previous.token)
        if entry_id in self._entries:
            raise ValueError("Artwork preparation is already active")
        preparation = Preparation(
            entry_id,
            flow_id,
            owner_id,
            refresh_token_id,
            self.hass.loop.time() + PREPARATION_LIFETIME,
        )
        preparation.timer = self.hass.loop.call_later(
            PREPARATION_LIFETIME, self.dispose, preparation
        )
        self._entries[entry_id] = preparation
        return preparation

    @callback
    def get(self, token: str) -> Preparation | None:
        """Return a live preparation without extending its lifetime."""
        for preparation in tuple(self._entries.values()):
            if preparation.token != token:
                continue
            if self.hass.loop.time() >= preparation.expires_at:
                self.dispose(preparation)
            return None if preparation.disposed else preparation
        return None

    @callback
    def dispose(self, preparation: Preparation) -> None:
        """Clear bytes now; retain a processing admission until its job ends."""
        preparation.disposed = True
        preparation.source = b""
        preparation.image = None
        preparation.thumbnail = b""
        if preparation.timer is not None:
            preparation.timer.cancel()
            preparation.timer = None
        if not preparation.processing and self._entries.get(preparation.entry_id) is preparation:
            self._entries.pop(preparation.entry_id)

    @callback
    def processing_finished(self, preparation: Preparation) -> None:
        """Release cancelled or expired admissions only after executor completion."""
        preparation.processing = False
        if preparation.disposed:
            self.dispose(preparation)

    @callback
    def dispose_entry(self, entry_id: str) -> None:
        """Invalidate this Frame's preparation on unload."""
        if preparation := self._entries.get(entry_id):
            self.dispose(preparation)

    @callback
    def sign(self, preparation: Preparation) -> str:
        """Bind the temporary bearer link to the initiating refresh token."""
        return async_sign_path(
            self.hass,
            PREVIEW_PATH.format(token=preparation.token),
            timedelta(seconds=max(0, preparation.expires_at - self.hass.loop.time())),
            refresh_token_id=preparation.refresh_token_id,
        )


@callback
def preview_store(hass: HomeAssistant) -> PreviewStore:
    """Return the process-local preparation store."""
    if _STORE not in hass.data:
        hass.data[_STORE] = PreviewStore(hass)
    return hass.data[_STORE]


@callback
def register_preview(hass: HomeAssistant) -> None:
    """Register the authenticated read-only view once."""
    store = preview_store(hass)
    if not store.registered:
        hass.http.register_view(ArtworkPreviewView(store))
        store.registered = True


class ArtworkPreviewView(HomeAssistantView):
    """Serve exactly the prepared JPEG, with no shared or browser cache."""

    url = PREVIEW_PATH
    name = "api:localthings:artwork_preview"

    def __init__(self, store: PreviewStore) -> None:
        self._store = store

    async def get(self, request: web.Request, token: str) -> web.Response:
        """Serve only a live preparation under its administrator identity."""
        preparation = self._store.get(token)
        user = request.get("hass_user")
        if (
            preparation is None
            or (preparation.image is None and not preparation.thumbnail)
            or user is None
            or not user.is_admin
            or user.id != preparation.owner_id
        ):
            raise web.HTTPNotFound
        return web.Response(
            body=(
                preparation.image.data if preparation.image is not None else preparation.thumbnail
            ),
            content_type=(
                "image/jpeg" if preparation.image is not None else preparation.thumbnail_mime_type
            ),
            headers={"Cache-Control": "no-store"},
        )
