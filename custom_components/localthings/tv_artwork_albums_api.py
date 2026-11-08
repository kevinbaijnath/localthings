"""Administrator-only album metadata and explicit saved-album playback."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv

from .tv_artwork_slideshow import INTERVALS
from .tv_artwork_slideshow_api import _coordinator

_PREFIX = "localthings/frame_albums/"


def _revision(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise vol.Invalid("Expected a nonnegative integer revision")
    return value


def _interval(value: Any) -> int:
    if type(value) is not int or value not in INTERVALS:
        raise vol.Invalid("Choose a supported album interval")
    return value


@callback
def register_albums(hass: HomeAssistant) -> None:
    """Register the process-wide commands with the integration."""
    for handler in (websocket_list, websocket_save, websocket_append, websocket_start):
        websocket_api.async_register_command(hass, handler)


@websocket_api.websocket_command(
    {vol.Required("type"): _PREFIX + "list", vol.Required("entity_id"): cv.entity_id}
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_list(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    """List saved definitions without refreshing the artwork cache."""
    result = await _coordinator(hass, msg["entity_id"]).albums.async_list()
    connection.send_result(msg["id"], result)


@websocket_api.websocket_command(
    {
        vol.Required("type"): _PREFIX + "save",
        vol.Required("entity_id"): cv.entity_id,
        vol.Required("expected_revision"): _revision,
        vol.Optional("album_id"): vol.All(str, vol.Length(min=1)),
        vol.Required("name"): str,
        vol.Required("content_ids"): vol.All([str], vol.Length(min=1, max=100)),
        vol.Required("interval_minutes"): _interval,
        vol.Required("shuffle"): bool,
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_save(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    """Save one edit against the version the editor read."""
    result = await _coordinator(hass, msg["entity_id"]).albums.async_save(
        expected_revision=msg["expected_revision"],
        album_id=msg.get("album_id"),
        name=msg["name"],
        content_ids=msg["content_ids"],
        interval_minutes=msg["interval_minutes"],
        shuffle=msg["shuffle"],
    )
    connection.send_result(msg["id"], result)


@websocket_api.websocket_command(
    {
        vol.Required("type"): _PREFIX + "append",
        vol.Required("entity_id"): cv.entity_id,
        vol.Required("album_id"): vol.All(str, vol.Length(min=1)),
        vol.Required("content_id"): str,
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_append(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    """Merge upload membership without retrying a physical upload."""
    result = await _coordinator(hass, msg["entity_id"]).albums.async_append(
        msg["album_id"], msg["content_id"]
    )
    connection.send_result(msg["id"], result)


@websocket_api.websocket_command(
    {
        vol.Required("type"): _PREFIX + "start",
        vol.Required("entity_id"): cv.entity_id,
        vol.Required("album_id"): vol.All(str, vol.Length(min=1)),
        vol.Required("expected_revision"): _revision,
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_start(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    """Start a frozen saved definition through the existing guarded engine."""
    coordinator = _coordinator(hass, msg["entity_id"])
    album = await coordinator.albums.async_snapshot(msg["album_id"], msg["expected_revision"])
    result = await coordinator.slideshow.async_start(
        album["content_ids"], album["interval_minutes"], album["shuffle"]
    )
    connection.send_result(msg["id"], result)
