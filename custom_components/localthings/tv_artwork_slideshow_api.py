"""Administrator-only WebSocket access to a loaded Frame's HA playlist."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er

from .const import CONF_TRANSPORT, DOMAIN, TRANSPORT_TV_WS
from .tv_artwork_slideshow import INTERVALS
from .tv_coordinator import TvCoordinator

_PREFIX = "localthings/frame_slideshow/"


def _coordinator(hass: HomeAssistant, entity_id: str) -> TvCoordinator:
    entity = er.async_get(hass).async_get(entity_id)
    entry = (
        hass.config_entries.async_get_entry(entity.config_entry_id)
        if entity is not None and entity.config_entry_id is not None
        else None
    )
    coordinator = hass.data.get(DOMAIN, {}).get(entry.entry_id) if entry is not None else None
    if (
        entity is None
        or entity.domain != "media_player"
        or entity.platform != DOMAIN
        or entity.disabled
        or entry is None
        or entry.domain != DOMAIN
        or entry.state is not ConfigEntryState.LOADED
        or entry.data.get(CONF_TRANSPORT) != TRANSPORT_TV_WS
        or not isinstance(coordinator, TvCoordinator)
        or entity.unique_id != f"{DOMAIN}_tv_{coordinator.identity.device_id}_media_player"
    ):
        raise HomeAssistantError("Choose a loaded LocalThings Frame media player")
    return coordinator


@callback
def register_slideshow(hass: HomeAssistant) -> None:
    """Register process-wide commands once during integration setup."""
    for handler in (websocket_status, websocket_start, websocket_stop):
        websocket_api.async_register_command(hass, handler)


@websocket_api.websocket_command(
    {vol.Required("type"): _PREFIX + "status", vol.Required("entity_id"): cv.entity_id}
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_status(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    """Return memory-only HA status; never query the TV."""
    connection.send_result(msg["id"], _coordinator(hass, msg["entity_id"]).slideshow.status())


@websocket_api.websocket_command(
    {
        vol.Required("type"): _PREFIX + "start",
        vol.Required("entity_id"): cv.entity_id,
        vol.Required("content_ids"): vol.All([str], vol.Length(min=2)),
        vol.Required("interval_minutes"): vol.All(int, vol.In(INTERVALS)),
        vol.Required("shuffle"): bool,
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_start(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    """Start one exact-list rotation using the selected Frame only."""
    result = await _coordinator(hass, msg["entity_id"]).slideshow.async_start(
        msg["content_ids"], msg["interval_minutes"], msg["shuffle"]
    )
    connection.send_result(msg["id"], result)


@websocket_api.websocket_command(
    {vol.Required("type"): _PREFIX + "stop", vol.Required("entity_id"): cv.entity_id}
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_stop(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    """Drain the HA playlist, then check and stop native playback."""
    result = await _coordinator(hass, msg["entity_id"]).slideshow.async_stop()
    connection.send_result(msg["id"], result)
