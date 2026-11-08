"""Button platform for Local Things."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import CONF_TRANSPORT, DOMAIN, TRANSPORT_TV_WS
from .coordinator import LocalThingsCoordinator
from .entity import LocalThingsEntity, _is_included
from .registry.entities import ButtonDesc
from .tv_coordinator import TvCoordinator
from .tv_entity import TvEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    if entry.data.get(CONF_TRANSPORT) == TRANSPORT_TV_WS:
        async_add_entities([FrameArtworkRefreshButton(hass.data[DOMAIN][entry.entry_id])])
        return
    coordinator: LocalThingsCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        LocalThingsButton(coordinator, b)
        for b in coordinator.bound
        if isinstance(b.desc, ButtonDesc) and _is_included(b, coordinator)
    )


class LocalThingsButton(LocalThingsEntity, ButtonEntity):
    def __init__(self, coordinator: LocalThingsCoordinator, bound) -> None:
        super().__init__(coordinator, bound)
        self._payload = bound.desc.payload
        self._payload_fn = bound.desc.payload_fn

    async def async_press(self) -> None:
        payload = self._payload_fn(dt_util.now()) if self._payload_fn else self._payload
        await self.coordinator.async_send_command(self._bound, payload)


class FrameArtworkRefreshButton(TvEntity, ButtonEntity):
    """Explicitly refresh the cached artwork list while the TV is awake."""

    _attr_translation_key = "tv_artwork_refresh"
    _attr_icon = "mdi:image-refresh"

    def __init__(self, coordinator: TvCoordinator) -> None:
        super().__init__(coordinator, "artwork_refresh")

    async def async_press(self) -> None:
        await self.coordinator.artwork.async_refresh()
