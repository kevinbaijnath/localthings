"""Switch platform for Local Things."""

from __future__ import annotations

from typing import Any, cast

from homeassistant.components.switch import SwitchDeviceClass, SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_TRANSPORT, DOMAIN, TRANSPORT_TV_WS
from .coordinator import LocalThingsCoordinator
from .entity import LocalThingsEntity, _is_included
from .registry.entities import SwitchDesc
from .tv import FrameMode
from .tv_coordinator import TvCoordinator
from .tv_entity import TvEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    if entry.data.get(CONF_TRANSPORT) == TRANSPORT_TV_WS:
        async_add_entities([FrameArtModeSwitch(hass.data[DOMAIN][entry.entry_id])])
        return
    coordinator: LocalThingsCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        LocalThingsSwitch(coordinator, b)
        for b in coordinator.bound
        if isinstance(b.desc, SwitchDesc) and _is_included(b, coordinator)
    )


class LocalThingsSwitch(LocalThingsEntity, SwitchEntity):
    def __init__(self, coordinator: LocalThingsCoordinator, bound) -> None:
        super().__init__(coordinator, bound)
        desc = cast(SwitchDesc, bound.desc)
        self._attr_device_class = (
            SwitchDeviceClass(desc.device_class) if desc.device_class else None
        )

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return entity specific state attributes."""
        desc = cast(SwitchDesc, self._bound.desc)
        if desc.extra_state_attributes_fn is not None:
            rep = self.coordinator.entity_rep(self._bound.href or "")
            resources = self._resources
            return desc.extra_state_attributes_fn(rep, resources)
        return None

    @property
    def is_on(self):
        return (self.coordinator.data or {}).get(self._state_key)

    async def async_turn_on(self, **kwargs) -> None:
        await self.coordinator.async_send_command(self._bound, "On")

    async def async_turn_off(self, **kwargs) -> None:
        await self.coordinator.async_send_command(self._bound, "Off")


class FrameArtModeSwitch(TvEntity, SwitchEntity):
    """Set Art Mode explicitly without treating Art off as full power-off."""

    _attr_translation_key = "tv_art_mode"
    _attr_icon = "mdi:palette"

    def __init__(self, coordinator: TvCoordinator) -> None:
        super().__init__(coordinator, "art_mode")

    @property
    def available(self) -> bool:
        return super().available or (
            self.coordinator.last_update_success and self.coordinator.state.mode is FrameMode.OFF
        )

    @property
    def is_on(self) -> bool | None:
        mode = self.coordinator.state.mode
        return None if mode is FrameMode.UNKNOWN else mode is FrameMode.ART

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self.coordinator.async_set_art_mode(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self.coordinator.async_set_art_mode(False)
