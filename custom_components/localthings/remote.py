"""Allowlisted ordinary Frame TV remote keys."""

from collections.abc import Iterable
from typing import Any

from homeassistant.components.remote import RemoteEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .tv import FrameMode
from .tv_coordinator import TvCoordinator
from .tv_entity import TvEntity

_REMOTE_KEYS = frozenset(
    {
        "KEY_UP",
        "KEY_DOWN",
        "KEY_LEFT",
        "KEY_RIGHT",
        "KEY_ENTER",
        "KEY_RETURN",
        "KEY_HOME",
        "KEY_MENU",
        "KEY_SOURCE",
        "KEY_PLAY",
        "KEY_PAUSE",
        "KEY_STOP",
        "KEY_FF",
        "KEY_REWIND",
        "KEY_VOLUP",
        "KEY_VOLDOWN",
        "KEY_CHUP",
        "KEY_CHDOWN",
    }
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: TvCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([FrameRemote(coordinator)])


class FrameRemote(TvEntity, RemoteEntity):
    """Power actions use the coordinator instead of blind toggle keys."""

    _attr_translation_key = "tv_remote"

    def __init__(self, coordinator: TvCoordinator) -> None:
        super().__init__(coordinator, "remote")

    @property
    def is_on(self) -> bool | None:
        mode = self.coordinator.state.mode
        return None if mode is FrameMode.UNKNOWN else mode in {FrameMode.TV, FrameMode.ART}

    async def async_turn_on(self, **kwargs: Any) -> None:
        await self.coordinator.async_turn_on()

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self.coordinator.async_turn_off()

    async def async_send_command(self, command: Iterable[str], **kwargs: Any) -> None:
        commands = tuple(command)
        if any(key not in _REMOTE_KEYS for key in commands):
            raise HomeAssistantError("Unsupported TV remote command")
        if (
            kwargs.get("num_repeats", 1) != 1
            or kwargs.get("hold_secs", 0) != 0
            or kwargs.get("delay_secs", 0) != 0
            or kwargs.get("device") is not None
        ):
            raise HomeAssistantError("TV remote supports single key presses without options")
        for key in commands:
            await self.coordinator.async_send_key(key)
