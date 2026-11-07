"""Observed Frame TV state and local power and volume controls."""

from homeassistant.components.media_player import (
    MediaPlayerDeviceClass,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .tv import FrameMode
from .tv_coordinator import TvCoordinator
from .tv_entity import TvEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: TvCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([FrameMediaPlayer(coordinator)])


class FrameMediaPlayer(TvEntity, MediaPlayerEntity):
    """Art Mode stays awake while unknown state retains a wake action."""

    _attr_name = None
    _attr_device_class = MediaPlayerDeviceClass.TV

    def __init__(self, coordinator: TvCoordinator) -> None:
        super().__init__(coordinator, "media_player")

    @property
    def available(self) -> bool:
        return True

    @property
    def state(self) -> MediaPlayerState | None:
        return {
            FrameMode.TV: MediaPlayerState.ON,
            FrameMode.ART: MediaPlayerState.IDLE,
            FrameMode.OFF: MediaPlayerState.OFF,
        }.get(self.coordinator.state.mode)

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        return {"frame_mode": self.coordinator.state.mode.value}

    @property
    def supported_features(self) -> MediaPlayerEntityFeature:
        if self.coordinator.state.mode not in {FrameMode.TV, FrameMode.ART}:
            return MediaPlayerEntityFeature.TURN_ON
        features = (
            MediaPlayerEntityFeature.TURN_ON
            | MediaPlayerEntityFeature.TURN_OFF
            | MediaPlayerEntityFeature.VOLUME_STEP
        )
        if self.coordinator.state.muted is not None:
            features |= MediaPlayerEntityFeature.VOLUME_MUTE
        return features

    @property
    def is_volume_muted(self) -> bool | None:
        return self.coordinator.state.muted

    async def async_turn_on(self) -> None:
        await self.coordinator.async_turn_on()

    async def async_turn_off(self) -> None:
        await self.coordinator.async_turn_off()

    async def async_volume_up(self) -> None:
        await self.coordinator.async_send_key("KEY_VOLUP")

    async def async_volume_down(self) -> None:
        await self.coordinator.async_send_key("KEY_VOLDOWN")

    async def async_mute_volume(self, mute: bool) -> None:
        await self.coordinator.async_set_mute(mute)
