"""Observed Frame TV state and local power and volume controls."""

from homeassistant.components.media_player import (
    BrowseMedia,
    MediaPlayerDeviceClass,
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
)
from homeassistant.components.media_player.const import MediaClass, MediaType
from homeassistant.components.media_player.errors import BrowseError
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError

from .tv import FrameMode
from .tv_coordinator import TvCoordinator
from .tv_entity import TvEntity


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
        attributes = {"frame_mode": self.coordinator.state.mode.value}
        observation = self.coordinator.artwork.current
        if self.coordinator.state.mode == FrameMode.ART and observation is not None:
            attributes["artwork_checked_at"] = observation.checked_at.isoformat()
        return attributes

    @property
    def media_content_id(self) -> str | None:
        observation = self.coordinator.artwork.current
        if self.coordinator.state.mode == FrameMode.ART and observation is not None:
            return observation.content_id
        return None

    @property
    def media_content_type(self) -> str | None:
        return MediaType.IMAGE if self.media_content_id is not None else None

    @property
    def media_title(self) -> str | None:
        if self.media_content_id is not None:
            return self.coordinator.artwork.current_title()
        return None

    @callback
    def _handle_coordinator_update(self) -> None:
        if self.coordinator.state.mode != FrameMode.ART:
            self.coordinator.artwork.clear_current()
        super()._handle_coordinator_update()

    @property
    def supported_features(self) -> MediaPlayerEntityFeature:
        features = MediaPlayerEntityFeature.TURN_ON | MediaPlayerEntityFeature.BROWSE_MEDIA
        if self.coordinator.state.mode not in {FrameMode.TV, FrameMode.ART}:
            return features
        features |= (
            MediaPlayerEntityFeature.PLAY_MEDIA
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

    async def async_browse_media(
        self, media_content_type: str | None = None, media_content_id: str | None = None
    ) -> BrowseMedia:
        if media_content_type not in (None, MediaClass.DIRECTORY) or media_content_id not in (
            None,
            "artwork",
        ):
            raise BrowseError("Artwork folder not found")
        try:
            catalog = await self.coordinator.artwork.async_catalog()
        except HomeAssistantError as error:
            raise BrowseError(str(error)) from None
        if self.coordinator.state.mode == FrameMode.ART:
            await self.coordinator.artwork.async_check_current()
        else:
            self.coordinator.artwork.clear_current()
        self.async_write_ha_state()
        awake = self.coordinator.state.mode in {FrameMode.TV, FrameMode.ART}
        return BrowseMedia(
            media_class=MediaClass.DIRECTORY,
            media_content_type=MediaClass.DIRECTORY,
            media_content_id="artwork",
            title="Artwork",
            can_play=False,
            can_expand=True,
            children_media_class=MediaClass.IMAGE,
            children=[
                BrowseMedia(
                    media_class=MediaClass.IMAGE,
                    media_content_type=MediaType.IMAGE,
                    media_content_id=picture.content_id,
                    title=picture.title,
                    can_play=awake,
                    can_expand=False,
                    thumbnail=(
                        None
                        if self.coordinator.artwork.thumbnail_failed(picture.content_id)
                        else self.get_browse_image_url(
                            MediaType.IMAGE,
                            picture.content_id,
                            self.coordinator.artwork.image_revision(picture.content_id),
                        )
                    ),
                )
                for picture in catalog
            ],
        )

    async def async_get_browse_image(
        self,
        media_content_type: str,
        media_content_id: str,
        media_image_id: str | None = None,
    ) -> tuple[bytes | None, str | None]:
        if media_content_type != MediaType.IMAGE:
            return None, None
        return await self.coordinator.artwork.async_thumbnail(media_content_id)

    async def async_play_media(self, media_type: str, media_id: str, **kwargs) -> None:
        if media_type != MediaType.IMAGE:
            raise HomeAssistantError("Only cached artwork can be selected")
        if self.coordinator.state.mode not in {FrameMode.TV, FrameMode.ART}:
            raise HomeAssistantError("Turn on the TV before selecting artwork")
        await self.coordinator.slideshow.async_manual_select(media_id)
        await self.coordinator.async_request_refresh()
