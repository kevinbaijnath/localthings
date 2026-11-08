"""Serialized local Frame TV polling and actions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import timedelta
from typing import Any, cast

from aiohttp import ClientSession
from homeassistant.config_entries import ConfigEntry, ConfigFlowContext
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import CONF_TV_TOKEN
from .tv import (
    FrameMode,
    TvAuthorizationIncomplete,
    TvClient,
    TvConnectionError,
    TvError,
    TvFailure,
    TvIdentity,
    TvPairingRejected,
    TvState,
    async_read_mute,
    async_read_tv_identity,
    async_set_mute,
)
from .tv_artwork import TvArtworkClient
from .tv_artwork_albums import ArtworkAlbums
from .tv_artwork_cache import ArtworkCache
from .tv_artwork_slideshow import ArtworkSlideshow
from .tv_power import TvPowerAuthorizationIncomplete, TvPowerClient, TvPowerPairingRejected

_LOGGER = logging.getLogger(__name__)
_UNKNOWN = TvState(FrameMode.UNKNOWN, None, None)
_POWER_OFF_TIMEOUT_SECONDS = 10.0
_WAKE_TIMEOUT_SECONDS = 30.0
_ART_MODE_TIMEOUT_SECONDS = 10.0


def _authorization_label(error: type[TvError] | None) -> str:
    if error is None:
        return "clear"
    return "rejected" if issubclass(error, TvPairingRejected) else "incomplete"


async def _async_wake_pause() -> None:
    await asyncio.sleep(1)


async def _async_power_off_pause() -> None:
    await asyncio.sleep(1)


async def _async_art_mode_pause() -> None:
    await asyncio.sleep(1)


class TvCoordinator(DataUpdateCoordinator[TvState]):
    """One lock owns all TV reads, commands, and connection cleanup."""

    def __init__(
        self,
        hass: HomeAssistant,
        session: ClientSession,
        *,
        host: str,
        token: str | None,
        power_token: str,
        identity: TvIdentity,
        config_entry: ConfigEntry | None,
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name="LocalThings TV",
            update_interval=timedelta(seconds=5),
        )
        self._session = session
        self._host = host
        self._token = token
        self._identity = identity
        self._client: TvClient | None = None
        self._power_client = TvPowerClient(session, host, power_token)
        self._power_authorization_error: (
            type[TvPowerPairingRejected | TvPowerAuthorizationIncomplete] | None
        ) = None
        self._power_reauth_requested = False
        self._authorization_error: type[TvPairingRejected | TvAuthorizationIncomplete] | None = None
        self._last_websocket_failure: TvFailure | None = None
        self._reported_websocket_failure: TvFailure | None = None
        self._reported_latching_failure = False
        self._reauth_requested = False
        self._command_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self._connected = False
        self.data = _UNKNOWN
        self.slideshow = ArtworkSlideshow(hass, self)
        self.artwork = ArtworkCache(
            hass,
            config_entry.entry_id if config_entry is not None else identity.device_id,
            TvArtworkClient(host),
            self.async_artwork_operation,
            lambda: self.state.mode in {FrameMode.TV, FrameMode.ART},
            device_id=identity.device_id,
            run_verified_operation=self.async_verified_artwork_operation,
            slideshow_running=lambda: self.slideshow.running,
        )
        self.albums = ArtworkAlbums(
            hass,
            config_entry.entry_id if config_entry is not None else identity.device_id,
            identity.device_id,
            self.artwork,
        )
        self._unsub_stop: Callable[[], None] | None = hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP, self._async_on_stop
        )

    @property
    def host(self) -> str:
        """The configured local host."""
        return self._host

    @property
    def identity(self) -> TvIdentity:
        """The configured identity, independent of current state confidence."""
        return self._identity

    @property
    def state(self) -> TvState:
        """The latest observed state; commands never set desired state here."""
        return self.data

    @property
    def connected(self) -> bool:
        """Whether any endpoint answered the latest read, without proving mode."""
        return self._connected

    @property
    def connection_diagnostics(self) -> dict[str, Any]:
        """Read cached, non-secret connection evidence without any TV request."""
        return {
            "websocket_authorization": _authorization_label(self._authorization_error),
            "power_authorization": _authorization_label(self._power_authorization_error),
            "websocket_retry_suppressed": self._authorization_error is not None,
            "last_websocket_failure": (
                asdict(self._last_websocket_failure)
                if self._last_websocket_failure is not None
                else None
            ),
        }

    @callback
    def async_update_listeners(self) -> None:
        """Invalidate a playlist on mode loss, including ordinary polling results."""
        self.slideshow.observe_mode()
        super().async_update_listeners()

    def _ensure_open(self, *, websocket: bool = False) -> None:
        if self._closing:
            raise HomeAssistantError("TV coordinator is closed")
        if websocket and self._authorization_error is TvPairingRejected:
            raise ConfigEntryAuthFailed("TV authorization was rejected") from None
        if websocket and self._authorization_error is TvAuthorizationIncomplete:
            raise HomeAssistantError(
                "TV channel authorization is suspended; reauthenticate or reload before retrying"
            ) from None

    async def _async_executor[T](self, operation: Callable[..., T], *args: Any) -> T:
        future = self.hass.async_add_executor_job(operation, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Cancelling a waiter cannot stop its thread or release a held key.
            # Keep the caller's lock until the SDK job and its cleanup finish.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not future.cancelled():
                future.exception()
            raise

    async def _async_close_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await self._async_executor(client.close)
            except TvError:
                _LOGGER.warning("TV connection cleanup failed")

    def _capture_sdk_state(self) -> None:
        if self._client is None:
            return
        error = self._client.authorization_error
        failure = self._client.last_failure
        if failure is not None:
            self._last_websocket_failure = failure
            if self._reported_websocket_failure is None or (
                error is not None
                and not self._reported_latching_failure
                and failure != self._reported_websocket_failure
            ):
                _LOGGER.warning(
                    "TV WebSocket failure: channel=%s stage=%s error=%s authorization=%s",
                    failure.channel,
                    failure.stage,
                    failure.error,
                    _authorization_label(error),
                )
                self._reported_websocket_failure = failure
            if error is not None:
                self._reported_latching_failure = True
        if error in (TvPairingRejected, TvAuthorizationIncomplete):
            self._authorization_error = error
        token = self._client.token
        if not isinstance(token, str) or not token or token == self._token:
            return
        self._token = token
        if self.config_entry is not None and token != self.config_entry.data.get(CONF_TV_TOKEN):
            self.hass.config_entries.async_update_entry(
                self.config_entry, data={**self.config_entry.data, CONF_TV_TOKEN: token}
            )

    def _request_reauth(self, *, power: bool = False) -> None:
        if power:
            if not self._power_reauth_requested and self.config_entry is not None:
                self._power_reauth_requested = True
                self.config_entry.async_start_reauth_if_available(
                    self.hass, context=cast(ConfigFlowContext, {"tv_auth_channel": "power"})
                )
            return
        if not self._reauth_requested and self.config_entry is not None:
            self._reauth_requested = True
            self.config_entry.async_start_reauth_if_available(self.hass)

    async def _async_sdk(self, method: str, *args: Any) -> Any:
        if self._authorization_error is TvPairingRejected:
            self._connected = False
            raise ConfigEntryAuthFailed("TV authorization was rejected") from None
        if self._authorization_error is TvAuthorizationIncomplete:
            raise TvAuthorizationIncomplete("TV channel authorization is suspended") from None
        try:
            if self._client is None:
                self._client = await self._async_executor(TvClient, self._host, self._token)
            try:
                return await self._async_executor(getattr(self._client, method), *args)
            finally:
                self._capture_sdk_state()
        except asyncio.CancelledError:
            if self._authorization_error is not None:
                if self._authorization_error is TvPairingRejected:
                    self._connected = False
                    self._request_reauth()
                await self._async_close_client()
            raise
        except TvPairingRejected:
            self._authorization_error = TvPairingRejected
            self._connected = False
            self.async_set_updated_data(_UNKNOWN)
            await self._async_close_client()
            self._request_reauth()
            raise ConfigEntryAuthFailed("TV authorization was rejected") from None
        except TvAuthorizationIncomplete:
            self._authorization_error = TvAuthorizationIncomplete
            await self._async_close_client()
            raise
        except TvConnectionError:
            await self._async_close_client()
            raise

    async def _async_power(self, method: str = "async_read_power", *args: Any) -> Any:
        if self._power_authorization_error is not None:
            raise self._power_authorization_error("TV power authorization is suspended") from None
        try:
            return await getattr(self._power_client, method)(*args)
        except TvPowerPairingRejected:
            self._power_authorization_error = TvPowerPairingRejected
            self._request_reauth(power=True)
            raise
        except TvPowerAuthorizationIncomplete:
            self._power_authorization_error = TvPowerAuthorizationIncomplete
            raise

    async def _async_identity(self) -> TvIdentity:
        identity = await async_read_tv_identity(self._session, self._host)
        if identity.device_id != self._identity.device_id:
            self._connected = False
            self.async_set_updated_data(_UNKNOWN)
            raise UpdateFailed("TV identity does not match the configured device")
        return identity

    async def _async_read_evidence(
        self, *, verify_off_identity: bool = False, read_mute: bool = True
    ) -> tuple[bool | None, TvState]:
        power: str | None = None
        art: bool | None = None
        muted: bool | None = None
        picture: str | None = None
        self._connected = False
        dedicated_power: bool | None = None
        if self._power_authorization_error is None:
            try:
                dedicated_power = await self._async_power()
            except TvError:
                pass
            else:
                self._connected = True
                if dedicated_power is False:
                    if not verify_off_identity:
                        return False, TvState(FrameMode.OFF, None, None)
                else:
                    power = "on"
        try:
            identity = await self._async_identity()
        except TvError:
            pass
        else:
            if dedicated_power is None:
                power = identity.power_state
            self._connected = True
        if dedicated_power is False:
            return False, TvState(FrameMode.OFF, None, None)
        for method in ("async_read_art_mode", "async_read_picture_mode"):
            try:
                value = await self._async_power(method)
            except TvError:
                continue
            self._connected = True
            if method == "async_read_art_mode":
                art = value
            else:
                picture = value
        if read_mute:
            try:
                muted = await async_read_mute(self._session, self._host)
            except TvError:
                pass
            else:
                self._connected = True
        mode = FrameMode.UNKNOWN
        if power is not None and power.casefold() == "on" and picture:
            ambient = picture == "Ambient"
            if art is True and ambient:
                mode = FrameMode.ART
            elif art is False and not ambient:
                mode = FrameMode.TV
        elif power is not None and power.casefold() in {"off", "standby"} and art is None:
            mode = FrameMode.OFF
        return dedicated_power, TvState(mode, art, muted)

    async def _async_read_state(self) -> TvState:
        _, state = await self._async_read_evidence()
        return state

    async def _async_update_data(self) -> TvState:
        async with self._command_lock:
            self._ensure_open()
            try:
                return await self._async_read_state()
            except asyncio.CancelledError:
                self.async_set_updated_data(_UNKNOWN)
                raise

    async def _async_publish_state(self) -> None:
        self.async_set_updated_data(await self._async_read_state())

    async def _async_command(self, method: str, *args: Any) -> None:
        try:
            await self._async_sdk(method, *args)
            await self._async_publish_state()
        except ConfigEntryAuthFailed:
            self._request_reauth()
            raise
        except asyncio.CancelledError:
            self.async_set_updated_data(_UNKNOWN)
            raise
        except TvAuthorizationIncomplete:
            self.async_set_updated_data(_UNKNOWN)
            raise HomeAssistantError(
                "TV channel authorization did not complete; "
                "reauthenticate or reload before retrying"
            ) from None
        except TvError:
            self.async_set_updated_data(_UNKNOWN)
            raise HomeAssistantError("TV command failed; it was not retried") from None

    async def _async_wake(self) -> TvState:
        self.async_set_updated_data(_UNKNOWN)
        self._connected = False
        await self._async_sdk("send_key", "KEY_POWER")
        try:
            async with asyncio.timeout(_WAKE_TIMEOUT_SECONDS):
                while True:
                    power, state = await self._async_read_evidence(verify_off_identity=True)
                    self._ensure_open(websocket=True)
                    if self._power_authorization_error is not None:
                        raise self._power_authorization_error(
                            "TV power authorization is suspended"
                        ) from None
                    if power is True and state.mode in {FrameMode.TV, FrameMode.ART}:
                        self.async_set_updated_data(state)
                        return state
                    await _async_wake_pause()
        except TimeoutError:
            raise HomeAssistantError("TV did not wake before the deadline") from None

    async def _async_change_mode(self, enabled: bool, *, wake: bool) -> None:
        try:
            power, state = await self._async_read_evidence(verify_off_identity=True)
            self._ensure_open()
            self.async_set_updated_data(state)
            if self._power_authorization_error is not None:
                raise self._power_authorization_error("TV power authorization is suspended")
            if not wake and state.mode in {FrameMode.OFF, FrameMode.TV}:
                return
            if power is False:
                state = await self._async_wake()
                if state.art_mode is enabled:
                    return
            elif state.mode not in {FrameMode.TV, FrameMode.ART}:
                raise HomeAssistantError("TV mode is unknown; wake requires confirmed power off")
            await self._async_power("async_set_art_mode", enabled)
            self.async_set_updated_data(_UNKNOWN)
            async with asyncio.timeout(_ART_MODE_TIMEOUT_SECONDS):
                while True:
                    _, state = await self._async_read_evidence(
                        verify_off_identity=True, read_mute=False
                    )
                    if self._power_authorization_error is not None:
                        raise self._power_authorization_error(
                            "TV power authorization is suspended"
                        ) from None
                    expected = FrameMode.ART if enabled else FrameMode.TV
                    if state.mode is expected:
                        self.async_set_updated_data(state)
                        return
                    await _async_art_mode_pause()
        except TimeoutError:
            self._connected = False
            self.async_set_updated_data(_UNKNOWN)
            raise HomeAssistantError("TV mode could not be confirmed before the deadline") from None
        except ConfigEntryAuthFailed:
            self._request_reauth()
            self._connected = False
            self.async_set_updated_data(_UNKNOWN)
            raise
        except (asyncio.CancelledError, HomeAssistantError):
            self._connected = False
            self.async_set_updated_data(_UNKNOWN)
            raise
        except TvError:
            self._connected = False
            self.async_set_updated_data(_UNKNOWN)
            raise HomeAssistantError("TV mode command failed; it was not retried") from None

    async def async_turn_on(self, *, art_mode: bool = False) -> None:
        """Use fresh evidence to wake once, then request an explicit mode."""
        async with self._command_lock:
            self._ensure_open()
            await self._async_change_mode(art_mode, wake=True)

    async def async_turn_off(self) -> None:
        """Send full off once, then observe power without repeating the setter."""
        async with self._command_lock:
            self._ensure_open()
            if self.data.mode is FrameMode.OFF:
                return
            try:
                await self._async_power("async_turn_off")
                self.async_set_updated_data(_UNKNOWN)
                self._connected = False
                async with asyncio.timeout(_POWER_OFF_TIMEOUT_SECONDS):
                    while True:
                        try:
                            power = await self._async_power()
                        except (TvPowerPairingRejected, TvPowerAuthorizationIncomplete):
                            raise
                        except TvError:
                            self._connected = False
                        else:
                            self._connected = True
                            if power is False:
                                self.async_set_updated_data(TvState(FrameMode.OFF, None, None))
                                return
                        await _async_power_off_pause()
            except asyncio.CancelledError:
                self._connected = False
                self.async_set_updated_data(_UNKNOWN)
                raise
            except (TvPowerPairingRejected, TvPowerAuthorizationIncomplete):
                self._connected = False
                self.async_set_updated_data(_UNKNOWN)
                raise HomeAssistantError(
                    "TV power authorization is suspended; reauthenticate before retrying"
                ) from None
            except TimeoutError:
                self.async_set_updated_data(_UNKNOWN)
                raise HomeAssistantError(
                    "TV power off could not be confirmed before the deadline"
                ) from None
            except TvError:
                self._connected = False
                self.async_set_updated_data(_UNKNOWN)
                raise HomeAssistantError("TV power command failed; it was not retried") from None

    async def async_set_art_mode(self, enabled: bool) -> None:
        """Art on may wake; Art off never wakes an already off TV."""
        if enabled:
            await self.async_turn_on(art_mode=True)
            return
        async with self._command_lock:
            self._ensure_open()
            await self._async_change_mode(False, wake=False)

    async def async_set_mute(self, muted: bool) -> None:
        """Set explicit mute and read it back without a remote toggle."""
        async with self._command_lock:
            self._ensure_open()
            try:
                await async_set_mute(self._session, self._host, muted)
                await self._async_publish_state()
            except (TvError, asyncio.CancelledError) as error:
                self.async_set_updated_data(TvState(self.data.mode, self.data.art_mode, None))
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise HomeAssistantError("TV mute command failed; it was not retried") from None

    async def async_send_key(self, key: str) -> None:
        """Send one key without replaying an uncertain result."""
        async with self._command_lock:
            self._ensure_open(websocket=True)
            await self._async_command("send_key", key)

    async def async_artwork_operation[T](self, operation: Callable[[], T]) -> T:
        """Keep artwork failures separate from working control permissions."""
        async with self._command_lock:
            self._ensure_open()
            if self.state.mode not in {FrameMode.TV, FrameMode.ART}:
                raise HomeAssistantError("Turn on the TV before requesting artwork")
            return await self._async_executor(operation)

    async def async_verified_artwork_operation[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Keep fresh identity, final guards and the write under the command lock."""
        async with self._command_lock:
            self._ensure_open()
            try:
                identity = await self._async_identity()
            except (TvError, UpdateFailed):
                raise HomeAssistantError("TV identity could not be verified") from None
            if (
                self.state.mode not in {FrameMode.TV, FrameMode.ART}
                or identity.power_state is None
                or identity.power_state.casefold() != "on"
            ):
                raise HomeAssistantError("TV must be on for this artwork operation")
            self._ensure_open()
            return await operation()

    async def async_slideshow_operation[T](
        self, operation: Callable[[], Awaitable[T]], *, require_art: bool = True
    ) -> T:
        """Read fresh identity and mode under the same lock as a slideshow write."""
        async with self._command_lock:
            self._ensure_open()
            try:
                identity = await self._async_identity()
                power = await self._async_power()
                art = await self._async_power("async_read_art_mode")
                picture = await self._async_power("async_read_picture_mode")
            except (TvError, UpdateFailed):
                self.async_set_updated_data(_UNKNOWN)
                raise HomeAssistantError(
                    "TV mode could not be verified; playlist stopped"
                ) from None
            mode = FrameMode.UNKNOWN
            if power is False:
                mode = FrameMode.OFF
            elif (
                power is True
                and identity.power_state is not None
                and identity.power_state.casefold() == "on"
            ):
                if art is True and picture == "Ambient":
                    mode = FrameMode.ART
                elif art is False and picture and picture != "Ambient":
                    mode = FrameMode.TV
            self.async_set_updated_data(TvState(mode, art, self.state.muted))
            if mode not in ({FrameMode.ART} if require_art else {FrameMode.ART, FrameMode.TV}):
                raise HomeAssistantError(
                    "TV must be in Art Mode for this playlist"
                    if require_art
                    else "TV must be awake to stop native rotation"
                )
            self._ensure_open()
            try:
                return await operation()
            finally:
                self.async_set_updated_data(self.state)

    async def async_config_entry_first_refresh(self) -> None:
        """Close owned SDK sockets if initial setup cannot complete."""
        try:
            await super().async_config_entry_first_refresh()
        except BaseException:
            await self.async_close()
            raise

    async def _async_on_stop(self, event: Event) -> None:
        self._unsub_stop = None
        await self.async_close()

    async def async_shutdown(self) -> None:
        """Let configuration-entry unload close the same owned SDK sockets."""
        await self.async_close()

    async def async_close(self) -> None:
        """Stop polling and wait for serialized work; never close the HA session."""
        self._closing = True
        await self.albums.async_close()
        await self.slideshow.async_close()
        await super().async_shutdown()
        if self._unsub_stop is not None:
            self._unsub_stop()
            self._unsub_stop = None
        await self.artwork.async_close()
        async with self._command_lock:
            if self._closed:
                return
            try:
                await self._async_close_client()
            finally:
                self._closed = True
