"""The native gallery uses protected cached images and explicit selections."""

import asyncio
import threading
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.components.media_player import DATA_COMPONENT, MediaPlayerEntityFeature
from homeassistant.components.media_player.const import MediaClass, MediaType
from homeassistant.components.media_player.errors import BrowseError
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.exceptions import HomeAssistantError

from custom_components.localthings.const import DOMAIN
from custom_components.localthings.tv import FrameMode
from custom_components.localthings.tv_artwork import Artwork, ArtworkAuthorizationError
from custom_components.localthings.tv_artwork_cache import ArtworkObservation, artwork_store
from custom_components.localthings.tv_coordinator import TvCoordinator

from .test_tv_entities import setup_tv as setup_tv


def artwork_manager():
    manager = Mock()
    manager.async_catalog = AsyncMock(
        return_value=[Mock(content_id="ART_ONE", title="First picture", revision="v1")]
    )
    manager.async_thumbnail = AsyncMock(return_value=(b"synthetic-image", "image/jpeg"))
    manager.async_select = AsyncMock(return_value="ART_ONE")
    manager.async_refresh = AsyncMock()
    manager.async_check_current = AsyncMock()
    manager.current = None
    manager.image_revision.return_value = "v1"
    manager.thumbnail_failed.return_value = False
    return manager


@pytest.mark.parametrize("mode", list(FrameMode))
async def test_current_check_only_runs_in_art_mode(hass, setup_tv, mode):
    coordinator, ids, actions, _ = await setup_tv(mode)
    manager = artwork_manager()
    player = hass.data[DATA_COMPONENT].get_entity(ids["media_player"])
    with patch.object(coordinator, "artwork", manager):
        await player.async_browse_media()
    if mode == FrameMode.ART:
        manager.async_check_current.assert_awaited_once_with()
    else:
        manager.async_check_current.assert_not_awaited()
        manager.clear_current.assert_called_once_with()
    actions["async_turn_on"].assert_not_awaited()


async def test_checked_picture_fields_publish_and_clear_on_mode_change(hass, setup_tv):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    manager = artwork_manager()
    checked_at = datetime(2026, 1, 1, tzinfo=UTC)
    manager.current = ArtworkObservation("ART_ONE", checked_at)
    manager.current_title.return_value = "First picture"
    manager.clear_current.side_effect = lambda: setattr(manager, "current", None)
    player = hass.data[DATA_COMPONENT].get_entity(ids["media_player"])
    with patch.object(coordinator, "artwork", manager):
        await player.async_browse_media()
        attributes = hass.states.get(ids["media_player"]).attributes
        assert attributes["media_content_id"] == "ART_ONE"
        assert attributes["media_content_type"] == "image"
        assert attributes["media_title"] == "First picture"
        assert attributes["artwork_checked_at"] == checked_at.isoformat()
        coordinator.async_set_updated_data(replace(coordinator.state, mode=FrameMode.TV))
        assert manager.current is None
        assert "artwork_checked_at" not in hass.states.get(ids["media_player"]).attributes
        coordinator.async_set_updated_data(replace(coordinator.state, mode=FrameMode.ART))
        assert player.media_content_id is None


@pytest.mark.parametrize("mode", list(FrameMode))
async def test_gallery_keeps_browse_feature_in_every_tv_mode(hass, setup_tv, mode):
    _, ids, _, _ = await setup_tv(mode)
    assert hass.states.get(ids["media_player"]).attributes["supported_features"] & (
        MediaPlayerEntityFeature.BROWSE_MEDIA
    )


@pytest.mark.parametrize("mode", list(FrameMode))
async def test_cached_gallery_uses_native_image_tiles(hass, setup_tv, mode):
    coordinator, ids, actions, _ = await setup_tv(mode)
    manager = artwork_manager()
    player = hass.data[DATA_COMPONENT].get_entity(ids["media_player"])
    with patch.object(coordinator, "artwork", manager, create=True):
        gallery = await player.async_browse_media()
    assert gallery.media_class == MediaClass.DIRECTORY
    assert len(gallery.children) == 1
    picture = gallery.children[0]
    assert picture.media_class == MediaClass.IMAGE
    assert picture.media_content_id == "ART_ONE"
    assert picture.title == "First picture"
    assert picture.can_play == (mode in {FrameMode.TV, FrameMode.ART})
    assert not picture.can_expand
    assert picture.thumbnail.startswith(f"/api/media_player_proxy/{ids['media_player']}/")
    manager.async_thumbnail.assert_not_awaited()
    actions["async_turn_on"].assert_not_awaited()


@pytest.mark.parametrize("mode", [FrameMode.OFF, FrameMode.UNKNOWN])
async def test_cached_thumbnail_hook_remains_available_offline(hass, setup_tv, mode):
    coordinator, ids, actions, _ = await setup_tv(mode)
    manager = artwork_manager()
    player = hass.data[DATA_COMPONENT].get_entity(ids["media_player"])
    with patch.object(coordinator, "artwork", manager, create=True):
        assert await player.async_get_browse_image("image", "ART_ONE", "v1") == (
            b"synthetic-image",
            "image/jpeg",
        )
    manager.async_thumbnail.assert_awaited_once_with("ART_ONE")
    actions["async_turn_on"].assert_not_awaited()


async def test_known_missing_preview_keeps_artwork_selectable(hass, setup_tv):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    manager = artwork_manager()
    manager.thumbnail_failed.return_value = True
    player = hass.data[DATA_COMPONENT].get_entity(ids["media_player"])
    with (
        patch.object(coordinator, "artwork", manager),
        patch.object(coordinator, "async_request_refresh", new=AsyncMock()),
    ):
        gallery = await player.async_browse_media()
        picture = gallery.children[0]
        assert picture.thumbnail is None
        assert picture.media_class == MediaClass.IMAGE
        assert picture.can_play
        await player.async_play_media(picture.media_content_type, picture.media_content_id)
    manager.async_select.assert_awaited_once_with("ART_ONE")
    manager.async_thumbnail.assert_not_awaited()


@pytest.mark.parametrize("mode", [FrameMode.TV, FrameMode.ART])
async def test_image_play_dispatches_one_artwork_selection(hass, setup_tv, mode):
    coordinator, ids, actions, _ = await setup_tv(mode)
    manager = artwork_manager()
    with (
        patch.object(coordinator, "artwork", manager, create=True),
        patch.object(coordinator, "async_request_refresh", new=AsyncMock()),
    ):
        await hass.services.async_call(
            "media_player",
            "play_media",
            {
                "entity_id": ids["media_player"],
                "media_content_type": "image",
                "media_content_id": "ART_ONE",
            },
            blocking=True,
        )
    manager.async_select.assert_awaited_once_with("ART_ONE")
    actions["async_send_key"].assert_not_awaited()


@pytest.mark.parametrize("mode", [FrameMode.OFF, FrameMode.UNKNOWN])
async def test_gallery_selection_does_not_wake_tv(hass, setup_tv, mode):
    coordinator, ids, actions, _ = await setup_tv(mode)
    manager = artwork_manager()
    player = hass.data[DATA_COMPONENT].get_entity(ids["media_player"])
    with (
        patch.object(coordinator, "artwork", manager, create=True),
        pytest.raises(HomeAssistantError),
    ):
        await player.async_play_media(MediaType.IMAGE, "ART_ONE")
    manager.async_select.assert_not_awaited()
    actions["async_turn_on"].assert_not_awaited()


async def test_gallery_rejects_unsupported_media_type(hass, setup_tv):
    coordinator, ids, _, _ = await setup_tv()
    manager = artwork_manager()
    player = hass.data[DATA_COMPONENT].get_entity(ids["media_player"])
    with patch.object(coordinator, "artwork", manager, create=True):
        with pytest.raises(HomeAssistantError):
            await player.async_play_media(MediaType.MUSIC, "https://invalid.example/song")
        with pytest.raises(BrowseError):
            await player.async_browse_media("image", "unrelated-folder")
        assert await player.async_get_browse_image("music", "ART_ONE") == (None, None)
    manager.async_select.assert_not_awaited()
    manager.async_thumbnail.assert_not_awaited()


async def test_tv_refresh_button_is_explicit(hass, setup_tv):
    coordinator, ids, _, _ = await setup_tv()
    manager = artwork_manager()
    with patch.object(coordinator, "artwork", manager, create=True):
        await hass.services.async_call(
            "button", "press", {"entity_id": ids["button"]}, blocking=True
        )
    manager.async_refresh.assert_awaited_once_with()


async def test_artwork_error_does_not_change_control_state_or_tokens(hass, setup_tv):
    coordinator, _, _, entry = await setup_tv()
    state = coordinator.state
    entry_data = dict(entry.data)
    operation = Mock(side_effect=HomeAssistantError("Artwork unavailable"))
    with pytest.raises(HomeAssistantError, match="Artwork unavailable"):
        await coordinator.async_artwork_operation(operation)
    assert coordinator.state is state
    assert dict(entry.data) == entry_data
    assert coordinator._authorization_error is None
    assert coordinator._power_authorization_error is None


async def test_artwork_cancellation_keeps_lock_until_worker_cleanup(hass, setup_tv):
    coordinator, _, _, _ = await setup_tv()
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def operation():
        started.set()
        release.wait(timeout=5)
        finished.set()

    task = hass.async_create_task(coordinator.async_artwork_operation(operation))
    try:
        await hass.async_add_executor_job(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert coordinator._command_lock.locked()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not coordinator._command_lock.locked()
    finally:
        release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)


async def test_gallery_websocket_and_protected_thumbnail_work_while_off(
    hass, setup_tv, hass_ws_client, hass_client_no_auth
):
    coordinator, ids, actions, _ = await setup_tv(FrameMode.OFF)
    manager = artwork_manager()
    websocket = await hass_ws_client(hass)
    client = await hass_client_no_auth()
    with patch.object(coordinator, "artwork", manager):
        await websocket.send_json(
            {"id": 1, "type": "media_player/browse_media", "entity_id": ids["media_player"]}
        )
        result = await websocket.receive_json()
        assert result["success"]
        picture = result["result"]["children"][0]
        assert picture["media_class"] == "image"
        assert picture["media_content_id"] == "ART_ONE"
        assert not picture["can_play"]
        url = picture["thumbnail"]
        response = await client.get(url)
        assert response.status == 200
        assert response.content_type == "image/jpeg"
        assert await response.read() == b"synthetic-image"
        manager.async_thumbnail.assert_awaited_once_with("ART_ONE")
        manager.async_thumbnail.reset_mock()
        response = await client.get(url.split("?")[0])
        assert response.status == 403
        manager.async_thumbnail.assert_not_awaited()
    actions["async_turn_on"].assert_not_awaited()


async def test_catalog_error_is_a_browse_error(hass, setup_tv):
    coordinator, ids, _, _ = await setup_tv()
    manager = artwork_manager()
    manager.async_catalog.side_effect = HomeAssistantError("Artwork list unavailable")
    player = hass.data[DATA_COMPONENT].get_entity(ids["media_player"])
    with (
        patch.object(coordinator, "artwork", manager),
        pytest.raises(BrowseError, match="Artwork list unavailable"),
    ):
        await player.async_browse_media()


async def test_artwork_authorization_is_separate_from_remote_permissions(hass, setup_tv):
    coordinator, _, _, entry = await setup_tv()
    state, credentials = coordinator.state, dict(entry.data)
    with (
        patch.object(
            coordinator.artwork._client, "list_artwork", side_effect=ArtworkAuthorizationError()
        ),
        patch.object(entry, "async_start_reauth_if_available") as reauth,
    ):
        with pytest.raises(HomeAssistantError):
            await coordinator.artwork.async_refresh()
        reauth.assert_not_called()
    assert coordinator.state is state
    assert dict(entry.data) == credentials
    assert coordinator._authorization_error is None
    assert coordinator._power_authorization_error is None


async def test_entry_removal_closes_and_removes_only_its_artwork_store(hass, setup_tv):
    coordinator, _, _, entry = await setup_tv()
    store = artwork_store(hass, entry.entry_id)
    await store.async_save({"sentinel": True})
    with patch.object(
        coordinator.artwork, "async_close", wraps=coordinator.artwork.async_close
    ) as close:
        assert await hass.config_entries.async_remove(entry.entry_id)
    close.assert_awaited()
    assert entry.entry_id not in hass.data[DOMAIN]
    assert await artwork_store(hass, entry.entry_id).async_load() is None


async def test_core_shutdown_closes_artwork_manager(hass, setup_tv):
    coordinator, _, _, _ = await setup_tv()
    with patch.object(
        coordinator.artwork, "async_close", wraps=coordinator.artwork.async_close
    ) as close:
        hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
        await hass.async_block_till_done()
    close.assert_awaited_once_with()
    assert coordinator._closed


async def test_thumbnail_burst_does_not_queue_ahead_of_power_controls(hass, setup_tv):
    coordinator, _, _, _ = await setup_tv()
    cache = coordinator.artwork
    client = cache._client
    started, release = asyncio.Event(), asyncio.Event()
    order = []

    async def execute(operation, *args):
        return operation(*args)

    def thumbnail(content_id):
        order.append(content_id)
        return b"synthetic-image", "image/jpeg"

    async def gated_execute(operation, *args):
        if not started.is_set():
            started.set()
            await release.wait()
        return operation(*args)

    async def key(*_):
        order.append("power-control")

    with (
        patch.object(
            client,
            "list_artwork",
            return_value=[Artwork("ART_ONE", "First", "v1"), Artwork("ART_TWO", "Second", "v1")],
        ),
        patch.object(client, "thumbnail", side_effect=thumbnail),
        patch.object(coordinator, "_async_executor", side_effect=execute) as executor,
        patch.object(coordinator, "_async_command", side_effect=key),
    ):
        await cache.async_refresh()
        executor.side_effect = gated_execute
        first = asyncio.create_task(cache.async_thumbnail("ART_ONE"))
        await started.wait()
        second = asyncio.create_task(cache.async_thumbnail("ART_TWO"))
        await asyncio.sleep(0)
        control = asyncio.create_task(TvCoordinator.async_send_key(coordinator, "KEY_HOME"))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second, control)
    assert order == ["ART_ONE", "power-control", "ART_TWO"]
