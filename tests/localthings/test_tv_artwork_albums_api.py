"""Use the real dispatcher to prove album authorization and local-only metadata."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.localthings.const import DOMAIN
from custom_components.localthings.tv import FrameMode
from custom_components.localthings.tv_artwork import Artwork
from custom_components.localthings.tv_artwork_albums import ArtworkAlbums

from .test_tv_entities import setup_tv as setup_tv

PREFIX = "localthings/frame_albums/"
IDS = ["MY_F0002", "MY_F0001"]


def message(action, entity_id="media_player.missing", **changes):
    result = {"type": PREFIX + action, "entity_id": entity_id}
    if action == "save":
        result.update(
            expected_revision=0,
            name="Nature",
            content_ids=IDS,
            interval_minutes=3,
            shuffle=False,
        )
    if action in {"append", "start"}:
        result["album_id"] = "unknown"
    if action == "append":
        result["content_id"] = IDS[0]
    if action == "start":
        result["expected_revision"] = 1
    result.update(changes)
    return result


async def request(ws, data):
    await ws.send_json_auto_id(data)
    return await ws.receive_json()


@pytest.mark.parametrize("action", ["list", "save", "append", "start"])
async def test_non_admin_rejected(
    hass, setup_tv, hass_ws_client, hass_read_only_access_token, action
):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    ws = await hass_ws_client(hass, access_token=hass_read_only_access_token)
    with patch.object(coordinator.slideshow, "async_start", new=AsyncMock()) as start:
        result = await request(ws, message(action, ids["media_player"]))
    assert not result["success"]
    assert result["error"]["code"] == "unauthorized"
    start.assert_not_awaited()


@pytest.mark.parametrize("action", ["list", "save", "append", "start"])
@pytest.mark.parametrize(
    "target", ["missing", "remote", "foreign", "appliance", "unloaded", "disabled", "wrong_tv"]
)
async def test_only_loaded_matching_frame_routes(hass, setup_tv, hass_ws_client, action, target):
    coordinator, ids, _, entry = await setup_tv(FrameMode.ART)
    entity_id = "media_player.missing"
    registry = er.async_get(hass)
    if target == "remote":
        entity_id = ids["remote"]
    elif target in {"foreign", "appliance"}:
        domain = "other" if target == "foreign" else DOMAIN
        other = MockConfigEntry(domain=domain, data={})
        other.add_to_hass(hass)
        other.mock_state(hass, ConfigEntryState.LOADED)
        entity_id = registry.async_get_or_create(
            "media_player", domain, "synthetic-appliance", config_entry=other
        ).entity_id
        hass.data.setdefault(DOMAIN, {})[other.entry_id] = Mock()
    elif target in {"unloaded", "disabled", "wrong_tv"}:
        entity_id = ids["media_player"]
        if target == "unloaded":
            entry.mock_state(hass, ConfigEntryState.NOT_LOADED)
        elif target == "disabled":
            registry.async_update_entity(entity_id, disabled_by=er.RegistryEntryDisabler.USER)
        else:
            registry.async_update_entity(
                entity_id, new_unique_id="localthings_tv_other_media_player"
            )
    ws = await hass_ws_client(hass)
    with patch.object(coordinator.slideshow, "async_start", new=AsyncMock()) as start:
        result = await request(ws, message(action, entity_id))
    assert not result["success"]
    assert result["error"]["message"] == "Choose a loaded LocalThings Frame media player"
    start.assert_not_awaited()
    if target == "unloaded":
        entry.mock_state(hass, ConfigEntryState.LOADED)


@pytest.mark.parametrize(
    "action,change",
    [
        ("list", {"extra": 1}),
        ("save", {"content_ids": [1]}),
        ("save", {"name": 1}),
        ("save", {"expected_revision": -1}),
        ("save", {"expected_revision": True}),
        ("save", {"shuffle": "false"}),
        ("save", {"interval_minutes": True}),
        ("start", {"expected_revision": "1"}),
        ("append", {"content_id": 1}),
    ],
)
async def test_malformed_requests_fail_before_actions(
    hass, setup_tv, hass_ws_client, action, change
):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    ws = await hass_ws_client(hass)
    with patch.object(coordinator.slideshow, "async_start", new=AsyncMock()) as start:
        result = await request(ws, message(action, ids["media_player"], **change))
    assert not result["success"]
    assert result["error"]["code"] == "invalid_format"
    start.assert_not_awaited()


async def test_offline_metadata_round_trip_append_stale_and_no_tv_calls(
    hass, setup_tv, hass_ws_client
):
    coordinator, ids, _, _ = await setup_tv(FrameMode.OFF)
    coordinator.artwork._catalog = {value: Artwork(value, value, "v1") for value in IDS}
    ws = await hass_ws_client(hass)
    with (
        patch.object(coordinator, "async_artwork_operation", new=AsyncMock()) as artwork,
        patch.object(coordinator, "async_slideshow_operation", new=AsyncMock()) as slideshow,
        patch.object(coordinator.artwork, "_client") as client,
    ):
        empty = await request(ws, message("list", ids["media_player"]))
        assert empty["result"] == {"revision": 0, "albums": []}
        created = await request(ws, message("save", ids["media_player"], content_ids=[IDS[0]]))
        assert created["success"]
        album_id = created["result"]["albums"][0]["album_id"]
        appended = await request(
            ws, message("append", ids["media_player"], album_id=album_id, content_id=IDS[1])
        )
        assert appended["success"]
        assert appended["result"]["revision"] == 2
        assert appended["result"]["albums"][0]["content_ids"] == IDS
        coordinator.artwork._catalog.pop(IDS[1])
        repeated = await request(
            ws, message("append", ids["media_player"], album_id=album_id, content_id=IDS[1])
        )
        assert repeated["result"] == appended["result"]
        stale = await request(
            ws, message("save", ids["media_player"], expected_revision=1, album_id=album_id)
        )
        assert not stale["success"]
        assert "changed" in stale["error"]["message"]
        listed = await request(ws, message("list", ids["media_player"]))
        assert listed["result"] == appended["result"]
        artwork.assert_not_awaited()
        slideshow.assert_not_awaited()
        assert not client.mock_calls


async def test_start_freezes_exact_order_and_uses_existing_status(hass, setup_tv, hass_ws_client):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    coordinator.artwork._catalog = {value: Artwork(value, value, "v1") for value in IDS}
    ws = await hass_ws_client(hass)
    created = await request(
        ws, message("save", ids["media_player"], interval_minutes=15, shuffle=True)
    )
    album_id = created["result"]["albums"][0]["album_id"]
    status = {"running": True, "content_ids": IDS, "existing_status": "unchanged"}
    with patch.object(
        coordinator.slideshow, "async_start", new=AsyncMock(return_value=status)
    ) as start:
        result = await request(ws, message("start", ids["media_player"], album_id=album_id))
        assert result["result"] == status
        start.assert_awaited_once_with(IDS, 15, True)
        start.call_args.args[0].clear()
        listed = await request(ws, message("list", ids["media_player"]))
        assert listed["result"]["albums"][0]["content_ids"] == IDS


@pytest.mark.parametrize("reason", ["missing", "stale", "single", "unknown"])
async def test_start_rejects_before_physical_boundary(hass, setup_tv, hass_ws_client, reason):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    coordinator.artwork._catalog = {value: Artwork(value, value, "v1") for value in IDS}
    ws = await hass_ws_client(hass)
    created = await request(
        ws, message("save", ids["media_player"], content_ids=IDS[:1] if reason == "single" else IDS)
    )
    album_id = created["result"]["albums"][0]["album_id"]
    if reason == "missing":
        coordinator.artwork._catalog.pop(IDS[1])
    with patch.object(coordinator.slideshow, "async_start", new=AsyncMock()) as start:
        result = await request(
            ws,
            message(
                "start",
                ids["media_player"],
                album_id="absent" if reason == "unknown" else album_id,
                expected_revision=0 if reason == "stale" else 1,
            ),
        )
    assert not result["success"]
    start.assert_not_awaited()


async def test_entry_unload_closes_album_requests_but_keeps_definitions(
    hass, setup_tv, hass_ws_client
):
    coordinator, ids, _, entry = await setup_tv(FrameMode.OFF)
    coordinator.artwork._catalog = {value: Artwork(value, value, "v1") for value in IDS}
    ws = await hass_ws_client(hass)
    created = await request(ws, message("save", ids["media_player"]))
    assert created["success"]
    assert await hass.config_entries.async_unload(entry.entry_id)
    with pytest.raises(HomeAssistantError, match="closed"):
        await coordinator.albums.async_list()
    restored = ArtworkAlbums(hass, entry.entry_id, coordinator.identity.device_id, Mock())
    await restored.async_load()
    assert await restored.async_list() == created["result"]
    assert not coordinator.slideshow.running
    rejected = await request(ws, message("list", ids["media_player"]))
    assert not rejected["success"]
    await restored.async_close()
