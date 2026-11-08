"""Exercise the real HA WebSocket dispatcher, authentication and TV routing."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.localthings.const import DOMAIN
from custom_components.localthings.tv import FrameMode
from custom_components.localthings.tv_artwork import Artwork

from .test_tv_entities import setup_tv as setup_tv

PREFIX = "localthings/frame_slideshow/"
IDS = ["MY_F0001", "MY_F0002"]


def message(action, entity_id="media_player.missing"):
    result = {"type": PREFIX + action, "entity_id": entity_id}
    if action == "start":
        result.update(content_ids=IDS, interval_minutes=3, shuffle=False)
    return result


async def request(ws, data):
    await ws.send_json_auto_id(data)
    return await ws.receive_json()


@pytest.mark.parametrize("action", ["status", "start", "stop"])
async def test_non_admin_refused_by_actual_dispatcher(
    hass, setup_tv, hass_ws_client, hass_read_only_access_token, action
):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    ws = await hass_ws_client(hass, access_token=hass_read_only_access_token)
    with patch.object(coordinator, "async_slideshow_operation", new=AsyncMock()) as run:
        result = await request(ws, message(action, ids["media_player"]))
    assert not result["success"]
    assert result["error"]["code"] == "unauthorized"
    run.assert_not_awaited()


async def test_status_is_cached_and_honest_while_off(hass, setup_tv, hass_ws_client):
    coordinator, ids, _, _ = await setup_tv(FrameMode.OFF)
    ws = await hass_ws_client(hass)
    with patch.object(coordinator, "async_slideshow_operation", new=AsyncMock()) as run:
        result = await request(ws, message("status", ids["media_player"]))
    assert result["success"]
    assert result["result"] == coordinator.slideshow.status()
    assert "native rotation was not checked" in result["result"]["reason"]
    run.assert_not_awaited()


@pytest.mark.parametrize("action", ["status", "start", "stop"])
@pytest.mark.parametrize("target", ["missing", "remote", "foreign", "appliance", "unloaded"])
async def test_commands_only_route_to_loaded_frame_media_player(
    hass, setup_tv, hass_ws_client, action, target
):
    coordinator, ids, _, entry = await setup_tv(FrameMode.ART)
    entity_id = "media_player.missing"
    if target == "remote":
        entity_id = ids["remote"]
    elif target in {"foreign", "appliance"}:
        domain = "other" if target == "foreign" else DOMAIN
        other = MockConfigEntry(domain=domain, data={})
        other.add_to_hass(hass)
        other.mock_state(hass, ConfigEntryState.LOADED)
        entity_id = (
            er.async_get(hass)
            .async_get_or_create("media_player", domain, "synthetic-appliance", config_entry=other)
            .entity_id
        )
        hass.data.setdefault(DOMAIN, {})[other.entry_id] = Mock()
    elif target == "unloaded":
        entity_id = ids["media_player"]
        entry.mock_state(hass, ConfigEntryState.NOT_LOADED)
    ws = await hass_ws_client(hass)
    with patch.object(coordinator, "async_slideshow_operation", new=AsyncMock()) as run:
        result = await request(ws, message(action, entity_id))
    assert not result["success"]
    assert result["error"]["message"] == "Choose a loaded LocalThings Frame media player"
    run.assert_not_awaited()
    if target == "unloaded":
        entry.mock_state(hass, ConfigEntryState.LOADED)


@pytest.mark.parametrize(
    "change",
    [
        {"content_ids": []},
        {"interval_minutes": 2},
        {"interval_minutes": "3"},
        {"shuffle": "false"},
        {"extra": "not accepted"},
        {"content_ids": [1, 2]},
    ],
)
async def test_start_schema_rejects_invalid_inputs(hass, setup_tv, hass_ws_client, change):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    ws = await hass_ws_client(hass)
    data = message("start", ids["media_player"])
    data.update(change)
    with patch.object(coordinator, "async_slideshow_operation", new=AsyncMock()) as run:
        result = await request(ws, data)
    assert not result["success"]
    assert result["error"]["code"] == "invalid_format"
    run.assert_not_awaited()


async def test_real_websocket_start_status_stop_and_error_delivery(hass, setup_tv, hass_ws_client):
    coordinator, ids, _, _ = await setup_tv(FrameMode.ART)
    client = Mock()
    client.current.return_value = "MY_OLD"
    client.select.side_effect = lambda value: value
    client.slideshow_active.return_value = True
    coordinator.artwork._client = client
    coordinator.artwork._catalog = {value: Artwork(value, value, "v1") for value in IDS}

    async def run(operation, **_kwargs):
        return await operation()

    ws = await hass_ws_client(hass)
    with patch.object(coordinator, "async_slideshow_operation", side_effect=run) as operation:
        started = await request(ws, message("start", ids["media_player"]))
        assert started["success"]
        assert started["result"]["running"]
        assert started["result"]["content_ids"] == IDS
        assert started["result"]["current_content_id"] == IDS[0]
        status = await request(ws, message("status", ids["media_player"]))
        assert status["result"] == started["result"]
        assert operation.await_count == 1
        stopped = await request(ws, message("stop", ids["media_player"]))
        assert stopped["success"]
        assert not stopped["result"]["running"]
        assert stopped["result"]["content_ids"] == IDS
        client.stop_slideshow.assert_called()
        operation.side_effect = HomeAssistantError("TV must be in Art Mode for this playlist")
        failed = await request(ws, message("start", ids["media_player"]))
        assert not failed["success"]
        assert failed["error"]["message"] == "TV must be in Art Mode for this playlist"
        assert not coordinator.slideshow.running
