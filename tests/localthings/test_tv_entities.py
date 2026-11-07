"""TV entities expose only observed state and executable local controls."""

from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import ClientSession
from homeassistant.components.media_player import MediaPlayerEntityFeature
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.service_info.dhcp import DhcpServiceInfo

from custom_components.localthings.const import DOMAIN
from custom_components.localthings.tv import FrameMode, TvConnectionError, TvIdentity, TvState
from custom_components.localthings.tv_coordinator import TvCoordinator

from .test_tv_config_flow import IDENTITY, tv_entry


@pytest.fixture
async def setup_tv(hass):
    entries = []
    with ExitStack() as patches:

        async def setup(mode=FrameMode.TV, *, mac=IDENTITY.mac, muted=False):
            entry = tv_entry(hass, mac=mac)
            entries.append(entry)
            art_mode = None if mode is FrameMode.UNKNOWN else mode is FrameMode.ART
            coordinator = TvCoordinator(
                hass,
                Mock(spec=ClientSession),
                host=entry.data["host"],
                token="synthetic-token",
                power_token="synthetic-power-token",
                identity=TvIdentity(IDENTITY.device_id, IDENTITY.model, mac),
                config_entry=entry,
            )
            patches.enter_context(
                patch.object(
                    coordinator,
                    "_async_update_data",
                    new=AsyncMock(return_value=TvState(mode, art_mode, muted)),
                )
            )
            actions = {
                name: patches.enter_context(patch.object(coordinator, name, new=AsyncMock()))
                for name in (
                    "async_turn_on",
                    "async_turn_off",
                    "async_set_art_mode",
                    "async_set_mute",
                    "async_send_key",
                )
            }
            with patch("custom_components.localthings.TvCoordinator", return_value=coordinator):
                assert await hass.config_entries.async_setup(entry.entry_id)
                await hass.async_block_till_done()
            ids = {
                row.domain: row.entity_id
                for row in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
            }
            return coordinator, ids, actions, entry

        yield setup
        for entry in entries:
            await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()


@pytest.mark.parametrize(
    ("mode", "state"),
    [
        (FrameMode.TV, "on"),
        (FrameMode.ART, "idle"),
        (FrameMode.OFF, "off"),
        (FrameMode.UNKNOWN, "unknown"),
    ],
)
async def test_media_state_mapping(hass, setup_tv, mode, state):
    _, ids, _, _ = await setup_tv(mode)
    player = hass.states.get(ids["media_player"])
    assert player.state == state
    assert player.attributes["frame_mode"] == mode.value


@pytest.mark.parametrize("mac", [IDENTITY.mac, None])
async def test_turn_on_service_works_while_tv_unknown(hass, setup_tv, mac):
    _, ids, actions, _ = await setup_tv(FrameMode.UNKNOWN, mac=mac, muted=None)
    player = hass.states.get(ids["media_player"])
    assert player.state == "unknown"
    assert player.attributes["supported_features"] == MediaPlayerEntityFeature.TURN_ON
    await hass.services.async_call(
        "media_player", "turn_on", {"entity_id": ids["media_player"]}, blocking=True
    )
    actions["async_turn_on"].assert_awaited_once_with()
    assert hass.states.get(ids["remote"]).state == "unavailable"
    assert hass.states.get(ids["switch"]).state == "unavailable"
    for domain, service in (("remote", "send_command"), ("switch", "turn_on")):
        data = {"entity_id": ids[domain]}
        if domain == "remote":
            data["command"] = ["KEY_UP"]
        await hass.services.async_call(domain, service, data, blocking=True)
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            "media_player", "volume_up", {"entity_id": ids["media_player"]}, blocking=True
        )
    actions["async_send_key"].assert_not_awaited()
    actions["async_set_art_mode"].assert_not_awaited()


@pytest.mark.parametrize("mac", [IDENTITY.mac, None])
async def test_confirmed_off_features_and_controls(hass, setup_tv, mac):
    _, ids, actions, _ = await setup_tv(FrameMode.OFF, mac=mac)
    player = hass.states.get(ids["media_player"])
    assert player.state == "off"
    assert player.attributes["supported_features"] == MediaPlayerEntityFeature.TURN_ON
    assert hass.states.get(ids["remote"]).state == "unavailable"
    assert hass.states.get(ids["switch"]).state == "off"
    for domain in ("media_player", "switch"):
        await hass.services.async_call(domain, "turn_on", {"entity_id": ids[domain]}, blocking=True)
    actions["async_turn_on"].assert_awaited_once_with()
    actions["async_set_art_mode"].assert_awaited_once_with(True)


@pytest.mark.parametrize("mode", [FrameMode.TV, FrameMode.ART])
async def test_volume_steps_and_explicit_mute(hass, setup_tv, mode):
    _, ids, actions, _ = await setup_tv(mode, muted=True)
    player = hass.states.get(ids["media_player"])
    assert player.attributes["is_volume_muted"] is True
    for service, key in (("volume_up", "KEY_VOLUP"), ("volume_down", "KEY_VOLDOWN")):
        await hass.services.async_call(
            "media_player", service, {"entity_id": ids["media_player"]}, blocking=True
        )
        actions["async_send_key"].assert_awaited_once_with(key)
        actions["async_send_key"].reset_mock()
    for mute in (False, True):
        await hass.services.async_call(
            "media_player",
            "volume_mute",
            {"entity_id": ids["media_player"], "is_volume_muted": mute},
            blocking=True,
        )
        actions["async_set_mute"].assert_awaited_once_with(mute)
        actions["async_set_mute"].reset_mock()
    assert "volume_level" not in player.attributes
    assert not player.attributes["supported_features"] & MediaPlayerEntityFeature.VOLUME_SET


async def test_unknown_mute_is_not_false_or_a_mute_control(hass, setup_tv):
    coordinator, ids, _, _ = await setup_tv(muted=None)
    player = hass.states.get(ids["media_player"])
    assert "is_volume_muted" not in player.attributes
    assert not player.attributes["supported_features"] & MediaPlayerEntityFeature.VOLUME_MUTE
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
    await hass.async_block_till_done()
    player = hass.states.get(ids["media_player"])
    assert player.attributes["is_volume_muted"] is False
    assert player.attributes["supported_features"] & MediaPlayerEntityFeature.VOLUME_MUTE


@pytest.mark.parametrize("domain", ["media_player", "remote"])
async def test_explicit_power_actions(hass, setup_tv, domain):
    _, ids, actions, _ = await setup_tv(FrameMode.ART)
    for service in ("turn_on", "turn_off"):
        await hass.services.async_call(domain, service, {"entity_id": ids[domain]}, blocking=True)
        actions[f"async_{service}"].assert_awaited_once_with()
    actions["async_send_key"].assert_not_awaited()
    actions["async_set_art_mode"].assert_not_awaited()


@pytest.mark.parametrize("mode", [FrameMode.TV, FrameMode.ART, FrameMode.OFF])
@pytest.mark.parametrize("mac", [IDENTITY.mac, None])
async def test_explicit_art_mode_actions(hass, setup_tv, mode, mac):
    _, ids, actions, _ = await setup_tv(mode, mac=mac)
    assert hass.states.get(ids["switch"]).state == ("on" if mode is FrameMode.ART else "off")
    for service, enabled in (("turn_on", True), ("turn_off", False)):
        await hass.services.async_call(
            "switch", service, {"entity_id": ids["switch"]}, blocking=True
        )
        actions["async_set_art_mode"].assert_awaited_once_with(enabled)
        actions["async_set_art_mode"].reset_mock()
    actions["async_turn_off"].assert_not_awaited()
    actions["async_send_key"].assert_not_awaited()


@pytest.mark.parametrize("mode", [FrameMode.TV, FrameMode.ART])
async def test_remote_navigation_and_media_keys(hass, setup_tv, mode):
    _, ids, actions, _ = await setup_tv(mode)
    for key in (
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
    ):
        await hass.services.async_call(
            "remote", "send_command", {"entity_id": ids["remote"], "command": key}, blocking=True
        )
        actions["async_send_key"].assert_awaited_once_with(key)
        actions["async_send_key"].reset_mock()


@pytest.mark.parametrize(
    "key",
    [
        "KEY_FACTORY",
        "KEY_SERVICE",
        "KEY_POWER",
        "KEY_POWERON",
        "KEY_POWEROFF",
        "KEY_ARTMODE",
        "KEY_MUTE",
        "KEY_HDMI1",
        "anything",
    ],
)
async def test_remote_rejects_unsafe_or_unverified_keys_before_sending(hass, setup_tv, key):
    _, ids, actions, _ = await setup_tv()
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            "remote",
            "send_command",
            {"entity_id": ids["remote"], "command": ["KEY_UP", key]},
            blocking=True,
        )
    actions["async_send_key"].assert_not_awaited()


@pytest.mark.parametrize(
    "option", [{"num_repeats": 2}, {"hold_secs": 1}, {"delay_secs": 0.2}, {"device": "other"}]
)
async def test_remote_rejects_unsupported_options(hass, setup_tv, option):
    _, ids, actions, _ = await setup_tv()
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            "remote",
            "send_command",
            {"entity_id": ids["remote"], "command": "KEY_UP", **option},
            blocking=True,
        )
    actions["async_send_key"].assert_not_awaited()


@pytest.mark.parametrize("mac", [IDENTITY.mac, None])
async def test_entity_inventory_identity_and_no_unverified_features(hass, setup_tv, mac):
    _, ids, _, entry = await setup_tv(mac=mac)
    rows = er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
    assert set(ids) == {"media_player", "remote", "switch"}
    assert len(rows) == 3
    assert {row.unique_id for row in rows} == {
        f"localthings_tv_{IDENTITY.device_id}_{suffix}"
        for suffix in ("media_player", "remote", "art_mode")
    }
    devices = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
    assert len(devices) == 1
    assert devices[0].identifiers == {(DOMAIN, f"tv_{IDENTITY.device_id}")}
    assert {row.device_id for row in rows} == {devices[0].id}
    player = hass.states.get(ids["media_player"])
    assert "source" not in player.attributes
    assert "source_list" not in player.attributes
    assert not player.attributes["supported_features"] & MediaPlayerEntityFeature.SELECT_SOURCE


async def test_state_changes_update_features_and_availability(hass, setup_tv):
    coordinator, ids, _, _ = await setup_tv()
    coordinator.async_set_updated_data(TvState(FrameMode.UNKNOWN, None, None))
    await hass.async_block_till_done()
    assert hass.states.get(ids["media_player"]).state == "unknown"
    assert hass.states.get(ids["media_player"]).attributes["supported_features"] == (
        MediaPlayerEntityFeature.TURN_ON
    )
    assert hass.states.get(ids["remote"]).state == "unavailable"
    assert hass.states.get(ids["switch"]).state == "unavailable"
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    await hass.async_block_till_done()
    assert hass.states.get(ids["media_player"]).state == "idle"
    assert hass.states.get(ids["remote"]).state == "on"
    assert hass.states.get(ids["switch"]).state == "on"


async def test_art_command_failure_does_not_fall_back_to_remote_or_power(hass, setup_tv):
    _, ids, actions, _ = await setup_tv(FrameMode.ART)
    actions["async_set_art_mode"].side_effect = HomeAssistantError("TV command failed")
    with pytest.raises(HomeAssistantError, match="TV command failed"):
        await hass.services.async_call(
            "switch", "turn_off", {"entity_id": ids["switch"]}, blocking=True
        )
    actions["async_set_art_mode"].assert_awaited_once_with(False)
    actions["async_send_key"].assert_not_awaited()
    actions["async_turn_off"].assert_not_awaited()
    assert hass.states.get(ids["switch"]).state == "on"


async def test_dhcp_registry_match_moves_host_and_preserves_identity_after_reload(hass, setup_tv):
    coordinator, ids, _, entry = await setup_tv()
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)
    before = {
        row.entity_id: (row.unique_id, row.device_id)
        for row in er.async_entries_for_config_entry(entity_registry, entry.entry_id)
    }
    sighting = DhcpServiceInfo(ip="10.0.0.99", hostname="tv", macaddress="aabbcc000202")
    matches = device_registry.async_get_devices(
        connections={(dr.CONNECTION_NETWORK_MAC, dr.format_mac(sighting.macaddress))}
    )
    assert {device.config_entry_id for device in matches} == {entry.entry_id}
    assert entry.domain == DOMAIN
    with patch.object(hass.config_entries, "async_schedule_reload") as schedule_reload:
        for _ in range(2):
            result = await hass.config_entries.flow.async_init(
                DOMAIN, context={"source": "dhcp"}, data=sighting
            )
            assert result["type"] is FlowResultType.ABORT
            assert result["reason"] == "already_configured"
    schedule_reload.assert_called_once_with(entry.entry_id)
    assert entry.data["host"] == sighting.ip

    with ExitStack() as patches:

        def reload_coordinator(*args, **kwargs):
            reloaded = TvCoordinator(*args, **kwargs)
            patches.enter_context(
                patch.object(
                    reloaded, "_async_update_data", new=AsyncMock(return_value=coordinator.state)
                )
            )
            return reloaded

        with patch("custom_components.localthings.TvCoordinator", side_effect=reload_coordinator):
            assert await hass.config_entries.async_reload(entry.entry_id)
            await hass.async_block_till_done()
            assert hass.data[DOMAIN][entry.entry_id].host == sighting.ip
            assert {
                row.entity_id: (row.unique_id, row.device_id)
                for row in er.async_entries_for_config_entry(entity_registry, entry.entry_id)
            } == before
            assert all(
                hass.states.get(entity_id).state != "unavailable" for entity_id in ids.values()
            )
            assert (
                device_registry.async_get_devices(
                    connections={(dr.CONNECTION_NETWORK_MAC, dr.format_mac(sighting.macaddress))}
                )
                == matches
            )
            assert await hass.config_entries.async_unload(entry.entry_id)


async def test_without_mac_registers_no_network_connection(hass, setup_tv):
    _, _, _, entry = await setup_tv(mac=None)
    registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(registry, entry.entry_id)
    assert len(devices) == 1
    assert devices[0].connections == set()
    assert IDENTITY.mac is not None
    assert not registry.async_get_devices(
        connections={(dr.CONNECTION_NETWORK_MAC, dr.format_mac(IDENTITY.mac))}
    )


@pytest.mark.parametrize(
    ("initial_mode", "domain", "art_mode", "final_state"),
    [
        (FrameMode.OFF, "media_player", False, "on"),
        (FrameMode.UNKNOWN, "media_player", False, "on"),
        (FrameMode.OFF, "switch", True, "idle"),
    ],
)
async def test_wake_service_without_mac_reaches_guarded_key_and_observed_mode(
    hass, initial_mode, domain, art_mode, final_state
):
    from custom_components.localthings import tv_coordinator as module

    entry = tv_entry(hass, mac=None)
    hass.config_entries.async_update_entry(
        entry, data={key: value for key, value in entry.data.items() if key != "mac"}
    )
    power_on = False
    observed_art = False

    def send_key(key):
        nonlocal power_on
        power_on = True

    def set_art_mode(enabled):
        nonlocal observed_art
        observed_art = enabled

    with (
        patch.object(module, "TvPowerClient", autospec=True) as power_factory,
        patch.object(module, "TvClient", autospec=True) as websocket_factory,
        patch.object(module, "async_read_tv_identity", new=AsyncMock()) as read_identity,
        patch.object(module, "async_read_mute", new=AsyncMock(return_value=False)) as read_mute,
        patch.object(module, "_async_wake_pause", new=AsyncMock()),
    ):
        power = power_factory.return_value
        websocket = websocket_factory.return_value
        websocket.token = "synthetic-token"
        websocket.authorization_error = None
        websocket.last_failure = None
        power.async_read_power.return_value = False
        if initial_mode is FrameMode.UNKNOWN:
            power.async_read_power.side_effect = TvConnectionError()
            read_identity.side_effect = TvConnectionError()
            power.async_read_art_mode.side_effect = TvConnectionError()
            power.async_read_picture_mode.side_effect = TvConnectionError()
            read_mute.side_effect = TvConnectionError()
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        ids = {
            row.domain: row.entity_id
            for row in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        }
        assert hass.states.get(ids["media_player"]).state == initial_mode.value
        power.async_read_power.side_effect = lambda: power_on
        read_identity.side_effect = lambda *_: TvIdentity(
            IDENTITY.device_id, IDENTITY.model, None, "on" if power_on else "off"
        )
        power.async_read_art_mode.side_effect = lambda: observed_art
        power.async_read_picture_mode.side_effect = lambda: "Ambient" if observed_art else "Movie"
        read_mute.side_effect = None
        websocket.send_key.side_effect = send_key
        power.async_set_art_mode.side_effect = set_art_mode
        power.async_read_power.reset_mock()
        read_identity.reset_mock()
        await hass.services.async_call(domain, "turn_on", {"entity_id": ids[domain]}, blocking=True)
        await hass.async_block_till_done()
        websocket.send_key.assert_called_once_with("KEY_POWER")
        if art_mode:
            power.async_set_art_mode.assert_awaited_once_with(True)
        else:
            power.async_set_art_mode.assert_not_awaited()
        websocket.read_art_mode.assert_not_called()
        websocket.set_art_mode.assert_not_called()
        power.async_turn_off.assert_not_awaited()
        assert power.async_read_power.await_count >= 2
        assert read_identity.await_count >= 1
        assert hass.states.get(ids["media_player"]).state == final_state
        assert entry.data["tv_token"] == "synthetic-token"
        assert entry.data["tv_power_token"] == "synthetic-power-token"
        assert "mac" not in entry.data
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()


@pytest.mark.parametrize("domain", ["media_player", "remote"])
async def test_turn_off_service_reaches_power_setter_and_observed_state(hass, domain):
    from custom_components.localthings import tv_coordinator as module

    entry = tv_entry(hass)
    with (
        patch.object(module, "TvPowerClient") as power_factory,
        patch.object(module, "TvClient") as websocket_factory,
        patch.object(module, "async_read_tv_identity", new=AsyncMock(return_value=IDENTITY)),
        patch.object(module, "async_read_mute", new=AsyncMock(return_value=False)),
    ):
        power = power_factory.return_value
        power.async_read_power = AsyncMock(side_effect=[True, False])
        power.async_turn_off = AsyncMock()
        websocket_factory.return_value.authorization_error = None
        websocket_factory.return_value.last_failure = None
        power.async_read_art_mode = AsyncMock(return_value=False)
        power.async_read_picture_mode = AsyncMock(return_value="Movie")
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        ids = {
            row.domain: row.entity_id
            for row in er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)
        }
        assert hass.states.get(ids["media_player"]).state == "on"
        await hass.services.async_call(
            domain, "turn_off", {"entity_id": ids[domain]}, blocking=True
        )
        assert hass.states.get(ids["media_player"]).state == "off"
        power.async_turn_off.assert_awaited_once()
        assert power.async_read_power.await_count == 2
        websocket_factory.return_value.hold_power.assert_not_called()
        websocket_factory.return_value.send_key.assert_not_called()
        assert await hass.config_entries.async_unload(entry.entry_id)
