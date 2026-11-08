"""TV lifecycle never reads or mutates appliance state."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp.test_utils import make_mocked_request
from homeassistant.config_entries import ConfigEntryState
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.http import current_request

from custom_components import localthings as integration
from custom_components.localthings.const import DOMAIN
from custom_components.localthings.diagnostics import async_get_config_entry_diagnostics
from custom_components.localthings.tv import FrameMode, TvState

from .test_tv_config_flow import IDENTITY, tv_entry


@pytest.fixture
def tv_coordinator():
    coord = Mock(
        spec=[
            "async_config_entry_first_refresh",
            "async_close",
            "identity",
            "state",
            "connected",
            "connection_diagnostics",
            "artwork",
            "albums",
        ]
    )
    coord.async_config_entry_first_refresh = AsyncMock()
    coord.async_close = AsyncMock()
    coord.artwork = Mock(async_load=AsyncMock())
    coord.albums = Mock(async_load=AsyncMock())
    coord.identity = IDENTITY
    coord.state = TvState(FrameMode.ART, True, None)
    coord.connected = True
    coord.connection_diagnostics = {
        "websocket_authorization": "clear",
        "power_authorization": "clear",
        "websocket_retry_suppressed": False,
        "last_websocket_failure": None,
    }
    with patch(
        "custom_components.localthings.TvCoordinator", create=True, return_value=coord
    ) as factory:
        yield coord, factory


async def test_tv_setup_forwards_only_tv_platforms(hass, tv_coordinator):
    entry = tv_entry(hass)
    with (
        patch.object(hass.config_entries, "async_forward_entry_setups", new=AsyncMock()) as forward,
        patch("custom_components.localthings.LocalThingsCoordinator") as appliance,
        patch("custom_components.localthings._drop_sensors_superseded_by_selects") as cleanup,
        patch("custom_components.localthings.snapshot_store") as store,
    ):
        assert await integration.async_setup_entry(hass, entry)
    forward.assert_awaited_once_with(entry, ["media_player", "remote", "switch", "button"])
    tv_coordinator[0].artwork.async_load.assert_awaited_once_with()
    tv_coordinator[0].albums.async_load.assert_awaited_once_with()
    assert not appliance.called and not cleanup.called and not store.called
    assert tv_coordinator[1].call_args.kwargs["token"] == "synthetic-token"
    assert tv_coordinator[1].call_args.kwargs["power_token"] == "synthetic-power-token"
    assert tv_coordinator[1].call_args.kwargs["identity"] == IDENTITY
    assert tv_coordinator[1].call_args.kwargs["config_entry"] is entry
    assert not tv_coordinator[1].call_args.args[1].closed
    with patch.object(
        hass.config_entries, "async_unload_platforms", new=AsyncMock(return_value=True)
    ) as unload:
        assert await integration.async_unload_entry(hass, entry)
    unload.assert_awaited_once_with(entry, ["media_player", "remote", "switch", "button"])
    tv_coordinator[0].async_close.assert_awaited_once()


@pytest.mark.parametrize("where", ["refresh", "forward"])
async def test_tv_setup_failure_closes_resources(hass, tv_coordinator, where):
    entry = tv_entry(hass)
    failure = (
        ConfigEntryNotReady("offline") if where == "refresh" else RuntimeError("platform failed")
    )
    if where == "refresh":
        tv_coordinator[0].async_config_entry_first_refresh.side_effect = failure
    with (
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            new=AsyncMock(side_effect=failure if where == "forward" else None),
        ),
        pytest.raises(type(failure)),
    ):
        await integration.async_setup_entry(hass, entry)
    tv_coordinator[0].async_close.assert_awaited_once()
    assert entry.entry_id not in hass.data.get(DOMAIN, {})


async def test_failed_tv_unload_keeps_live_coordinator(hass, tv_coordinator):
    entry = tv_entry(hass)
    hass.data[DOMAIN] = {entry.entry_id: tv_coordinator[0]}
    with patch.object(
        hass.config_entries, "async_unload_platforms", new=AsyncMock(return_value=False)
    ):
        assert not await integration.async_unload_entry(hass, entry)
    assert hass.data[DOMAIN][entry.entry_id] is tv_coordinator[0]
    tv_coordinator[0].async_close.assert_not_called()


async def test_tv_options_do_not_touch_appliance_attributes(hass, tv_coordinator):
    entry = tv_entry(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    hass.data[DOMAIN] = {entry.entry_id: tv_coordinator[0]}
    request = make_mocked_request("POST", "/api/config/config_entries/options/flow")
    request["hass_user"] = SimpleNamespace(id="admin", is_admin=True)
    request["hass_refresh_token_id"] = "refresh"
    token = current_request.set(request)
    try:
        result = await hass.config_entries.options.async_init(entry.entry_id)
    finally:
        current_request.reset(token)
    assert result["type"] == "menu"
    assert result["menu_options"] == [
        "upload_artwork",
        "delete_artwork",
        "delete_artworks",
        "delete_status",
        "delete_artworks_status",
        "upload_finish",
    ]
    await integration._async_options_updated(hass, entry)


async def test_tv_diagnostics_allowlist(hass, tv_coordinator):
    entry = tv_entry(hass, account={"secret": "raw-frame"})
    hass.data[DOMAIN] = {entry.entry_id: tv_coordinator[0]}
    result = await async_get_config_entry_diagnostics(hass, entry)
    assert result == {
        "transport": "tv_ws",
        "model": "Frame test",
        "connected": True,
        "state": {"mode": "art", "art_mode": True, "muted": None},
        "connection": {
            "websocket_authorization": "clear",
            "power_authorization": "clear",
            "websocket_retry_suppressed": False,
            "last_websocket_failure": None,
        },
    }
    encoded = json.dumps(result)
    for secret in (
        "synthetic-token",
        "synthetic-power-token",
        "synthetic-tv",
        "aa:bb",
        "raw-frame",
        "account",
        "10.0.0",
    ):
        assert secret not in encoded


@pytest.mark.parametrize("loaded", [True, False])
async def test_tv_device_removal_keeps_live_device_only(hass, tv_coordinator, loaded):
    entry = tv_entry(hass)
    if loaded:
        hass.data[DOMAIN] = {entry.entry_id: tv_coordinator[0]}
    registry = dr.async_get(hass)
    live = registry.async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={(DOMAIN, "tv_synthetic-tv")}
    )
    stale = registry.async_get_or_create(
        config_entry_id=entry.entry_id, identifiers={(DOMAIN, "tv_stale")}
    )
    assert await integration.async_remove_config_entry_device(hass, entry, live) is (not loaded)
    assert await integration.async_remove_config_entry_device(hass, entry, stale)


async def test_tv_entry_removal_does_not_touch_appliance_stores(hass):
    entry = tv_entry(hass)
    with (
        patch("custom_components.localthings.snapshot_store") as snapshots,
        patch("custom_components.localthings.mode_spec_store") as modes,
        patch("custom_components.localthings.album_store") as albums,
    ):
        albums.return_value.async_remove = AsyncMock()
        await integration.async_remove_entry(hass, entry)
    snapshots.assert_not_called()
    modes.assert_not_called()
    albums.assert_called_once_with(hass, entry.entry_id)
    albums.return_value.async_remove.assert_awaited_once_with()


@pytest.mark.parametrize("version,success", [(4, True), (5, False)])
async def test_tv_migration_version_guard(hass, version, success):
    entry = tv_entry(hass)
    hass.config_entries.async_update_entry(entry, version=version)
    before = dict(entry.data)
    with (
        patch("custom_components.localthings._repair_placeholder_keys") as repair,
        patch("custom_components.localthings._relabel_particulate_statistics") as statistics,
    ):
        assert await integration.async_migrate_entry(hass, entry) is success
    assert dict(entry.data) == before
    assert entry.version == version
    repair.assert_not_called()
    statistics.assert_not_called()


@pytest.mark.parametrize("credential", [None, "", "absent"])
async def test_missing_power_credential_requests_only_explicit_form(
    hass, tv_coordinator, credential
):
    entry = tv_entry(hass, tv_power_token=credential)
    if credential == "absent":
        hass.config_entries.async_update_entry(
            entry, data={key: value for key, value in entry.data.items() if key != "tv_power_token"}
        )
    before = dict(entry.data)
    with patch.object(entry, "async_start_reauth_if_available") as reauth:
        with pytest.raises(ConfigEntryAuthFailed):
            await integration.async_setup_entry(hass, entry)
        reauth.assert_called_once_with(hass, context={"tv_auth_channel": "power"})
    tv_coordinator[1].assert_not_called()
    assert dict(entry.data) == before
    result = await entry.start_reauth_flow(hass, context={"tv_auth_channel": "power"})
    assert result["step_id"] == "tv_power_pair"
    assert result["type"] == "form"
