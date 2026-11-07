"""TV setup stays separate from appliance credentials and discovery."""

import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.service_info.dhcp import DhcpServiceInfo
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.localthings.config_flow import LocalThingsConfigFlow
from custom_components.localthings.const import DOMAIN
from custom_components.localthings.tv import TvIdentity, TvPairingRejected, TvPairingTimeout

from .conftest import ENTRY_DATA, MOCK_HOST


@pytest.fixture(autouse=True)
def no_entry_setup(hass):
    with patch.object(hass.config_entries, "async_setup", new=AsyncMock(return_value=True)):
        yield


TV_DATA = {
    "transport": "tv_ws",
    "host": MOCK_HOST,
    "port": 8002,
    "tv_token": "synthetic-token",
    "tv_power_token": "synthetic-power-token",
    "device_key": "synthetic-tv",
    "model": "Frame test",
    "mac": "aa:bb:cc:00:02:02",
}
IDENTITY = TvIdentity("synthetic-tv", "Frame test", "aa:bb:cc:00:02:02")


def tv_entry(hass, **overrides):
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={**TV_DATA, **overrides},
        unique_id="localthings_tv_synthetic-tv",
        version=4,
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def tv_pairing():
    with (
        patch(
            "custom_components.localthings.config_flow.async_read_tv_identity",
            create=True,
            new=AsyncMock(return_value=IDENTITY),
        ) as identity,
        patch("custom_components.localthings.config_flow.TvClient", create=True) as client,
        patch("custom_components.localthings.config_flow.TvPowerClient", autospec=True) as power,
    ):
        client.return_value.pair.return_value = "new-synthetic-token"
        power.return_value.async_pair = AsyncMock(return_value="new-synthetic-power-token")
        yield identity, client, power


async def submit_pairing(hass, flow_id):
    result = await hass.config_entries.flow.async_configure(flow_id, {})
    if result["type"] == FlowResultType.SHOW_PROGRESS:
        await hass.async_block_till_done()
        result = await hass.config_entries.flow.async_configure(flow_id)
    if result.get("step_id") == "tv_power_pair":
        result = await hass.config_entries.flow.async_configure(flow_id, {})
        if result["type"] == FlowResultType.SHOW_PROGRESS:
            await hass.async_block_till_done()
            result = await hass.config_entries.flow.async_configure(flow_id)
    return result


async def start_tv(hass):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    return await hass.config_entries.flow.async_configure(
        result["flow_id"], {"host": MOCK_HOST, "device_kind": "tv"}
    )


@pytest.mark.parametrize("has_appliance", [False, True])
async def test_setup_choice_and_tv_bypasses_appliance_probe(hass, tv_pairing, has_appliance):
    if has_appliance:
        MockConfigEntry(domain=DOMAIN, data=ENTRY_DATA).add_to_hass(hass)
    with patch("custom_components.localthings.config_flow._probe_and_validate") as probe:
        result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == ("user_reuse" if has_appliance else "user")
        assert "device_kind" in result["data_schema"].schema
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"host": MOCK_HOST, "device_kind": "tv"}
        )
        assert result["step_id"] == "tv_pair"
        result = await submit_pairing(hass, result["flow_id"])
    assert not probe.called
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["result"].unique_id == "localthings_tv_synthetic-tv"
    assert result["data"] == {
        **TV_DATA,
        "tv_token": "new-synthetic-token",
        "tv_power_token": "new-synthetic-power-token",
    }
    tv_pairing[1].return_value.close.assert_called_once()


async def test_tv_entry_is_excluded_from_appliance_credential_reuse(hass, mock_probe):
    tv_entry(hass)
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    assert result["step_id"] == "user"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"host": "10.0.0.20"}
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert mock_probe.call_args.args[1:4] == ("", "", None)


@pytest.mark.parametrize("tv_first", [True, False])
async def test_mixed_entry_reuses_appliance_certificate(hass, mock_probe, tv_first):
    appliance = MockConfigEntry(
        domain=DOMAIN, data=ENTRY_DATA, unique_id="localthings_fridge", version=4
    )
    if tv_first:
        tv_entry(hass)
    appliance.add_to_hass(hass)
    if not tv_first:
        tv_entry(hass)
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    assert result["step_id"] == "user_reuse"
    await hass.config_entries.flow.async_configure(result["flow_id"], {"host": "10.0.0.20"})
    assert mock_probe.call_args.args[3] == (ENTRY_DATA["leaf_cert_pem"], ENTRY_DATA["leaf_key_pem"])


@pytest.mark.parametrize("source", ["user", "reauth"])
@pytest.mark.parametrize(
    "failure,key",
    [(TvPairingRejected(), "tv_pairing_rejected"), (TvPairingTimeout(), "tv_pairing_timeout")],
)
async def test_pairing_failure_preserves_entry(hass, tv_pairing, failure, key, source):
    entry = tv_entry(hass) if source == "reauth" else None
    before = dict(entry.data) if entry else None
    tv_pairing[1].return_value.pair.side_effect = failure
    result = await entry.start_reauth_flow(hass) if entry else await start_tv(hass)
    assert result["step_id"] == "tv_pair"
    result = await submit_pairing(hass, result["flow_id"])
    assert result["errors"] == {"base": key}
    if entry:
        assert dict(entry.data) == before
    else:
        assert not hass.config_entries.async_entries(DOMAIN)
    tv_pairing[1].return_value.close.assert_called_once()


async def test_reauth_different_identity_preserves_entry(hass, tv_pairing):
    entry = tv_entry(hass)
    tv_pairing[0].return_value = TvIdentity("different-tv", "Frame test", None)
    result = await entry.start_reauth_flow(hass)
    result = await submit_pairing(hass, result["flow_id"])
    assert result["errors"] == {"base": "wrong_device"}
    assert entry.data["tv_token"] == "synthetic-token"
    assert not tv_pairing[1].called


async def test_reauth_matches_identity_and_replaces_only_token(hass, tv_pairing):
    entry = tv_entry(hass)
    with patch.object(hass.config_entries, "async_schedule_reload") as reload:
        result = await entry.start_reauth_flow(hass)
        result = await submit_pairing(hass, result["flow_id"])
    assert result["reason"] == "reauth_successful"
    assert dict(entry.data) == {**TV_DATA, "tv_token": "new-synthetic-token"}
    reload.assert_called_once_with(entry.entry_id)


@pytest.mark.parametrize("matched", [True, False])
@pytest.mark.parametrize("mac", [IDENTITY.mac, None])
async def test_tv_reconfigure_requires_identity(hass, tv_pairing, matched, mac):
    entry = tv_entry(hass)
    tv_pairing[0].return_value = TvIdentity(
        IDENTITY.device_id if matched else "different-tv", IDENTITY.model, mac
    )
    with patch.object(hass.config_entries, "async_schedule_reload") as reload:
        result = await entry.start_reconfigure_flow(hass)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"host": "10.0.0.99"}
        )
    if matched:
        assert result["reason"] == "reconfigure_successful"
        assert dict(entry.data) == {**TV_DATA, "host": "10.0.0.99"}
        reload.assert_called_once_with(entry.entry_id)
    else:
        assert result["errors"] == {"base": "wrong_device"}
        assert dict(entry.data) == TV_DATA
        reload.assert_not_called()
    assert entry.unique_id == "localthings_tv_synthetic-tv"
    assert not tv_pairing[1].called


async def test_tv_dhcp_moves_host_once_without_changing_entity_identity(hass):
    entry = tv_entry(hass)
    registry = er.async_get(hass)
    entity = registry.async_get_or_create(
        "media_player", DOMAIN, "synthetic-tv_media_player", config_entry=entry
    )
    from homeassistant.config_entries import ConfigEntryState

    entry.mock_state(hass, ConfigEntryState.LOADED)
    with patch.object(hass.config_entries, "async_schedule_reload") as reload:
        for _ in range(2):
            await hass.config_entries.flow.async_init(
                DOMAIN,
                context={"source": "dhcp"},
                data=DhcpServiceInfo(ip="10.0.0.99", hostname="tv", macaddress="aabbcc000202"),
            )
    reload.assert_called_once_with(entry.entry_id)
    assert entry.data["host"] == "10.0.0.99"
    assert registry.async_get(entity.entity_id).unique_id == entity.unique_id
    entry.mock_state(hass, ConfigEntryState.NOT_LOADED)


async def test_tv_pairing_cancel_closes_client(hass, tv_pairing):
    started = threading.Event()
    release = threading.Event()

    def pair():
        started.set()
        release.wait()
        return "discarded-synthetic-token"

    tv_pairing[1].return_value.pair.side_effect = pair
    flow = LocalThingsConfigFlow()
    flow.hass = hass
    flow._host = MOCK_HOST
    task = hass.async_create_task(flow._request_tv_token())
    await hass.async_add_executor_job(started.wait)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    tv_pairing[1].return_value.close.assert_called_once()
    assert not hass.config_entries.async_entries(DOMAIN)


async def test_missing_mac_completes_setup_without_wake_address(hass, tv_pairing):
    tv_pairing[0].return_value = TvIdentity("synthetic-tv", "Frame test", None)
    result = await start_tv(hass)
    result = await submit_pairing(hass, result["flow_id"])
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {
        key: value
        for key, value in {
            **TV_DATA,
            "tv_token": "new-synthetic-token",
            "tv_power_token": "new-synthetic-power-token",
        }.items()
        if key != "mac"
    }
    assert result["result"].unique_id == "localthings_tv_synthetic-tv"


async def test_tv_pairing_dialog_cancel_does_not_save_entry(hass, tv_pairing):
    started = threading.Event()
    release = threading.Event()

    def pair():
        started.set()
        release.wait()
        return "discarded-synthetic-token"

    tv_pairing[1].return_value.pair.side_effect = pair
    result = await start_tv(hass)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["type"] == FlowResultType.SHOW_PROGRESS
    await hass.async_add_executor_job(started.wait)
    hass.config_entries.flow.async_abort(result["flow_id"])
    release.set()
    await hass.async_block_till_done()
    tv_pairing[1].return_value.close.assert_called_once()
    assert not hass.config_entries.async_entries(DOMAIN)


async def test_websocket_pairing_requires_separate_power_consent(hass, tv_pairing):
    result = await start_tv(hass)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    await hass.async_block_till_done()
    result = await hass.config_entries.flow.async_configure(result["flow_id"])
    assert result["step_id"] == "tv_power_pair"
    assert result["type"] is FlowResultType.FORM
    assert not hass.config_entries.async_entries(DOMAIN)
    tv_pairing[2].assert_not_called()
    result = await submit_pairing(hass, result["flow_id"])
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["tv_power_token"] == "new-synthetic-power-token"
    tv_pairing[2].return_value.async_pair.assert_awaited_once()


@pytest.mark.parametrize(
    "failure,key",
    [
        ("rejected", "tv_power_pairing_rejected"),
        ("timeout", "tv_power_pairing_incomplete"),
        ("network", "cannot_connect"),
    ],
)
async def test_power_reauth_failure_preserves_both_credentials(hass, tv_pairing, failure, key):
    from custom_components.localthings.tv import TvConnectionError
    from custom_components.localthings.tv_power import (
        TvPowerAuthorizationIncomplete,
        TvPowerPairingRejected,
    )

    errors = {
        "rejected": TvPowerPairingRejected(),
        "timeout": TvPowerAuthorizationIncomplete(),
        "network": TvConnectionError(),
    }
    tv_pairing[2].return_value.async_pair.side_effect = errors[failure]
    entry = tv_entry(hass)
    before = dict(entry.data)
    result = await entry.start_reauth_flow(hass, context={"tv_auth_channel": "power"})
    assert result["step_id"] == "tv_power_pair"
    result = await submit_pairing(hass, result["flow_id"])
    assert result["errors"] == {"base": key}
    assert dict(entry.data) == before
    tv_pairing[1].assert_not_called()


async def test_power_reauth_replaces_only_power_credential(hass, tv_pairing):
    entry = tv_entry(hass)
    with patch.object(hass.config_entries, "async_schedule_reload") as reload:
        result = await entry.start_reauth_flow(hass, context={"tv_auth_channel": "power"})
        assert result["step_id"] == "tv_power_pair"
        result = await submit_pairing(hass, result["flow_id"])
    assert result["reason"] == "reauth_successful"
    assert dict(entry.data) == {**TV_DATA, "tv_power_token": "new-synthetic-power-token"}
    assert "tv_auth_channel" not in entry.data
    tv_pairing[1].assert_not_called()
    reload.assert_called_once_with(entry.entry_id)


async def test_power_reauth_wrong_identity_never_pairs(hass, tv_pairing):
    entry = tv_entry(hass)
    tv_pairing[0].return_value = TvIdentity("other-tv", "Frame test", None)
    result = await entry.start_reauth_flow(hass, context={"tv_auth_channel": "power"})
    assert result["step_id"] == "tv_power_pair"
    result = await submit_pairing(hass, result["flow_id"])
    assert result["errors"] == {"base": "wrong_device"}
    assert dict(entry.data) == TV_DATA
    tv_pairing[1].assert_not_called()
    tv_pairing[2].assert_not_called()


@pytest.mark.parametrize("source", ["user", "reauth"])
async def test_power_pairing_cancel_preserves_entries_and_closes_exchange(hass, tv_pairing, source):
    started = asyncio.Event()
    closed = asyncio.Event()

    async def pair():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    tv_pairing[2].return_value.async_pair.side_effect = pair
    entry = tv_entry(hass) if source == "reauth" else None
    result = (
        await entry.start_reauth_flow(hass, context={"tv_auth_channel": "power"})
        if entry
        else await start_tv(hass)
    )
    if not entry:
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
        await hass.async_block_till_done()
        result = await hass.config_entries.flow.async_configure(result["flow_id"])
    assert result["step_id"] == "tv_power_pair"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    await started.wait()
    hass.config_entries.flow.async_abort(result["flow_id"])
    await hass.async_block_till_done()
    assert closed.is_set()
    assert dict(entry.data) == TV_DATA if entry else not hass.config_entries.async_entries(DOMAIN)


async def test_initial_power_pair_failure_keeps_websocket_token_for_explicit_retry(
    hass, tv_pairing
):
    from custom_components.localthings.tv_power import TvPowerAuthorizationIncomplete

    tv_pairing[2].return_value.async_pair.side_effect = [
        TvPowerAuthorizationIncomplete(),
        "retried-power-token",
    ]
    result = await start_tv(hass)
    result = await submit_pairing(hass, result["flow_id"])
    assert result["step_id"] == "tv_power_pair"
    assert result["errors"] == {"base": "tv_power_pairing_incomplete"}
    assert not hass.config_entries.async_entries(DOMAIN)
    result = await submit_pairing(hass, result["flow_id"])
    assert result["data"]["tv_token"] == "new-synthetic-token"
    assert result["data"]["tv_power_token"] == "retried-power-token"
    tv_pairing[1].return_value.pair.assert_called_once()
    assert tv_pairing[2].call_count == 2


async def test_initial_power_pair_checks_identity_again(hass, tv_pairing):
    result = await start_tv(hass)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    await hass.async_block_till_done()
    result = await hass.config_entries.flow.async_configure(result["flow_id"])
    assert result["step_id"] == "tv_power_pair"
    tv_pairing[0].return_value = TvIdentity("different-tv", "Frame test", None)
    result = await submit_pairing(hass, result["flow_id"])
    assert result["errors"] == {"base": "wrong_device"}
    tv_pairing[2].assert_not_called()
    assert not hass.config_entries.async_entries(DOMAIN)
