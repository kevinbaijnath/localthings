"""TV coordinator behavior with synthetic identity and mocked transports."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady, HomeAssistantError
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed

from custom_components.localthings import tv
from custom_components.localthings import tv_coordinator as coordinator_module
from custom_components.localthings.const import CONF_TV_TOKEN
from custom_components.localthings.tv import (
    FrameMode,
    TvConnectionError,
    TvIdentity,
    TvPairingRejected,
    TvState,
)
from custom_components.localthings.tv_coordinator import TvCoordinator

IDENTITY = TvIdentity("synthetic-tv", "Synthetic Frame", "02:00:00:00:00:01", "on")
HOST = "192.0.2.1"


@pytest.fixture
def boundaries() -> Iterator[dict[str, MagicMock | AsyncMock]]:
    """Prevent all TV, socket, and delay calls from crossing test boundaries."""
    with (
        patch.object(coordinator_module, "TvClient") as factory,
        patch.object(coordinator_module, "TvPowerClient", autospec=True) as power_factory,
        patch.object(
            coordinator_module, "_async_power_off_pause", new_callable=AsyncMock
        ) as power_pause,
        patch.object(
            coordinator_module, "async_read_tv_identity", new_callable=AsyncMock
        ) as identity,
        patch.object(coordinator_module, "async_read_mute", new_callable=AsyncMock) as mute,
        patch.object(coordinator_module, "async_set_mute", new_callable=AsyncMock) as set_mute,
        patch.object(coordinator_module, "_async_wake_pause", new_callable=AsyncMock) as pause,
        patch.object(
            coordinator_module, "_async_art_mode_pause", new_callable=AsyncMock
        ) as art_pause,
    ):

        async def yield_pause():
            await asyncio.sleep(0)

        art_pause.side_effect = yield_pause
        factory.return_value.last_failure = None
        factory.return_value.authorization_error = None
        power_factory.return_value.async_read_power = AsyncMock(return_value=True)
        power_factory.return_value.async_turn_off = AsyncMock()
        power_factory.return_value.async_read_art_mode = AsyncMock(return_value=False)
        power_factory.return_value.async_read_picture_mode = AsyncMock(
            side_effect=lambda: (
                "Ambient"
                if power_factory.return_value.async_read_art_mode.return_value is True
                else "Movie"
            )
        )

        async def set_art(enabled):
            power_factory.return_value.async_read_art_mode.return_value = enabled

        power_factory.return_value.async_set_art_mode = AsyncMock(side_effect=set_art)
        identity.return_value = IDENTITY
        mute.return_value = False
        yield {
            "client": factory.return_value,
            "power": power_factory.return_value,
            "power_factory": power_factory,
            "power_pause": power_pause,
            "factory": factory,
            "identity": identity,
            "mute": mute,
            "set_mute": set_mute,
            "pause": pause,
            "art_pause": art_pause,
        }


@pytest.fixture
async def coordinator(hass, boundaries):
    session = MagicMock()
    entry = MockConfigEntry(domain="localthings", data={}, state=ConfigEntryState.SETUP_IN_PROGRESS)
    entry.add_to_hass(hass)
    value = TvCoordinator(
        hass,
        session,
        host=HOST,
        token="synthetic-token",
        power_token="synthetic-power-token",
        identity=IDENTITY,
        config_entry=entry,
    )
    yield value
    await value.async_close()
    session.close.assert_not_called()


async def test_offline_refresh_does_not_infer_off(coordinator, boundaries):
    boundaries["power"].async_read_power.side_effect = TvConnectionError("power unavailable")
    await coordinator.async_refresh()
    boundaries["identity"].side_effect = TvConnectionError("offline")
    boundaries["power"].async_read_art_mode.side_effect = TvConnectionError("offline")
    boundaries["mute"].side_effect = TvConnectionError("offline")
    boundaries["power"].async_read_picture_mode.side_effect = TvConnectionError("offline")

    await coordinator.async_refresh()

    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator.connected


async def test_polling_never_opens_art_or_remote_websocket(coordinator, boundaries, caplog):
    for _ in range(4):
        await coordinator.async_refresh()
    boundaries["factory"].assert_not_called()
    boundaries["power"].async_read_art_mode.assert_awaited()
    assert "TV Art receive trace:" not in caplog.text


async def test_art_false_does_not_infer_off(coordinator, boundaries):
    boundaries["power"].async_read_power.side_effect = TvConnectionError("power unavailable")
    boundaries["identity"].side_effect = TvConnectionError("offline")

    await coordinator.async_refresh()

    assert coordinator.data.mode is FrameMode.UNKNOWN
    assert coordinator.data.art_mode is False
    assert coordinator.connected


async def test_mute_failure_does_not_erase_known_mode(coordinator, boundaries):
    boundaries["mute"].side_effect = TvConnectionError("mute unavailable")

    await coordinator.async_refresh()

    assert coordinator.data == TvState(FrameMode.TV, False, None)


async def test_art_read_failure_clears_previous_mode(coordinator, boundaries):
    await coordinator.async_refresh()
    boundaries["power"].async_read_art_mode.side_effect = TvConnectionError("art unavailable")

    await coordinator.async_refresh()

    assert coordinator.data.mode is FrameMode.UNKNOWN
    assert coordinator.data.art_mode is None


async def test_explicit_standby_is_off(coordinator, boundaries):
    boundaries["power"].async_read_power.side_effect = TvConnectionError("power unavailable")
    boundaries["identity"].return_value = replace(IDENTITY, power_state="standby")
    boundaries["power"].async_read_art_mode.side_effect = TvConnectionError("art unavailable")

    await coordinator.async_refresh()

    assert coordinator.data.mode is FrameMode.OFF


async def test_art_off_with_unknown_offline_state_does_not_wake(coordinator, boundaries):
    boundaries["power"].async_read_power.side_effect = TvConnectionError("power unavailable")
    boundaries["identity"].side_effect = TvConnectionError("offline")
    boundaries["power"].async_read_art_mode.side_effect = TvConnectionError("offline")
    boundaries["mute"].side_effect = TvConnectionError("offline")
    boundaries["power"].async_read_picture_mode.side_effect = TvConnectionError("offline")

    with pytest.raises(HomeAssistantError, match="unknown"):
        await coordinator.async_set_art_mode(False)

    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_not_called()


async def test_power_off_does_not_wake_an_off_tv(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.OFF, False, None))

    await coordinator.async_turn_off()

    boundaries["client"].send_key.assert_not_called()
    boundaries["client"].hold_power.assert_not_called()


async def test_positive_art_evidence_without_http_is_art(coordinator, boundaries):
    boundaries["identity"].side_effect = TvConnectionError("http unavailable")
    boundaries["power"].async_read_art_mode.return_value = True

    await coordinator.async_refresh()

    assert coordinator.data.mode is FrameMode.ART


async def test_identity_mismatch_discards_evidence(coordinator, boundaries):
    boundaries["identity"].return_value = replace(IDENTITY, device_id="another-synthetic-tv")

    await coordinator.async_refresh()

    assert not coordinator.last_update_success
    assert coordinator.data.mode is FrameMode.UNKNOWN
    boundaries["power"].async_read_art_mode.assert_not_called()


async def test_commands_and_reads_are_serialized(coordinator, boundaries, hass):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_identity(*args):
        started.set()
        await release.wait()
        return IDENTITY

    boundaries["identity"].side_effect = blocked_identity
    read = asyncio.create_task(coordinator.async_refresh())
    await started.wait()
    command = asyncio.create_task(coordinator.async_send_key("KEY_HOME"))
    await hass.async_block_till_done(wait_background_tasks=False)
    boundaries["client"].send_key.assert_not_called()

    release.set()
    await asyncio.gather(read, command)

    boundaries["client"].send_key.assert_called_once_with("KEY_HOME")


async def test_wake_waits_for_power_and_art_readiness(coordinator, boundaries):
    events = []
    boundaries["power"].async_read_power.side_effect = [False, False, True, True, True]
    boundaries["identity"].side_effect = [
        IDENTITY,
        TvConnectionError("not ready"),
        IDENTITY,
        IDENTITY,
        IDENTITY,
    ]
    boundaries["power"].async_read_art_mode.side_effect = [
        TvConnectionError("art not ready"),
        False,
        True,
    ]
    boundaries["power"].async_read_picture_mode.side_effect = ["Movie", "Movie", "Ambient"]
    boundaries["client"].send_key.side_effect = lambda key: events.append(key)
    boundaries["power"].async_set_art_mode.side_effect = lambda value: events.append(value)

    await coordinator.async_turn_on(art_mode=True)

    assert events == ["KEY_POWER", True]
    assert boundaries["pause"].await_count == 2
    assert coordinator.data.mode is FrameMode.ART
    boundaries["client"].hold_power.assert_not_called()


async def test_art_off_does_not_wake_an_already_off_tv(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.OFF, False, None))
    boundaries["power"].async_read_power.return_value = False

    await coordinator.async_set_art_mode(False)

    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_not_called()


async def test_turn_on_leaves_art_mode(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))

    await coordinator.async_turn_on()

    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_called_once_with(False)


async def test_wake_identity_mismatch_sends_no_art_command(coordinator, boundaries):
    boundaries["identity"].return_value = replace(IDENTITY, device_id="another-synthetic-tv")

    with pytest.raises(HomeAssistantError, match="identity"):
        await coordinator.async_turn_on(art_mode=True)

    boundaries["power"].async_set_art_mode.assert_not_called()


async def test_wake_without_mac_uses_saved_remote_permission(coordinator, boundaries):
    coordinator._identity = replace(IDENTITY, mac=None)
    boundaries["power"].async_read_power.side_effect = [False, True, True]
    await coordinator.async_turn_on()
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")


@pytest.mark.parametrize("enabled, expected", [(False, FrameMode.TV), (True, FrameMode.ART)])
async def test_wake_succeeds_without_rewriting_fresh_matching_mode(
    coordinator, boundaries, enabled, expected
):
    coordinator.async_set_updated_data(TvState(FrameMode.UNKNOWN, None, None))
    boundaries["power"].async_read_power.side_effect = [False, True]
    boundaries["power"].async_read_art_mode.return_value = enabled
    boundaries["power"].async_set_art_mode.side_effect = TvConnectionError("post-wake write failed")

    await coordinator.async_turn_on(art_mode=enabled)

    assert coordinator.data == TvState(expected, enabled, False)
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["power"].async_set_art_mode.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
async def test_wake_reports_required_mode_change_failure(coordinator, boundaries, enabled):
    boundaries["power"].async_read_power.side_effect = [False, True]
    boundaries["power"].async_read_art_mode.return_value = not enabled
    boundaries["power"].async_set_art_mode.side_effect = TvConnectionError("mode write failed")

    with pytest.raises(HomeAssistantError, match="TV mode command failed; it was not retried"):
        await coordinator.async_turn_on(art_mode=enabled)

    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator.connected
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["power"].async_set_art_mode.assert_called_once_with(enabled)


async def test_wake_deadline_is_bounded(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
    boundaries["power"].async_read_power.return_value = False

    async def pause():
        await asyncio.sleep(0)

    boundaries["pause"].side_effect = pause
    with (
        patch.object(coordinator_module, "_WAKE_TIMEOUT_SECONDS", 0),
        pytest.raises(HomeAssistantError, match="wake"),
    ):
        await coordinator.async_turn_on(art_mode=True)

    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["power"].async_set_art_mode.assert_not_called()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)


async def test_command_loss_is_not_replayed(coordinator, boundaries):
    boundaries["client"].send_key.side_effect = TvConnectionError("lost reply")

    with pytest.raises(HomeAssistantError):
        await coordinator.async_send_key("KEY_HOME")

    boundaries["client"].send_key.assert_called_once_with("KEY_HOME")
    assert coordinator.data.mode is FrameMode.UNKNOWN


async def test_successful_setter_cannot_claim_unobserved_mode(coordinator, boundaries):
    boundaries["power"].async_read_art_mode.return_value = True
    boundaries["power"].async_set_art_mode.side_effect = None
    with (
        patch.object(coordinator_module, "_ART_MODE_TIMEOUT_SECONDS", 0),
        pytest.raises(HomeAssistantError, match="confirm"),
    ):
        await coordinator.async_turn_on()
    boundaries["power"].async_set_art_mode.assert_awaited_once_with(False)
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)


async def test_mute_action_uses_desired_state_and_reads_back(coordinator, boundaries):
    boundaries["mute"].return_value = True

    await coordinator.async_set_mute(True)

    boundaries["set_mute"].assert_awaited_once_with(coordinator._session, HOST, True)
    assert coordinator.data.muted is True
    boundaries["client"].send_key.assert_not_called()


async def test_failed_mute_action_preserves_known_mode(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
    boundaries["set_mute"].side_effect = TvConnectionError("lost reply")

    with pytest.raises(HomeAssistantError):
        await coordinator.async_set_mute(True)

    assert coordinator.data == TvState(FrameMode.TV, False, None)
    boundaries["set_mute"].assert_awaited_once()


async def test_cancelled_mute_action_clears_only_mute_confidence(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
    boundaries["set_mute"].side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await coordinator.async_set_mute(True)

    assert coordinator.data == TvState(FrameMode.TV, False, None)


@pytest.mark.parametrize("readback", ["identity", "art"])
async def test_cancelled_mode_readback_clears_confidence(coordinator, boundaries, readback):
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    operation = (
        boundaries["identity"]
        if readback == "identity"
        else boundaries["power"].async_read_art_mode
    )
    if readback == "art":
        boundaries["power"].async_read_picture_mode.return_value = "Ambient"
        boundaries["power"].async_read_picture_mode.side_effect = None
    operation.side_effect = [IDENTITY if readback == "identity" else True, asyncio.CancelledError()]

    with pytest.raises(asyncio.CancelledError):
        await coordinator.async_turn_on()

    boundaries["power"].async_set_art_mode.assert_called_once_with(False)
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator._command_lock.locked()


@pytest.mark.parametrize("readback", ["identity", "art"])
async def test_cancelled_mute_readback_clears_mute_confidence(coordinator, boundaries, readback):
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
    operation = (
        boundaries["identity"]
        if readback == "identity"
        else boundaries["power"].async_read_art_mode
    )
    operation.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await coordinator.async_set_mute(True)

    boundaries["set_mute"].assert_awaited_once()
    assert coordinator.data == TvState(FrameMode.TV, False, None)
    assert not coordinator._command_lock.locked()


async def test_closed_remote_socket_is_recreated_on_next_key(coordinator, boundaries):
    first = boundaries["client"]
    second = MagicMock()
    second.last_failure = None
    second.authorization_error = None
    boundaries["factory"].side_effect = [first, second]
    first.send_key.side_effect = TvConnectionError("closed socket")
    with pytest.raises(HomeAssistantError):
        await coordinator.async_send_key("KEY_HOME")
    await coordinator.async_refresh()
    assert boundaries["factory"].call_count == 1
    await coordinator.async_send_key("KEY_HOME")
    first.send_key.assert_called_once_with("KEY_HOME")
    first.close.assert_called_once()
    second.send_key.assert_called_once_with("KEY_HOME")
    assert coordinator.data.mode is FrameMode.TV


async def test_first_refresh_needs_no_remote_authorization(coordinator, boundaries):
    boundaries["client"].read_art_mode.side_effect = TvPairingRejected("rejected")
    await coordinator.async_config_entry_first_refresh()
    assert coordinator.data.mode is FrameMode.TV
    boundaries["factory"].assert_not_called()


async def test_command_authorization_failure_requests_reauth(coordinator, boundaries):
    boundaries["client"].send_key.side_effect = TvPairingRejected("rejected")

    with pytest.raises(ConfigEntryAuthFailed):
        await coordinator.async_send_key("KEY_HOME")


async def test_failed_art_setter_does_not_keep_old_state(coordinator, boundaries):
    from custom_components.localthings.tv_power import TvPowerPairingRejected

    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    boundaries["power"].async_set_art_mode.side_effect = TvPowerPairingRejected("rejected")
    with pytest.raises(HomeAssistantError):
        await coordinator.async_turn_on()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator.connected
    boundaries["power"].async_set_art_mode.assert_awaited_once_with(False)


async def test_close_on_setup_failure(coordinator, boundaries):
    await coordinator.async_send_key("KEY_HOME")
    with (
        patch.object(
            coordinator, "_async_update_data", side_effect=RuntimeError("synthetic failure")
        ),
        pytest.raises(ConfigEntryNotReady),
    ):
        await coordinator.async_config_entry_first_refresh()

    boundaries["client"].close.assert_called_once()


async def test_close_on_core_stop(coordinator, boundaries, hass):
    await coordinator.async_send_key("KEY_HOME")

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()

    boundaries["client"].close.assert_called_once()


async def test_close_on_config_entry_unload(hass, boundaries):
    entry = MockConfigEntry(domain="localthings", data={})
    entry.add_to_hass(hass)
    value = TvCoordinator(
        hass,
        MagicMock(),
        host=HOST,
        token="synthetic-token",
        power_token="synthetic-power-token",
        identity=IDENTITY,
        config_entry=entry,
    )
    await value.async_send_key("KEY_HOME")

    await entry._async_process_on_unload(hass)
    await hass.async_block_till_done()
    await value.async_close()

    boundaries["client"].close.assert_called_once()


async def test_close_is_idempotent_and_rejects_new_commands(coordinator, boundaries):
    await coordinator.async_send_key("KEY_HOME")

    await coordinator.async_close()
    await coordinator.async_close()
    with pytest.raises(HomeAssistantError, match="closed"):
        await coordinator.async_send_key("KEY_HOME")

    boundaries["client"].close.assert_called_once()


async def test_polling_runs_only_with_subscribers(coordinator):
    assert coordinator.update_interval == timedelta(seconds=5)
    assert coordinator._unsub_refresh is None

    remove = coordinator.async_add_listener(MagicMock())
    assert coordinator._unsub_refresh is not None
    remove()

    assert coordinator._unsub_refresh is None


async def test_entity_facing_properties(coordinator):
    assert coordinator.host == HOST
    assert coordinator.identity == IDENTITY
    assert coordinator.state == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator.connected


async def test_close_waits_for_bounded_inflight_work(coordinator, boundaries, hass):
    started = asyncio.Event()
    release = asyncio.Event()
    original_executor = hass.async_add_executor_job

    async def key_work():
        started.set()
        await release.wait()

    def executor(operation, *args):
        if operation == boundaries["client"].send_key:
            return asyncio.create_task(key_work())
        return original_executor(operation, *args)

    with patch.object(hass, "async_add_executor_job", side_effect=executor):
        command = asyncio.create_task(coordinator.async_send_key("KEY_HOME"))
        await started.wait()
        command.cancel()
        close = asyncio.create_task(coordinator.async_close())
        await hass.async_block_till_done(wait_background_tasks=False)
        assert not close.done()
        boundaries["client"].close.assert_not_called()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await command
        await close

    boundaries["client"].close.assert_called_once()


@pytest.mark.parametrize("failure", [TvPairingRejected, tv.TvAuthorizationIncomplete])
async def test_remote_failure_keeps_power_art_polling_and_known_on_controls(
    coordinator, boundaries, hass, failure
):
    remove = coordinator.async_add_listener(MagicMock())
    client = boundaries["client"]
    client.send_key.side_effect = failure("synthetic-private-reply")
    with patch.object(coordinator.config_entry, "async_start_reauth_if_available") as reauth:
        with pytest.raises(HomeAssistantError):
            await coordinator.async_send_key("KEY_HOME")
        await coordinator.async_refresh()
        assert coordinator.data == TvState(FrameMode.TV, False, False)
        assert coordinator._unsub_refresh is not None
        assert coordinator.update_interval == timedelta(seconds=5)
        async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=10))
        await hass.async_block_till_done()
        assert boundaries["power"].async_read_power.await_count >= 2
        assert boundaries["power"].async_read_art_mode.await_count >= 2
        await coordinator.async_set_art_mode(True)
        assert coordinator.data.mode is FrameMode.ART
        for _ in range(2):
            with pytest.raises(HomeAssistantError):
                await coordinator.async_send_key("KEY_HOME")
        boundaries["power"].async_read_power.return_value = False
        with pytest.raises(HomeAssistantError):
            await coordinator.async_turn_on()
        client.send_key.assert_called_once_with("KEY_HOME")
        client.close.assert_called_once()
        assert boundaries["factory"].call_count == 1
        if failure is TvPairingRejected:
            reauth.assert_called_once_with(hass)
        else:
            reauth.assert_not_called()
    remove()


@pytest.mark.parametrize("rejected", [False, True])
async def test_connection_diagnostics_keep_failure_after_cleanup_without_io(
    coordinator, boundaries, rejected
):
    from custom_components.localthings.const import CONF_TRANSPORT, DOMAIN, TRANSPORT_TV_WS
    from custom_components.localthings.diagnostics import async_get_config_entry_diagnostics
    from custom_components.localthings.tv_power import (
        TvPowerAuthorizationIncomplete,
        TvPowerPairingRejected,
    )

    client = boundaries["client"]
    client.last_failure = tv.TvFailure(
        "remote", "channel_connect", "rejected" if rejected else "timeout"
    )
    client.authorization_error = tv.TvPairingRejected if rejected else tv.TvAuthorizationIncomplete
    client.send_key.side_effect = client.authorization_error("synthetic-private-reply")
    boundaries["power"].async_read_power.side_effect = (
        TvPowerPairingRejected if rejected else TvPowerAuthorizationIncomplete
    )()
    coordinator.hass.config_entries.async_update_entry(
        coordinator.config_entry, data={CONF_TRANSPORT: TRANSPORT_TV_WS}
    )
    coordinator.hass.data[DOMAIN] = {coordinator.config_entry.entry_id: coordinator}

    with (
        patch.object(coordinator.config_entry, "async_start_reauth_if_available"),
        pytest.raises(HomeAssistantError),
    ):
        await coordinator.async_send_key("KEY_HOME")
    with patch.object(coordinator.config_entry, "async_start_reauth_if_available"):
        await coordinator.async_refresh()
    assert coordinator._client is None
    read_count = boundaries["identity"].await_count
    result = await async_get_config_entry_diagnostics(coordinator.hass, coordinator.config_entry)

    assert result["connection"] == {
        "websocket_authorization": "rejected" if rejected else "incomplete",
        "power_authorization": "rejected" if rejected else "incomplete",
        "websocket_retry_suppressed": True,
        "last_websocket_failure": {
            "channel": "remote",
            "stage": "channel_connect",
            "error": "rejected" if rejected else "timeout",
        },
    }
    assert boundaries["identity"].await_count == read_count
    client.send_key.assert_called_once()
    assert boundaries["factory"].call_count == 1
    client.read_art_mode.assert_not_called()


async def test_runtime_failure_warnings_are_bounded_and_contain_only_labels(
    coordinator, boundaries, caplog
):
    client = boundaries["client"]
    failures = [
        tv.TvFailure("remote", "transport", "timeout"),
        tv.TvFailure("remote", "upgrade", "connection_error"),
        tv.TvFailure("remote", "channel_connect", "closed"),
    ]
    for failure in failures:
        client.last_failure = failure
        client.authorization_error = (
            tv.TvAuthorizationIncomplete if failure is failures[-1] else None
        )
        client.send_key.side_effect = (
            tv.TvAuthorizationIncomplete if failure is failures[-1] else TvConnectionError
        )("synthetic-private-reply")
        with pytest.raises(HomeAssistantError):
            await coordinator.async_send_key("KEY_HOME")
    await coordinator.async_refresh()

    records = [
        record
        for record in caplog.records
        if record.name == "custom_components.localthings.tv_coordinator"
        and record.levelname == "WARNING"
    ]
    assert [record.getMessage() for record in records] == [
        "TV WebSocket failure: channel=remote stage=transport error=timeout authorization=clear",
        "TV WebSocket failure: channel=remote stage=channel_connect error=closed "
        "authorization=incomplete",
    ]
    assert all(record.exc_info is None for record in records)
    assert "synthetic-private-reply" not in caplog.text
    assert "synthetic-token" not in caplog.text
    assert HOST not in caplog.text
    assert client.send_key.call_count == 3


@pytest.mark.parametrize(
    "failure", [None, TvConnectionError("read failed"), TvPairingRejected("rejected")]
)
async def test_issued_token_persists_before_client_cleanup(coordinator, boundaries, hass, failure):
    entry = coordinator.config_entry
    hass.config_entries.async_update_entry(
        entry, data={CONF_TV_TOKEN: "synthetic-token", "preserved": "value"}
    )

    def key(_key):
        boundaries["client"].token = "synthetic-issued-token"
        if failure:
            raise failure
        return False

    boundaries["client"].send_key.side_effect = key
    tokens_at_close = []
    boundaries["client"].close.side_effect = lambda: tokens_at_close.append(
        entry.data[CONF_TV_TOKEN]
    )
    if failure:
        with pytest.raises(HomeAssistantError):
            await coordinator.async_send_key("KEY_HOME")
    else:
        await coordinator.async_send_key("KEY_HOME")
    assert entry.data == {CONF_TV_TOKEN: "synthetic-issued-token", "preserved": "value"}
    if failure:
        assert tokens_at_close == ["synthetic-issued-token"]
    if isinstance(failure, TvConnectionError):
        with pytest.raises(HomeAssistantError):
            await coordinator.async_send_key("KEY_HOME")
        assert boundaries["factory"].call_args.args == (HOST, "synthetic-issued-token")
    restarted = TvCoordinator(
        hass,
        MagicMock(),
        host=HOST,
        token=entry.data[CONF_TV_TOKEN],
        power_token="synthetic-power-token",
        identity=IDENTITY,
        config_entry=entry,
    )
    try:
        if failure:
            with pytest.raises(HomeAssistantError):
                await restarted.async_send_key("KEY_HOME")
        else:
            await restarted.async_send_key("KEY_HOME")
        assert boundaries["factory"].call_args.args == (HOST, "synthetic-issued-token")
    finally:
        await restarted.async_close()


@pytest.mark.parametrize("failure", ["connection", "rejected", "incomplete"])
async def test_issued_token_persists_after_cancelled_worker_drains(
    coordinator, boundaries, hass, failure
):
    entry = coordinator.config_entry
    hass.config_entries.async_update_entry(entry, data={CONF_TV_TOKEN: "synthetic-token"})
    started = asyncio.Event()
    release = asyncio.Event()
    original_executor = hass.async_add_executor_job

    async def key_work():
        started.set()
        await release.wait()
        boundaries["client"].token = "synthetic-issued-token"
        error = {
            "connection": TvConnectionError,
            "rejected": TvPairingRejected,
            "incomplete": tv.TvAuthorizationIncomplete,
        }[failure]
        boundaries["client"].authorization_error = None if failure == "connection" else error
        raise error("later failure")

    def executor(operation, *args):
        if operation == boundaries["client"].send_key:
            return asyncio.create_task(key_work())
        return original_executor(operation, *args)

    with patch.object(hass, "async_add_executor_job", side_effect=executor):
        command = asyncio.create_task(coordinator.async_send_key("KEY_HOME"))
        await started.wait()
        command.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await command
    assert entry.data[CONF_TV_TOKEN] == "synthetic-issued-token"
    if failure != "connection":
        with pytest.raises(HomeAssistantError):
            await coordinator.async_send_key("KEY_HOME")
        boundaries["client"].send_key.assert_not_called()
        boundaries["client"].close.assert_called_once()


@pytest.mark.parametrize("failure", [TvPairingRejected, tv.TvAuthorizationIncomplete])
async def test_suspended_remote_authorization_prevents_wake_before_key(
    coordinator, boundaries, failure
):
    coordinator._authorization_error = failure
    boundaries["power"].async_read_power.return_value = False
    for _ in range(2):
        with pytest.raises(HomeAssistantError):
            await coordinator.async_turn_on()
    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_not_called()


@pytest.mark.parametrize("operation", ["async_read_art_mode", "async_read_picture_mode"])
async def test_panel_or_art_rejection_latches_power_reauth_once(
    coordinator, boundaries, operation, hass
):
    from custom_components.localthings.tv_power import TvPowerPairingRejected

    getattr(boundaries["power"], operation).side_effect = TvPowerPairingRejected("private")
    with patch.object(coordinator.config_entry, "async_start_reauth_if_available") as reauth:
        await coordinator.async_refresh()
        for _ in range(2):
            with pytest.raises(HomeAssistantError):
                await coordinator.async_set_art_mode(True)
        reauth.assert_called_once_with(hass, context={"tv_auth_channel": "power"})
        assert coordinator._power_authorization_error is TvPowerPairingRejected
        assert coordinator.update_interval == timedelta(seconds=5)
    assert getattr(boundaries["power"], operation).await_count == 1
    boundaries["power"].async_set_art_mode.assert_not_awaited()
    boundaries["factory"].assert_not_called()


async def test_dedicated_off_overrides_stale_art_and_skips_other_reads(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    boundaries["power"].async_read_power.return_value = False
    boundaries["power"].async_read_art_mode.return_value = True
    await coordinator.async_refresh()
    assert coordinator.data == TvState(FrameMode.OFF, None, None)
    assert coordinator.connected
    boundaries["identity"].assert_not_awaited()
    boundaries["power"].async_read_art_mode.assert_not_called()
    boundaries["mute"].assert_not_awaited()


async def test_dedicated_on_cannot_be_overridden_by_stale_http_off(coordinator, boundaries):
    boundaries["identity"].return_value = replace(IDENTITY, power_state="standby")
    await coordinator.async_refresh()
    assert coordinator.data.mode is FrameMode.TV


async def test_dedicated_on_without_art_remains_unknown(coordinator, boundaries):
    boundaries["identity"].return_value = replace(IDENTITY, power_state="standby")
    boundaries["power"].async_read_art_mode.side_effect = TvConnectionError("missing art")
    await coordinator.async_refresh()
    assert coordinator.data.mode is FrameMode.UNKNOWN
    assert coordinator.connected


async def test_dedicated_failure_does_not_turn_off_or_request_permission(
    coordinator, boundaries, hass
):
    boundaries["power"].async_read_power.side_effect = TvConnectionError("offline")
    boundaries["identity"].side_effect = TvConnectionError("offline")
    boundaries["power"].async_read_art_mode.side_effect = TvConnectionError("offline")
    boundaries["mute"].side_effect = TvConnectionError("offline")
    boundaries["power"].async_read_picture_mode.side_effect = TvConnectionError("offline")
    with patch.object(coordinator.config_entry, "async_start_reauth_if_available") as reauth:
        await coordinator.async_refresh()
        assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
        assert not coordinator.connected
        reauth.assert_not_called()
    boundaries["power"].async_pair.assert_not_called()


async def test_full_off_observes_delayed_transition_with_one_setter(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    boundaries["power"].async_read_power.side_effect = [True, True, False]
    await coordinator.async_turn_off()
    assert coordinator.data == TvState(FrameMode.OFF, None, None)
    boundaries["power"].async_turn_off.assert_awaited_once()
    assert boundaries["power"].async_read_power.await_count == 3
    boundaries["power_pause"].assert_awaited()
    boundaries["client"].hold_power.assert_not_called()
    boundaries["client"].send_key.assert_not_called()


@pytest.mark.parametrize("cancel", [False, True])
async def test_uncertain_off_never_replays_setter_and_clears_state(coordinator, boundaries, cancel):
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    failure = asyncio.CancelledError() if cancel else TvConnectionError("lost reply")
    boundaries["power"].async_turn_off.side_effect = failure
    with pytest.raises(asyncio.CancelledError if cancel else HomeAssistantError):
        await coordinator.async_turn_off()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    boundaries["power"].async_turn_off.assert_awaited_once()
    boundaries["power"].async_read_power.assert_not_awaited()
    assert not coordinator._command_lock.locked()


async def test_off_confirmation_deadline_clears_unconfirmed_state(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))

    async def read():
        await asyncio.sleep(0)
        return True

    boundaries["power"].async_read_power.side_effect = read
    with (
        patch.object(coordinator_module, "_POWER_OFF_TIMEOUT_SECONDS", 0),
        pytest.raises(HomeAssistantError, match="confirm"),
    ):
        await coordinator.async_turn_off()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    boundaries["power"].async_turn_off.assert_awaited_once()


@pytest.mark.parametrize("during_setter", [False, True])
async def test_power_rejection_suspends_only_power_interface(
    coordinator, boundaries, hass, during_setter
):
    from custom_components.localthings.tv_power import TvPowerPairingRejected

    error = TvPowerPairingRejected("rejected")
    if during_setter:
        coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
        boundaries["power"].async_turn_off.side_effect = error
    else:
        boundaries["power"].async_read_power.side_effect = error
    with patch.object(coordinator.config_entry, "async_start_reauth_if_available") as reauth:
        if during_setter:
            with pytest.raises(HomeAssistantError):
                await coordinator.async_turn_off()
        else:
            await coordinator.async_refresh()
        await coordinator.async_refresh()
        assert coordinator.data == TvState(FrameMode.UNKNOWN, None, False)
        await coordinator.async_send_key("KEY_HOME")
        with pytest.raises(HomeAssistantError):
            await coordinator.async_turn_off()
        reauth.assert_called_once_with(hass, context={"tv_auth_channel": "power"})
    assert boundaries["power"].async_read_power.await_count == int(not during_setter)
    assert boundaries["power"].async_turn_off.await_count == int(during_setter)
    assert coordinator.update_interval == timedelta(seconds=5)
    boundaries["client"].send_key.assert_called_once_with("KEY_HOME")


async def test_power_refusal_keeps_permission_retryable(coordinator, boundaries, hass):
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
    boundaries["power"].async_turn_off.side_effect = tv.TvError("refused")
    with patch.object(coordinator.config_entry, "async_start_reauth_if_available") as reauth:
        with pytest.raises(HomeAssistantError):
            await coordinator.async_turn_off()
        await coordinator.async_refresh()
        assert coordinator.data.mode is FrameMode.TV
        reauth.assert_not_called()
    boundaries["power"].async_read_power.assert_awaited_once()


async def test_unload_cancels_no_owned_session_and_waits_for_power(coordinator, boundaries):
    started = asyncio.Event()
    release = asyncio.Event()

    async def read():
        started.set()
        await release.wait()
        return False

    boundaries["power"].async_read_power.side_effect = read
    refresh = asyncio.create_task(coordinator.async_refresh())
    await asyncio.sleep(0)
    try:
        assert started.is_set()
        close = asyncio.create_task(coordinator.async_close())
        await asyncio.sleep(0)
        assert not close.done()
        release.set()
        await asyncio.gather(refresh, close)
        assert coordinator._closed
        coordinator._session.close.assert_not_called()
    finally:
        release.set()
        await refresh


async def test_reload_reuses_saved_power_permission_without_pairing(hass, boundaries):
    session = MagicMock()
    entry = MockConfigEntry(
        domain="localthings",
        data={"tv_token": "saved-ws-token", "tv_power_token": "saved-power-token"},
    )
    entry.add_to_hass(hass)
    for _ in range(2):
        coordinator = TvCoordinator(
            hass,
            session,
            host=HOST,
            token=entry.data["tv_token"],
            power_token=entry.data["tv_power_token"],
            identity=IDENTITY,
            config_entry=entry,
        )
        await coordinator.async_refresh()
        assert coordinator.data.mode is FrameMode.TV
        await coordinator.async_close()
    assert boundaries["power_factory"].call_count == 2
    assert all(
        call.args == (session, HOST, "saved-power-token")
        for call in boundaries["power_factory"].call_args_list
    )
    boundaries["power"].async_pair.assert_not_called()
    assert boundaries["power"].async_read_power.await_count == 2
    assert dict(entry.data) == {"tv_token": "saved-ws-token", "tv_power_token": "saved-power-token"}
    session.close.assert_not_called()


async def test_cancelled_off_observation_clears_state_without_replaying_setter(
    coordinator, boundaries
):
    started = asyncio.Event()

    async def read():
        started.set()
        await asyncio.Event().wait()

    boundaries["power"].async_read_power.side_effect = read
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    command = asyncio.create_task(coordinator.async_turn_off())
    await started.wait()
    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator._command_lock.locked()
    boundaries["power"].async_turn_off.assert_awaited_once()
    boundaries["client"].hold_power.assert_not_called()


async def test_off_observation_retries_reads_after_network_failure(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
    boundaries["power"].async_read_power.side_effect = [TvConnectionError("lost status"), False]
    await coordinator.async_turn_off()
    assert coordinator.data == TvState(FrameMode.OFF, None, None)
    boundaries["power"].async_turn_off.assert_awaited_once()
    assert boundaries["power"].async_read_power.await_count == 2
    boundaries["power_pause"].assert_awaited_once()


async def test_incomplete_power_authorization_suspends_power_without_repair_prompt(
    coordinator, boundaries, hass
):
    from custom_components.localthings.tv_power import TvPowerAuthorizationIncomplete

    boundaries["power"].async_read_power.side_effect = TvPowerAuthorizationIncomplete()
    with patch.object(coordinator.config_entry, "async_start_reauth_if_available") as reauth:
        await coordinator.async_refresh()
        await coordinator.async_refresh()
        assert coordinator.data == TvState(FrameMode.UNKNOWN, None, False)
        with pytest.raises(HomeAssistantError):
            await coordinator.async_turn_off()
        reauth.assert_not_called()
    boundaries["power"].async_read_power.assert_awaited_once()
    boundaries["power"].async_turn_off.assert_not_awaited()


@pytest.mark.parametrize("cached", [FrameMode.TV, FrameMode.ART, FrameMode.OFF])
async def test_fresh_off_wakes_once_despite_stale_cache(coordinator, boundaries, cached):
    coordinator.async_set_updated_data(TvState(cached, cached is FrameMode.ART, False))
    boundaries["power"].async_read_power.side_effect = [False, True, True]
    await coordinator.async_turn_on()
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["client"].hold_power.assert_not_called()
    boundaries["power"].async_turn_off.assert_not_awaited()
    boundaries["power"].async_set_art_mode.assert_not_called()
    assert coordinator.data == TvState(FrameMode.TV, False, False)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("cached", [FrameMode.TV, FrameMode.ART, FrameMode.OFF])
async def test_art_directions_use_fresh_evidence(coordinator, boundaries, enabled, cached):
    coordinator.async_set_updated_data(TvState(cached, cached is FrameMode.ART, False))
    boundaries["power"].async_read_art_mode.return_value = not enabled
    await coordinator.async_set_art_mode(enabled)
    boundaries["power"].async_read_power.assert_awaited()
    boundaries["identity"].assert_awaited()
    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_called_once_with(enabled)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("identity_available", [False, True])
async def test_missing_power_still_requires_identity_on_before_art_command(
    coordinator, boundaries, enabled, identity_available
):
    boundaries["power"].async_read_power.side_effect = TvConnectionError("power unavailable")
    boundaries["power"].async_read_art_mode.return_value = not enabled
    if not identity_available:
        boundaries["identity"].side_effect = TvConnectionError("identity unavailable")
        with pytest.raises(HomeAssistantError, match="unknown"):
            await coordinator.async_set_art_mode(enabled)
        boundaries["power"].async_set_art_mode.assert_not_awaited()
    else:
        await coordinator.async_set_art_mode(enabled)
        boundaries["power"].async_set_art_mode.assert_awaited_once_with(enabled)
    boundaries["client"].send_key.assert_not_called()


@pytest.mark.parametrize("command", ["tv", "art_on", "art_off"])
@pytest.mark.parametrize("conflict", [False, True])
async def test_unknown_or_conflicting_fresh_evidence_cannot_send_a_key_or_setter(
    coordinator, boundaries, command, conflict
):
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    boundaries["power"].async_read_power.side_effect = TvConnectionError("power unavailable")
    if conflict:
        boundaries["identity"].return_value = replace(IDENTITY, power_state="standby")
    else:
        boundaries["identity"].side_effect = TvConnectionError("identity unavailable")
    boundaries["power"].async_read_art_mode.return_value = conflict
    action = {
        "tv": coordinator.async_turn_on,
        "art_on": lambda: coordinator.async_set_art_mode(True),
        "art_off": lambda: coordinator.async_set_art_mode(False),
    }[command]
    with pytest.raises(HomeAssistantError, match="unknown"):
        await action()
    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_not_called()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)


@pytest.mark.parametrize("cached", [FrameMode.TV, FrameMode.ART, FrameMode.OFF])
@pytest.mark.parametrize("fresh_off", [False, True])
async def test_art_off_uses_fresh_state_before_returning(
    coordinator, boundaries, cached, fresh_off
):
    coordinator.async_set_updated_data(TvState(cached, cached is FrameMode.ART, False))
    boundaries["power"].async_read_power.return_value = not fresh_off
    await coordinator.async_set_art_mode(False)
    boundaries["power"].async_read_power.assert_awaited_once()
    boundaries["identity"].assert_awaited_once()
    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_not_called()
    assert coordinator.data.mode is (FrameMode.OFF if fresh_off else FrameMode.TV)


@pytest.mark.parametrize("power", [False, True, None])
async def test_identity_mismatch_aborts_fresh_mode_preflight(coordinator, boundaries, power):
    if power is None:
        boundaries["power"].async_read_power.side_effect = TvConnectionError("power unavailable")
    else:
        boundaries["power"].async_read_power.return_value = power
    boundaries["identity"].return_value = replace(IDENTITY, device_id="another-synthetic-tv")
    boundaries["power"].async_read_art_mode.return_value = True
    with pytest.raises(HomeAssistantError, match="identity"):
        await coordinator.async_turn_on()
    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_not_called()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)


@pytest.mark.parametrize("cancel", [False, True])
async def test_uncertain_wake_key_never_replays_or_sets_mode(coordinator, boundaries, cancel):
    coordinator.async_set_updated_data(TvState(FrameMode.TV, False, False))
    boundaries["power"].async_read_power.return_value = False
    boundaries["client"].send_key.side_effect = (
        asyncio.CancelledError() if cancel else TvConnectionError("lost wake reply")
    )
    with pytest.raises(asyncio.CancelledError if cancel else HomeAssistantError):
        await coordinator.async_turn_on()
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["power"].async_read_power.assert_awaited_once()
    boundaries["power"].async_set_art_mode.assert_not_called()
    boundaries["client"].hold_power.assert_not_called()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator._command_lock.locked()


@pytest.mark.parametrize("failure", ["rejected", "incomplete"])
async def test_power_authorization_suspension_stops_wake_observation(
    coordinator, boundaries, failure
):
    from custom_components.localthings.tv_power import (
        TvPowerAuthorizationIncomplete,
        TvPowerPairingRejected,
    )

    error = TvPowerPairingRejected if failure == "rejected" else TvPowerAuthorizationIncomplete
    boundaries["power"].async_read_power.side_effect = [False, error("suspended")]
    with pytest.raises(HomeAssistantError):
        await coordinator.async_turn_on()
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["power"].async_set_art_mode.assert_not_awaited()
    boundaries["pause"].assert_not_awaited()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)


async def test_wake_observation_requires_dedicated_on_even_with_fresh_art(coordinator, boundaries):
    boundaries["power"].async_read_power.side_effect = [
        False,
        TvConnectionError("power temporarily unavailable"),
        True,
        True,
    ]
    boundaries["power"].async_read_art_mode.return_value = True
    await coordinator.async_turn_on()
    boundaries["pause"].assert_awaited_once()
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["power"].async_set_art_mode.assert_called_once_with(False)


async def test_cancelled_wake_observation_clears_confidence(coordinator, boundaries):
    started = asyncio.Event()

    async def pause():
        started.set()
        await asyncio.Event().wait()

    boundaries["power"].async_read_power.return_value = False
    boundaries["pause"].side_effect = pause
    command = asyncio.create_task(coordinator.async_turn_on())
    await started.wait()
    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["power"].async_set_art_mode.assert_not_called()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator._command_lock.locked()


@pytest.mark.parametrize("art_mode", [False, True])
async def test_fresh_on_never_wakes_from_cached_off(coordinator, boundaries, art_mode):
    coordinator.async_set_updated_data(TvState(FrameMode.OFF, None, None))
    await coordinator.async_turn_on(art_mode=art_mode)
    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_called_once_with(art_mode)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", ["rejected", "incomplete"])
async def test_power_permission_failure_blocks_art_even_with_identity_on(
    coordinator, boundaries, enabled, failure
):
    from custom_components.localthings.tv_power import (
        TvPowerAuthorizationIncomplete,
        TvPowerPairingRejected,
    )

    error = TvPowerPairingRejected if failure == "rejected" else TvPowerAuthorizationIncomplete
    boundaries["power"].async_read_power.side_effect = error("power unavailable")
    with pytest.raises(HomeAssistantError):
        await coordinator.async_set_art_mode(enabled)
    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_not_awaited()
    boundaries["power"].async_read_art_mode.assert_not_awaited()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)


async def test_identity_mismatch_during_wake_observation_aborts_without_replay(
    coordinator, boundaries
):
    boundaries["power"].async_read_power.side_effect = [False, True]
    boundaries["identity"].side_effect = [
        IDENTITY,
        replace(IDENTITY, device_id="another-synthetic-tv"),
    ]
    with pytest.raises(HomeAssistantError, match="identity"):
        await coordinator.async_turn_on()
    boundaries["client"].send_key.assert_called_once_with("KEY_POWER")
    boundaries["power"].async_set_art_mode.assert_not_called()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)


async def test_cancelled_mode_preflight_clears_cache_without_transmission(coordinator, boundaries):
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    boundaries["identity"].side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await coordinator.async_set_art_mode(False)
    boundaries["client"].send_key.assert_not_called()
    boundaries["power"].async_set_art_mode.assert_not_called()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator._command_lock.locked()


@pytest.mark.parametrize(
    "art,picture,expected",
    [
        (True, "Ambient", FrameMode.ART),
        (False, "Movie", FrameMode.TV),
        (True, "Movie", FrameMode.UNKNOWN),
        (False, "Ambient", FrameMode.UNKNOWN),
        (None, "Ambient", FrameMode.UNKNOWN),
        (True, "", FrameMode.UNKNOWN),
    ],
)
async def test_ip_art_requires_consistent_panel_evidence(
    coordinator, boundaries, art, picture, expected
):
    boundaries["power"].async_read_art_mode = AsyncMock(return_value=art)
    boundaries["power"].async_read_picture_mode = AsyncMock(return_value=picture)
    await coordinator.async_refresh()
    assert coordinator.data.mode is expected
    boundaries["factory"].assert_not_called()


async def test_ip_art_acknowledgement_requires_fresh_confirmation(coordinator, boundaries):
    boundaries["power"].async_read_art_mode = AsyncMock(return_value=False)
    boundaries["power"].async_read_picture_mode = AsyncMock(return_value="Movie")
    boundaries["power"].async_set_art_mode = AsyncMock()
    with (
        patch.object(coordinator_module, "_ART_MODE_TIMEOUT_SECONDS", 0, create=True),
        pytest.raises(HomeAssistantError, match="confirm"),
    ):
        await coordinator.async_set_art_mode(True)
    boundaries["power"].async_set_art_mode.assert_awaited_once_with(True)
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    boundaries["factory"].assert_not_called()


@pytest.mark.parametrize("operation", ["async_read_art_mode", "async_read_picture_mode"])
async def test_missing_ip_mode_evidence_is_unknown(coordinator, boundaries, operation):
    getattr(boundaries["power"], operation).side_effect = TvConnectionError("unavailable")
    await coordinator.async_refresh()
    assert coordinator.data.mode is FrameMode.UNKNOWN
    assert coordinator.connected
    boundaries["factory"].assert_not_called()


@pytest.mark.parametrize("cancel", [False, True])
async def test_art_setter_loss_never_replays_or_falls_back(coordinator, boundaries, cancel):
    boundaries["power"].async_set_art_mode.side_effect = (
        asyncio.CancelledError() if cancel else TvConnectionError("private reply")
    )
    with pytest.raises(asyncio.CancelledError if cancel else HomeAssistantError):
        await coordinator.async_set_art_mode(True)
    boundaries["power"].async_set_art_mode.assert_awaited_once_with(True)
    boundaries["power"].async_read_art_mode.assert_awaited_once()
    boundaries["factory"].assert_not_called()
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator._command_lock.locked()


async def test_art_confirmation_retains_command_lock_and_closes_cleanly(
    coordinator, boundaries, hass
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def setter(_enabled):
        started.set()
        await release.wait()
        boundaries["power"].async_read_art_mode.return_value = True

    boundaries["power"].async_set_art_mode.side_effect = setter
    command = asyncio.create_task(coordinator.async_set_art_mode(True))
    await started.wait()
    refresh = asyncio.create_task(coordinator.async_refresh())
    close = asyncio.create_task(coordinator.async_close())
    await hass.async_block_till_done(wait_background_tasks=False)
    assert coordinator._command_lock.locked()
    assert not close.done()
    boundaries["power"].async_read_power.assert_awaited_once()
    release.set()
    await command
    await asyncio.gather(refresh, close)
    assert coordinator._closed
    assert not coordinator._command_lock.locked()
    coordinator._session.close.assert_not_called()
    boundaries["factory"].assert_not_called()


@pytest.mark.parametrize("enabled, expected", [(False, FrameMode.TV), (True, FrameMode.ART)])
async def test_art_confirmation_cannot_wait_for_optional_mute(
    coordinator, boundaries, enabled, expected
):
    setter_sent = False
    mute_started = asyncio.Event()
    boundaries["power"].async_read_art_mode.return_value = not enabled

    async def setter(value):
        nonlocal setter_sent
        setter_sent = True
        boundaries["power"].async_read_art_mode.return_value = value

    async def mute(*_):
        if setter_sent:
            mute_started.set()
            await asyncio.Event().wait()
        return False

    boundaries["power"].async_set_art_mode.side_effect = setter
    boundaries["mute"].side_effect = mute
    command = asyncio.create_task(coordinator.async_set_art_mode(enabled))
    blocked_mute = asyncio.create_task(mute_started.wait())
    try:
        done, _ = await asyncio.wait({command, blocked_mute}, return_when=asyncio.FIRST_COMPLETED)
        assert command in done, "Optional mute prevented mode confirmation"
        await command
        assert coordinator.data == TvState(expected, enabled, None)
        boundaries["power"].async_set_art_mode.assert_awaited_once_with(enabled)
        assert boundaries["identity"].await_count == 2
        assert boundaries["power"].async_read_power.await_count == 2
        boundaries["mute"].assert_awaited_once()
        assert not coordinator._command_lock.locked()
        boundaries["factory"].assert_not_called()
        boundaries["mute"].side_effect = None
        boundaries["mute"].return_value = True
        await coordinator.async_refresh()
        assert coordinator.data == TvState(expected, enabled, True)
    finally:
        for task in (command, blocked_mute):
            task.cancel()
        await asyncio.gather(command, blocked_mute, return_exceptions=True)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", ["missing", "conflicting", "delayed", "identity"])
async def test_art_confirmation_still_requires_fresh_matching_evidence(
    coordinator, boundaries, enabled, failure
):
    setter_sent = False

    async def setter(value):
        nonlocal setter_sent
        setter_sent = True

    async def identity(*_):
        if setter_sent and failure == "identity":
            return replace(IDENTITY, device_id="another-synthetic-tv")
        return IDENTITY

    async def art():
        if setter_sent and failure == "missing":
            raise TvConnectionError("Art evidence unavailable")
        return enabled if setter_sent else not enabled

    async def picture():
        if setter_sent and failure == "delayed":
            await asyncio.Event().wait()
        ambient = enabled if setter_sent else not enabled
        if setter_sent and failure == "conflicting":
            ambient = not ambient
        return "Ambient" if ambient else "Movie"

    boundaries["power"].async_set_art_mode.side_effect = setter
    boundaries["power"].async_read_art_mode.side_effect = art
    boundaries["power"].async_read_picture_mode.side_effect = picture
    boundaries["identity"].side_effect = identity
    with (
        patch.object(coordinator_module, "_ART_MODE_TIMEOUT_SECONDS", 0),
        pytest.raises(HomeAssistantError),
    ):
        await coordinator.async_set_art_mode(enabled)
    assert coordinator.data == TvState(FrameMode.UNKNOWN, None, None)
    assert not coordinator._command_lock.locked()
    boundaries["power"].async_set_art_mode.assert_awaited_once_with(enabled)
    boundaries["mute"].assert_awaited_once()
    boundaries["factory"].assert_not_called()
