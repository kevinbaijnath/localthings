"""An exact-list timer never wakes the TV or replays a failed selection."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict
from hashlib import sha256
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError

from custom_components.localthings.tv import FrameMode, TvConnectionError, TvState
from custom_components.localthings.tv_artwork import (
    Artwork,
    ArtworkAuthorizationError,
    ArtworkError,
)
from custom_components.localthings.tv_artwork_slideshow import ArtworkSlideshow

from .test_tv_coordinator import boundaries as boundaries
from .test_tv_coordinator import coordinator as coordinator

IDS = ["MY_F0001", "MY_F0002", "MY_F0003"]


@pytest.fixture
def slideshow_env(coordinator, boundaries, monkeypatch):
    timers = []

    def later(_hass, seconds, callback):
        timer = SimpleNamespace(seconds=seconds, callback=callback, cancel=Mock())
        timers.append(timer)
        return timer.cancel

    monkeypatch.setattr(
        "custom_components.localthings.tv_artwork_slideshow.async_call_later", later
    )
    boundaries["power"].async_read_art_mode.return_value = True
    coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    client = Mock()
    current = SimpleNamespace(value="MY_OLD")

    def select(content_id):
        current.value = content_id
        return content_id

    client.select.side_effect = select
    client.current.side_effect = lambda: current.value
    client.slideshow_active.return_value = False
    coordinator.artwork._client = client
    coordinator.artwork._catalog = {value: Artwork(value, value, "v1") for value in IDS}
    return SimpleNamespace(
        runner=coordinator.slideshow,
        coordinator=coordinator,
        client=client,
        current=current,
        timers=timers,
        boundaries=boundaries,
    )


async def advance(env):
    env.timers[-1].callback(None)
    assert env.runner._task is not None
    await env.runner._task


async def test_defaults_and_ordered_cycle(slideshow_env):
    env = slideshow_env
    assert isinstance(env.runner, ArtworkSlideshow)
    assert env.runner.status() == {
        "running": False,
        "content_ids": [],
        "interval_minutes": 3,
        "shuffle": False,
        "current_content_id": None,
        "next_change_at": None,
        "reason": "Home Assistant playlist is stopped; native rotation was not checked.",
    }
    await env.runner.async_start(IDS, 3, False)
    assert env.timers[-1].seconds == 180
    assert env.runner.status()["next_change_at"].endswith("+00:00")
    assert env.coordinator.artwork.current.content_id == IDS[0]
    for _ in range(3):
        await advance(env)
    assert [call.args[0] for call in env.client.select.call_args_list] == [*IDS, IDS[0]]
    assert env.boundaries["power"].async_read_art_mode.await_count == 4
    env.boundaries["power"].async_turn_off.assert_not_awaited()


async def test_shuffle_covers_each_cycle_without_boundary_repeat(slideshow_env, monkeypatch):
    env = slideshow_env
    cycles = []

    def shuffle(values):
        if not cycles or len(cycles) % 2 == 0:
            values.reverse()
        cycles.append(list(values))

    monkeypatch.setattr(
        "custom_components.localthings.tv_artwork_slideshow.random.shuffle", shuffle
    )
    await env.runner.async_start(IDS, 1, True)
    for _ in range(8):
        await advance(env)
    chosen = [call.args[0] for call in env.client.select.call_args_list]
    for offset in (0, 3, 6):
        assert set(chosen[offset : offset + 3]) == set(IDS)
    assert all(first != second for first, second in pairwise(chosen))
    assert cycles[0][-1] == cycles[1][0]


@pytest.mark.parametrize(
    ("ids", "interval", "shuffle"),
    [
        ([], 3, False),
        ([IDS[0]], 3, False),
        ([IDS[0]] * 2, 3, False),
        ([IDS[0], "STORE_ONE"], 3, False),
        ([IDS[0], "MY-C0002"], 3, False),
        ([IDS[0], "MY_UNKNOWN"], 3, False),
        (IDS, 2, False),
        (IDS, True, False),
        (IDS, 3, "false"),
    ],
)
async def test_invalid_start_refuses_before_any_tv_request(slideshow_env, ids, interval, shuffle):
    env = slideshow_env
    with pytest.raises(HomeAssistantError):
        await env.runner.async_start(ids, interval, shuffle)
    env.client.select.assert_not_called()
    env.client.stop_slideshow.assert_not_called()
    env.boundaries["identity"].assert_not_awaited()
    assert not env.timers


async def test_duplicate_starts_keep_one_timer_and_one_selection(slideshow_env):
    env = slideshow_env
    await asyncio.gather(*(env.runner.async_start(IDS, 3, False) for _ in range(3)))
    assert len(env.timers) == 1
    env.client.select.assert_called_once_with(IDS[0])


async def test_concurrent_duplicate_start_does_not_replay_uncertain_selection(slideshow_env):
    env = slideshow_env
    entered, release = threading.Event(), threading.Event()

    def fail(_content_id):
        entered.set()
        assert release.wait(5)
        raise ArtworkError("uncertain response")

    env.client.select.side_effect = fail
    first = asyncio.create_task(env.runner.async_start(IDS, 3, False))
    assert await asyncio.to_thread(entered.wait, 5)
    second = asyncio.create_task(env.runner.async_start(IDS, 3, False))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, HomeAssistantError) for result in results)
    env.client.select.assert_called_once_with(IDS[0])
    assert not env.runner.running
    assert not env.timers


async def test_start_replaces_native_once_before_select(slideshow_env):
    env = slideshow_env
    env.client.slideshow_active.return_value = True
    await env.runner.async_start(IDS, 3, False)
    env.client.stop_slideshow.assert_called_once_with()
    names = [call[0] for call in env.client.mock_calls]
    assert names.index("stop_slideshow") < names.index("select")


@pytest.mark.parametrize("failure", ["native_state", "native_stop"])
async def test_start_requires_native_off_confirmation_before_selection(slideshow_env, failure):
    env = slideshow_env
    if failure == "native_state":
        env.client.slideshow_active.side_effect = ArtworkError("invalid status")
    else:
        env.client.slideshow_active.return_value = True
        env.client.stop_slideshow.side_effect = ArtworkError("uncertain stop")
    with pytest.raises(HomeAssistantError):
        await env.runner.async_start(IDS, 3, False)
    assert not env.runner.running
    assert not env.timers
    env.client.select.assert_not_called()
    assert env.client.stop_slideshow.call_count == (failure == "native_stop")


@pytest.mark.parametrize("mode", [FrameMode.TV, FrameMode.OFF, FrameMode.UNKNOWN])
async def test_start_requires_fresh_art_mode_despite_cached_art(slideshow_env, mode):
    env = slideshow_env
    power = env.boundaries["power"]
    power.async_read_art_mode.return_value = mode is not FrameMode.TV
    if mode == FrameMode.OFF:
        power.async_read_power.return_value = False
    if mode == FrameMode.UNKNOWN:
        power.async_read_art_mode.side_effect = TvConnectionError("offline")
    with pytest.raises(HomeAssistantError):
        await env.runner.async_start(IDS, 3, False)
    env.client.select.assert_not_called()
    env.client.stop_slideshow.assert_not_called()
    assert not env.runner.running


@pytest.mark.parametrize("conflict", ["native", "manual", "mode", "authorization", "select"])
async def test_tick_stops_on_conflict_without_retry(slideshow_env, conflict):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    if conflict == "native":
        env.client.slideshow_active.return_value = True
    elif conflict == "manual":
        env.current.value = "MY_REMOTE"
    elif conflict == "mode":
        env.boundaries["power"].async_read_art_mode.return_value = False
    elif conflict == "authorization":
        env.client.select.side_effect = ArtworkAuthorizationError("secret response")
    else:
        env.client.select.side_effect = ArtworkError("secret response")
    await advance(env)
    assert not env.runner.running
    assert env.runner.status()["next_change_at"] is None
    assert len(env.timers) == 1
    assert env.client.select.call_count == (2 if conflict in {"authorization", "select"} else 1)
    assert "secret" not in env.runner.status()["reason"]
    env.client.stop_slideshow.assert_not_called()
    if conflict == "authorization":
        with pytest.raises(HomeAssistantError, match="authorization"):
            await env.runner.async_start(IDS, 3, False)
        assert env.client.select.call_count == 2


async def test_stop_drains_admitted_selection_then_stops_native(slideshow_env):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    entered, release = threading.Event(), threading.Event()

    def blocked(content_id):
        entered.set()
        assert release.wait(5)
        env.current.value = content_id
        return content_id

    env.client.select.side_effect = blocked
    env.timers[-1].callback(None)
    await asyncio.to_thread(entered.wait, 5)
    stop = asyncio.create_task(env.runner.async_stop())
    await asyncio.sleep(0)
    assert not stop.done()
    env.client.stop_slideshow.assert_not_called()
    release.set()
    await stop
    assert not env.runner.running
    assert len(env.timers) == 1
    count = env.client.select.call_count
    env.timers[0].callback(None)
    await asyncio.sleep(0)
    assert env.client.select.call_count == count


async def test_explicit_stop_checks_native_even_without_ha_run(slideshow_env):
    env = slideshow_env
    env.client.slideshow_active.return_value = True
    await env.runner.async_stop()
    env.client.stop_slideshow.assert_called_once_with()
    env.client.select.assert_not_called()


async def test_native_stop_failure_keeps_ha_stopped_and_surfaces_error(slideshow_env):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    env.client.slideshow_active.return_value = True
    env.client.stop_slideshow.side_effect = ArtworkAuthorizationError("secret")
    with pytest.raises(HomeAssistantError):
        await env.runner.async_stop()
    assert not env.runner.running
    assert env.runner.status()["next_change_at"] is None
    assert "could not" in env.runner.status()["reason"]


async def test_unload_drains_own_work_without_native_request(slideshow_env):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    await env.coordinator.async_close()
    env.client.slideshow_active.assert_called_once_with()
    env.client.stop_slideshow.assert_not_called()
    with pytest.raises(HomeAssistantError, match="closed"):
        await env.runner.async_start(IDS, 3, False)


async def test_unload_waits_for_selection_without_stopping_native(slideshow_env):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    entered, release = threading.Event(), threading.Event()

    def blocked(content_id):
        entered.set()
        assert release.wait(5)
        return content_id

    env.client.select.side_effect = blocked
    env.timers[-1].callback(None)
    assert await asyncio.to_thread(entered.wait, 5)
    close = asyncio.create_task(env.coordinator.async_close())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    await close
    assert not env.runner.running
    assert len(env.timers) == 1
    env.client.stop_slideshow.assert_not_called()
    assert not env.coordinator._command_lock.locked()


async def test_unload_during_mode_read_prevents_first_selection(slideshow_env):
    env = slideshow_env
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_mode():
        entered.set()
        await release.wait()
        return True

    env.boundaries["power"].async_read_art_mode.side_effect = blocked_mode
    start = asyncio.create_task(env.runner.async_start(IDS, 3, False))
    await entered.wait()
    close = asyncio.create_task(env.coordinator.async_close())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    with pytest.raises(HomeAssistantError):
        await start
    await close
    env.client.select.assert_not_called()
    env.client.stop_slideshow.assert_not_called()
    assert not env.timers


async def test_manual_select_stops_playlist_before_using_cache(slideshow_env):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    await env.runner.async_manual_select(IDS[2])
    assert not env.runner.running
    assert env.current.value == IDS[2]
    env.client.stop_slideshow.assert_not_called()


async def test_deletion_guard_refuses_active_ha_playlist(slideshow_env):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    with pytest.raises(HomeAssistantError, match="Home Assistant"):
        env.coordinator.artwork._ensure_delete_available()


async def test_start_racing_final_delete_save_prevents_delete_send(slideshow_env, monkeypatch):
    env = slideshow_env
    cache = env.coordinator.artwork
    env.client.list_artwork.return_value = list(cache._catalog.values())
    env.client.thumbnail.return_value = (b"verified-thumbnail", "image/jpeg")
    cache._uploads[IDS[1]] = {
        "device_id": env.coordinator.identity.device_id,
        "metadata": asdict(cache._catalog[IDS[1]]),
        "thumbnail_sha256": sha256(b"verified-thumbnail").hexdigest(),
        "record_token": "synthetic-record",
    }
    save = AsyncMock()
    monkeypatch.setattr(cache, "_async_save", save)
    preview = await cache.async_prepare_delete(IDS[1])
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_save():
        entered.set()
        await release.wait()

    save.side_effect = blocked_save
    delete = asyncio.create_task(cache.async_delete(preview))
    await entered.wait()
    start = asyncio.create_task(env.runner.async_start(IDS, 3, False))
    await asyncio.sleep(0)
    assert env.runner.running
    assert not start.done()
    release.set()
    with pytest.raises(HomeAssistantError, match="Home Assistant"):
        await delete
    await start
    env.client.delete.assert_not_called()
    env.client.select.assert_called_once_with(IDS[0])


@pytest.mark.parametrize("mode", [FrameMode.TV, FrameMode.OFF])
async def test_polled_mode_loss_stops_run_and_art_return_does_not_resume(slideshow_env, mode):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    timer = env.timers[-1]
    with patch.object(
        env.coordinator, "_async_update_data", return_value=TvState(mode, None, None)
    ):
        await env.coordinator.async_refresh()
    assert not env.runner.running
    timer.cancel.assert_called_once_with()
    env.coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    timer.callback(None)
    await asyncio.sleep(0)
    assert not env.runner.running
    env.client.select.assert_called_once_with(IDS[0])


@pytest.mark.parametrize("uncertainty", ["unknown", "failed_poll"])
async def test_uncertain_poll_keeps_due_and_recovers_without_write(slideshow_env, uncertainty):
    from homeassistant.helpers.update_coordinator import UpdateFailed

    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    due = env.runner.status()["next_change_at"]
    generation = env.runner._generation
    timer = env.timers[-1]
    if uncertainty == "unknown":
        env.boundaries["power"].async_read_picture_mode.side_effect = TvConnectionError(
            "unavailable"
        )
        await env.coordinator.async_refresh()
        assert env.coordinator.state.mode is FrameMode.UNKNOWN
    else:
        env.coordinator.async_set_update_error(UpdateFailed("unavailable"))
    assert env.runner.running
    assert "waiting" in env.runner.status()["reason"]
    assert env.runner.status()["next_change_at"] == due
    assert env.runner._generation == generation
    timer.cancel.assert_not_called()
    env.client.select.assert_called_once_with(IDS[0])
    env.client.stop_slideshow.assert_not_called()
    env.boundaries["power"].async_read_picture_mode.return_value = "Ambient"
    env.boundaries["power"].async_read_picture_mode.side_effect = None
    await env.coordinator.async_refresh()
    assert env.coordinator.state.mode is FrameMode.ART
    assert env.runner.status()["reason"] == "Home Assistant playlist is running."
    assert env.runner.status()["next_change_at"] == due
    assert len(env.timers) == 1
    env.client.select.assert_called_once_with(IDS[0])
    await advance(env)
    assert [call.args[0] for call in env.client.select.call_args_list] == IDS[:2]
    assert len(env.timers) == 2


async def test_unknown_at_existing_due_stops_without_write_or_retry_timer(slideshow_env):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    env.boundaries["power"].async_read_picture_mode.side_effect = TvConnectionError("unavailable")
    await env.coordinator.async_refresh()
    assert env.runner.running
    await advance(env)
    assert not env.runner.running
    assert env.runner.status()["next_change_at"] is None
    assert "waiting" not in env.runner.status()["reason"]
    assert len(env.timers) == 1
    env.client.select.assert_called_once_with(IDS[0])
    env.client.current.assert_called_once_with()
    env.client.slideshow_active.assert_called_once_with()
    env.client.stop_slideshow.assert_not_called()
    env.timers[0].callback(None)
    await asyncio.sleep(0)
    env.client.select.assert_called_once_with(IDS[0])


@pytest.mark.parametrize("conflict", ["manual", "native", "authorization"])
async def test_recovered_art_still_checks_picture_native_and_authorization(slideshow_env, conflict):
    env = slideshow_env
    await env.runner.async_start(IDS, 3, False)
    env.coordinator.async_set_updated_data(TvState(FrameMode.UNKNOWN, None, None))
    env.coordinator.async_set_updated_data(TvState(FrameMode.ART, True, False))
    assert env.runner.running
    env.client.select.assert_called_once_with(IDS[0])
    if conflict == "manual":
        env.current.value = "MY_REMOTE"
    elif conflict == "native":
        env.client.slideshow_active.return_value = True
    else:
        env.coordinator.artwork._authorization_suspended = True
    await advance(env)
    assert not env.runner.running
    assert env.runner.status()["next_change_at"] is None
    assert len(env.timers) == 1
    env.client.select.assert_called_once_with(IDS[0])
    env.client.stop_slideshow.assert_not_called()


async def test_actual_ha_stop_drains_cancelled_selection_and_closes_clients(hass, slideshow_env):
    env = slideshow_env
    env.coordinator._client = env.boundaries["client"]
    await env.runner.async_start(IDS, 3, False)
    entered, release = threading.Event(), threading.Event()

    def blocked(content_id):
        entered.set()
        assert release.wait(5)
        return content_id

    env.client.select.side_effect = blocked
    env.timers[-1].callback(None)
    assert await asyncio.to_thread(entered.wait, 5)
    owned = env.runner._task
    assert owned is not None
    stopping = asyncio.Event()

    @callback
    def on_stop(_event):
        stopping.set()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, on_stop)
    stop = asyncio.create_task(hass.async_stop(force=True))
    try:
        async with asyncio.timeout(5):
            await stopping.wait()
        assert owned.cancelling()
        assert not stop.done()
        assert not env.coordinator._closed
        env.boundaries["client"].close.assert_not_called()
    finally:
        release.set()
    await stop
    assert owned.cancelled()
    assert env.coordinator._closed
    assert env.coordinator.artwork._closed
    env.boundaries["client"].close.assert_called_once_with()
    env.client.stop_slideshow.assert_not_called()
    assert not env.coordinator._command_lock.locked()


async def test_drain_preserves_caller_cancellation_even_if_owned_task_is_cancelled(slideshow_env):
    env = slideshow_env
    owned = asyncio.create_task(asyncio.Event().wait())
    env.runner._task = owned
    drain = asyncio.create_task(env.runner._async_drain("Stopped for test."))
    await asyncio.sleep(0)
    owned.cancel()
    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain
    await asyncio.gather(owned, return_exceptions=True)
