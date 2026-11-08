"""Persistent artwork data stays usable without contacting an offline TV."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from importlib import import_module
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.core import CoreState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util.file import WriteError
from homeassistant.util.json import SerializationError

# Keep the real write method before the Home Assistant fixture patches storage.
_REAL_STORE_WRITE_DATA = Store._async_write_data

if TYPE_CHECKING:
    from custom_components.localthings.tv_artwork import TvArtworkClient


def test_cache_module_exists():
    assert import_module("custom_components.localthings.tv_artwork_cache").ArtworkCache


@pytest.fixture
def cache_env(hass, monkeypatch):
    module = import_module("custom_components.localthings.tv_artwork_cache")
    transport = import_module("custom_components.localthings.tv_artwork")
    clock = SimpleNamespace(now=datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(module.dt_util, "utcnow", lambda: clock.now)
    saved = SimpleNamespace(payload=None)

    async def save(payload):
        saved.payload = deepcopy(payload)

    store = SimpleNamespace(
        async_load=AsyncMock(side_effect=lambda: deepcopy(saved.payload)),
        async_save=AsyncMock(side_effect=save),
        async_remove=AsyncMock(),
    )
    monkeypatch.setattr(module, "artwork_store", lambda *_: store)
    timer = SimpleNamespace(callback=None, unsubscribe=Mock())

    def register(_hass, callback, interval):
        assert interval == timedelta(hours=1)
        timer.callback = callback
        return timer.unsubscribe

    monkeypatch.setattr(module, "async_track_time_interval", register)
    artwork = transport.Artwork("picture-a", "First picture", "revision-1")
    other = transport.Artwork("picture-b", "Second picture", "revision-1")
    client = SimpleNamespace(
        list_artwork=Mock(return_value=[artwork, other]),
        thumbnail=Mock(return_value=(b"first-thumbnail", "image/jpeg")),
        current=Mock(return_value="picture-a"),
        select=Mock(side_effect=lambda content_id: content_id),
        upload=Mock(return_value="uploaded-picture"),
    )
    awake = SimpleNamespace(value=True)

    async def run(operation):
        return operation()

    runner = AsyncMock(side_effect=run)

    async def verified(operation):
        return await operation()

    def create():
        return module.ArtworkCache(
            hass,
            "synthetic-entry",
            cast("TvArtworkClient", client),
            runner,
            lambda: awake.value,
            device_id="synthetic-tv",
            run_verified_operation=verified,
        )

    return SimpleNamespace(
        module=module,
        transport=transport,
        clock=clock,
        store=store,
        saved=saved,
        timer=timer,
        client=client,
        awake=awake,
        runner=runner,
        create=create,
        artwork=artwork,
        other=other,
    )


async def test_load_only_reads_storage(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    env.client.list_artwork.assert_not_called()
    env.runner.assert_not_called()
    await cache.async_close()
    env.timer.unsubscribe.assert_called_once()


async def test_current_read_is_transient_and_deduplicated(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await asyncio.gather(cache.async_check_current(), cache.async_check_current())
    env.client.current.assert_called_once_with()
    assert cache.current.content_id == "picture-a"
    assert cache.current.checked_at == env.clock.now
    assert cache.current_title() == "First picture"
    assert "current" not in env.saved.payload
    env.clock.now += timedelta(seconds=29)
    await cache.async_check_current()
    env.client.current.assert_called_once_with()
    env.clock.now += timedelta(seconds=1)
    env.client.current.return_value = "picture-b"
    await cache.async_check_current()
    assert env.client.current.call_count == 2
    assert cache.current.content_id == "picture-b"
    await cache.async_close()
    reloaded = env.create()
    await reloaded.async_load()
    assert reloaded.current is None
    await reloaded.async_close()


async def test_current_failure_clears_observation_without_breaking_catalog(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_check_current()
    env.clock.now += timedelta(seconds=30)
    env.client.current.side_effect = env.transport.ArtworkError("synthetic error")
    await cache.async_check_current()
    assert cache.current is None
    assert await cache.async_catalog() == [env.artwork, env.other]
    await cache.async_check_current()
    assert env.client.current.call_count == 2
    await cache.async_close()


async def test_current_authorization_failure_pauses_other_artwork_requests(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    env.client.current.side_effect = env.transport.ArtworkAuthorizationError("synthetic auth")
    await cache.async_check_current()
    env.clock.now += timedelta(minutes=1)
    await cache.async_check_current()
    assert await cache.async_thumbnail("picture-a") == (None, None)
    env.client.current.assert_called_once_with()
    env.client.thumbnail.assert_not_called()
    env.client.current.side_effect = None
    await cache.async_refresh()
    await cache.async_check_current()
    assert cache.current.content_id == "picture-a"
    await cache.async_close()


async def test_current_check_does_not_contact_sleeping_tv(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    env.awake.value = False
    await cache.async_check_current()
    assert cache.current is None
    env.runner.assert_not_called()
    await cache.async_close()


async def test_current_inflight_mode_change_and_cancellation_cannot_publish(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed(operation):
        started.set()
        await release.wait()
        return operation()

    env.runner.side_effect = delayed
    task = asyncio.create_task(cache.async_check_current())
    await started.wait()
    cache.clear_current()
    release.set()
    await task
    assert cache.current is None
    env.client.current.assert_not_called()
    started.clear()
    release.clear()
    task = asyncio.create_task(cache.async_check_current())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cache.current is None
    await cache.async_close()


async def test_current_result_after_mode_change_is_discarded(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()

    async def changed_after_read(operation):
        result = operation()
        cache.clear_current()
        return result

    env.runner.side_effect = changed_after_read
    await cache.async_check_current()
    assert cache.current is None
    env.client.current.assert_called_once_with()
    await cache.async_close()


async def test_selection_confirmation_sets_observation_and_failure_clears_it(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_select("picture-b")
    assert cache.current.content_id == "picture-b"
    env.client.current.assert_not_called()
    env.client.select.side_effect = env.transport.ArtworkError("uncertain selection")
    with pytest.raises(HomeAssistantError):
        await cache.async_select("picture-a")
    assert cache.current is None
    await cache.async_close()


async def test_upload_invalidates_current_without_assuming_new_photo_is_displayed(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_check_current()
    await cache.async_upload(b"jpeg", b"thumbnail")
    assert cache.current is None
    await cache.async_close()


async def test_failed_uncached_preview_is_hidden_until_manual_refresh(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    env.client.thumbnail.side_effect = env.transport.ArtworkError("Unavailable")
    assert await cache.async_thumbnail("picture-a") == (None, None)
    assert cache.thumbnail_failed("picture-a")
    assert not cache.thumbnail_failed("picture-b")
    assert await cache.async_select("picture-a") == "picture-a"
    await cache.async_refresh()
    assert not cache.thumbnail_failed("picture-a")
    await cache.async_close()


async def test_failed_refresh_keeps_stale_preview_visible(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    expected = await cache.async_thumbnail("picture-a")
    env.clock.now += timedelta(days=31)
    env.client.thumbnail.side_effect = env.transport.ArtworkError("Unavailable")
    assert await cache.async_thumbnail("picture-a") == expected
    assert not cache.thumbnail_failed("picture-a")
    await cache.async_close()


async def test_catalog_refreshes_after_seven_days(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    assert await cache.async_catalog() == [env.artwork, env.other]
    env.clock.now += timedelta(days=6, hours=23)
    assert await cache.async_catalog() == [env.artwork, env.other]
    env.client.list_artwork.assert_called_once()
    env.clock.now += timedelta(hours=1)
    await cache.async_catalog()
    assert env.client.list_artwork.call_count == 2
    await cache.async_close()


async def test_reload_serves_offline_catalog_and_thumbnail(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    assert await cache.async_thumbnail("picture-a") == (b"first-thumbnail", "image/jpeg")
    await cache.async_close()
    env.awake.value = False
    env.clock.now += timedelta(days=40)
    env.client.list_artwork.reset_mock()
    env.client.thumbnail.reset_mock()
    reloaded = env.create()
    await reloaded.async_load()
    assert await reloaded.async_catalog() == [env.artwork, env.other]
    assert await reloaded.async_thumbnail("picture-a") == (b"first-thumbnail", "image/jpeg")
    env.client.list_artwork.assert_not_called()
    env.client.thumbnail.assert_not_called()
    await reloaded.async_close()


async def test_thumbnail_expires_independently_after_thirty_days(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_thumbnail("picture-a")
    env.clock.now += timedelta(days=29)
    await cache.async_thumbnail("picture-a")
    env.client.thumbnail.assert_called_once()
    env.clock.now += timedelta(days=1)
    env.client.thumbnail.return_value = (b"new-thumbnail", "image/jpeg")
    assert await cache.async_thumbnail("picture-a") == (b"new-thumbnail", "image/jpeg")
    assert env.client.thumbnail.call_count == 2
    env.client.list_artwork.assert_called_once()
    await cache.async_close()


async def test_failed_automatic_refresh_keeps_cache_and_attempt_guard_on_reload(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    env.clock.now += timedelta(days=7)
    env.client.list_artwork.side_effect = env.transport.ArtworkError("synthetic failure")
    assert await cache.async_catalog() == [env.artwork, env.other]
    await cache.async_catalog()
    assert env.client.list_artwork.call_count == 2
    await cache.async_close()
    reloaded = env.create()
    await reloaded.async_load()
    assert await reloaded.async_catalog() == [env.artwork, env.other]
    assert env.client.list_artwork.call_count == 2
    await reloaded.async_close()


async def test_hourly_check_refreshes_only_due_awake_catalog(cache_env, hass):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    env.awake.value = False
    env.timer.callback(env.clock.now)
    await hass.async_block_till_done()
    env.client.list_artwork.assert_not_called()
    env.awake.value = True
    env.timer.callback(env.clock.now)
    await hass.async_block_till_done()
    env.client.list_artwork.assert_called_once()
    env.timer.callback(env.clock.now)
    await hass.async_block_till_done()
    env.client.list_artwork.assert_called_once()
    await cache.async_close()


async def test_manual_refresh_resets_authorization_and_thumbnail_failure_guards(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    env.client.thumbnail.side_effect = env.transport.ArtworkAuthorizationError("synthetic auth")
    assert await cache.async_thumbnail("picture-a") == (None, None)
    assert await cache.async_thumbnail("picture-b") == (None, None)
    env.client.thumbnail.assert_called_once()
    with pytest.raises(HomeAssistantError, match="authorization"):
        await cache.async_select("picture-a")
    env.client.select.assert_not_called()
    env.client.thumbnail.side_effect = None
    await cache.async_refresh()
    assert await cache.async_thumbnail("picture-a") == (b"first-thumbnail", "image/jpeg")
    assert env.client.thumbnail.call_count == 2
    assert await cache.async_select("picture-a") == "picture-a"
    await cache.async_close()


async def test_failed_thumbnail_returns_stale_bytes_without_retry_loop(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_thumbnail("picture-a")
    env.clock.now += timedelta(days=30)
    env.client.thumbnail.side_effect = env.transport.ArtworkError("synthetic failure")
    assert await cache.async_thumbnail("picture-a") == (b"first-thumbnail", "image/jpeg")
    assert await cache.async_thumbnail("picture-a") == (b"first-thumbnail", "image/jpeg")
    assert env.client.thumbnail.call_count == 2
    await cache.async_close()


@pytest.mark.parametrize("operation", ["catalog", "thumbnail"])
async def test_offline_coordinator_admission_keeps_stale_cache(cache_env, operation):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_thumbnail("picture-a")
    env.clock.now += timedelta(days=30)

    async def offline(_operation):
        env.awake.value = False
        raise HomeAssistantError("Turn on the TV before requesting artwork")

    env.runner.side_effect = offline
    if operation == "catalog":
        assert await cache.async_catalog() == [env.artwork, env.other]
    else:
        assert await cache.async_thumbnail("picture-a") == (b"first-thumbnail", "image/jpeg")
        env.awake.value = True
        env.runner.side_effect = lambda operation: operation()
        env.client.thumbnail.return_value = (b"fresh-thumbnail", "image/jpeg")
        assert await cache.async_thumbnail("picture-a") == (b"fresh-thumbnail", "image/jpeg")
    await cache.async_close()


async def test_changed_revision_invalidates_only_changed_thumbnail(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_thumbnail("picture-a")
    await cache.async_thumbnail("picture-b")
    revision = cache.image_revision("picture-a")
    env.client.list_artwork.return_value = [
        env.transport.Artwork("picture-a", "First picture", "revision-2"),
        env.other,
    ]
    await cache.async_refresh()
    env.client.thumbnail.return_value = (b"changed-thumbnail", "image/jpeg")
    assert await cache.async_thumbnail("picture-a") == (b"changed-thumbnail", "image/jpeg")
    assert cache.image_revision("picture-a") != revision
    assert await cache.async_thumbnail("picture-b") == (b"first-thumbnail", "image/jpeg")
    assert env.client.thumbnail.call_count == 3
    await cache.async_close()


@pytest.mark.parametrize("branch", ["admission", "artwork_authorization", "artwork_request"])
async def test_thumbnail_warning_distinguishes_failure_branch_without_private_data(
    cache_env, caplog, branch
):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    caplog.clear()
    private = "synthetic-secret-token-and-protocol-reply"
    if branch == "admission":
        env.runner.side_effect = HomeAssistantError(private)
    elif branch == "artwork_authorization":
        env.client.thumbnail.side_effect = env.transport.ArtworkAuthorizationError(private)
    else:
        env.client.thumbnail.side_effect = env.transport.ArtworkError(private)

    assert await cache.async_thumbnail("picture-a") == (None, None)
    records = [row for row in caplog.records if row.name.endswith("tv_artwork_cache")]
    assert len(records) == 1
    assert records[0].args == (branch,)
    assert records[0].exc_info is None
    assert private not in caplog.text
    assert "picture-a" not in caplog.text
    assert cache._authorization_suspended is (branch == "artwork_authorization")
    assert ("picture-a" in cache._failed_thumbnails) is (branch != "admission")
    await cache.async_close()


async def test_title_change_keeps_in_flight_thumbnail(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed(operation):
        if operation is not env.client.list_artwork:
            entered.set()
            await release.wait()
        return operation()

    env.runner.side_effect = delayed
    fetching = asyncio.create_task(cache.async_thumbnail("picture-a"))
    await entered.wait()
    env.client.list_artwork.return_value = [
        env.transport.Artwork("picture-a", "Renamed picture", "revision-1"),
        env.other,
    ]
    await cache.async_refresh()
    release.set()
    assert await fetching == (b"first-thumbnail", "image/jpeg")
    await cache.async_close()


@pytest.mark.parametrize("catalog_change", ["revision", "removed"])
async def test_failed_in_flight_thumbnail_does_not_return_invalidated_bytes(
    cache_env, catalog_change
):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_thumbnail("picture-a")
    env.clock.now += timedelta(days=30)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_failure(operation):
        if operation is env.client.list_artwork:
            return operation()
        entered.set()
        await release.wait()
        raise env.transport.ArtworkError("synthetic failure")

    env.runner.side_effect = delayed_failure
    fetching = asyncio.create_task(cache.async_thumbnail("picture-a"))
    await entered.wait()
    env.client.list_artwork.return_value = [env.other]
    if catalog_change == "revision":
        env.client.list_artwork.return_value.append(
            env.transport.Artwork("picture-a", "Changed picture", "revision-2")
        )
    await cache.async_refresh()
    release.set()
    assert await fetching == (None, None)
    env.awake.value = False
    assert await cache.async_thumbnail("picture-a") == (None, None)
    await cache.async_close()


async def test_unknown_ids_and_offline_selection_never_reach_client(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    with pytest.raises(HomeAssistantError, match="Unknown artwork"):
        await cache.async_select("https://example.invalid/picture.jpg")
    assert await cache.async_thumbnail("../outside") == (None, None)
    env.awake.value = False
    with pytest.raises(HomeAssistantError, match="on"):
        await cache.async_select("picture-a")
    env.client.thumbnail.assert_not_called()
    env.client.select.assert_not_called()
    await cache.async_close()


async def test_oldest_thumbnail_evicted_at_total_byte_limit(cache_env, monkeypatch):
    env = cache_env
    monkeypatch.setattr(env.module, "MAX_CACHE_BYTES", 20)
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    env.client.thumbnail.return_value = (b"0123456789ab", "image/jpeg")
    await cache.async_thumbnail("picture-a")
    env.clock.now += timedelta(seconds=1)
    await cache.async_thumbnail("picture-b")
    env.awake.value = False
    assert await cache.async_thumbnail("picture-a") == (None, None)
    assert await cache.async_thumbnail("picture-b") == (b"0123456789ab", "image/jpeg")
    await cache.async_close()


async def test_close_waits_for_active_fetch_and_blocks_queued_requests(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed(operation):
        entered.set()
        await release.wait()
        return operation()

    env.runner.side_effect = delayed
    fetching = asyncio.create_task(cache.async_thumbnail("picture-a"))
    await entered.wait()
    queued = asyncio.create_task(cache.async_thumbnail("picture-b"))
    closing = asyncio.create_task(cache.async_close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await fetching
    await closing
    with pytest.raises(HomeAssistantError, match="closed"):
        await queued
    env.client.thumbnail.assert_called_once_with("picture-a")
    with pytest.raises(HomeAssistantError, match="closed"):
        await cache.async_catalog()
    with pytest.raises(HomeAssistantError, match="closed"):
        cache.image_revision("picture-a")


async def test_remove_closes_manager_and_removes_store(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    await cache.async_remove()
    env.timer.unsubscribe.assert_called_once()
    env.store.async_remove.assert_awaited_once()


async def test_upload_confirms_once_and_seeds_thumbnail_with_catalog_revision(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    uploaded = env.transport.Artwork("uploaded-picture", "Uploaded", "actual-revision")
    env.client.list_artwork.return_value = [env.artwork, env.other, uploaded]
    env.client.list_artwork.reset_mock()

    result = await cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail")

    assert result == env.module.ArtworkUploadResult("uploaded-picture", True)
    env.client.upload.assert_called_once_with(b"prepared-jpeg")
    env.client.list_artwork.assert_called_once_with()
    assert await cache.async_thumbnail("uploaded-picture") == (b"prepared-thumbnail", "image/jpeg")
    env.client.thumbnail.assert_called_once_with("uploaded-picture")
    assert env.saved.payload["thumbnails"]["uploaded-picture"]["revision"] == "actual-revision"
    await cache.async_close()


@pytest.mark.parametrize("confirmation", ["missing", "failure", "authorization", "storage"])
async def test_accepted_upload_remains_accepted_when_confirmation_fails(cache_env, confirmation):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    env.client.list_artwork.reset_mock()
    if confirmation in {"failure", "authorization"}:
        error_type = (
            env.transport.ArtworkAuthorizationError
            if confirmation == "authorization"
            else env.transport.ArtworkError
        )
        env.client.list_artwork.side_effect = error_type("synthetic-private-error")
    elif confirmation == "storage":
        env.store.async_save.side_effect = OSError("synthetic-private-error")

    result = await cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail")

    assert result == env.module.ArtworkUploadResult("uploaded-picture", False)
    env.client.upload.assert_called_once()
    env.client.list_artwork.assert_called_once()
    env.client.select.assert_not_called()
    if confirmation == "authorization":
        with pytest.raises(env.module.ArtworkUploadNotSentError, match="was not sent"):
            await cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail")
        env.client.upload.assert_called_once()
    await cache.async_close()


@pytest.mark.parametrize("failure", ["offline", "authorization", "closed"])
async def test_upload_admission_never_sends_when_unavailable(cache_env, failure):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    if failure == "offline":
        env.awake.value = False
    elif failure == "authorization":
        env.client.list_artwork.side_effect = env.transport.ArtworkAuthorizationError("denied")
        with pytest.raises(HomeAssistantError):
            await cache.async_catalog()
    else:
        await cache.async_close()
    env.runner.reset_mock()
    with pytest.raises(env.module.ArtworkUploadNotSentError, match="was not sent"):
        await cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail")
    env.client.upload.assert_not_called()
    env.runner.assert_not_called()
    await cache.async_close()


async def test_upload_runner_admission_failure_is_known_not_sent(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    env.runner.side_effect = HomeAssistantError("synthetic-private-admission-reason")

    with pytest.raises(env.module.ArtworkUploadNotSentError) as caught:
        await cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail")

    assert str(caught.value) == "Artwork upload was not sent"
    assert caught.value.__suppress_context__
    env.client.upload.assert_not_called()
    env.client.list_artwork.assert_not_called()
    await cache.async_close()


async def test_upload_runner_failure_after_callback_starts_remains_uncertain(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()

    async def run(operation):
        operation()
        raise HomeAssistantError("synthetic-private-post-send-reason")

    env.runner.side_effect = run
    with pytest.raises(HomeAssistantError) as caught:
        await cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail")

    assert not isinstance(caught.value, env.module.ArtworkUploadNotSentError)
    assert str(caught.value) == "Artwork could not be uploaded"
    env.client.upload.assert_called_once()
    env.client.list_artwork.assert_not_called()
    await cache.async_close()


@pytest.mark.parametrize("failure", ["artwork", "unknown"])
async def test_unknown_runner_failure_is_not_claimed_to_be_unsent(cache_env, failure):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    error = (
        env.transport.ArtworkError("synthetic-private-error")
        if failure == "artwork"
        else RuntimeError("synthetic-private-error")
    )
    env.runner.side_effect = error

    with pytest.raises((HomeAssistantError, RuntimeError)) as caught:
        await cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail")

    assert not isinstance(caught.value, env.module.ArtworkUploadNotSentError)
    env.client.upload.assert_not_called()
    env.client.list_artwork.assert_not_called()
    await cache.async_close()


@pytest.mark.parametrize("authorization", [False, True])
async def test_upload_failure_is_not_retried_or_followed_by_catalog_read(cache_env, authorization):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    error_type = (
        env.transport.ArtworkAuthorizationError if authorization else env.transport.ArtworkError
    )
    env.client.upload.side_effect = error_type("synthetic-private-error")
    with pytest.raises(HomeAssistantError, match="could not be uploaded"):
        await cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail")
    env.client.upload.assert_called_once()
    env.client.list_artwork.assert_not_called()
    await cache.async_close()


async def test_close_waits_for_upload_and_blocks_queued_upload_and_selection(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    env.client.list_artwork.reset_mock()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed(operation):
        entered.set()
        await release.wait()
        return operation()

    env.runner.side_effect = delayed
    uploading = asyncio.create_task(cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail"))
    await entered.wait()
    queued_upload = asyncio.create_task(cache.async_upload(b"other-jpeg", b"other-thumbnail"))
    queued_selection = asyncio.create_task(cache.async_select("picture-a"))
    closing = asyncio.create_task(cache.async_close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    assert await uploading == env.module.ArtworkUploadResult("uploaded-picture", False)
    await closing
    with pytest.raises(env.module.ArtworkUploadNotSentError, match="was not sent"):
        await queued_upload
    with pytest.raises(HomeAssistantError, match="closed"):
        await queued_selection
    env.client.upload.assert_called_once()
    env.client.list_artwork.assert_not_called()
    env.client.select.assert_not_called()


async def test_cancelled_upload_drains_executor_before_releasing_cache_locks(
    cache_env, hass, monkeypatch
):
    from custom_components.localthings.tv_coordinator import TvCoordinator

    env = cache_env
    cache = env.create()
    await cache.async_load()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def work(operation):
        entered.set()
        await release.wait()
        return operation()

    monkeypatch.setattr(
        hass, "async_add_executor_job", lambda operation: asyncio.create_task(work(operation))
    )
    coordinator = cast("TvCoordinator", SimpleNamespace(hass=hass))

    async def run(operation):
        return await TvCoordinator._async_executor(coordinator, operation)

    env.runner.side_effect = run
    uploading = asyncio.create_task(cache.async_upload(b"prepared-jpeg", b"prepared-thumbnail"))
    await entered.wait()
    uploading.cancel()
    closing = asyncio.create_task(cache.async_close())
    await asyncio.sleep(0)
    uploading.cancel()
    await asyncio.sleep(0)
    assert not uploading.done()
    assert not closing.done()
    env.client.upload.assert_not_called()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await uploading
    await closing
    env.client.upload.assert_called_once()
    env.client.list_artwork.assert_not_called()


async def test_delete_old_store_has_no_ownership(cache_env):
    env = cache_env
    cache = env.create()
    await cache.async_load()
    await cache.async_catalog()
    with pytest.raises(env.module.ArtworkDeleteNotSentError):
        await cache.async_prepare_delete("picture-a")
    env.client.select.assert_not_called()
    await cache.async_close()


async def test_upload_ownership_requires_fresh_remote_match(cache_env, monkeypatch):
    env = cache_env
    env.client.upload.return_value = "picture-a"
    monkeypatch.setattr(env.module, "_thumbnail_matches_upload", lambda *_: True, raising=False)
    cache = env.create()
    await cache.async_load()
    result = await cache.async_upload(b"source-jpeg", b"prepared-thumbnail")
    assert result.catalog_confirmed
    env.client.thumbnail.assert_called_once_with("picture-a")
    record = env.saved.payload["uploads"]["picture-a"]
    assert record["device_id"] == "synthetic-tv"
    assert record["record_token"]
    assert (
        record["thumbnail_sha256"] == __import__("hashlib").sha256(b"first-thumbnail").hexdigest()
    )
    assert b"source-jpeg" not in repr(record).encode()
    assert b"first-thumbnail" not in repr(record).encode()
    await cache.async_close()


@pytest.mark.parametrize("failure", ["mismatch", "thumbnail", "save"])
async def test_upload_accepted_without_ownership_on_failed_evidence(
    cache_env, monkeypatch, failure
):
    env = cache_env
    env.client.upload.return_value = "picture-a"
    monkeypatch.setattr(
        env.module, "_thumbnail_matches_upload", lambda *_: failure != "mismatch", raising=False
    )
    if failure == "thumbnail":
        env.client.thumbnail.side_effect = env.transport.ArtworkError("private")
    elif failure == "save":
        env.store.async_save.side_effect = OSError("private")
    cache = env.create()
    await cache.async_load()
    result = await cache.async_upload(b"source-jpeg", b"thumbnail")
    assert result.content_id == "picture-a"
    assert "picture-a" not in cache._uploads
    env.client.upload.assert_called_once()
    await cache.async_close()


@pytest.fixture
async def owned(cache_env, monkeypatch):
    env = cache_env
    env.client.delete = Mock()
    env.client.slideshow_active = Mock(return_value=False)
    env.client.current.return_value = "picture-b"
    env.client.upload.return_value = "picture-a"
    monkeypatch.setattr(env.module, "_thumbnail_matches_upload", lambda *_: True, raising=False)
    cache = env.create()
    await cache.async_load()
    await cache.async_upload(b"source-jpeg", b"prepared-thumbnail")
    env.cache = cache
    yield env
    await cache.async_close()


@pytest.mark.parametrize(
    "guard", ["current", "slideshow", "unknown_slideshow", "thumbnail", "metadata", "offline"]
)
async def test_delete_refuses_changed_or_unsafe_state(owned, guard):
    env, cache = owned, owned.cache
    if guard == "current":
        env.client.current.return_value = "picture-a"
    elif guard == "slideshow":
        env.client.slideshow_active.return_value = True
    elif guard == "unknown_slideshow":
        env.client.slideshow_active.side_effect = env.transport.ArtworkError("private")
    elif guard == "thumbnail":
        env.client.thumbnail.return_value = (b"changed-with-same-date", "image/jpeg")
    elif guard == "metadata":
        env.client.list_artwork.return_value = [
            env.transport.Artwork("picture-a", "Changed", "revision-1")
        ]
    else:
        env.awake.value = False
    with pytest.raises(env.module.ArtworkDeleteNotSentError):
        await cache.async_prepare_delete("picture-a")
    env.client.delete.assert_not_called()
    if guard in {"thumbnail", "metadata"}:
        assert "picture-a" not in env.saved.payload["uploads"]


async def test_observed_absence_permanently_retires_reappearing_id(owned):
    env, cache = owned, owned.cache
    env.client.list_artwork.return_value = [env.other]
    await cache.async_refresh()
    env.client.list_artwork.return_value = [env.artwork, env.other]
    await cache.async_refresh()
    with pytest.raises(env.module.ArtworkDeleteNotSentError):
        await cache.async_prepare_delete("picture-a")
    env.client.delete.assert_not_called()


async def test_pending_save_failure_never_sends(owned):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    env.store.async_save.side_effect = OSError("private")
    with pytest.raises(env.module.ArtworkDeleteNotSentError):
        await cache.async_delete(preview)
    env.client.delete.assert_not_called()


@pytest.mark.parametrize("outcome", ["absent", "present", "uncertain"])
async def test_lost_reply_always_reconciles_once_and_never_resends(owned, outcome):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")

    def sent(_content_id):
        assert next(iter(env.saved.payload["deletions"].values()))["outcome"] == "pending"
        if outcome == "absent":
            env.client.list_artwork.return_value = []
        elif outcome == "uncertain":
            env.client.list_artwork.side_effect = env.transport.ArtworkError("private")
        raise env.transport.ArtworkError("private lost reply")

    env.client.delete.side_effect = sent
    result = await cache.async_delete(preview)
    assert result == env.module.ArtworkDeleteResult("picture-a", outcome)
    assert cache.current is None
    env.client.delete.assert_called_once_with("picture-a")
    result2 = await cache.async_delete(preview)
    assert result2.outcome == outcome
    env.client.delete.assert_called_once()
    await cache.async_close()
    reloaded = env.create()
    await reloaded.async_load()
    result3 = await reloaded.async_delete_status("picture-a")
    assert result3.outcome == outcome
    env.client.delete.assert_called_once()
    await reloaded.async_close()


async def test_two_confirmations_admit_one_send(owned):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    env.client.delete.side_effect = lambda _: setattr(env.client.list_artwork, "return_value", [])
    results = await asyncio.gather(cache.async_delete(preview), cache.async_delete(preview))
    assert [result.outcome for result in results] == ["absent", "absent"]
    env.client.delete.assert_called_once_with("picture-a")


async def test_pending_reboot_recovery_reads_without_send(owned):
    env, cache = owned, owned.cache
    token = env.saved.payload["uploads"]["picture-a"]["record_token"]
    env.saved.payload["deletions"] = {
        token: {
            "content_id": "picture-a",
            "record_token": token,
            "created_at": env.clock.now.timestamp(),
            "device_id": "synthetic-tv",
            "outcome": "pending",
        }
    }
    await cache.async_close()
    reloaded = env.create()
    await reloaded.async_load()
    assert reloaded._deletions[token]["outcome"] == "uncertain"
    result = await reloaded.async_delete_status("picture-a")
    assert result.outcome == "present"
    env.client.delete.assert_not_called()
    assert await reloaded.async_delete_status("no-operation") is None
    await reloaded.async_close()


@pytest.mark.parametrize("change", ["current", "slideshow", "thumbnail", "ownership"])
async def test_confirmation_rechecks_every_guard(owned, change):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    if change == "current":
        env.client.current.return_value = "picture-a"
    elif change == "slideshow":
        env.client.slideshow_active.return_value = True
    elif change == "thumbnail":
        env.client.thumbnail.return_value = (b"same-date-new-image", "image/jpeg")
    else:
        cache._uploads["picture-a"]["record_token"] = "new-generation"
    with pytest.raises(env.module.ArtworkDeleteNotSentError):
        await cache.async_delete(preview)
    env.client.delete.assert_not_called()
    assert not cache._deletions


async def test_unknown_status_creates_no_operation_or_tv_request(owned):
    env, cache = owned, owned.cache
    env.client.list_artwork.reset_mock()
    assert await cache.async_delete_status("picture-a") is None
    assert not cache._deletions
    env.client.delete.assert_not_called()
    env.client.list_artwork.assert_not_called()


async def test_interrupted_pending_save_never_replays_after_reload(owned):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")

    async def saved_then_stopped(payload):
        env.saved.payload = deepcopy(payload)
        raise asyncio.CancelledError

    env.store.async_save.side_effect = saved_then_stopped
    with pytest.raises(asyncio.CancelledError):
        await cache.async_delete(preview)
    env.client.delete.assert_not_called()
    assert next(iter(env.saved.payload["deletions"].values()))["outcome"] == "pending"
    env.store.async_save.side_effect = None
    await cache.async_close()
    reloaded = env.create()
    await reloaded.async_load()
    assert (await reloaded.async_delete_status("picture-a")).outcome == "present"
    env.client.delete.assert_not_called()
    await reloaded.async_close()


async def test_identity_failure_after_preparation_never_sends(owned):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    cache._run_verified_operation = AsyncMock(side_effect=HomeAssistantError("private identity"))
    with pytest.raises(env.module.ArtworkDeleteNotSentError) as caught:
        await cache.async_delete(preview)
    assert "private" not in str(caught.value)
    env.client.delete.assert_not_called()


async def test_ownership_token_changes_for_new_confirmed_upload(owned):
    env, cache = owned, owned.cache
    first = env.saved.payload["uploads"]["picture-a"]["record_token"]
    await cache.async_upload(b"second-source", b"second-thumbnail")
    assert env.saved.payload["uploads"]["picture-a"]["record_token"] != first


@pytest.mark.parametrize(
    "stage", ["thumbnail", "current", "slideshow_active", "delete", "list_artwork"]
)
async def test_delete_authorization_suspends_followup_requests(owned, stage):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    if stage == "list_artwork":
        env.client.list_artwork.side_effect = env.transport.ArtworkAuthorizationError("private")
    else:
        getattr(env.client, stage).side_effect = env.transport.ArtworkAuthorizationError("private")
    if stage == "delete":
        result = await cache.async_delete(preview)
        assert result.outcome == "uncertain"
    else:
        with pytest.raises(env.module.ArtworkDeleteNotSentError) as caught:
            await cache.async_delete(preview)
        assert cache._authorization_suspended, str(caught.value)
    assert cache._authorization_suspended
    reads = env.client.list_artwork.call_count
    thumbs = env.client.thumbnail.call_count
    with pytest.raises(env.module.ArtworkDeleteNotSentError):
        await cache.async_prepare_delete("picture-a")
    assert env.client.list_artwork.call_count == reads
    assert env.client.thumbnail.call_count == thumbs
    if stage == "delete":
        assert reads == 3  # Upload confirmation, preview and final guards, no post-auth request.


async def test_old_device_operation_status_is_uncertain_without_tv_read(owned):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    await cache.async_delete(preview)
    assert env.saved.payload["deletions"][preview.record_token]["device_id"] == "synthetic-tv"
    await cache.async_close()
    reloaded = env.create()
    reloaded._device_id = "different-tv"
    await reloaded.async_load()
    env.client.list_artwork.reset_mock()
    env.client.delete.reset_mock()
    assert (await reloaded.async_delete_status("picture-a")).outcome == "uncertain"
    env.client.list_artwork.assert_not_called()
    env.client.delete.assert_not_called()
    assert "picture-a" not in reloaded._uploads
    await reloaded.async_close()


async def test_status_uses_newest_generation_for_reused_id(owned):
    cache = owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    await cache.async_delete(preview)
    await cache.async_upload(b"new-source", b"new-thumbnail")
    second = await cache.async_prepare_delete("picture-a")
    await cache.async_delete(second)
    old, new = cache._deletions[preview.record_token], cache._deletions[second.record_token]
    old["outcome"] = "uncertain"
    new["outcome"] = "uncertain"
    assert (await cache.async_delete_status("picture-a")).outcome == "present"
    assert new["outcome"] == "present"
    assert old["outcome"] == "uncertain"


async def test_exact_status_never_falls_back_to_reused_id_generation(owned):
    env, cache = owned, owned.cache
    first = await cache.async_prepare_delete("picture-a")
    await cache.async_delete(first)
    await cache.async_upload(b"new-source", b"new-thumbnail")
    second = await cache.async_prepare_delete("picture-a")
    await cache.async_delete(second)
    old = cache._deletions[first.record_token]
    new = cache._deletions[second.record_token]
    old["outcome"] = new["outcome"] = "uncertain"
    upload = deepcopy(cache._uploads["picture-a"])
    env.client.list_artwork.reset_mock()
    result = await cache.async_delete_status("picture-a", record_token=first.record_token)
    assert result.outcome == "uncertain"
    assert new["outcome"] == "uncertain"
    assert cache._uploads["picture-a"] == upload
    assert await cache.async_delete_status("picture-a", record_token="missing-record") is None
    assert await cache.async_delete_status("different-id", record_token=second.record_token) is None
    env.client.list_artwork.assert_not_called()
    env.client.delete.assert_called_with("picture-a")


async def test_exact_status_reconciles_only_matching_record(owned):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    await cache.async_delete(preview)
    env.client.delete.reset_mock()
    env.client.list_artwork.return_value = []
    result = await cache.async_delete_status("picture-a", record_token=preview.record_token)
    assert result.outcome == "absent"
    assert cache._deletions[preview.record_token]["outcome"] == "absent"
    env.client.delete.assert_not_called()


async def test_exact_old_record_does_not_reconcile_a_later_retired_generation(owned):
    env, cache = owned, owned.cache
    first = await cache.async_prepare_delete("picture-a")
    await cache.async_delete(first)
    await cache.async_upload(b"second-source", b"second-thumbnail")
    second = await cache.async_prepare_delete("picture-a")
    env.client.delete.side_effect = lambda _: setattr(env.client.list_artwork, "return_value", [])
    assert (await cache.async_delete(second)).outcome == "absent"
    old, new = cache._deletions[first.record_token], cache._deletions[second.record_token]
    old["outcome"] = "uncertain"
    assert "picture-a" not in cache._uploads
    env.client.list_artwork.reset_mock()
    result = await cache.async_delete_status("picture-a", record_token=first.record_token)
    assert result.outcome == "uncertain"
    assert old["outcome"] == "uncertain"
    assert new["outcome"] == "absent"
    env.client.list_artwork.assert_not_called()


async def test_upload_fingerprint_processing_runs_outside_event_loop(cache_env, monkeypatch):
    import threading

    env = cache_env
    main_thread = threading.get_ident()
    env.client.upload.return_value = "picture-a"

    def compare(_data, _thumbnail):
        assert threading.get_ident() != main_thread
        return True

    monkeypatch.setattr(env.module, "_thumbnail_matches_upload", compare)
    cache = env.create()
    await cache.async_load()
    await cache.async_upload(b"source-jpeg", b"prepared-thumbnail")
    assert "picture-a" in env.saved.payload["uploads"]
    await cache.async_close()


async def test_suspended_status_does_not_call_identity_or_artwork_runner(owned):
    env, cache = owned, owned.cache
    preview = await cache.async_prepare_delete("picture-a")
    await cache.async_delete(preview)
    cache._authorization_suspended = True
    cache._run_verified_operation = AsyncMock()
    env.client.list_artwork.reset_mock()
    assert (await cache.async_delete_status("picture-a")).outcome == "uncertain"
    cache._run_verified_operation.assert_not_called()
    env.client.list_artwork.assert_not_called()


@pytest.fixture
async def real_store_cache(hass, monkeypatch):
    import json

    module = import_module("custom_components.localthings.tv_artwork_cache")
    transport = import_module("custom_components.localthings.tv_artwork")
    monkeypatch.setattr(module, "_thumbnail_matches_upload", lambda *_: True)
    artwork = transport.Artwork("owned-photo", "Owned photo", "revision-1")
    client = SimpleNamespace(
        list_artwork=Mock(return_value=[artwork]),
        thumbnail=Mock(return_value=(b"fresh-thumbnail", "image/jpeg")),
        upload=Mock(return_value="owned-photo"),
        current=Mock(return_value="other-photo"),
        slideshow_active=Mock(return_value=False),
        delete=Mock(),
    )

    async def run(operation):
        return operation()

    async def verified(operation):
        return await operation()

    cache = module.ArtworkCache(
        hass,
        "real-store-entry",
        cast("TvArtworkClient", client),
        run,
        lambda: True,
        device_id="synthetic-tv",
        run_verified_operation=verified,
    )
    saved = SimpleNamespace(payload=None, fail=None, error=WriteError)

    def write(_mode, json_data):
        payload = json.loads(json_data)["data"]
        if saved.fail is not None and saved.fail(payload):
            raise saved.error("private-write-failure-source-path")
        saved.payload = payload

    # Exercise Store's real serialization, executor and error-catching write path.
    monkeypatch.setattr(cache._store, "_write_prepared_data", Mock(side_effect=write))
    await cache.async_load()
    with patch.object(Store, "_async_write_data", _REAL_STORE_WRITE_DATA):
        yield SimpleNamespace(cache=cache, client=client, module=module, saved=saved, hass=hass)
        await cache.async_close()


@pytest.mark.parametrize("error", [WriteError, SerializationError])
async def test_real_store_failed_pending_write_never_sends(real_store_cache, error, caplog):
    env = real_store_cache
    await env.cache.async_upload(b"source-jpeg", b"thumbnail")
    preview = await env.cache.async_prepare_delete("owned-photo")
    env.saved.error = error
    env.saved.fail = lambda payload: bool(payload["deletions"])
    with pytest.raises(env.module.ArtworkDeleteNotSentError) as caught:
        await env.cache.async_delete(preview)
    env.client.delete.assert_not_called()
    assert "private-write-failure-source-path" not in str(caught.value)
    assert "private-write-failure-source-path" not in caplog.text
    assert not env.cache._deletions
    assert not env.saved.payload["deletions"]


@pytest.mark.parametrize("error", [WriteError, SerializationError])
async def test_real_store_failed_ownership_write_keeps_acceptance_without_ownership(
    real_store_cache, error, caplog
):
    env = real_store_cache
    env.saved.error = error
    env.saved.fail = lambda payload: bool(payload["uploads"])
    result = await env.cache.async_upload(b"source-jpeg", b"thumbnail")
    assert result == env.module.ArtworkUploadResult("owned-photo", True)
    assert not env.cache._uploads
    assert not env.saved.payload["uploads"]
    env.client.upload.assert_called_once()
    env.client.delete.assert_not_called()
    assert "private-write-failure-source-path" not in caplog.text


async def test_real_store_stopping_state_never_admits_delete(real_store_cache, monkeypatch):
    env = real_store_cache
    await env.cache.async_upload(b"source-jpeg", b"thumbnail")
    preview = await env.cache.async_prepare_delete("owned-photo")
    monkeypatch.setattr(env.hass, "state", CoreState.stopping)
    with pytest.raises(env.module.ArtworkDeleteNotSentError):
        await env.cache.async_delete(preview)
    env.client.delete.assert_not_called()
    assert not env.cache._deletions
    assert not env.saved.payload["deletions"]


async def test_real_store_stopping_state_does_not_grant_upload_ownership(
    real_store_cache, monkeypatch
):
    env = real_store_cache
    monkeypatch.setattr(env.hass, "state", CoreState.stopping)
    result = await env.cache.async_upload(b"source-jpeg", b"thumbnail")
    assert result.content_id == "owned-photo"
    assert not env.cache._uploads
    env.client.upload.assert_called_once()
    env.client.delete.assert_not_called()


async def test_real_store_stopping_save_never_queues_a_deferred_write(
    real_store_cache, monkeypatch
):
    env = real_store_cache
    monkeypatch.setattr(env.hass, "state", CoreState.stopping)
    with pytest.raises(OSError, match="Home Assistant is stopping"):
        await env.cache._store.async_save({"pending": True})
    assert env.cache._store._data is None
    assert env.saved.payload is None
    env.cache._store._write_prepared_data.assert_not_called()


async def test_real_store_stop_after_pending_write_still_sends_nothing(
    real_store_cache, monkeypatch
):
    env = real_store_cache
    await env.cache.async_upload(b"source-jpeg", b"thumbnail")
    preview = await env.cache.async_prepare_delete("owned-photo")
    original_write = env.cache._store._async_write_data

    async def write_then_stop(data):
        await original_write(data)
        if data["data"]["deletions"]:
            monkeypatch.setattr(env.hass, "state", CoreState.stopping)

    monkeypatch.setattr(env.cache._store, "_async_write_data", write_then_stop)
    with pytest.raises(env.module.ArtworkDeleteNotSentError):
        await env.cache.async_delete(preview)
    env.client.delete.assert_not_called()
    assert next(iter(env.saved.payload["deletions"].values()))["outcome"] == "pending"
