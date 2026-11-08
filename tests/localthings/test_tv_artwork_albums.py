"""Album metadata is local, durable and serialized across editors."""

import asyncio
import json
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.core import CoreState
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util.file import WriteError
from homeassistant.util.json import SerializationError

_REAL_STORE_WRITE_DATA = Store._async_write_data

IDS = ["MY_F0002", "MY_F0001"]


@pytest.fixture
def album_env(hass, monkeypatch):
    module = import_module("custom_components.localthings.tv_artwork_albums")
    saved = {}

    def create(entry="entry-one", device="device-one"):
        async def save(payload):
            saved[entry] = deepcopy(payload)

        store = SimpleNamespace(
            async_load=AsyncMock(side_effect=lambda: deepcopy(saved.get(entry))),
            async_save=AsyncMock(side_effect=save),
            async_remove=AsyncMock(side_effect=lambda: saved.pop(entry, None)),
        )
        monkeypatch.setattr(module, "album_store", lambda *_: store)
        artwork = Mock()
        artwork.image_revision.side_effect = lambda value: "v1" if value in IDS else ""
        albums = module.ArtworkAlbums(hass, entry, device, artwork)
        return SimpleNamespace(albums=albums, store=store, artwork=artwork)

    return SimpleNamespace(create=create, saved=saved, module=module)


async def save(env, **changes):
    fields = {
        "expected_revision": 0,
        "name": "Nature",
        "content_ids": IDS,
        "interval_minutes": 3,
        "shuffle": False,
    }
    fields.update(changes)
    return await env.albums.async_save(**fields)


async def test_round_trip_identity_isolation_and_deep_copies(album_env):
    env = album_env.create()
    await env.albums.async_load()
    assert await env.albums.async_list() == {"revision": 0, "albums": []}
    result = await save(env, name="  Nature  ")
    original = deepcopy(result)
    assert result["revision"] == 1
    assert result["albums"][0]["name"] == "Nature"
    assert result["albums"][0]["content_ids"] == IDS
    result["albums"][0]["content_ids"].clear()
    listed = await env.albums.async_list()
    listed["albums"].clear()
    assert await env.albums.async_list() == original
    restarted = album_env.create()
    await restarted.albums.async_load()
    assert await restarted.albums.async_list() == original
    for isolated in (album_env.create(entry="other"), album_env.create(device="replacement")):
        await isolated.albums.async_load()
        assert await isolated.albums.async_list() == {"revision": 0, "albums": []}


@pytest.mark.parametrize(
    "change",
    [
        {"name": " "},
        {"name": "x" * 61},
        {"name": 1},
        {"content_ids": []},
        {"content_ids": IDS * 2},
        {"content_ids": ["MY_missing"]},
        {"content_ids": ["SAM_1"]},
        {"content_ids": [1]},
        {"content_ids": ["MY_" + str(i) for i in range(101)]},
        {"interval_minutes": 2},
        {"interval_minutes": True},
        {"shuffle": 0},
        {"expected_revision": True},
        {"expected_revision": -1},
        {"album_id": "unknown"},
    ],
)
async def test_invalid_boundary_never_writes(album_env, change):
    env = album_env.create()
    await env.albums.async_load()
    with pytest.raises(HomeAssistantError):
        await save(env, **change)
    env.store.async_save.assert_not_awaited()
    assert await env.albums.async_list() == {"revision": 0, "albums": []}


async def test_limits_names_and_missing_members_can_be_repaired(album_env):
    env = album_env.create()
    await env.albums.async_load()
    doc = await save(env)
    album_id = doc["albums"][0]["album_id"]
    with pytest.raises(HomeAssistantError, match="name"):
        await save(env, expected_revision=1, name="NATURE")
    env.artwork.image_revision.return_value = ""
    env.artwork.image_revision.side_effect = None
    doc = await save(env, expected_revision=1, album_id=album_id, name="Renamed")
    assert doc["albums"][0]["content_ids"] == IDS
    with pytest.raises(HomeAssistantError, match="personal"):
        await save(env, expected_revision=2, name="Other")
    doc = await save(env, expected_revision=2, album_id=album_id, content_ids=[IDS[1]])
    assert doc["albums"][0]["content_ids"] == [IDS[1]]
    with pytest.raises(HomeAssistantError, match="personal"):
        await save(env, expected_revision=3, album_id=album_id)


async def test_fifty_albums_and_hundred_members_are_the_exact_limits(album_env):
    env = album_env.create()
    env.artwork.image_revision.side_effect = None
    env.artwork.image_revision.return_value = "v1"
    await env.albums.async_load()
    members = ["MY_" + str(i) for i in range(100)]
    for revision in range(50):
        doc = await save(env, expected_revision=revision, name=str(revision), content_ids=members)
    assert doc["revision"] == 50
    assert len(doc["albums"]) == 50
    with pytest.raises(HomeAssistantError, match="50"):
        await save(env, expected_revision=50, name="Too many")
    await save(env, expected_revision=50, album_id=doc["albums"][0]["album_id"])


async def test_stale_concurrent_edit_cannot_overwrite(album_env):
    env = album_env.create()
    await env.albums.async_load()
    outcomes = await asyncio.gather(save(env), save(env, name="Second"), return_exceptions=True)
    assert sum(isinstance(value, HomeAssistantError) for value in outcomes) == 1
    assert (await env.albums.async_list())["revision"] == 1
    env.store.async_save.assert_awaited_once()


async def test_entry_removal_after_failed_unload_drains_album_writes(hass, album_env):
    from custom_components import localthings as integration
    from custom_components.localthings.const import DOMAIN

    from .test_tv_config_flow import tv_entry

    entry = tv_entry(hass)
    env = album_env.create(entry=entry.entry_id)
    await env.albums.async_load()
    doc = await save(env)
    hass.data[DOMAIN] = {entry.entry_id: SimpleNamespace(albums=env.albums)}
    entered, release = asyncio.Event(), asyncio.Event()
    writer = env.store.async_save.side_effect

    async def delayed(payload):
        entered.set()
        await release.wait()
        await writer(payload)

    env.store.async_save.side_effect = delayed
    with (
        patch.object(
            hass.config_entries, "async_unload_platforms", new=AsyncMock(return_value=False)
        ),
        patch.object(integration, "album_store", return_value=env.store),
        patch.object(
            integration, "artwork_store", return_value=SimpleNamespace(async_remove=AsyncMock())
        ),
    ):
        write = asyncio.create_task(
            save(env, expected_revision=1, album_id=doc["albums"][0]["album_id"])
        )
        await entered.wait()
        assert not await integration.async_unload_entry(hass, entry)
        queued = asyncio.create_task(save(env, expected_revision=2))
        removal = asyncio.create_task(integration.async_remove_entry(hass, entry))
        await asyncio.sleep(0)
        removal_waited = not removal.done()
        release.set()
        results = await asyncio.gather(write, queued, removal, return_exceptions=True)
    assert removal_waited
    assert isinstance(results[1], HomeAssistantError)
    assert "closed" in str(results[1])
    assert results[2] is None
    assert entry.entry_id not in album_env.saved
    with pytest.raises(HomeAssistantError, match="closed"):
        await env.albums.async_list()


async def test_failed_write_does_not_publish_and_can_retry(album_env):
    env = album_env.create()
    await env.albums.async_load()
    writer = env.store.async_save.side_effect
    env.store.async_save.side_effect = OSError("disk full")
    with pytest.raises(HomeAssistantError, match="saved"):
        await save(env)
    assert await env.albums.async_list() == {"revision": 0, "albums": []}
    env.store.async_save.side_effect = writer
    assert (await save(env))["revision"] == 1


async def test_cancelled_save_drains_disk_before_next_edit(album_env):
    env = album_env.create()
    await env.albums.async_load()
    entered, release = asyncio.Event(), asyncio.Event()
    writer = env.store.async_save.side_effect

    async def delayed(payload):
        entered.set()
        await release.wait()
        await writer(payload)

    env.store.async_save.side_effect = delayed
    first = asyncio.create_task(save(env))
    await entered.wait()
    first.cancel()
    second = asyncio.create_task(save(env, name="Second"))
    await asyncio.sleep(0)
    first.cancel()
    assert not second.done()
    assert "entry-one" not in album_env.saved
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(HomeAssistantError, match="changed"):
        await second
    doc = await env.albums.async_list()
    assert doc["albums"][0]["name"] == "Nature"
    restarted = album_env.create()
    await restarted.albums.async_load()
    assert await restarted.albums.async_list() == doc


async def test_append_merges_latest_order_and_repeat_does_not_write(album_env):
    env = album_env.create()
    await env.albums.async_load()
    doc = await save(env, content_ids=[IDS[0]], interval_minutes=60, shuffle=True)
    album_id = doc["albums"][0]["album_id"]
    doc = await env.albums.async_append(album_id, IDS[1])
    assert doc["albums"][0]["content_ids"] == IDS
    assert doc["albums"][0]["interval_minutes"] == 60
    assert doc["albums"][0]["shuffle"] is True
    assert await env.albums.async_append(album_id, IDS[1]) == doc
    assert env.store.async_save.await_count == 2
    with pytest.raises(HomeAssistantError):
        await env.albums.async_append(album_id, "MY_unknown")


async def test_append_existing_unavailable_member_is_noop_without_cache_or_disk_access(album_env):
    env = album_env.create()
    await env.albums.async_load()
    original = await save(env)
    album_id = original["albums"][0]["album_id"]
    persisted = deepcopy(album_env.saved)
    env.store.async_save.reset_mock()
    env.artwork.image_revision.reset_mock()
    env.artwork.image_revision.side_effect = None
    env.artwork.image_revision.return_value = ""

    repeated = await env.albums.async_append(album_id, IDS[1])

    assert repeated == original
    assert album_env.saved == persisted
    env.store.async_save.assert_not_awaited()
    env.artwork.image_revision.assert_not_called()
    repeated["albums"][0]["content_ids"].clear()
    assert await env.albums.async_list() == original
    with pytest.raises(HomeAssistantError, match="cached personal"):
        await env.albums.async_append(album_id, "MY_missing_new")
    env.store.async_save.assert_not_awaited()


async def test_close_drains_writer_then_rejects_all_operations(album_env):
    env = album_env.create()
    await env.albums.async_load()
    entered, release = asyncio.Event(), asyncio.Event()
    writer = env.store.async_save.side_effect

    async def delayed(payload):
        entered.set()
        await release.wait()
        await writer(payload)

    env.store.async_save.side_effect = delayed
    task = asyncio.create_task(save(env))
    await entered.wait()
    closed = asyncio.create_task(env.albums.async_close())
    await asyncio.sleep(0)
    assert not closed.done()
    release.set()
    await task
    await closed
    for operation in (env.albums.async_list(), save(env), env.albums.async_append("x", IDS[0])):
        with pytest.raises(HomeAssistantError, match="closed"):
            await operation
    restarted = album_env.create()
    await restarted.albums.async_load()
    assert (await restarted.albums.async_list())["revision"] == 1


@pytest.mark.parametrize("failure", [OSError("disk"), {"bad": "document"}])
async def test_load_error_stays_local_and_blocks_destructive_empty_write(album_env, failure):
    env = album_env.create()
    if isinstance(failure, Exception):
        env.store.async_load.side_effect = failure
    else:
        env.store.async_load.side_effect = None
        env.store.async_load.return_value = failure
    await env.albums.async_load()
    with pytest.raises(HomeAssistantError, match="unavailable"):
        await env.albums.async_list()
    with pytest.raises(HomeAssistantError, match="unavailable"):
        await save(env)
    env.store.async_save.assert_not_awaited()


@pytest.mark.parametrize("interval", [1, 3, 5, 10, 15, 30, 60])
async def test_each_supported_interval_and_longest_trimmed_name(album_env, interval):
    env = album_env.create()
    await env.albums.async_load()
    doc = await save(env, name=" " + "x" * 60 + " ", interval_minutes=interval)
    assert doc["albums"][0]["interval_minutes"] == interval
    assert doc["albums"][0]["name"] == "x" * 60


async def test_simultaneous_appends_merge_and_make_existing_editor_stale(album_env):
    env = album_env.create()
    env.artwork.image_revision.side_effect = None
    env.artwork.image_revision.return_value = "v1"
    await env.albums.async_load()
    original = await save(env, content_ids=[IDS[0]])
    album_id = original["albums"][0]["album_id"]
    await asyncio.gather(
        env.albums.async_append(album_id, IDS[1]),
        env.albums.async_append(album_id, "MY_F0003"),
    )
    doc = await env.albums.async_list()
    assert doc["revision"] == 3
    assert doc["albums"][0]["content_ids"] == [*IDS, "MY_F0003"]
    with pytest.raises(HomeAssistantError, match="changed"):
        await save(env, album_id=album_id, expected_revision=1)
    assert await env.albums.async_list() == doc


@pytest.fixture
async def real_store_albums(hass, monkeypatch):
    module = import_module("custom_components.localthings.tv_artwork_albums")
    albums = module.ArtworkAlbums(hass, "real-albums", "synthetic-device", Mock())
    saved = SimpleNamespace(payload=None, error=None)

    def write(_mode, json_data):
        if saved.error:
            raise saved.error("private-source-path")
        saved.payload = json.loads(json_data)["data"]

    monkeypatch.setattr(albums._store, "_write_prepared_data", Mock(side_effect=write))
    await albums.async_load()
    with patch.object(Store, "_async_write_data", _REAL_STORE_WRITE_DATA):
        yield SimpleNamespace(albums=albums, saved=saved, hass=hass)
    await albums.async_close()


@pytest.mark.parametrize("error", [WriteError, SerializationError])
async def test_real_store_failure_does_not_publish_or_log_private_paths(
    real_store_albums, error, caplog
):
    env = real_store_albums
    env.saved.error = error
    with pytest.raises(HomeAssistantError, match="saved") as caught:
        await save(env)
    assert env.saved.payload is None
    assert await env.albums.async_list() == {"revision": 0, "albums": []}
    assert "private-source-path" not in str(caught.value)
    assert "private-source-path" not in caplog.text
    env.saved.error = None
    assert (await save(env))["revision"] == 1
    assert env.saved.payload["revision"] == 1


async def test_real_store_stopping_does_not_defer_or_acknowledge(real_store_albums, monkeypatch):
    env = real_store_albums
    monkeypatch.setattr(env.hass, "state", CoreState.stopping)
    with pytest.raises(HomeAssistantError, match="saved"):
        await save(env)
    assert env.albums._store._data is None
    assert env.saved.payload is None
    assert await env.albums.async_list() == {"revision": 0, "albums": []}


async def test_real_store_read_only_mode_does_not_acknowledge_unsaved_metadata(real_store_albums):
    env = real_store_albums
    env.albums._store.make_read_only()
    with pytest.raises(HomeAssistantError, match="saved"):
        await save(env)
    assert env.saved.payload is None
    assert await env.albums.async_list() == {"revision": 0, "albums": []}


async def test_real_store_selects_atomic_file_writer(hass, tmp_path, monkeypatch):
    module = import_module("custom_components.localthings.tv_artwork_albums")
    monkeypatch.setattr(hass.config, "config_dir", str(tmp_path))
    store = module.album_store(hass, "atomic-entry")
    with (
        patch.object(Store, "_async_write_data", _REAL_STORE_WRITE_DATA),
        patch("homeassistant.helpers.storage.write_utf8_file_atomic") as atomic,
        patch("homeassistant.helpers.storage.write_utf8_file") as direct,
    ):
        await store.async_save({"device_id": "synthetic", "revision": 1, "albums": []})
    atomic.assert_called_once()
    direct.assert_not_called()
    assert json.loads(atomic.call_args.args[1])["data"]["revision"] == 1
