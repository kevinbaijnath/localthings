"""Artwork deletion requires a private preview and a single explicit confirmation."""

import asyncio
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp.test_utils import make_mocked_request
from homeassistant.config_entries import ConfigEntryState
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.http import current_request
from PIL import Image
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.localthings.config_flow import LocalThingsConfigFlow
from custom_components.localthings.const import DOMAIN
from custom_components.localthings.tv_artwork_delete import ArtworkDeletePreview
from custom_components.localthings.tv_artwork_preview import preview_store


def request(user_id="admin", token_id="refresh", is_admin=True):
    result = make_mocked_request("POST", "/api/config/config_entries/options/flow")
    result["hass_user"] = SimpleNamespace(id=user_id, is_admin=is_admin)
    result["hass_refresh_token_id"] = token_id
    return result


@pytest.fixture
def deletion(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"transport": "tv_ws"}, options={"keep": True})
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    output = io.BytesIO()
    Image.new("RGB", (120, 80), "red").save(output, format="PNG")
    preview = ArtworkDeletePreview(
        content_id="MY_NEW",
        title="Uploaded photograph",
        record_token="owned-generation",
        thumbnail=output.getvalue(),
        mime_type="image/png",
    )
    artwork = SimpleNamespace(
        async_prepare_delete=AsyncMock(return_value=preview),
        async_delete=AsyncMock(return_value=SimpleNamespace(content_id="MY_NEW", outcome="absent")),
        async_delete_status=AsyncMock(
            return_value=SimpleNamespace(content_id="MY_NEW", outcome="uncertain")
        ),
    )
    coordinator = SimpleNamespace(artwork=artwork)
    hass.data[DOMAIN] = {entry.entry_id: coordinator}
    flow = LocalThingsConfigFlow.async_get_options_flow(entry)
    flow.hass = hass
    flow.handler = entry.entry_id
    flow.flow_id = "synthetic-delete-flow"
    token = current_request.set(request())
    user = SimpleNamespace(id="admin", is_admin=True, is_active=True)
    refresh_token = SimpleNamespace(id="refresh", user=user, expire_at=None)
    with (
        patch(
            "custom_components.localthings.tv_artwork_preview.async_sign_path",
            return_value="/api/localthings/artwork/preview/private?authSig=signed.jwt",
        ),
        patch.object(hass.auth, "async_get_user", AsyncMock(return_value=user)),
        patch.object(hass.auth, "async_get_refresh_token", Mock(return_value=refresh_token)),
    ):
        yield SimpleNamespace(
            flow=flow,
            artwork=artwork,
            preview=preview,
            entry=entry,
            coordinator=coordinator,
            user=user,
            refresh_token=refresh_token,
        )
    flow.async_remove()
    current_request.reset(token)
    entry.mock_state(hass, ConfigEntryState.NOT_LOADED)


async def prepare(hass, env):
    await env.flow.async_step_init()
    await env.flow.async_step_delete_artwork({"content_id": "MY_NEW"})
    await hass.async_block_till_done()
    result = await env.flow.async_step_delete_prepare()
    assert result["step_id"] == "delete_confirm"
    return await env.flow.async_step_delete_confirm()


async def test_delete_menu_and_selection_never_mutate(hass, deletion):
    env = deletion
    menu = await env.flow.async_step_init()
    assert "delete_artwork" in menu["menu_options"]
    selection = await env.flow.async_step_delete_artwork()
    assert selection["type"] == FlowResultType.FORM
    assert "content_id" in selection["data_schema"].schema
    form = await prepare(hass, env)
    assert form["type"] == FlowResultType.FORM
    assert form["description_placeholders"]["content_id"] == "MY_NEW"
    assert form["description_placeholders"]["title"] == "Uploaded photograph"
    assert form["description_placeholders"]["preview"].startswith(
        "/api/localthings/artwork/preview/"
    )
    env.artwork.async_prepare_delete.assert_awaited_once_with("MY_NEW")
    env.artwork.async_delete.assert_not_awaited()
    env.flow.async_remove()
    env.artwork.async_delete.assert_not_awaited()
    assert preview_store(hass)._entries == {}


async def test_delete_confirmation_is_single_send_and_keeps_entry_options(hass, deletion):
    env = deletion
    await prepare(hass, env)
    await env.flow.async_step_delete_confirm({})
    await hass.async_block_till_done()
    result = await env.flow.async_step_delete_confirm({})
    assert result["step_id"] == "delete_complete"
    env.artwork.async_delete.assert_awaited_once_with(replace(env.preview, thumbnail=b""))
    assert preview_store(hass)._entries == {}
    result = await env.flow.async_step_delete_complete()
    assert result["description_placeholders"]["content_id"] == "MY_NEW"
    result = await env.flow.async_step_delete_complete({})
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "delete_finished"
    assert dict(env.entry.options) == {"keep": True}


@pytest.mark.parametrize(
    "new_request", [request("other"), request(token_id="other"), request(is_admin=False)]
)
async def test_confirmation_requires_same_admin_session(hass, deletion, new_request):
    env = deletion
    await prepare(hass, env)
    token = current_request.set(new_request)
    try:
        result = await env.flow.async_step_delete_confirm({})
    finally:
        current_request.reset(token)
    assert result["reason"] == "delete_unauthorized"
    env.artwork.async_delete.assert_not_awaited()


async def test_expired_delete_preview_cannot_confirm(hass, deletion):
    env = deletion
    await prepare(hass, env)
    preparation = preview_store(hass)._entries[env.entry.entry_id]
    assert 0 < preparation.expires_at - hass.loop.time() <= 300
    assert preparation.timer is not None
    assert 0 < preparation.timer.when() - hass.loop.time() <= 300
    preparation.expires_at = hass.loop.time() - 1
    result = await env.flow.async_step_delete_confirm({})
    assert result["reason"] == "delete_expired"
    env.artwork.async_delete.assert_not_awaited()


async def test_integration_reload_invalidates_delete_confirmation(hass, deletion):
    env = deletion
    await prepare(hass, env)
    hass.data[DOMAIN][env.entry.entry_id] = SimpleNamespace(artwork=env.artwork)
    result = await env.flow.async_step_delete_confirm({})
    assert result["reason"] == "delete_expired"
    env.artwork.async_delete.assert_not_awaited()


async def test_unowned_photo_has_clear_refusal_without_delete(hass, deletion):
    env = deletion
    env.artwork.async_prepare_delete.side_effect = HomeAssistantError("No upload record")
    await env.flow.async_step_init()
    await env.flow.async_step_delete_artwork({"content_id": "MY_F0002"})
    await hass.async_block_till_done()
    result = await env.flow.async_step_delete_prepare()
    assert result["step_id"] == "delete_blocked"
    result = await env.flow.async_step_delete_blocked()
    assert result["description_placeholders"]["reason"] == "No upload record"
    env.artwork.async_delete.assert_not_awaited()


async def test_canceled_waiter_cannot_cancel_admitted_deletion(hass, deletion):
    env = deletion
    await prepare(hass, env)
    started, release = asyncio.Event(), asyncio.Event()

    async def remove(_preview):
        started.set()
        await release.wait()
        return SimpleNamespace(content_id="MY_NEW", outcome="absent")

    env.artwork.async_delete.side_effect = remove
    await env.flow.async_step_delete_confirm({})
    await started.wait()
    env.flow._delete_waiter.cancel()
    await asyncio.gather(env.flow._delete_waiter, return_exceptions=True)
    assert not env.flow._delete_task.cancelled()
    env.flow.async_remove()
    release.set()
    await hass.async_block_till_done()
    assert env.flow._delete_task.done()
    env.artwork.async_delete.assert_awaited_once()


async def test_closing_flow_before_job_starts_keeps_admitted_photo(hass, deletion):
    env = deletion
    await prepare(hass, env)
    create_task = hass.async_create_task

    def delayed(coroutine, name):
        return create_task(coroutine, name, eager_start=False)

    with patch.object(hass, "async_create_task", side_effect=delayed):
        await env.flow.async_step_delete_confirm({})
        env.flow.async_remove()
    await hass.async_block_till_done()
    env.artwork.async_delete.assert_awaited_once_with(replace(env.preview, thumbnail=b""))


@pytest.mark.parametrize("outcome", ["present", "uncertain", "absent"])
async def test_restart_status_path_is_read_only(hass, deletion, outcome):
    env = deletion
    env.artwork.async_delete_status.return_value = SimpleNamespace(
        content_id="MY_NEW", outcome=outcome
    )
    await env.flow.async_step_init()
    await env.flow.async_step_delete_status({"content_id": "MY_NEW"})
    await hass.async_block_till_done()
    result = await env.flow.async_step_delete_progress()
    assert (
        result["step_id"]
        == {
            "present": "delete_present",
            "uncertain": "delete_uncertain",
            "absent": "delete_complete",
        }[outcome]
    )
    env.artwork.async_delete_status.assert_awaited_once_with("MY_NEW")
    env.artwork.async_prepare_delete.assert_not_awaited()
    env.artwork.async_delete.assert_not_awaited()


async def test_missing_delete_status_never_creates_or_sends_operation(hass, deletion):
    env = deletion
    env.artwork.async_delete_status.return_value = None
    await env.flow.async_step_init()
    await env.flow.async_step_delete_status({"content_id": "MY_NEW"})
    await hass.async_block_till_done()
    result = await env.flow.async_step_delete_progress()
    assert result["step_id"] == "delete_stop"
    result = await env.flow.async_step_delete_stop()
    assert result["reason"] == "delete_missing"
    env.artwork.async_prepare_delete.assert_not_awaited()
    env.artwork.async_delete.assert_not_awaited()


async def test_native_manager_advances_through_one_confirmed_deletion(hass, deletion):
    env = deletion
    manager = hass.config_entries.options
    result = await manager.async_init(env.entry.entry_id)
    flow_id = result["flow_id"]
    try:
        result = await manager.async_configure(flow_id, {"next_step_id": "delete_artwork"})
        assert result["step_id"] == "delete_artwork"
        result = await manager.async_configure(flow_id, {"content_id": "MY_NEW"})
        assert result["type"] == FlowResultType.SHOW_PROGRESS
        await hass.async_block_till_done()
        result = await manager.async_configure(flow_id)
        assert result["step_id"] == "delete_confirm"
        assert result["description_placeholders"]["content_id"] == "MY_NEW"
        env.artwork.async_delete.assert_not_awaited()
        result = await manager.async_configure(flow_id, {})
        assert result["type"] == FlowResultType.SHOW_PROGRESS
        await hass.async_block_till_done()
        result = await manager.async_configure(flow_id)
        assert result["step_id"] == "delete_complete"
        env.artwork.async_delete.assert_awaited_once_with(replace(env.preview, thumbnail=b""))
        result = await manager.async_configure(flow_id, {})
        assert result["type"] == FlowResultType.ABORT
        assert result["reason"] == "delete_finished"
    finally:
        if flow_id in manager._progress:
            manager.async_abort(flow_id)


async def test_unexpected_delete_failure_stays_uncertain_and_consumed(hass, deletion):
    env = deletion
    env.artwork.async_delete.side_effect = TimeoutError("synthetic private server detail")
    await prepare(hass, env)
    await env.flow.async_step_delete_confirm({})
    await hass.async_block_till_done()
    result = await env.flow.async_step_delete_confirm({})
    assert result["step_id"] == "delete_uncertain"
    result = await env.flow.async_step_delete_uncertain()
    assert result["description_placeholders"]["content_id"] == "MY_NEW"
    assert result["description_placeholders"]["reason"] == ""
    assert rows(result)[0]["outcome"] == "uncertain"
    env.artwork.async_delete.assert_awaited_once()


async def test_not_sent_delete_preserves_safe_reason(hass, deletion):
    from custom_components.localthings.tv_artwork_cache import ArtworkDeleteNotSentError

    env = deletion
    env.artwork.async_delete.side_effect = ArtworkDeleteNotSentError("TV is asleep")
    await prepare(hass, env)
    await env.flow.async_step_delete_confirm({})
    await hass.async_block_till_done()
    result = await env.flow.async_step_delete_progress()
    assert result["step_id"] == "delete_not_sent"
    result = await env.flow.async_step_delete_not_sent()
    assert result["description_placeholders"]["reason"] == "TV is asleep"
    env.artwork.async_delete.assert_awaited_once()


def batch_previews(env):
    second = ArtworkDeletePreview(
        "MY_SECOND", "Second photo", "second-generation", env.preview.thumbnail, "image/png"
    )
    env.artwork.async_prepare_delete.side_effect = [env.preview, second]
    return (env.preview, second)


async def prepare_batch(hass, env):
    await env.flow.async_step_init()
    result = await env.flow.async_step_delete_artworks({"content_ids": ["MY_NEW", "MY_SECOND"]})
    assert result["step_id"] == "delete_prepare"
    await hass.async_block_till_done()
    return await env.flow.async_step_delete_prepare()


def rows(result):
    return json.loads(result["description_placeholders"]["items"])


@pytest.mark.parametrize(
    "value",
    [
        None,
        "MY_NEW",
        [],
        ["MY_NEW", "MY_NEW"],
        [1],
        [""],
        ["SAM_BUILTIN"],
        [f"MY_{i}" for i in range(11)],
    ],
)
async def test_batch_rejects_invalid_selection_without_preparing(hass, deletion, value):
    env = deletion
    await env.flow.async_step_init()
    result = await env.flow.async_step_delete_artworks({"content_ids": value})
    assert result["step_id"] == "delete_artworks"
    assert result["errors"] == {"base": "delete_invalid_selection"}
    env.artwork.async_prepare_delete.assert_not_awaited()
    env.artwork.async_delete.assert_not_awaited()


async def test_batch_checks_whole_group_and_freezes_exact_order_before_confirmation(hass, deletion):
    env = deletion
    previews = batch_previews(env)
    assert (await prepare_batch(hass, env))["step_id"] == "delete_confirm"
    result = await env.flow.async_step_delete_confirm()
    assert result["description_placeholders"]["count"] == "2"
    assert rows(result) == [
        {
            "content_id": "MY_NEW",
            "title": "Uploaded photograph",
            "record_token": "owned-generation",
            "outcome": "ready",
            "reason": "",
        },
        {
            "content_id": "MY_SECOND",
            "title": "Second photo",
            "record_token": "second-generation",
            "outcome": "ready",
            "reason": "",
        },
    ]
    assert [c.args[0] for c in env.artwork.async_prepare_delete.await_args_list] == [
        p.content_id for p in previews
    ]
    preparation = preview_store(hass)._entries[env.entry.entry_id]
    with Image.open(io.BytesIO(preparation.thumbnail)) as image:
        image.load()
        assert image.format == "JPEG"
        assert image.width > 120
    env.artwork.async_delete.assert_not_awaited()


async def test_one_bad_photo_blocks_original_whole_group(hass, deletion):
    env = deletion
    env.artwork.async_prepare_delete.side_effect = [
        env.preview,
        HomeAssistantError("Displayed photo"),
    ]
    assert (await prepare_batch(hass, env))["step_id"] == "delete_blocked"
    result = await env.flow.async_step_delete_blocked()
    assert [item["content_id"] for item in rows(result)] == ["MY_NEW", "MY_SECOND"]
    assert rows(result)[1]["reason"] == "Displayed photo"
    assert result["description_placeholders"]["content_id"] == "MY_SECOND"
    assert (await env.flow.async_step_delete_confirm({}))["reason"] == "delete_expired"
    env.artwork.async_delete.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["absent", "present", "uncertain", "not_sent", "exception"])
async def test_batch_continues_only_after_absence_and_never_repeats(hass, deletion, outcome):
    from custom_components.localthings.tv_artwork_delete import ArtworkDeleteNotSentError

    env = deletion
    previews = batch_previews(env)
    await prepare_batch(hass, env)

    async def remove(preview):
        if outcome == "not_sent":
            raise ArtworkDeleteNotSentError("Record save failed")
        if outcome == "exception":
            raise TimeoutError("private")
        return SimpleNamespace(content_id=preview.content_id, outcome=outcome)

    env.artwork.async_delete.side_effect = remove
    await env.flow.async_step_delete_confirm({})
    await hass.async_block_till_done()
    result = await env.flow.async_step_delete_confirm({})
    expected_step = "delete_complete" if outcome == "absent" else "delete_partial"
    assert result["step_id"] == expected_step
    result = await getattr(env.flow, f"async_step_{expected_step}")()
    expected = "uncertain" if outcome == "exception" else outcome
    assert [item["outcome"] for item in rows(result)] == [
        expected,
        "absent" if outcome == "absent" else "not_attempted",
    ]
    assert env.artwork.async_delete.await_count == (2 if outcome == "absent" else 1)
    assert env.artwork.async_delete.await_args_list[0].args[0] == replace(
        previews[0], thumbnail=b""
    )
    assert preview_store(hass)._entries == {}


@pytest.mark.parametrize(
    "change", ["expiry", "logout", "demotion", "inactive", "unload", "coordinator", "shutdown"]
)
async def test_batch_live_admission_stops_later_photo(hass, deletion, change):
    from homeassistant.core import CoreState

    env = deletion
    batch_previews(env)
    await prepare_batch(hass, env)
    preparation = preview_store(hass)._entries[env.entry.entry_id]

    async def remove(preview):
        assert preparation.processing and not preparation.disposed
        assert not preparation.thumbnail
        with pytest.raises(ValueError):
            preview_store(hass).reserve(env.entry.entry_id, "other", "admin", "refresh")
        if change == "expiry":
            preparation.expires_at = hass.loop.time() - 1
        elif change == "logout":
            hass.auth.async_get_refresh_token.return_value = None
        elif change == "demotion":
            env.user.is_admin = False
        elif change == "inactive":
            env.user.is_active = False
        elif change == "unload":
            preview_store(hass).dispose_entry(env.entry.entry_id)
        elif change == "coordinator":
            hass.data[DOMAIN][env.entry.entry_id] = SimpleNamespace(artwork=env.artwork)
        else:
            hass.set_state(CoreState.stopping)
        return SimpleNamespace(content_id=preview.content_id, outcome="absent")

    env.artwork.async_delete.side_effect = remove
    await env.flow.async_step_delete_confirm({})
    await hass.async_block_till_done()
    env.artwork.async_delete.assert_awaited_once()
    assert env.flow._delete_task.done()
    assert preview_store(hass)._entries == {}
    if change == "shutdown":
        hass.set_state(CoreState.not_running)


async def test_exact_batch_recovery_uses_references_and_missing_is_uncertain(hass, deletion):
    env = deletion
    env.artwork.async_delete_status.side_effect = [
        None,
        SimpleNamespace(content_id="MY_SECOND", outcome="absent"),
    ]
    references = [
        {"content_id": "MY_NEW", "record_token": "old-generation"},
        {"content_id": "MY_SECOND", "record_token": "second-generation"},
    ]
    await env.flow.async_step_init()
    await env.flow.async_step_delete_artworks_status({"references": references})
    await hass.async_block_till_done()
    assert (await env.flow.async_step_delete_progress())["step_id"] == "delete_partial"
    result = await env.flow.async_step_delete_partial()
    assert [item["outcome"] for item in rows(result)] == ["uncertain", "absent"]
    assert "record" in rows(result)[0]["reason"].lower()
    assert [
        (call.args, call.kwargs) for call in env.artwork.async_delete_status.await_args_list
    ] == [
        (("MY_NEW",), {"record_token": "old-generation"}),
        (("MY_SECOND",), {"record_token": "second-generation"}),
    ]
    env.artwork.async_prepare_delete.assert_not_awaited()
    env.artwork.async_delete.assert_not_awaited()


def test_contact_sheet_labels_exact_checked_previews_in_order(deletion):
    from PIL import ImageDraw

    from custom_components.localthings.tv_artwork_delete import delete_contact_sheet

    previews = batch_previews(deletion)
    blue = io.BytesIO()
    Image.new("RGB", (120, 80), "blue").save(blue, format="PNG")
    previews = (previews[0], replace(previews[1], thumbnail=blue.getvalue()))
    labels = []
    draw_text = ImageDraw.ImageDraw.text

    def draw(self, xy, text, *args, **kwargs):
        labels.append(text)
        return draw_text(self, xy, text, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "text", draw):
        data = delete_contact_sheet(previews)
    assert labels == ["1. MY_NEW", "2. MY_SECOND"]
    with Image.open(io.BytesIO(data)) as image:
        first = image.getpixel((220, 100))
        second = image.getpixel((700, 100))
        assert isinstance(first, tuple) and first[0] > 200
        assert isinstance(second, tuple) and second[2] > 200


def test_contact_sheet_never_clips_the_full_samsung_id(deletion):
    from PIL import ImageDraw

    from custom_components.localthings.tv_artwork_delete import delete_contact_sheet

    preview = replace(deletion.preview, content_id="MY_" + "W" * 240)
    labels = []
    draw_text = ImageDraw.ImageDraw.text

    def draw(self, xy, text, *args, **kwargs):
        labels.append(text)
        assert kwargs["font"].getlength(text) <= 440
        return draw_text(self, xy, text, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "text", draw):
        delete_contact_sheet((preview,))
    assert "".join(labels) == "1. " + preview.content_id


async def test_malformed_checked_thumbnail_blocks_confirmation(hass, deletion):
    env = deletion
    env.artwork.async_prepare_delete.return_value = ArtworkDeletePreview(
        "MY_NEW", "Photo", "owned-generation", b"damaged bytes", "image/jpeg"
    )
    await env.flow.async_step_init()
    await env.flow.async_step_delete_artwork({"content_id": "MY_NEW"})
    await hass.async_block_till_done()
    assert (await env.flow.async_step_delete_prepare())["step_id"] == "delete_blocked"
    env.artwork.async_delete.assert_not_awaited()


async def test_duplicate_generation_blocks_entire_group(hass, deletion):
    env = deletion
    env.artwork.async_prepare_delete.side_effect = [
        env.preview,
        ArtworkDeletePreview(
            "MY_SECOND", "Second", env.preview.record_token, env.preview.thumbnail, "image/png"
        ),
    ]
    assert (await prepare_batch(hass, env))["step_id"] == "delete_blocked"
    env.artwork.async_delete.assert_not_awaited()


@pytest.mark.parametrize("step", ["delete_artworks", "delete_artworks_status"])
async def test_batch_input_schema_serializes_in_home_assistant(hass, deletion, step):
    from homeassistant.helpers import config_validation as cv
    from probatio import to_field_list

    result = await getattr(deletion.flow, f"async_step_{step}")()
    serialized = to_field_list(result["data_schema"], custom_serializer=cv.custom_serializer)
    assert serialized
    json.dumps(serialized)


async def test_cancelled_batch_preserves_uncertainty_and_does_not_admit_next(hass, deletion):
    env = deletion
    batch_previews(env)
    await prepare_batch(hass, env)
    entered, release = asyncio.Event(), asyncio.Event()

    async def remove(_preview):
        entered.set()
        await release.wait()

    env.artwork.async_delete.side_effect = remove
    await env.flow.async_step_delete_confirm({})
    await entered.wait()
    env.flow._delete_task.cancel()
    await asyncio.gather(env.flow._delete_task, env.flow._delete_waiter, return_exceptions=True)
    assert [item["outcome"] for item in env.flow._delete_items] == ["uncertain", "not_attempted"]
    env.artwork.async_delete.assert_awaited_once()
    assert preview_store(hass)._entries == {}


async def test_cancelled_contact_sheet_keeps_reservation_until_executor_drains(hass, deletion):
    import threading

    env = deletion
    started, release = asyncio.Event(), threading.Event()

    def sheet(_previews):
        hass.loop.call_soon_threadsafe(started.set)
        release.wait(5)
        return b"sheet"

    with patch(
        "custom_components.localthings.tv_artwork_delete_flow.delete_contact_sheet", new=sheet
    ):
        await env.flow.async_step_delete_artwork({"content_id": "MY_NEW"})
        await started.wait()
        preparation = preview_store(hass)._entries[env.entry.entry_id]
        env.flow._delete_prepare_task.cancel()
        env.flow.async_remove()
        await asyncio.sleep(0)
        try:
            with pytest.raises(ValueError):
                preview_store(hass).reserve(env.entry.entry_id, "other", "admin", "refresh")
            assert not env.flow._delete_prepare_task.done()
        finally:
            release.set()
        await asyncio.gather(
            env.flow._delete_prepare_task, env.flow._delete_prepare_waiter, return_exceptions=True
        )
    assert preparation.disposed
    assert preview_store(hass)._entries == {}


async def test_confirm_clears_checked_bytes_but_keeps_frozen_references(hass, deletion):
    env = deletion
    batch_previews(env)
    await prepare_batch(hass, env)
    entered, release = asyncio.Event(), asyncio.Event()

    async def remove(preview):
        entered.set()
        await release.wait()
        return SimpleNamespace(content_id=preview.content_id, outcome="absent")

    env.artwork.async_delete.side_effect = remove
    await env.flow.async_step_delete_confirm({})
    await entered.wait()
    try:
        assert env.flow._delete_previews
        assert all(not preview.thumbnail for preview in env.flow._delete_previews)
    finally:
        release.set()
        await hass.async_block_till_done()
    assert [item["record_token"] for item in env.flow._delete_items] == [
        "owned-generation",
        "second-generation",
    ]


async def test_close_after_confirmation_keeps_server_owned_group_running(hass, deletion):
    env = deletion
    batch_previews(env)
    await prepare_batch(hass, env)
    entered, release = asyncio.Event(), asyncio.Event()

    async def remove(preview):
        entered.set()
        await release.wait()
        return SimpleNamespace(content_id=preview.content_id, outcome="absent")

    env.artwork.async_delete.side_effect = remove
    await env.flow.async_step_delete_confirm({})
    await entered.wait()
    env.flow._delete_waiter.cancel()
    await asyncio.gather(env.flow._delete_waiter, return_exceptions=True)
    env.flow.async_remove()
    assert not env.flow._delete_preparation.disposed
    release.set()
    await hass.async_block_till_done()
    assert [item["outcome"] for item in env.flow._delete_items] == ["absent", "absent"]
    assert env.artwork.async_delete.await_count == 2
    assert preview_store(hass)._entries == {}


@pytest.mark.parametrize("change", ["logout", "demotion", "inactive", "token_expiry"])
async def test_first_photo_requires_live_authorization_after_confirmation(hass, deletion, change):
    env = deletion
    batch_previews(env)
    await prepare_batch(hass, env)
    if change == "logout":
        hass.auth.async_get_refresh_token.return_value = None
    elif change == "demotion":
        env.user.is_admin = False
    elif change == "inactive":
        env.user.is_active = False
    else:
        env.refresh_token.expire_at = 1
    await env.flow.async_step_delete_confirm({})
    await hass.async_block_till_done()
    assert [item["outcome"] for item in env.flow._delete_items] == [
        "not_attempted",
        "not_attempted",
    ]
    env.artwork.async_delete.assert_not_awaited()
    assert preview_store(hass)._entries == {}
