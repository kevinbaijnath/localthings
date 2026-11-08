"""Private browser artwork preparation never sends before confirmation."""

import asyncio
import io
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp.test_utils import make_mocked_request
from homeassistant.config_entries import ConfigEntryState
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.http import current_request
from PIL import Image
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.localthings.config_flow import LocalThingsConfigFlow
from custom_components.localthings.const import DOMAIN
from custom_components.localthings.tv_artwork_image import PreparedArtwork
from custom_components.localthings.tv_artwork_preview import preview_store


def admin_request(user_id="admin", token_id="refresh", is_admin=True):
    request = make_mocked_request("POST", "/api/config/config_entries/options/flow")
    request["hass_user"] = SimpleNamespace(id=user_id, is_admin=is_admin)
    request["hass_refresh_token_id"] = token_id
    return request


@pytest.fixture
def authenticated_flow(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"transport": "tv_ws"}, options={"keep": True})
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    artwork = SimpleNamespace(async_upload=AsyncMock())
    hass.data[DOMAIN] = {entry.entry_id: SimpleNamespace(artwork=artwork, async_close=AsyncMock())}
    flow = LocalThingsConfigFlow.async_get_options_flow(entry)
    flow.hass = hass
    flow.handler = entry.entry_id
    flow.flow_id = "synthetic-flow"
    token = current_request.set(admin_request())
    yield flow, artwork, entry
    current_request.reset(token)
    flow.async_remove()
    entry.mock_state(hass, ConfigEntryState.NOT_LOADED)


def prepared(crop_axis="none"):
    return PreparedArtwork(
        b"exact prepared JPEG", b"local thumbnail", 4000, 3000, 4000, 2250, crop_axis, "digest"
    )


async def seed_preparation(hass, flow, image):
    await flow.async_step_init()
    preparation = preview_store(hass).reserve(
        flow.config_entry.entry_id, flow.flow_id, "admin", "refresh"
    )
    preparation.image = image
    preparation.source = b"source for recrop"
    flow._preparation = preparation
    return preparation


async def test_tv_options_requires_authenticated_request(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"transport": "tv_ws"})
    entry.add_to_hass(hass)
    flow = LocalThingsConfigFlow.async_get_options_flow(entry)
    flow.hass = hass
    flow.handler = entry.entry_id
    flow.flow_id = "synthetic-flow"
    result = await flow.async_step_init()
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "upload_unauthorized"


async def test_tv_options_offers_upload_without_contacting_tv(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"transport": "tv_ws"})
    entry.add_to_hass(hass)
    entry.mock_state(hass, ConfigEntryState.LOADED)
    artwork = SimpleNamespace(async_upload=AsyncMock())
    hass.data[DOMAIN] = {entry.entry_id: SimpleNamespace(artwork=artwork)}
    flow = LocalThingsConfigFlow.async_get_options_flow(entry)
    flow.hass = hass
    flow.handler = entry.entry_id
    flow.flow_id = "synthetic-flow"
    token = current_request.set(admin_request())
    try:
        result = await flow.async_step_init()
    finally:
        current_request.reset(token)
    assert result["type"] == FlowResultType.MENU
    assert result["menu_options"] == [
        "upload_artwork",
        "delete_artwork",
        "delete_artworks",
        "delete_status",
        "delete_artworks_status",
        "upload_finish",
    ]
    artwork.async_upload.assert_not_awaited()


async def test_recrop_returns_directly_to_exact_prepared_review(hass, authenticated_flow):
    flow, artwork, _ = authenticated_flow
    preparation = await seed_preparation(hass, flow, prepared("vertical"))
    original_token = preparation.token
    with patch(
        "custom_components.localthings.tv_artwork_image.prepare_image",
        return_value=prepared("vertical"),
    ) as decode:
        result = await flow.async_step_upload_crop({"position": "top"})
        assert result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE)
        await hass.async_block_till_done()
        result = await flow.async_step_upload_prepare()
    assert result["step_id"] == "upload_review"
    assert preparation.token != original_token
    assert preview_store(hass).get(original_token) is None
    assert preparation.image.data == b"exact prepared JPEG"
    decode.assert_called_once_with(b"source for recrop", "top")
    artwork.async_upload.assert_not_awaited()


@pytest.mark.parametrize(
    "confirmed,step", [(True, "upload_complete"), (False, "upload_unconfirmed")]
)
async def test_confirmation_sends_once_and_preserves_options(
    hass, authenticated_flow, confirmed, step
):
    flow, artwork, entry = authenticated_flow
    preparation = await seed_preparation(hass, flow, prepared())
    artwork.async_upload.return_value = SimpleNamespace(
        content_id="MY_F0001", catalog_confirmed=confirmed
    )
    result = await flow.async_step_upload_review()
    assert result["step_id"] == "upload_review"
    result = await flow.async_step_upload_confirm()
    assert result["type"] == FlowResultType.FORM
    artwork.async_upload.assert_not_awaited()
    result = await flow.async_step_upload_confirm({})
    assert preview_store(hass).get(preparation.token) is None
    assert preparation.image is None
    assert preparation.source == b""
    await hass.async_block_till_done()
    result = await flow.async_step_upload_confirm({})
    assert result["step_id"] == step
    artwork.async_upload.assert_awaited_once_with(b"exact prepared JPEG", b"local thumbnail")
    result = await flow._result_step(None)
    assert result["description_placeholders"]["content_id"] == "MY_F0001"
    result = await flow._result_step({})
    assert result["type"] == FlowResultType.ABORT
    assert dict(entry.options) == {"keep": True}


async def test_unknown_upload_failure_remains_consumed(hass, authenticated_flow):
    flow, artwork, _ = authenticated_flow
    await seed_preparation(hass, flow, prepared())
    artwork.async_upload.side_effect = TimeoutError
    await flow.async_step_upload_confirm({})
    await hass.async_block_till_done()
    result = await flow.async_step_upload_confirm({})
    assert result["step_id"] == "upload_uncertain"
    await flow.async_step_upload_artwork({"file": "never-reprocess"})
    artwork.async_upload.assert_awaited_once()


async def test_cancelled_browser_does_not_cancel_in_flight_upload(hass, authenticated_flow):
    flow, artwork, _ = authenticated_flow
    await seed_preparation(hass, flow, prepared())
    started = asyncio.Event()
    release = asyncio.Event()

    async def upload(data, thumbnail):
        started.set()
        await release.wait()
        return SimpleNamespace(content_id="MY_F0001", catalog_confirmed=True)

    artwork.async_upload.side_effect = upload
    await flow.async_step_upload_confirm({})
    await started.wait()
    flow._upload_waiter.cancel()
    flow.async_remove()
    await asyncio.sleep(0)
    assert not flow._upload_task.cancelled()
    assert not flow._upload_task.done()
    release.set()
    await flow._upload_task
    artwork.async_upload.assert_awaited_once()


async def test_file_selector_consumes_native_file_without_upload(
    hass, authenticated_flow, tmp_path
):
    flow, artwork, _ = authenticated_flow
    await flow.async_step_init()
    path = tmp_path / "native-upload"
    path.write_bytes(b"synthetic original")

    @contextmanager
    def native_file(hass_arg, file_id):
        assert hass_arg is hass
        assert file_id == "native-id"
        try:
            yield path
        finally:
            path.unlink()

    with (
        patch("custom_components.localthings.tv_artwork_upload.process_uploaded_file", native_file),
        patch(
            "custom_components.localthings.tv_artwork_image.prepare_image", return_value=prepared()
        ) as decode,
    ):
        result = await flow.async_step_upload_artwork({"file": "native-id"})
        assert result["type"] in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE)
        await hass.async_block_till_done()
        result = await flow.async_step_upload_prepare()
    assert result["step_id"] == "upload_review"
    assert not path.exists()
    assert flow._preparation.source == b""
    decode.assert_called_once_with(b"synthetic original")
    artwork.async_upload.assert_not_awaited()


async def test_quality_failure_explains_the_problem(hass, authenticated_flow, tmp_path):
    from custom_components.localthings.tv_artwork_image import ImageValidationError

    flow, artwork, _ = authenticated_flow
    await flow.async_step_init()
    with patch(
        "custom_components.localthings.tv_artwork_upload._read_and_prepare",
        side_effect=ImageValidationError(
            "The crop must contain at least 3840 x 2160 source pixels."
        ),
    ):
        await flow.async_step_upload_artwork({"file": "native-id"})
        await hass.async_block_till_done()
        result = await flow.async_step_upload_prepare()
        assert result["step_id"] == "upload_stop"
        result = await flow.async_step_upload_stop()
    assert "3840" in result["description_placeholders"]["reason"]
    artwork.async_upload.assert_not_awaited()


async def test_cancelling_preparation_keeps_admission_until_executor_finishes(
    hass, authenticated_flow
):
    flow, artwork, entry = authenticated_flow
    await flow.async_step_init()
    started = threading.Event()
    release = threading.Event()

    def processing(hass_arg, file_id):
        started.set()
        assert release.wait(5)
        return b"private original", prepared()

    with patch("custom_components.localthings.tv_artwork_upload._read_and_prepare", processing):
        await flow.async_step_upload_artwork({"file": "native-id"})
        assert await hass.async_add_executor_job(started.wait, 5)
        preparation = flow._preparation
        flow._prepare_waiter.cancel()
        flow.async_remove()
        with pytest.raises(ValueError):
            preview_store(hass).reserve(entry.entry_id, "another-flow", "admin", "refresh")
        release.set()
        await flow._prepare_task
    replacement = preview_store(hass).reserve(entry.entry_id, "another-flow", "admin", "refresh")
    assert preparation.image is None
    assert preparation.source == b""
    flow.async_remove()
    assert preview_store(hass).get(replacement.token) is replacement
    preview_store(hass).dispose(replacement)
    artwork.async_upload.assert_not_awaited()


async def test_unloaded_entry_blocks_send_and_clears_preview(hass, authenticated_flow):
    from custom_components import localthings as integration

    flow, artwork, entry = authenticated_flow
    preparation = await seed_preparation(hass, flow, prepared())
    coordinator = hass.data[DOMAIN][entry.entry_id]
    coordinator.async_close = AsyncMock()
    with patch.object(
        hass.config_entries, "async_unload_platforms", new=AsyncMock(return_value=True)
    ):
        assert await integration.async_unload_entry(hass, entry)
    assert preview_store(hass).get(preparation.token) is None
    result = await flow.async_step_upload_confirm({})
    assert result["type"] == FlowResultType.ABORT
    artwork.async_upload.assert_not_awaited()


@pytest.mark.parametrize("change", ["other_user", "other_token", "nonadmin", "missing"])
async def test_later_confirmation_rechecks_initiating_identity(hass, authenticated_flow, change):
    flow, artwork, _ = authenticated_flow
    await seed_preparation(hass, flow, prepared())
    request = admin_request(
        user_id="other" if change == "other_user" else "admin",
        is_admin=change != "nonadmin",
        token_id="other" if change == "other_token" else "refresh",
    )
    token = current_request.set(None if change == "missing" else request)
    try:
        result = await flow.async_step_upload_confirm({})
    finally:
        current_request.reset(token)
    assert result["reason"] == "upload_unauthorized"
    artwork.async_upload.assert_not_awaited()


@pytest.mark.parametrize(
    "outcome", ["upload_complete", "upload_unconfirmed", "upload_uncertain", "upload_not_sent"]
)
async def test_native_manager_does_not_skip_immediate_upload_result(
    hass, authenticated_flow, outcome
):
    _, artwork, entry = authenticated_flow
    result = await hass.config_entries.options.async_init(entry.entry_id)
    flow_id = result["flow_id"]
    try:
        await hass.config_entries.options.async_configure(
            flow_id, {"next_step_id": "upload_artwork"}
        )
        with patch(
            "custom_components.localthings.tv_artwork_upload._read_and_prepare",
            return_value=(b"source", prepared()),
        ):
            result = await hass.config_entries.options.async_configure(
                flow_id, {"file": "55a1ce6f542b4c3bbefb3d9be34d6bc1"}
            )
            if result["type"] == FlowResultType.SHOW_PROGRESS:
                await hass.async_block_till_done()
                result = await hass.config_entries.options.async_configure(flow_id)
        assert result["step_id"] == "upload_review"
        result = await hass.config_entries.options.async_configure(
            flow_id, {"next_step_id": "upload_confirm"}
        )
    except BaseException:
        hass.config_entries.options.async_abort(flow_id)
        raise
    assert result["type"] == FlowResultType.FORM
    if outcome == "upload_uncertain":
        artwork.async_upload.side_effect = TimeoutError
    elif outcome == "upload_not_sent":
        from custom_components.localthings.tv_artwork_cache import ArtworkUploadNotSentError

        artwork.async_upload.side_effect = ArtworkUploadNotSentError("Artwork upload was not sent")
    else:
        artwork.async_upload.return_value = SimpleNamespace(
            content_id="MY_F0001", catalog_confirmed=outcome == "upload_complete"
        )
    result = await hass.config_entries.options.async_configure(flow_id, {})
    assert result["type"] == FlowResultType.SHOW_PROGRESS
    await hass.async_block_till_done()
    result = await hass.config_entries.options.async_configure(flow_id)
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == outcome
    artwork.async_upload.assert_awaited_once()
    hass.config_entries.options.async_abort(flow_id)


async def test_native_manager_shows_quality_error_after_preparation_progress(
    hass, authenticated_flow
):
    from custom_components.localthings.tv_artwork_image import ImageValidationError

    _, artwork, entry = authenticated_flow
    result = await hass.config_entries.options.async_init(entry.entry_id)
    flow_id = result["flow_id"]
    await hass.config_entries.options.async_configure(flow_id, {"next_step_id": "upload_artwork"})
    with patch(
        "custom_components.localthings.tv_artwork_upload._read_and_prepare",
        side_effect=ImageValidationError("The crop must contain 4K source pixels."),
    ):
        await hass.config_entries.options.async_configure(
            flow_id, {"file": "55a1ce6f542b4c3bbefb3d9be34d6bc1"}
        )
        await hass.async_block_till_done()
        try:
            result = await hass.config_entries.options.async_configure(flow_id)
        except BaseException:
            hass.config_entries.options.async_abort(flow_id)
            raise
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "upload_invalid_image"
    assert result["description_placeholders"]["reason"] == "The crop must contain 4K source pixels."
    artwork.async_upload.assert_not_awaited()


@pytest.mark.parametrize("oversize", [False, True])
async def test_native_file_cleanup_and_bounded_read(hass, tmp_path, oversize):
    from homeassistant.components.file_upload import FileUploadData

    from custom_components.localthings.tv_artwork_image import ImageValidationError
    from custom_components.localthings.tv_artwork_upload import _read_and_prepare

    file_id = "55a1ce6f542b4c3bbefb3d9be34d6bc1"
    directory = tmp_path / file_id
    directory.mkdir()
    (directory / "source.jpg").write_bytes(b"12345" if oversize else b"123")
    upload_data = FileUploadData(tmp_path, {file_id: "source.jpg"})
    hass.data["file_upload"] = upload_data
    with (
        patch("custom_components.localthings.tv_artwork_upload._MAX_SOURCE_BYTES", 4),
        patch(
            "custom_components.localthings.tv_artwork_image.prepare_image", return_value=prepared()
        ) as decode,
    ):
        if oversize:
            with pytest.raises(ImageValidationError):
                await hass.async_add_executor_job(_read_and_prepare, hass, file_id)
            decode.assert_not_called()
        else:
            source, image = await hass.async_add_executor_job(_read_and_prepare, hass, file_id)
            assert source == b"123"
            assert image.data == b"exact prepared JPEG"
            decode.assert_called_once_with(b"123")
    assert file_id not in upload_data.files
    assert not directory.exists()


async def test_confirm_during_entry_unload_does_not_admit_send(hass, authenticated_flow):
    flow, artwork, entry = authenticated_flow
    await seed_preparation(hass, flow, prepared())
    entry.mock_state(hass, ConfigEntryState.UNLOAD_IN_PROGRESS)
    result = await flow.async_step_upload_confirm({})
    assert result["reason"] == "not_loaded"
    artwork.async_upload.assert_not_awaited()


async def test_definite_presend_rejection_reports_not_sent_and_remains_consumed(
    hass, authenticated_flow
):
    from custom_components.localthings.tv_artwork_cache import ArtworkUploadNotSentError

    flow, artwork, _ = authenticated_flow
    preparation = await seed_preparation(hass, flow, prepared())
    artwork.async_upload.side_effect = ArtworkUploadNotSentError("Artwork upload was not sent")
    await flow.async_step_upload_confirm({})
    await hass.async_block_till_done()
    result = await flow.async_step_upload_confirm({})
    assert result["step_id"] == "upload_not_sent"
    assert preview_store(hass).get(preparation.token) is None
    assert preparation.source == b""
    assert preparation.image is None
    result = await flow.async_step_upload_not_sent()
    assert result["type"] == FlowResultType.FORM
    assert "not sent" in result["description_placeholders"]["reason"]
    artwork.async_upload.assert_awaited_once()


async def test_native_flow_prepares_real_image_and_reviews_requested_crop(
    hass, authenticated_flow, tmp_path
):
    from homeassistant.components.file_upload import FileUploadData

    from custom_components.localthings.tv_artwork_preview import ArtworkPreviewView

    _, artwork, entry = authenticated_flow
    file_id = "55a1ce6f542b4c3bbefb3d9be34d6bc1"
    directory = tmp_path / file_id
    directory.mkdir()
    with Image.new("RGB", (4000, 3000), "black") as source:
        source.paste("red", (0, 0, 4000, 750))
        source.paste("blue", (0, 2250, 4000, 3000))
        source.save(directory / "source.png", format="PNG")
    hass.data["file_upload"] = FileUploadData(tmp_path, {file_id: "source.png"})
    result = await hass.config_entries.options.async_init(entry.entry_id)
    flow_id = result["flow_id"]
    try:
        result = await hass.config_entries.options.async_configure(
            flow_id, {"next_step_id": "upload_artwork"}
        )
        assert result["step_id"] == "upload_artwork"
        result = await hass.config_entries.options.async_configure(flow_id, {"file": file_id})
        assert result["type"] == FlowResultType.SHOW_PROGRESS
        await hass.async_block_till_done()
        result = await hass.config_entries.options.async_configure(flow_id)
        assert result["step_id"] == "upload_crop"
        result = await hass.config_entries.options.async_configure(flow_id, {"position": "bottom"})
        assert result["type"] == FlowResultType.SHOW_PROGRESS
        await hass.async_block_till_done()
        result = await hass.config_entries.options.async_configure(flow_id)
        assert result["step_id"] == "upload_review"
        flow = hass.config_entries.options._progress[flow_id]
        preparation = flow._preparation
        request = make_mocked_request("GET", "/api/localthings/artwork/preview/synthetic")
        request["hass_user"] = SimpleNamespace(id="admin", is_admin=True)
        preview = await ArtworkPreviewView(preview_store(hass)).get(request, preparation.token)
        assert preview.body == preparation.image.data
        assert isinstance(preview.body, bytes)
        with Image.open(io.BytesIO(preview.body)) as image:
            assert image.size == (3840, 2160)
            assert image.format == "JPEG"
            assert image.getpixel((100, 100)) == (0, 0, 0)
            bottom_pixel = image.getpixel((100, 2100))
            assert isinstance(bottom_pixel, tuple)
            assert bottom_pixel[2] > 240
        assert not directory.exists()
        assert preparation.image.crop_width == 4000
        assert preparation.image.crop_height == 2250
        artwork.async_upload.assert_not_awaited()
    finally:
        hass.config_entries.options.async_abort(flow_id)
