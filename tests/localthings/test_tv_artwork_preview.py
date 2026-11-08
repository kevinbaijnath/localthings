"""Prepared artwork stays private and expires without contacting a TV."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from custom_components.localthings.tv_artwork_image import PreparedArtwork


def prepared(data):
    return PreparedArtwork(data, b"thumbnail", 3840, 2160, 3840, 2160, "none", "synthetic-digest")


async def test_preview_requires_owner_and_disposal_removes_bytes(hass):
    from custom_components.localthings.tv_artwork_preview import (
        ArtworkPreviewView,
        preview_store,
    )

    store = preview_store(hass)
    preparation = store.reserve("frame", "flow", "owner", "refresh")
    preparation.image = prepared(b"exact prepared jpeg")
    view = ArtworkPreviewView(store)
    request = make_mocked_request("GET", "/api/localthings/artwork/preview/synthetic")
    request["hass_user"] = SimpleNamespace(id="other", is_admin=True)
    with pytest.raises(web.HTTPNotFound):
        await view.get(request, preparation.token)
    request["hass_user"] = SimpleNamespace(id="owner", is_admin=True)
    response = await view.get(request, preparation.token)
    assert response.body == b"exact prepared jpeg"
    assert response.content_type == "image/jpeg"
    assert response.headers["Cache-Control"] == "no-store"
    store.dispose(preparation)
    assert preparation.image is None
    assert preparation.source == b""
    with pytest.raises(web.HTTPNotFound):
        await view.get(request, preparation.token)


async def test_processing_reservation_survives_cancel_until_executor_finishes(hass):
    from custom_components.localthings.tv_artwork_preview import preview_store

    store = preview_store(hass)
    preparation = store.reserve("frame", "first", "owner", "refresh")
    preparation.processing = True
    store.dispose(preparation)
    with pytest.raises(ValueError):
        store.reserve("frame", "second", "owner", "refresh")
    store.processing_finished(preparation)
    replacement = store.reserve("frame", "second", "owner", "refresh")
    store.dispose(preparation)
    assert store.get(replacement.token) is replacement
    store.dispose_entry("frame")
    assert store.get(replacement.token) is None


async def test_preview_expiry_clears_original_and_prepared_bytes(hass):
    from custom_components.localthings.tv_artwork_preview import ArtworkPreviewView, preview_store

    store = preview_store(hass)
    preparation = store.reserve("frame", "flow", "owner", "refresh")
    preparation.source = b"private original"
    preparation.image = prepared(b"private JPEG")
    preparation.expires_at = hass.loop.time() - 1
    request = make_mocked_request("GET", "/api/localthings/artwork/preview/synthetic")
    request["hass_user"] = SimpleNamespace(id="owner", is_admin=True)
    with pytest.raises(web.HTTPNotFound):
        await ArtworkPreviewView(store).get(request, preparation.token)
    assert preparation.source == b""
    assert preparation.image is None
    assert preparation.timer is None


async def test_signing_uses_initiating_refresh_token_and_view_registers_once(hass):
    from custom_components.localthings.tv_artwork_preview import preview_store, register_preview

    store = preview_store(hass)
    preparation = store.reserve("frame", "flow", "owner", "specific-refresh-token")
    with patch("custom_components.localthings.tv_artwork_preview.async_sign_path") as sign:
        store.sign(preparation)
    assert sign.call_args.kwargs["refresh_token_id"] == "specific-refresh-token"
    assert 0 < sign.call_args.args[2].total_seconds() <= 600
    with patch.object(type(hass), "http", new=Mock(register_view=Mock())) as http:
        register_preview(hass)
        register_preview(hass)
    view = http.register_view.call_args.args[0]
    assert view.requires_auth is True
    assert not hasattr(view, "post")
    http.register_view.assert_called_once()
    store.dispose(preparation)


@pytest.mark.parametrize("user", [None, SimpleNamespace(id="owner", is_admin=False)])
async def test_preview_denies_missing_or_nonadmin_identity(hass, user):
    from custom_components.localthings.tv_artwork_preview import ArtworkPreviewView, preview_store

    store = preview_store(hass)
    preparation = store.reserve("frame", "flow", "owner", "refresh")
    preparation.image = prepared(b"private")
    request = make_mocked_request("GET", "/api/localthings/artwork/preview/synthetic")
    request["hass_user"] = user
    with pytest.raises(web.HTTPNotFound):
        await ArtworkPreviewView(store).get(request, preparation.token)
    store.dispose(preparation)


async def test_delete_thumbnail_preview_is_private_and_disposed(hass):
    from custom_components.localthings.tv_artwork_preview import ArtworkPreviewView, preview_store

    store = preview_store(hass)
    preparation = store.reserve("frame", "flow", "owner", "refresh")
    preparation.thumbnail = b"fresh checked thumbnail"
    preparation.thumbnail_mime_type = "image/png"
    request = make_mocked_request("GET", "/api/localthings/artwork/preview/synthetic")
    request["hass_user"] = SimpleNamespace(id="owner", is_admin=True)
    response = await ArtworkPreviewView(store).get(request, preparation.token)
    assert response.body == b"fresh checked thumbnail"
    assert response.content_type == "image/png"
    assert response.headers["Cache-Control"] == "no-store"
    request["hass_user"] = SimpleNamespace(id="other", is_admin=True)
    with pytest.raises(web.HTTPNotFound):
        await ArtworkPreviewView(store).get(request, preparation.token)
    store.dispose(preparation)
    assert preparation.thumbnail == b""
    with pytest.raises(web.HTTPNotFound):
        await ArtworkPreviewView(store).get(request, preparation.token)
