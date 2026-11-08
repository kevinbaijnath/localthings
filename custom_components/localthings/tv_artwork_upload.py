"""Native administrator-only preparation and single-send artwork upload flow."""

from __future__ import annotations

import asyncio
import secrets
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.components.file_upload import process_uploaded_file
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.selector import (
    FileSelector,
    FileSelectorConfig,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .tv_artwork_cache import ArtworkUploadNotSentError
from .tv_artwork_delete_flow import TvArtworkDeleteFlow
from .tv_artwork_preview import Preparation, preview_store

if TYPE_CHECKING:
    from .tv_artwork_image import PreparedArtwork

_MAX_SOURCE_BYTES = 30 * 1024 * 1024


def _read_and_prepare(hass: HomeAssistant, file_id: str) -> tuple[bytes, PreparedArtwork]:
    """Consume HA's temporary file entirely inside its executor context."""
    from .tv_artwork_image import ImageValidationError, prepare_image

    with process_uploaded_file(hass, file_id) as path, path.open("rb") as source_file:
        source = source_file.read(_MAX_SOURCE_BYTES + 1)
        if len(source) > _MAX_SOURCE_BYTES:
            raise ImageValidationError("The original file exceeds 30 MiB")
        return source, prepare_image(source)


async def _wait_for(task: asyncio.Task[Any]) -> None:
    """Let the flow manager cancel its waiter without cancelling the operation."""
    await asyncio.shield(task)


class TvArtworkUploadFlow(TvArtworkDeleteFlow):
    """Prepare privately, confirm once, and finish without changing options."""

    def __init__(self) -> None:
        super().__init__()
        self._preparation: Preparation | None = None
        self._prepare_task: asyncio.Task[str | None] | None = None
        self._prepare_waiter: asyncio.Task[None] | None = None
        self._upload_task: asyncio.Task[None] | None = None
        self._upload_waiter: asyncio.Task[None] | None = None
        self._outcome = "upload_uncertain"
        self._content_id = ""
        self._not_sent_reason = ""
        self._position = "center"
        self._prepare_next = "upload_review"
        self._validation_reason = "Choose a valid standard sRGB JPEG or PNG."
        self._stop_reason = "upload_expired"
        self._stop_placeholders: dict[str, str] | None = None

    def _live_preparation(self) -> Preparation | None:
        preparation = self._preparation
        if preparation is None:
            return None
        if (
            preview_store(self.hass).get(preparation.token) is preparation
            and preparation.flow_id == self.flow_id
            and preparation.entry_id == self.config_entry.entry_id
            and self._owner == (preparation.owner_id, preparation.refresh_token_id)
        ):
            return preparation
        return None

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="upload_unauthorized")
        if self._coordinator() is None:
            return self.async_abort(reason="not_loaded")
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "upload_artwork",
                "delete_artwork",
                "delete_artworks",
                "delete_status",
                "delete_artworks_status",
                "upload_finish",
            ],
        )

    async def async_step_upload_artwork(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="upload_unauthorized")
        if self._upload_task is not None:
            return await self.async_step_upload_progress()
        if self._prepare_task is not None and not self._prepare_task.done():
            return await self.async_step_upload_prepare()
        if self._preparation is not None:
            preview_store(self.hass).dispose(self._preparation)
            self._preparation = None
        errors: dict[str, str] = {}
        if user_input is not None:
            assert self._owner is not None
            try:
                preparation = preview_store(self.hass).reserve(
                    self.config_entry.entry_id, self.flow_id, *self._owner
                )
            except ValueError:
                errors["base"] = "upload_busy"
            else:
                self._preparation = preparation
                self._position = "center"
                preparation.processing = True
                self._prepare_task = self.hass.async_create_task(
                    self._prepare_file(preparation, user_input["file"]), "Prepare Frame artwork"
                )
                self._prepare_waiter = self.hass.async_create_task(
                    _wait_for(self._prepare_task), "Wait for Frame artwork preparation"
                )
                return self._show_preparation_progress()
        return self.async_show_form(
            step_id="upload_artwork",
            data_schema=vol.Schema(
                {
                    vol.Required("file"): FileSelector(
                        FileSelectorConfig(accept="image/jpeg,image/png")
                    )
                }
            ),
            errors=errors,
        )

    async def _prepare_file(self, preparation: Preparation, file_id: str) -> str | None:
        from .tv_artwork_image import ImageValidationError

        try:
            source, image = await self.hass.async_add_executor_job(
                _read_and_prepare, self.hass, file_id
            )
            if preview_store(self.hass).get(preparation.token) is not preparation:
                return "upload_expired"
            preparation.source = source if image.crop_axis != "none" else b""
            preparation.image = image
            self._prepare_next = "upload_review" if image.crop_axis == "none" else "upload_crop"
        except ImageValidationError as err:
            self._validation_reason = str(err)
            preview_store(self.hass).dispose(preparation)
            return "upload_invalid_image"
        except (ValueError, OSError):
            preview_store(self.hass).dispose(preparation)
            return "upload_invalid_image"
        finally:
            preview_store(self.hass).processing_finished(preparation)
        return None

    async def async_step_upload_prepare(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self._stop_progress("upload_unauthorized")
        assert self._prepare_task is not None
        if not self._prepare_task.done():
            return self._show_preparation_progress()
        if error := self._prepare_task.result():
            return self._stop_progress(error, {"reason": self._validation_reason})
        preparation = self._live_preparation()
        if preparation is None or preparation.image is None:
            return self._stop_progress("upload_expired")
        return self.async_show_progress_done(next_step_id=self._prepare_next)

    def _show_preparation_progress(self) -> ConfigFlowResult:
        return self.async_show_progress(
            step_id="upload_prepare",
            progress_action="upload_prepare",
            progress_task=self._prepare_waiter,
        )

    async def async_step_upload_crop(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="upload_unauthorized")
        preparation = self._live_preparation()
        if preparation is None or preparation.image is None:
            return self.async_abort(reason="upload_expired")
        options = (
            ["left", "center", "right"]
            if preparation.image.crop_axis == "horizontal"
            else ["top", "center", "bottom"]
        )
        if user_input is not None:
            position = user_input["position"]
            if position not in options:
                return self.async_abort(reason="upload_invalid_image")
            if position == self._position:
                return await self.async_step_upload_review()
            self._position = position
            preparation.image = None
            preparation.processing = True
            self._prepare_task = self.hass.async_create_task(
                self._recrop(preparation, position), "Crop Frame artwork"
            )
            self._prepare_waiter = self.hass.async_create_task(
                _wait_for(self._prepare_task), "Wait for Frame artwork crop"
            )
            return self._show_preparation_progress()
        return self.async_show_form(
            step_id="upload_crop",
            data_schema=vol.Schema(
                {
                    vol.Required("position", default=self._position): SelectSelector(
                        SelectSelectorConfig(
                            options=options,
                            mode=SelectSelectorMode.DROPDOWN,
                            translation_key="artwork_crop",
                        )
                    )
                }
            ),
        )

    async def _recrop(self, preparation: Preparation, position: str) -> str | None:
        from .tv_artwork_image import ImageValidationError, prepare_image

        try:
            image = await self.hass.async_add_executor_job(
                prepare_image, preparation.source, position
            )
            if preview_store(self.hass).get(preparation.token) is not preparation:
                return "upload_expired"
            preparation.token = secrets.token_urlsafe(24)
            preparation.image = image
            self._prepare_next = "upload_review"
        except ImageValidationError as err:
            self._validation_reason = str(err)
            preview_store(self.hass).dispose(preparation)
            return "upload_invalid_image"
        except (ValueError, OSError):
            preview_store(self.hass).dispose(preparation)
            return "upload_invalid_image"
        finally:
            preview_store(self.hass).processing_finished(preparation)
        return None

    def _review_placeholders(self, preparation: Preparation) -> dict[str, str]:
        assert preparation.image is not None
        image = preparation.image
        return {
            "preview": preview_store(self.hass).sign(preparation),
            "source": f"{image.source_width} x {image.source_height}",
            "crop": f"{image.crop_width} x {image.crop_height}",
        }

    async def async_step_upload_review(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="upload_unauthorized")
        preparation = self._live_preparation()
        if preparation is None or preparation.image is None:
            return self.async_abort(reason="upload_expired")
        menu = ["upload_confirm"]
        if preparation.image.crop_axis != "none":
            menu.append("upload_crop")
        menu.extend(["upload_artwork", "upload_finish"])
        return self.async_show_menu(
            step_id="upload_review",
            menu_options=menu,
            description_placeholders=self._review_placeholders(preparation),
        )

    async def async_step_upload_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="upload_unauthorized")
        if self._upload_task is not None:
            return await self.async_step_upload_progress()
        preparation = self._live_preparation()
        if preparation is None or preparation.image is None:
            return self.async_abort(reason="upload_expired")
        if user_input is None:
            return self.async_show_form(
                step_id="upload_confirm",
                data_schema=vol.Schema({}),
                description_placeholders=self._review_placeholders(preparation),
            )
        coordinator = self._coordinator()
        if coordinator is None:
            return self.async_abort(reason="not_loaded")
        image = preparation.image
        preview_store(self.hass).dispose(preparation)
        self._upload_task = self.hass.async_create_task(
            self._upload(coordinator.artwork, image), "Upload Frame artwork once"
        )
        self._upload_waiter = self.hass.async_create_task(
            _wait_for(self._upload_task), "Wait for Frame artwork upload"
        )
        return self._show_upload_progress()

    async def _upload(self, artwork: Any, image: PreparedArtwork) -> None:
        try:
            result = await artwork.async_upload(image.data, image.thumbnail)
        except ArtworkUploadNotSentError as err:
            self._outcome = "upload_not_sent"
            self._not_sent_reason = str(err)
        except Exception:
            self._outcome = "upload_uncertain"
        else:
            self._content_id = result.content_id
            self._outcome = "upload_complete" if result.catalog_confirmed else "upload_unconfirmed"

    async def async_step_upload_progress(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self._stop_progress("upload_unauthorized")
        assert self._upload_task is not None
        if not self._upload_task.done():
            return self._show_upload_progress()
        return self.async_show_progress_done(next_step_id=self._outcome)

    def _show_upload_progress(self) -> ConfigFlowResult:
        return self.async_show_progress(
            step_id="upload_progress",
            progress_action="upload_progress",
            progress_task=self._upload_waiter,
        )

    def _stop_progress(
        self, reason: str, placeholders: dict[str, str] | None = None
    ) -> ConfigFlowResult:
        self._stop_reason = reason
        self._stop_placeholders = placeholders
        return self.async_show_progress_done(next_step_id="upload_stop")

    async def async_step_upload_stop(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return self.async_abort(
            reason=self._stop_reason, description_placeholders=self._stop_placeholders
        )

    async def _result_step(self, user_input: dict[str, Any] | None) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="upload_unauthorized")
        if user_input is not None:
            return await self.async_step_upload_finish()
        return self.async_show_form(
            step_id=self._outcome,
            data_schema=vol.Schema({}),
            description_placeholders={
                "content_id": self._content_id,
                "reason": self._not_sent_reason,
            },
        )

    async def async_step_upload_complete(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._result_step(user_input)

    async def async_step_upload_unconfirmed(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._result_step(user_input)

    async def async_step_upload_uncertain(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._result_step(user_input)

    async def async_step_upload_not_sent(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._result_step(user_input)

    async def async_step_upload_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="upload_unauthorized")
        self.async_remove()
        return self.async_abort(reason="upload_finished")

    @callback
    def async_remove(self) -> None:
        """Clear previews while allowing admitted executor and TV jobs to finish."""
        if self._preparation is not None:
            preview_store(self.hass).dispose(self._preparation)
        super().async_remove()
