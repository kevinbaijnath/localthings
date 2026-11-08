"""Administrator-bound, single-confirmation deletion of recorded Frame photos."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from dataclasses import replace
from typing import Any, TypeGuard

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.http import KEY_HASS_REFRESH_TOKEN_ID, KEY_HASS_USER
from homeassistant.config_entries import ConfigEntryState, ConfigFlowResult
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.http import current_request
from homeassistant.helpers.selector import ObjectSelector

from .const import DOMAIN
from .tv_artwork_delete import (
    ArtworkDeleteNotSentError,
    ArtworkDeletePreview,
    delete_contact_sheet,
)
from .tv_artwork_preview import Preparation, preview_store


async def _wait_for_delete(task: asyncio.Task[None]) -> None:
    await asyncio.shield(task)


def _valid_ids(value: Any) -> TypeGuard[list[str]]:
    return (
        isinstance(value, list)
        and 1 <= len(value) <= 10
        and all(isinstance(item, str) and 0 < len(item) <= 256 for item in value)
        and len(set(value)) == len(value)
    )


def _valid_references(value: Any, exact: bool) -> TypeGuard[list[dict[str, str]]]:
    if (
        not isinstance(value, list)
        or not all(isinstance(item, dict) for item in value)
        or not _valid_ids([item.get("content_id") for item in value])
    ):
        return False
    if not exact:
        return True
    tokens = [item.get("record_token") for item in value]
    return _valid_ids(tokens)


def _delete_item_text(item: dict[str, str]) -> str:
    outcome = item["outcome"].replace("_", " ")
    return f"{item['content_id']}: {outcome}. {item['reason']}".rstrip()


class TvArtworkDeleteFlow(config_entries.OptionsFlow):
    """Keep the reviewed group in memory and reuse guarded one-photo writes."""

    def __init__(self) -> None:
        self._owner: tuple[str, str] | None = None
        self._delete_previews: tuple[ArtworkDeletePreview, ...] = ()
        self._delete_items: list[dict[str, str]] = []
        self._delete_preparation: Preparation | None = None
        self._delete_coordinator: Any = None
        self._delete_prepare_task: asyncio.Task[None] | None = None
        self._delete_prepare_waiter: asyncio.Task[None] | None = None
        self._delete_task: asyncio.Task[None] | None = None
        self._delete_waiter: asyncio.Task[None] | None = None
        self._delete_id = ""
        self._delete_reason = ""
        self._delete_outcome = "delete_uncertain"
        self._delete_next = "delete_confirm"
        self._delete_stop_reason = "delete_expired"

    def _authorized(self) -> bool:
        request = current_request.get()
        if request is None:
            return False
        user = request.get(KEY_HASS_USER)
        refresh_token_id = request.get(KEY_HASS_REFRESH_TOKEN_ID)
        if user is None or not user.is_admin or not refresh_token_id:
            return False
        identity = (user.id, refresh_token_id)
        if self._owner is None:
            self._owner = identity
        return identity == self._owner

    def _coordinator(self):
        entry = self.hass.config_entries.async_get_entry(self.config_entry.entry_id)
        if entry is None or entry.state is not ConfigEntryState.LOADED:
            return None
        return self.hass.data.get(DOMAIN, {}).get(self.config_entry.entry_id)

    def _delete_live(self, coordinator: Any, preparation: Preparation) -> bool:
        return (
            not self.hass.is_stopping
            and preview_store(self.hass).get(preparation.token) is preparation
            and self._coordinator() is coordinator
            and coordinator is self._delete_coordinator
            and preparation.flow_id == self.flow_id
            and preparation.entry_id == self.config_entry.entry_id
            and self._owner == (preparation.owner_id, preparation.refresh_token_id)
        )

    async def _delete_admitted(self, coordinator: Any, preparation: Preparation) -> bool:
        assert self._owner is not None
        user = await self.hass.auth.async_get_user(self._owner[0])
        token = self.hass.auth.async_get_refresh_token(self._owner[1])
        return (
            user is not None
            and user.is_admin
            and user.is_active
            and token is not None
            and token.user.id == user.id
            and (token.expire_at is None or token.expire_at > time.time())
            and self._delete_live(coordinator, preparation)
        )

    def _delete_placeholders(self) -> dict[str, str]:
        return {
            "content_id": self._delete_id,
            "title": self._delete_items[0]["title"] if self._delete_items else "",
            "count": str(len(self._delete_items)),
            "items": json.dumps(self._delete_items),
            "reason": self._delete_reason,
            "photos": "\n".join(
                f"{index}. {item['title']} ({item['content_id']})"
                for index, item in enumerate(self._delete_items, 1)
            ),
            "results": "\n".join(_delete_item_text(item) for item in self._delete_items),
        }

    async def async_step_delete_artwork(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_selection_step("delete_artwork", user_input)

    async def async_step_delete_artworks(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_selection_step("delete_artworks", user_input)

    async def _delete_selection_step(
        self, step: str, user_input: dict[str, Any] | None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="delete_unauthorized")
        if self._delete_task is not None:
            return await self.async_step_delete_progress()
        if self._delete_prepare_task is not None:
            return await self.async_step_delete_prepare()
        coordinator = self._coordinator()
        if coordinator is None:
            return self.async_abort(reason="not_loaded")
        if user_input is None:
            return self._delete_selection(step)
        ids = (
            user_input.get("content_ids")
            if step == "delete_artworks"
            else [user_input.get("content_id")]
        )
        if not _valid_ids(ids) or not all(content_id.startswith("MY_") for content_id in ids):
            return self._delete_selection(step, {"base": "delete_invalid_selection"})
        self._delete_items = [
            {
                "content_id": content_id,
                "title": "",
                "record_token": "",
                "outcome": "not_attempted",
                "reason": "",
            }
            for content_id in ids
        ]
        self._delete_id = ids[0]
        assert self._owner is not None
        try:
            preparation = preview_store(self.hass).reserve(
                self.config_entry.entry_id, self.flow_id, *self._owner
            )
        except ValueError:
            self._delete_outcome = "delete_blocked"
            self._delete_reason = "Another photo preview is active. Close it first."
            return await self._delete_result(None)
        preparation.expires_at = self.hass.loop.time() + 300
        assert preparation.timer is not None
        preparation.timer.cancel()
        preparation.timer = self.hass.loop.call_later(
            300, preview_store(self.hass).dispose, preparation
        )
        preparation.processing = True
        self._delete_preparation = preparation
        self._delete_coordinator = coordinator
        self._delete_prepare_task = self.hass.async_create_task(
            self._prepare_delete(coordinator, preparation), "Check Frame photos for deletion"
        )
        self._delete_prepare_waiter = self.hass.async_create_task(
            _wait_for_delete(self._delete_prepare_task), "Wait for Frame deletion preview"
        )
        return self._show_delete_prepare()

    def _delete_selection(
        self, step: str, errors: dict[str, str] | None = None
    ) -> ConfigFlowResult:
        schema = {
            "delete_artworks": {vol.Required("content_ids"): ObjectSelector()},
            "delete_artworks_status": {vol.Required("references"): ObjectSelector()},
        }.get(step, {vol.Required("content_id"): str})
        return self.async_show_form(step_id=step, data_schema=vol.Schema(schema), errors=errors)

    async def _prepare_delete(self, coordinator: Any, preparation: Preparation) -> None:
        item = self._delete_items[0]
        try:
            previews = []
            for item in self._delete_items:
                preview = await coordinator.artwork.async_prepare_delete(item["content_id"])
                if not self._delete_live(coordinator, preparation):
                    self._delete_next = "delete_stop"
                    preview_store(self.hass).dispose(preparation)
                    return
                if (
                    preview.content_id != item["content_id"]
                    or not preview.record_token
                    or any(previous.record_token == preview.record_token for previous in previews)
                ):
                    raise ValueError("The checked photo does not match the selection")
                previews.append(preview)
                item.update(title=preview.title, record_token=preview.record_token)
            checked = tuple(previews)
            future = self.hass.async_add_executor_job(delete_contact_sheet, checked)
            try:
                thumbnail = await asyncio.shield(future)
            except asyncio.CancelledError:
                while not future.done():
                    with suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(future)
                if not future.cancelled():
                    future.exception()
                raise
            if not self._delete_live(coordinator, preparation):
                self._delete_next = "delete_stop"
                preview_store(self.hass).dispose(preparation)
                return
            self._delete_previews = checked
            for item in self._delete_items:
                item["outcome"] = "ready"
            preparation.thumbnail = thumbnail
            preparation.thumbnail_mime_type = "image/jpeg"
        except asyncio.CancelledError:
            preview_store(self.hass).dispose(preparation)
            raise
        except Exception as err:
            self._delete_next = self._delete_outcome = "delete_blocked"
            self._delete_id = item["content_id"]
            self._delete_reason = (
                str(err)
                if isinstance(err, HomeAssistantError)
                else "The photos could not be checked. Nothing was deleted."
            )
            item["reason"] = self._delete_reason
            preview_store(self.hass).dispose(preparation)
        finally:
            preview_store(self.hass).processing_finished(preparation)

    def _show_delete_prepare(self) -> ConfigFlowResult:
        return self.async_show_progress(
            step_id="delete_prepare",
            progress_action="delete_prepare",
            progress_task=self._delete_prepare_waiter,
            description_placeholders=self._delete_placeholders(),
        )

    async def async_step_delete_prepare(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self._stop_delete_progress("delete_unauthorized")
        assert self._delete_prepare_task is not None
        if not self._delete_prepare_task.done():
            return self._show_delete_prepare()
        return self.async_show_progress_done(next_step_id=self._delete_next)

    async def async_step_delete_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="delete_unauthorized")
        if self._delete_task is not None:
            return await self.async_step_delete_progress()
        preparation = self._delete_preparation
        coordinator = self._coordinator()
        if (
            preparation is None
            or not self._delete_previews
            or coordinator is None
            or not self._delete_live(coordinator, preparation)
        ):
            return self.async_abort(reason="delete_expired")
        if user_input is None:
            return self.async_show_form(
                step_id="delete_confirm",
                data_schema=vol.Schema({}),
                description_placeholders={
                    **self._delete_placeholders(),
                    "preview": preview_store(self.hass).sign(preparation),
                },
            )
        preparation.processing = True
        preparation.thumbnail = b""
        self._delete_previews = tuple(
            replace(preview, thumbnail=b"") for preview in self._delete_previews
        )
        for item in self._delete_items:
            item["outcome"] = "not_attempted"
        self._delete_task = self.hass.async_create_task(
            self._delete_group(coordinator, preparation), "Delete checked Frame photos once"
        )
        self._delete_waiter = self.hass.async_create_task(
            _wait_for_delete(self._delete_task), "Wait for Frame deletion results"
        )
        return self._show_delete_progress()

    async def _delete_group(self, coordinator: Any, preparation: Preparation) -> None:
        try:
            for preview, item in zip(self._delete_previews, self._delete_items, strict=True):
                if not await self._delete_admitted(coordinator, preparation):
                    item["reason"] = "The session, checked preview or Frame is no longer available."
                    break
                item["outcome"] = "pending"
                try:
                    result = await coordinator.artwork.async_delete(preview)
                except asyncio.CancelledError:
                    item["outcome"] = "uncertain"
                    raise
                except ArtworkDeleteNotSentError as err:
                    item.update(outcome="not_sent", reason=str(err))
                except Exception:
                    item["outcome"] = "uncertain"
                else:
                    item["outcome"] = (
                        result.outcome
                        if result.content_id == preview.content_id
                        and result.outcome in {"absent", "present", "uncertain"}
                        else "uncertain"
                    )
                if item["outcome"] != "absent":
                    break
        finally:
            self._set_group_result()
            self._delete_previews = ()
            preview_store(self.hass).dispose(preparation)
            preview_store(self.hass).processing_finished(preparation)

    def _set_group_result(self) -> None:
        outcomes = {item["outcome"] for item in self._delete_items}
        if len(outcomes) > 1:
            self._delete_outcome = "delete_partial"
        else:
            self._delete_outcome = {
                "absent": "delete_complete",
                "present": "delete_present",
                "uncertain": "delete_uncertain",
                "not_sent": "delete_not_sent",
                "not_attempted": "delete_not_sent",
            }.get(next(iter(outcomes)), "delete_uncertain")
        self._delete_reason = next(
            (item["reason"] for item in self._delete_items if item["reason"]), ""
        )

    async def async_step_delete_status(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_status_step("delete_status", user_input)

    async def async_step_delete_artworks_status(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_status_step("delete_artworks_status", user_input)

    async def _delete_status_step(
        self, step: str, user_input: dict[str, Any] | None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="delete_unauthorized")
        if self._delete_task is not None:
            return await self.async_step_delete_progress()
        coordinator = self._coordinator()
        if coordinator is None:
            return self.async_abort(reason="not_loaded")
        if user_input is None:
            return self._delete_selection(step)
        exact = step == "delete_artworks_status"
        references = (
            user_input.get("references")
            if exact
            else [{"content_id": user_input.get("content_id"), "record_token": ""}]
        )
        if not _valid_references(references, exact):
            return self._delete_selection(step, {"base": "delete_invalid_selection"})
        self._delete_items = [
            {
                "content_id": item["content_id"],
                "title": "",
                "record_token": item["record_token"],
                "outcome": "pending",
                "reason": "",
            }
            for item in references
        ]
        self._delete_id = self._delete_items[0]["content_id"]
        self._delete_task = self.hass.async_create_task(
            self._check_delete_status(coordinator, exact),
            "Reconcile Frame deletion without resending",
        )
        self._delete_waiter = self.hass.async_create_task(
            _wait_for_delete(self._delete_task), "Wait for Frame deletion status"
        )
        return self._show_delete_progress()

    async def _check_delete_status(self, coordinator: Any, exact: bool) -> None:
        for item in self._delete_items:
            try:
                if exact:
                    result = await coordinator.artwork.async_delete_status(
                        item["content_id"], record_token=item["record_token"]
                    )
                else:
                    result = await coordinator.artwork.async_delete_status(item["content_id"])
                if result is None:
                    item.update(
                        outcome="uncertain",
                        reason="No matching removal record was found. Its result is unknown.",
                    )
                else:
                    item["outcome"] = (
                        result.outcome
                        if result.content_id == item["content_id"]
                        and result.outcome in {"absent", "present", "uncertain"}
                        else "uncertain"
                    )
            except Exception:
                item["outcome"] = "uncertain"
        self._set_group_result()
        if not exact and self._delete_items[0]["reason"]:
            self._delete_outcome = "delete_stop"
            self._delete_stop_reason = "delete_missing"

    def _show_delete_progress(self) -> ConfigFlowResult:
        return self.async_show_progress(
            step_id="delete_progress",
            progress_action="delete_progress",
            progress_task=self._delete_waiter,
            description_placeholders=self._delete_placeholders(),
        )

    async def async_step_delete_progress(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if not self._authorized():
            return self._stop_delete_progress("delete_unauthorized")
        assert self._delete_task is not None
        if not self._delete_task.done():
            return self._show_delete_progress()
        return self.async_show_progress_done(next_step_id=self._delete_outcome)

    def _stop_delete_progress(self, reason: str) -> ConfigFlowResult:
        self._delete_stop_reason = reason
        return self.async_show_progress_done(next_step_id="delete_stop")

    async def async_step_delete_stop(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return self.async_abort(reason=self._delete_stop_reason)

    async def _delete_result(self, user_input: dict[str, Any] | None) -> ConfigFlowResult:
        if not self._authorized():
            return self.async_abort(reason="delete_unauthorized")
        if user_input is not None:
            self.async_remove()
            return self.async_abort(reason="delete_finished")
        return self.async_show_form(
            step_id=self._delete_outcome,
            data_schema=vol.Schema({}),
            description_placeholders=self._delete_placeholders(),
        )

    async def async_step_delete_complete(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_result(user_input)

    async def async_step_delete_present(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_result(user_input)

    async def async_step_delete_uncertain(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_result(user_input)

    async def async_step_delete_not_sent(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_result(user_input)

    async def async_step_delete_blocked(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_result(user_input)

    async def async_step_delete_partial(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._delete_result(user_input)

    @callback
    def async_remove(self) -> None:
        if self._delete_task is not None and not self._delete_task.done():
            return
        if self._delete_preparation is not None:
            preview_store(self.hass).dispose(self._delete_preparation)
        self._delete_previews = ()
