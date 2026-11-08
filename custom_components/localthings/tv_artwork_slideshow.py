"""Memory-only exact-photo rotation, with one admitted change at a time."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from .tv import FrameMode

if TYPE_CHECKING:
    from .tv_coordinator import TvCoordinator

INTERVALS = (1, 3, 5, 10, 15, 30, 60)


class ArtworkSlideshow:
    """Own the timer; leave the last confirmed photo visible when stopped."""

    def __init__(self, hass: HomeAssistant, coordinator: TvCoordinator) -> None:
        self._hass = hass
        self._coordinator = coordinator
        self._lock = asyncio.Lock()
        self._closed = False
        self._running = False
        self._content_ids: list[str] = []
        self._interval = 3
        self._shuffle = False
        self._current: str | None = None
        self._next: datetime | None = None
        self._reason = "Home Assistant playlist is stopped; native rotation was not checked."
        self._remaining: list[str] = []
        self._generation = 0
        self._cancel_timer: Callable[[], None] | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """Include the first admitted change in the deletion guard."""
        return self._running

    @callback
    def status(self) -> dict[str, Any]:
        """Return HA state without reading or implying native playback state."""
        return {
            "running": self._running,
            "content_ids": list(self._content_ids),
            "interval_minutes": self._interval,
            "shuffle": self._shuffle,
            "current_content_id": self._current,
            "next_change_at": self._next.isoformat() if self._next else None,
            "reason": self._reason,
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise HomeAssistantError("Home Assistant slideshow is closed")

    @callback
    def observe_mode(self) -> None:
        """Keep the due check through uncertainty; stop on a confirmed mode exit."""
        if not self.running:
            return
        if (
            not self._coordinator.last_update_success
            or self._coordinator.state.mode is FrameMode.UNKNOWN
        ):
            self._reason = "Home Assistant playlist is waiting for fresh Art Mode verification."
        elif self._coordinator.state.mode in {FrameMode.TV, FrameMode.OFF}:
            self._stop("Home Assistant playlist stopped: TV left Art Mode.")
        elif self._next is not None:
            self._reason = "Home Assistant playlist is running."

    def _stop(self, reason: str) -> None:
        self._running = False
        self._next = None
        self._reason = reason
        self._generation += 1
        if self._cancel_timer is not None:
            self._cancel_timer()
            self._cancel_timer = None

    async def _async_drain(self, reason: str) -> None:
        self._stop(reason)
        task = self._task
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # HA cancels background jobs before delivering its stop event.
                caller = asyncio.current_task()
                if not task.cancelled() or (caller is not None and caller.cancelling()):
                    raise

    async def async_start(
        self, content_ids: list[str], interval_minutes: int, shuffle: bool
    ) -> dict[str, Any]:
        """Replace a run once; identical concurrent starts share the current run."""
        self._ensure_open()
        generation = self._generation
        if type(interval_minutes) is not int or interval_minutes not in INTERVALS:
            raise HomeAssistantError("Choose a supported slideshow interval")
        if type(shuffle) is not bool:
            raise HomeAssistantError("Slideshow shuffle must be true or false")
        self._coordinator.artwork.validate_slideshow_ids(content_ids)
        async with self._lock:
            self._ensure_open()
            if (
                content_ids == self._content_ids
                and interval_minutes == self._interval
                and shuffle == self._shuffle
                and (self.running or generation != self._generation)
            ):
                # A queued duplicate must not replay an uncertain selection.
                if not self.running:
                    raise HomeAssistantError(self._reason)
                return self.status()
            await self._async_drain("Home Assistant playlist is being replaced.")
            self._ensure_open()
            self._content_ids = list(content_ids)
            self._interval = interval_minutes
            self._shuffle = shuffle
            self._remaining = []
            self._running = True
            self._reason = "Home Assistant playlist is starting."
            self._task = self._hass.async_create_background_task(
                self._async_step(start=True), "Start Frame selected-photo slideshow"
            )
            await asyncio.shield(self._task)
            if not self.running:
                raise HomeAssistantError(self._reason)
            return self.status()

    def _next_content(self) -> str:
        if not self._remaining:
            self._remaining = list(self._content_ids)
            if self._shuffle:
                random.shuffle(self._remaining)
                if self._remaining[0] == self._current:
                    self._remaining[0], self._remaining[1] = self._remaining[1], self._remaining[0]
        return self._remaining.pop(0)

    async def _async_step(self, *, start: bool = False) -> None:
        generation = self._generation
        try:
            selected = await self._coordinator.artwork.async_slideshow_select(
                self._next_content(),
                self._content_ids,
                expected_current=None if start else self._current,
                start=start,
                admitted=lambda: (
                    self.running and not self._closed and generation == self._generation
                ),
                run_operation=self._coordinator.async_slideshow_operation,
            )
        except HomeAssistantError as error:
            self._stop(str(error))
            return
        except Exception:
            self._stop("Home Assistant playlist stopped because a TV request failed.")
            return
        if selected is not None:
            self._current = selected
        if not self.running or self._closed or generation != self._generation:
            return
        self._reason = "Home Assistant playlist is running."
        self._next = dt_util.utcnow() + timedelta(minutes=self._interval)

        @callback
        def change(_now: datetime) -> None:
            if not self.running or self._closed or generation != self._generation:
                return
            self._cancel_timer = None
            self._next = None
            self._task = self._hass.async_create_background_task(
                self._async_step(), "Change Frame slideshow photo"
            )

        self._cancel_timer = async_call_later(self._hass, self._interval * 60, change)

    async def async_stop(self) -> dict[str, Any]:
        """Drain HA changes, then read and stop native rotation without waking."""
        self._ensure_open()
        async with self._lock:
            await self._async_drain("Home Assistant playlist is stopped.")
            self._ensure_open()
            try:
                await self._coordinator.artwork.async_stop_native_slideshow(
                    self._coordinator.async_slideshow_operation
                )
            except HomeAssistantError:
                self._reason = (
                    "Home Assistant playlist stopped; native rotation could not be stopped."
                )
                raise HomeAssistantError(self._reason) from None
            self._reason = "Home Assistant playlist and native rotation are stopped."
            return self.status()

    async def async_manual_select(self, content_id: str) -> str:
        """Serialize manual selection with Start so the user's choice wins."""
        async with self._lock:
            self._ensure_open()
            await self._async_drain("Home Assistant playlist stopped for a manual photo selection.")
            return await self._coordinator.artwork.async_select(content_id)

    async def async_close(self) -> None:
        """Invalidate pending changes immediately; never change native rotation."""
        self._closed = True
        self._stop("Home Assistant playlist stopped because the integration closed.")
        async with self._lock:
            await self._async_drain(self._reason)
