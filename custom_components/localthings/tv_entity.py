"""Shared identity and observed availability for Frame TV entities."""

from homeassistant.helpers.device_registry import CONNECTION_NETWORK_MAC, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .tv import FrameMode
from .tv_coordinator import TvCoordinator


class TvEntity(CoordinatorEntity[TvCoordinator]):
    """Keep all TV entities in the lifecycle's stable device namespace."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: TvCoordinator, suffix: str) -> None:
        super().__init__(coordinator)
        device_id = f"tv_{coordinator.identity.device_id}"
        self._attr_unique_id = f"{DOMAIN}_{device_id}_{suffix}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, device_id)},
            connections={(CONNECTION_NETWORK_MAC, coordinator.identity.mac)}
            if coordinator.identity.mac
            else set(),
            manufacturer="Samsung",
            model=coordinator.identity.model,
            name=coordinator.identity.model,
        )

    @property
    def available(self) -> bool:
        """Only confirmed awake modes allow ordinary remote commands."""
        return super().available and self.coordinator.state.mode in {FrameMode.TV, FrameMode.ART}
