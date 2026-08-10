"""Camera-motion compensation used by the vendored BoT-SORT tracker."""

from __future__ import annotations

from app.vendor.boxmot.motion.cmc.ecc import ECC


def get_cmc_method(name: str | None) -> type[ECC] | None:
    """Resolve the supported CMC method."""
    if name is None:
        return None
    if name.strip().lower() != "ecc":
        raise ValueError(f"Unknown cmc_method={name!r}. Supported values: ecc")
    return ECC


__all__ = ["get_cmc_method"]
