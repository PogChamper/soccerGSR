# Vendored from BoxMOT (https://github.com/mikel-brostrom/boxmot) — AGPL-3.0
# Original work: Mikel Broström. Modifications: stripped torch / ReID
# dependencies, kept only the BotSort + numpy/scipy/cv2 core.
#
# This package is redistributed under the AGPL-3.0 license. See LICENSE-AGPL.

from .trackers.botsort.botsort import BotSort  # noqa: F401

__all__ = ["BotSort"]
