"""Portable region classification matching the historical native implementation."""

from __future__ import annotations

import numpy as np

from .data import LabeledVolume, RegionClassificationVolume, SelectedCuttingPlanes


def classify_regions(
    volume: LabeledVolume,
    cutting_planes: SelectedCuttingPlanes,
    *,
    chunk_size: int = 1_000_000,
) -> RegionClassificationVolume:
    """Classify normal bone by the first plane whose positive half-space contains it."""

    if not isinstance(volume, LabeledVolume):
        raise TypeError("volume must be a LabeledVolume")
    if not isinstance(cutting_planes, SelectedCuttingPlanes):
        raise TypeError("cutting_planes must be SelectedCuttingPlanes")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    result = volume.data.astype(np.int32, copy=True)
    result[result == -1] = 2
    flat = result.ravel()
    dimensions = volume.shape

    for plane_index, equation in enumerate(cutting_planes.equations, start=1):
        remaining = np.flatnonzero(flat == 2)
        for start in range(0, len(remaining), chunk_size):
            positions = remaining[start : start + chunk_size]
            i, j, k = np.unravel_index(positions, dimensions)
            values = i * equation[0] + j * equation[1] + k * equation[2] + equation[3]
            flat[positions[values > 0.0]] = -plane_index

    return RegionClassificationVolume(result)
