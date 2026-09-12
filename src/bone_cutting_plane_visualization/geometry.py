"""Geometry operations used to prepare visualization primitives."""

from __future__ import annotations

import numpy as np


def plane_box_intersection(
    equation: np.ndarray,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Return the ordered polygon where an index-space plane cuts a volume box."""

    plane = np.asarray(equation, dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise ValueError("a plane equation must contain four finite values")
    normal = plane[:3]
    normal_length = float(np.linalg.norm(normal))
    if normal_length < 1e-12:
        raise ValueError("a plane normal must be non-zero")
    shape = np.asarray(volume_shape, dtype=np.int64)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError("volume_shape must contain three positive dimensions")

    upper = shape.astype(np.float64) - 1.0
    corners = np.asarray(
        [(x, y, z) for x in (0.0, upper[0]) for y in (0.0, upper[1]) for z in (0.0, upper[2])],
        dtype=np.float64,
    )
    edge_pairs = [
        (left, right)
        for left in range(8)
        for right in range(left + 1, 8)
        if (left ^ right).bit_count() == 1
    ]
    scale = max(1.0, normal_length * float(np.linalg.norm(upper)), abs(float(plane[3])))
    tolerance = scale * 1e-9
    intersections: list[np.ndarray] = []

    def append_unique(point: np.ndarray) -> None:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in intersections):
            intersections.append(point)

    for left, right in edge_pairs:
        point0 = corners[left]
        point1 = corners[right]
        value0 = float(np.dot(normal, point0) + plane[3])
        value1 = float(np.dot(normal, point1) + plane[3])
        if abs(value0) <= tolerance:
            append_unique(point0)
        if abs(value1) <= tolerance:
            append_unique(point1)
        if value0 * value1 < -(tolerance**2):
            fraction = value0 / (value0 - value1)
            append_unique(point0 + fraction * (point1 - point0))

    if len(intersections) < 3:
        return np.empty((0, 3), dtype=np.float64)

    points = np.asarray(intersections)
    centroid = points.mean(axis=0)
    unit_normal = normal / normal_length
    reference = np.eye(3)[int(np.argmin(np.abs(unit_normal)))]
    axis0 = np.cross(unit_normal, reference)
    axis0 /= np.linalg.norm(axis0)
    axis1 = np.cross(unit_normal, axis0)
    relative = points - centroid
    angles = np.arctan2(relative @ axis1, relative @ axis0)
    return points[np.argsort(angles)]
