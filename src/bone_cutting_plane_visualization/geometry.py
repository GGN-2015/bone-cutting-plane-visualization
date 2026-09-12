"""Geometry operations used to prepare visualization primitives."""

from __future__ import annotations

import numpy as np


def clip_convex_polygon_to_halfspaces(
    polygon: np.ndarray,
    halfspaces: np.ndarray,
) -> np.ndarray:
    """Clip an ordered 3D convex polygon to ``a*x + b*y + c*z + d <= 0``."""

    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        raise ValueError("polygon must have shape (N, 3) and contain only finite values")
    equations = np.asarray(halfspaces, dtype=np.float64)
    if (
        equations.ndim != 2
        or equations.shape[1:] != (4,)
        or not np.all(np.isfinite(equations))
    ):
        raise ValueError("halfspaces must have shape (N, 4) and contain only finite values")
    if len(points) < 3:
        return np.empty((0, 3), dtype=np.float64)
    if len(equations) == 0:
        return points.copy()

    normal_lengths = np.linalg.norm(equations[:, :3], axis=1)
    if np.any(normal_lengths < 1e-12):
        raise ValueError("halfspace normals must be non-zero")
    normalized = equations / normal_lengths[:, None]
    polygon_scale = max(1.0, float(np.linalg.norm(np.ptp(points, axis=0))))
    tolerance = polygon_scale * 1e-9
    result = points.copy()

    for equation in normalized:
        if len(result) < 3:
            return np.empty((0, 3), dtype=np.float64)
        result = _clip_polygon_to_halfspace(result, equation, tolerance)
        result = _clean_ordered_polygon(result, tolerance)

    return result


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


def _clip_polygon_to_halfspace(
    polygon: np.ndarray,
    equation: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    values = polygon @ equation[:3] + equation[3]
    values[np.abs(values) <= tolerance] = 0.0
    clipped: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(values[-1])
    previous_inside = previous_value <= 0.0

    for current, raw_value in zip(polygon, values, strict=True):
        current_value = float(raw_value)
        current_inside = current_value <= 0.0
        if current_inside != previous_inside:
            fraction = -previous_value / (current_value - previous_value)
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside

    if not clipped:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def _clean_ordered_polygon(polygon: np.ndarray, tolerance: float) -> np.ndarray:
    if len(polygon) < 3:
        return np.empty((0, 3), dtype=np.float64)

    unique: list[np.ndarray] = []
    for point in polygon:
        if not unique or np.linalg.norm(point - unique[-1]) > tolerance:
            unique.append(point)
    if len(unique) > 1 and np.linalg.norm(unique[0] - unique[-1]) <= tolerance:
        unique.pop()
    if len(unique) < 3:
        return np.empty((0, 3), dtype=np.float64)

    changed = True
    while changed and len(unique) >= 3:
        changed = False
        for index in range(len(unique)):
            previous = unique[index - 1]
            current = unique[index]
            following = unique[(index + 1) % len(unique)]
            edge0 = current - previous
            edge1 = following - current
            edge_scale = float(np.linalg.norm(edge0) * np.linalg.norm(edge1))
            if edge_scale <= tolerance**2 or np.linalg.norm(np.cross(edge0, edge1)) <= (
                1e-10 * edge_scale
            ):
                unique.pop(index)
                changed = True
                break

    if len(unique) < 3:
        return np.empty((0, 3), dtype=np.float64)
    result = np.asarray(unique, dtype=np.float64)
    area_vector = np.sum(np.cross(result, np.roll(result, -1, axis=0)), axis=0) * 0.5
    if np.linalg.norm(area_vector) <= tolerance**2:
        return np.empty((0, 3), dtype=np.float64)
    return result
