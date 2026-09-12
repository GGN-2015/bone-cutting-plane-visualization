"""Validated, renderer-independent data contracts for resection plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IDENTITY_DIRECTION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _finite_numeric_array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number) or np.issubdtype(
        result.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _readonly_copy(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _discrete_volume(value: np.ndarray, name: str, dtype: np.dtype) -> np.ndarray:
    result = _finite_numeric_array(value, name)
    if result.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional array")
    if not np.all(result == np.rint(result)):
        raise ValueError(f"{name} must contain integer-valued labels")
    return _readonly_copy(result, dtype)


@dataclass(frozen=True, eq=False)
class LabeledVolume:
    """A labeled LPS volume: -1 normal bone, 0 other, and 1 tumor."""

    data: np.ndarray

    def __post_init__(self) -> None:
        data = _discrete_volume(self.data, "labeled volume", np.int8)
        if not np.all(np.isin(data, (-1, 0, 1))):
            raise ValueError("labeled volume may contain only -1, 0, and 1")
        object.__setattr__(self, "data", data)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def normal_bone_mask(self) -> np.ndarray:
        return self.data == -1

    @property
    def tumor_mask(self) -> np.ndarray:
        return self.data == 1

    def to_numpy(self, *, copy: bool = False) -> np.ndarray:
        return self.data.copy() if copy else self.data


@dataclass(frozen=True, eq=False)
class SelectedCuttingPlanes:
    """Ordered index-space cutting planes stored as ``[a, b, c, d]`` rows."""

    equations: np.ndarray

    def __post_init__(self) -> None:
        equations = _finite_numeric_array(self.equations, "cutting-plane equations")
        if equations.ndim != 2 or equations.shape[1:] != (4,) or len(equations) == 0:
            raise ValueError("cutting-plane equations must have shape (N, 4) with N >= 1")
        equations = _readonly_copy(equations, np.float64)
        if np.any(np.linalg.norm(equations[:, :3], axis=1) < 1e-12):
            raise ValueError("cutting-plane normals must be non-zero")
        object.__setattr__(self, "equations", equations)

    @property
    def count(self) -> int:
        return len(self.equations)

    def to_numpy(self, *, copy: bool = False) -> np.ndarray:
        return self.equations.copy() if copy else self.equations


@dataclass(frozen=True, eq=False)
class RegionClassificationVolume:
    """Plane-classified volume: negative bone is retained and value 2 is resected."""

    data: np.ndarray

    def __post_init__(self) -> None:
        source = _finite_numeric_array(self.data, "region classification volume")
        if np.any(source < np.iinfo(np.int32).min) or np.any(source > np.iinfo(np.int32).max):
            raise ValueError("region classification labels exceed int32 range")
        data = _discrete_volume(source, "region classification volume", np.int32)
        if np.any(data > 2):
            raise ValueError("region classification labels may not exceed 2")
        object.__setattr__(self, "data", data)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def retained_mask(self) -> np.ndarray:
        return self.data <= -1

    @property
    def resected_mask(self) -> np.ndarray:
        return self.data == 2

    def to_numpy(self, *, copy: bool = False) -> np.ndarray:
        return self.data.copy() if copy else self.data


@dataclass(frozen=True)
class VolumeGeometry:
    """Map NumPy ``(L, P, S)`` indices into LPS physical millimetres."""

    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[tuple[float, float, float], ...] = IDENTITY_DIRECTION

    def __post_init__(self) -> None:
        spacing = np.asarray(self.spacing, dtype=np.float64)
        origin = np.asarray(self.origin, dtype=np.float64)
        direction = np.asarray(self.direction, dtype=np.float64)
        if spacing.shape != (3,) or origin.shape != (3,) or direction.shape != (3, 3):
            raise ValueError("spacing, origin, and direction must describe three dimensions")
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError("spacing must contain three finite positive values")
        if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
            raise ValueError("origin and direction must be finite")
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-8):
            raise ValueError("direction must be an orthonormal 3x3 matrix")
        object.__setattr__(self, "spacing", tuple(float(value) for value in spacing))
        object.__setattr__(self, "origin", tuple(float(value) for value in origin))
        object.__setattr__(
            self,
            "direction",
            tuple(tuple(float(value) for value in row) for row in direction),
        )

    @classmethod
    def from_mmpd(cls, mmpd: float) -> VolumeGeometry:
        value = float(mmpd)
        return cls(spacing=(value, value, value))

    @property
    def affine(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=np.float64) @ np.diag(self.spacing)

    def index_to_world(self, points: np.ndarray) -> np.ndarray:
        points = _points_array(points)
        return points @ self.affine.T + np.asarray(self.origin, dtype=np.float64)

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        points = _points_array(points)
        shifted = points - np.asarray(self.origin, dtype=np.float64)
        return np.linalg.solve(self.affine, shifted.T).T

    def plane_index_to_world(self, equation: np.ndarray) -> np.ndarray:
        equation = np.asarray(equation, dtype=np.float64)
        if equation.shape != (4,) or not np.all(np.isfinite(equation)):
            raise ValueError("a plane equation must contain four finite values")
        world_normal = np.linalg.solve(self.affine.T, equation[:3])
        norm = np.linalg.norm(world_normal)
        if norm < 1e-12:
            raise ValueError("a plane normal must be non-zero")
        world_offset = equation[3] - np.dot(world_normal, self.origin)
        return np.r_[world_normal / norm, world_offset / norm]


@dataclass(frozen=True, eq=False)
class TumorSafetyMarginData:
    """Planning volume and masks for an optional physical tumor safety margin."""

    planning_volume: LabeledVolume
    protected_mask: np.ndarray
    danger_bone_mask: np.ndarray
    margin_mm: float

    def __post_init__(self) -> None:
        if not isinstance(self.planning_volume, LabeledVolume):
            raise TypeError("planning_volume must be a LabeledVolume")
        protected = np.asarray(self.protected_mask, dtype=bool)
        danger = np.asarray(self.danger_bone_mask, dtype=bool)
        if (
            protected.shape != self.planning_volume.shape
            or danger.shape != self.planning_volume.shape
        ):
            raise ValueError("safety-margin masks must match planning_volume")
        margin = float(self.margin_mm)
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("margin_mm must be finite and non-negative")
        if np.any(danger & ~protected):
            raise ValueError("danger_bone_mask must be a subset of protected_mask")
        object.__setattr__(self, "protected_mask", _readonly_copy(protected, np.bool_))
        object.__setattr__(self, "danger_bone_mask", _readonly_copy(danger, np.bool_))
        object.__setattr__(self, "margin_mm", margin)

    @classmethod
    def compute(
        cls,
        volume: LabeledVolume,
        margin_mm: float,
        geometry: VolumeGeometry | None = None,
    ) -> TumorSafetyMarginData:
        if not isinstance(volume, LabeledVolume):
            raise TypeError("volume must be a LabeledVolume")
        geometry = geometry if geometry is not None else VolumeGeometry()
        margin = float(margin_mm)
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("margin_mm must be finite and non-negative")

        result = volume.to_numpy(copy=True)
        tumor_mask = volume.tumor_mask
        protected = tumor_mask.copy()
        danger = np.zeros(volume.shape, dtype=bool)
        if margin == 0:
            return cls(LabeledVolume(result), protected, danger, margin)
        if not np.any(tumor_mask):
            raise ValueError("labeled volume contains no tumor voxels")

        from scipy.ndimage import distance_transform_edt

        tumor_indices = np.nonzero(tumor_mask)
        voxel_radii = np.ceil(margin / np.asarray(geometry.spacing)).astype(np.int64)
        lower = np.maximum([int(axis.min()) for axis in tumor_indices] - voxel_radii, 0)
        upper = np.minimum(
            [int(axis.max()) + 1 for axis in tumor_indices] + voxel_radii,
            volume.shape,
        )
        region = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
        distance = distance_transform_edt(~tumor_mask[region], sampling=geometry.spacing)
        protected[region] = distance <= np.nextafter(margin, np.inf)
        danger = protected & volume.normal_bone_mask
        result[protected] = 1
        return cls(LabeledVolume(result), protected, danger, margin)


@dataclass(frozen=True, eq=False)
class ResectionPlanData:
    """A complete, cross-validated input for final resection visualization."""

    labeled_volume: LabeledVolume
    cutting_planes: SelectedCuttingPlanes
    classification: RegionClassificationVolume
    geometry: VolumeGeometry
    safety_margin: TumorSafetyMarginData | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.labeled_volume, LabeledVolume):
            raise TypeError("labeled_volume must be a LabeledVolume")
        if not isinstance(self.cutting_planes, SelectedCuttingPlanes):
            raise TypeError("cutting_planes must be SelectedCuttingPlanes")
        if not isinstance(self.classification, RegionClassificationVolume):
            raise TypeError("classification must be a RegionClassificationVolume")
        if not isinstance(self.geometry, VolumeGeometry):
            raise TypeError("geometry must be a VolumeGeometry")
        if self.classification.shape != self.labeled_volume.shape:
            raise ValueError("classification must have the same shape as labeled_volume")

        source = self.labeled_volume.data
        solved = self.classification.data
        normal = source == -1
        if not np.any(normal):
            raise ValueError("labeled_volume contains no normal bone voxels")
        if not np.any(source == 1):
            raise ValueError("labeled_volume contains no tumor voxels")
        if np.any(~((solved[normal] <= -1) | (solved[normal] == 2))):
            raise ValueError("normal bone must be classified as retained or resected")
        if np.any(solved[~normal] != source[~normal]):
            raise ValueError("classification may only change normal-bone voxels")
        retained_values = solved[normal & (solved <= -1)]
        if len(retained_values) and int(retained_values.min()) < -self.cutting_planes.count:
            raise ValueError("classification references a cutting plane that is not present")

        safety = self.safety_margin
        if safety is not None:
            if not isinstance(safety, TumorSafetyMarginData):
                raise TypeError("safety_margin must be TumorSafetyMarginData or None")
            if safety.planning_volume.shape != self.labeled_volume.shape:
                raise ValueError("safety-margin data must match labeled_volume")
            expected_danger = safety.protected_mask & normal
            if not np.array_equal(safety.danger_bone_mask, expected_danger):
                raise ValueError("danger_bone_mask must identify protected normal bone exactly")
            if np.any(self.labeled_volume.tumor_mask & ~safety.protected_mask):
                raise ValueError("protected_mask must contain every tumor voxel")
            if np.any(~safety.protected_mask & (safety.planning_volume.data != source)):
                raise ValueError("planning_volume may only change protected voxels")
            if np.any(safety.planning_volume.data[safety.protected_mask] != 1):
                raise ValueError("planning_volume must label every protected voxel as tumor")
            if np.any(solved[safety.danger_bone_mask] != 2):
                raise ValueError("all safety-margin bone must be resected")
            _ensure_mask_is_resected(safety.protected_mask, self.cutting_planes.equations)

    @property
    def keep_rate(self) -> float:
        normal = self.labeled_volume.normal_bone_mask
        retained = normal & self.classification.retained_mask
        return float(np.count_nonzero(retained) / np.count_nonzero(normal))


def _points_array(points: np.ndarray) -> np.ndarray:
    result = _finite_numeric_array(points, "points").astype(np.float64, copy=False)
    if result.shape[-1:] != (3,):
        raise ValueError("points must have a final dimension of length 3")
    return result


def _ensure_mask_is_resected(mask: np.ndarray, equations: np.ndarray) -> None:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return
    extent = np.asarray(mask.shape, dtype=np.float64) - 1.0
    for index, equation in enumerate(equations):
        scale = max(
            1.0,
            float(np.linalg.norm(equation[:3]) * np.linalg.norm(extent)),
            abs(float(equation[3])),
        )
        if np.any(coordinates @ equation[:3] + equation[3] > scale * 1e-9):
            raise ValueError(
                f"cutting plane {index + 1} does not fully resect the tumor safety margin"
            )
