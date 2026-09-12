"""Read and write resection data as compatible extended ``.ubd.npz`` files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ct_mri_dicom_nii_reader import BodyDataLoaderManager

from .data import (
    LabeledVolume,
    RegionClassificationVolume,
    ResectionPlanData,
    SelectedCuttingPlanes,
    TumorSafetyMarginData,
    VolumeGeometry,
)

_BCV_SCHEMA_VERSION = 1
_BASE_KEYS = {"body_data", "image_type", "mmpd"}
_PLAN_KEYS = {
    "bcv_schema_version",
    "bcv_cutting_planes",
    "bcv_region_classification",
    "bcv_spacing_mm",
    "bcv_origin_lps_mm",
    "bcv_direction_index_to_lps",
    "bcv_has_safety_margin",
}


class IncompleteResectionPlanError(ValueError):
    """Raised when a standard UBD volume has no complete BCV plan extension."""


def _ubd_path(path: str | Path, *, for_write: bool = False) -> Path:
    result = Path(path).expanduser()
    if not str(result).lower().endswith(".ubd.npz"):
        if for_write:
            result = Path(f"{result}.ubd.npz")
        else:
            raise ValueError("input path must end with .ubd.npz")
    return result.resolve()


def _load_body_data(path: Path):
    body_data = BodyDataLoaderManager().load_file(str(path))
    if body_data.get_type() != "mask":
        raise ValueError("resection visualization requires a UBD file with image_type='mask'")
    return body_data


def load_labeled_volume(path: str | Path) -> tuple[LabeledVolume, VolumeGeometry]:
    """Load a standard ``.ubd.npz`` mask and its isotropic LPS geometry."""

    resolved = _ubd_path(path)
    body_data = _load_body_data(resolved)
    volume = LabeledVolume(body_data.to_numpy(copy=True))
    geometry = VolumeGeometry.from_mmpd(body_data.get_mmpd())
    return volume, geometry


def load_resection_plan(path: str | Path) -> ResectionPlanData:
    """Load a complete plan from a BCV-extended, standard-compatible UBD file."""

    resolved = _ubd_path(path)
    body_data = _load_body_data(resolved)
    with np.load(resolved, allow_pickle=False) as archive:
        missing_base = _BASE_KEYS - set(archive.files)
        if missing_base:
            raise ValueError(f"invalid UBD file; missing fields: {sorted(missing_base)}")
        missing_plan = _PLAN_KEYS - set(archive.files)
        if missing_plan:
            raise IncompleteResectionPlanError(
                "UBD file contains a volume but not a complete resection plan; "
                f"missing fields: {sorted(missing_plan)}"
            )
        schema_version = int(archive["bcv_schema_version"].item())
        if schema_version != _BCV_SCHEMA_VERSION:
            raise ValueError(f"unsupported BCV UBD schema version: {schema_version}")

        labeled = LabeledVolume(body_data.to_numpy(copy=True))
        planes = SelectedCuttingPlanes(archive["bcv_cutting_planes"])
        classification = RegionClassificationVolume(archive["bcv_region_classification"])
        geometry = VolumeGeometry(
            spacing=tuple(float(value) for value in archive["bcv_spacing_mm"]),
            origin=tuple(float(value) for value in archive["bcv_origin_lps_mm"]),
            direction=tuple(
                tuple(float(value) for value in row)
                for row in archive["bcv_direction_index_to_lps"]
            ),
        )
        mmpd = float(body_data.get_mmpd())
        if not np.allclose(geometry.spacing, (mmpd, mmpd, mmpd), atol=1e-12):
            raise ValueError("BCV spacing must agree with the isotropic UBD mmpd value")

        safety: TumorSafetyMarginData | None = None
        has_safety = bool(archive["bcv_has_safety_margin"].item())
        if has_safety:
            safety_keys = {
                "bcv_safety_planning_volume",
                "bcv_safety_protected_mask",
                "bcv_safety_danger_bone_mask",
                "bcv_safety_margin_mm",
            }
            missing_safety = safety_keys - set(archive.files)
            if missing_safety:
                raise ValueError(
                    f"UBD safety-margin data is incomplete; missing fields: {sorted(missing_safety)}"
                )
            safety = TumorSafetyMarginData(
                planning_volume=LabeledVolume(archive["bcv_safety_planning_volume"]),
                protected_mask=archive["bcv_safety_protected_mask"],
                danger_bone_mask=archive["bcv_safety_danger_bone_mask"],
                margin_mm=float(archive["bcv_safety_margin_mm"].item()),
            )

    return ResectionPlanData(labeled, planes, classification, geometry, safety)


def save_resection_plan(plan: ResectionPlanData, path: str | Path) -> Path:
    """Write a plan as UBD v1 plus namespaced BCV extension fields."""

    if not isinstance(plan, ResectionPlanData):
        raise TypeError("plan must be a ResectionPlanData")
    spacing = np.asarray(plan.geometry.spacing, dtype=np.float64)
    if not np.allclose(spacing, spacing[0], atol=1e-12):
        raise ValueError(".ubd.npz requires isotropic spacing")

    resolved = _ubd_path(path, for_write=True)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int32),
        "body_data": plan.labeled_volume.data,
        "image_type": np.asarray("mask", dtype=np.str_),
        "mmpd": np.asarray(spacing[0], dtype=np.float64),
        "bcv_schema_version": np.asarray(_BCV_SCHEMA_VERSION, dtype=np.int32),
        "bcv_cutting_planes": plan.cutting_planes.equations,
        "bcv_region_classification": plan.classification.data,
        "bcv_spacing_mm": spacing,
        "bcv_origin_lps_mm": np.asarray(plan.geometry.origin, dtype=np.float64),
        "bcv_direction_index_to_lps": np.asarray(plan.geometry.direction, dtype=np.float64),
        "bcv_has_safety_margin": np.asarray(plan.safety_margin is not None, dtype=np.bool_),
    }
    if plan.safety_margin is not None:
        fields.update(
            {
                "bcv_safety_planning_volume": plan.safety_margin.planning_volume.data,
                "bcv_safety_protected_mask": plan.safety_margin.protected_mask,
                "bcv_safety_danger_bone_mask": plan.safety_margin.danger_bone_mask,
                "bcv_safety_margin_mm": np.asarray(plan.safety_margin.margin_mm, dtype=np.float64),
            }
        )
    np.savez_compressed(resolved, **fields)

    # Validate that the upstream package accepts the result as a normal UBD file.
    _load_body_data(resolved)
    return resolved
