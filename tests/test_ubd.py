from __future__ import annotations

import numpy as np
import pytest
from ct_mri_dicom_nii_reader import BodyData, BodyDataLoaderManager

from bone_cutting_plane_visualization import (
    load_labeled_volume,
    load_resection_plan,
    save_resection_plan,
)


def test_loads_standard_upstream_ubd(labeled_volume, tmp_path):
    body_data = BodyData()
    body_data.from_array(labeled_volume.to_numpy(copy=True), "mask", 2.5)
    body_data.save(str(tmp_path / "labels"))

    loaded, geometry = load_labeled_volume(tmp_path / "labels.ubd.npz")

    np.testing.assert_array_equal(loaded.data, labeled_volume.data)
    assert geometry.spacing == (2.5, 2.5, 2.5)
    assert geometry.origin == (0.0, 0.0, 0.0)


def test_resection_plan_round_trips_as_upstream_compatible_ubd(safety_plan, tmp_path):
    path = save_resection_plan(safety_plan, tmp_path / "case-plan")

    assert path.name == "case-plan.ubd.npz"
    upstream = BodyDataLoaderManager().load_file(str(path))
    assert upstream.get_type() == "mask"
    assert upstream.get_mmpd() == 1.5
    np.testing.assert_array_equal(upstream.to_numpy(), safety_plan.labeled_volume.data)

    restored = load_resection_plan(path)
    np.testing.assert_array_equal(
        restored.cutting_planes.equations, safety_plan.cutting_planes.equations
    )
    np.testing.assert_array_equal(restored.classification.data, safety_plan.classification.data)
    np.testing.assert_array_equal(
        restored.safety_margin.danger_bone_mask,
        safety_plan.safety_margin.danger_bone_mask,
    )
    assert restored.geometry == safety_plan.geometry
    assert restored.keep_rate == safety_plan.keep_rate


def test_plain_ubd_is_not_mistaken_for_complete_plan(labeled_volume, tmp_path):
    body_data = BodyData()
    body_data.from_array(labeled_volume.to_numpy(copy=True), "mask", 1.0)
    body_data.save(str(tmp_path / "labels.ubd.npz"))

    with pytest.raises(ValueError, match="not a complete resection plan"):
        load_resection_plan(tmp_path / "labels.ubd.npz")


def test_only_ubd_extension_is_accepted(tmp_path):
    with pytest.raises(ValueError, match="must end with .ubd.npz"):
        load_labeled_volume(tmp_path / "labels.npz")


def test_ubd_save_rejects_anisotropic_geometry(plan, tmp_path):
    from dataclasses import replace

    from bone_cutting_plane_visualization import VolumeGeometry

    anisotropic = replace(plan, geometry=VolumeGeometry(spacing=(1.0, 2.0, 3.0)))
    with pytest.raises(ValueError, match="requires isotropic"):
        save_resection_plan(anisotropic, tmp_path / "plan.ubd.npz")
