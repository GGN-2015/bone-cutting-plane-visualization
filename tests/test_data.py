from __future__ import annotations

import numpy as np
import pytest

from bone_cutting_plane_visualization import (
    LabeledVolume,
    RegionClassificationVolume,
    ResectionPlanData,
    SelectedCuttingPlanes,
    TumorSafetyMarginData,
    VolumeGeometry,
    classify_regions,
)


def test_labeled_volume_normalizes_and_owns_read_only_data():
    source = np.asarray([[[-1.0, 0.0, 1.0]]])
    volume = LabeledVolume(source)
    source[0, 0, 0] = 0

    assert volume.data.dtype == np.int8
    assert volume.data[0, 0, 0] == -1
    assert not volume.data.flags.writeable
    with pytest.raises(ValueError):
        volume.data[0, 0, 0] = 0


@pytest.mark.parametrize(
    "value, message",
    [
        (np.zeros((2, 2)), "three-dimensional"),
        (np.asarray([[[0.5]]]), "integer-valued"),
        (np.asarray([[[3]]]), "only -1, 0, and 1"),
    ],
)
def test_labeled_volume_rejects_invalid_data(value, message):
    with pytest.raises(ValueError, match=message):
        LabeledVolume(value)


def test_cutting_planes_preserve_order_and_reject_zero_normal():
    source = np.asarray([[1.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, -3.0]])
    planes = SelectedCuttingPlanes(source)
    source[0] = 0

    np.testing.assert_array_equal(planes.equations[0], np.asarray([1.0, 0.0, 0.0, -2.0]))
    assert not planes.equations.flags.writeable
    with pytest.raises(ValueError, match="non-zero"):
        SelectedCuttingPlanes(np.zeros((1, 4)))


def test_region_classifier_matches_ordered_positive_half_space_semantics(labeled_volume):
    planes = SelectedCuttingPlanes(np.asarray([[1.0, 0.0, 0.0, -4.5], [0.0, 1.0, 0.0, -5.5]]))
    solved = classify_regions(labeled_volume, planes, chunk_size=11)
    normal = labeled_volume.normal_bone_mask
    indices = np.indices(labeled_volume.shape)

    np.testing.assert_array_equal(solved.data[(normal) & (indices[0] >= 5)], -1)
    np.testing.assert_array_equal(solved.data[(normal) & (indices[0] < 5) & (indices[1] >= 6)], -2)
    assert np.all(solved.data[(normal) & (indices[0] < 5) & (indices[1] < 6)] == 2)
    assert np.all(solved.data[labeled_volume.tumor_mask] == 1)


def test_volume_geometry_round_trip_and_plane_transform():
    geometry = VolumeGeometry(
        spacing=(2.0, 3.0, 4.0),
        origin=(10.0, 20.0, 30.0),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(geometry.world_to_index(geometry.index_to_world(points)), points)
    world_plane = VolumeGeometry(spacing=(2, 1, 1), origin=(10, 0, 0)).plane_index_to_world(
        np.asarray([1, 0, 0, -2])
    )
    np.testing.assert_allclose(world_plane, [1, 0, 0, -14])


def test_safety_margin_respects_physical_spacing():
    array = np.zeros((5, 5, 5), dtype=np.int8)
    array[2, 3, 2] = -1
    array[3, 2, 2] = -1
    array[2, 2, 2] = 1
    volume = LabeledVolume(array)
    margin = TumorSafetyMarginData.compute(volume, 1.1, VolumeGeometry(spacing=(2.0, 1.0, 1.0)))

    assert margin.danger_bone_mask[2, 3, 2]
    assert not margin.danger_bone_mask[3, 2, 2]
    assert margin.planning_volume.data[2, 3, 2] == 1
    assert volume.data[2, 3, 2] == -1


def test_complete_plan_rejects_shape_and_classification_mismatch(labeled_volume, cutting_planes):
    with pytest.raises(ValueError, match="same shape"):
        ResectionPlanData(
            labeled_volume,
            cutting_planes,
            RegionClassificationVolume(np.zeros((2, 2, 2))),
            VolumeGeometry(),
        )

    solved = labeled_volume.to_numpy(copy=True).astype(np.int32)
    solved[0, 0, 0] = 2
    solved[labeled_volume.normal_bone_mask] = -1
    with pytest.raises(ValueError, match="only change normal-bone"):
        ResectionPlanData(
            labeled_volume,
            cutting_planes,
            RegionClassificationVolume(solved),
            VolumeGeometry(),
        )


def test_complete_plan_rejects_retained_safety_bone(labeled_volume):
    planes = SelectedCuttingPlanes(np.asarray([[1.0, 0.0, 0.0, -6.5]]))
    solved = classify_regions(labeled_volume, planes).to_numpy(copy=True)
    protected = labeled_volume.tumor_mask.copy()
    protected[6, 2, 2] = True
    planning = labeled_volume.to_numpy(copy=True)
    planning[protected] = 1
    safety = TumorSafetyMarginData(
        LabeledVolume(planning), protected, protected & labeled_volume.normal_bone_mask, 2.0
    )
    solved[6, 2, 2] = -1

    with pytest.raises(ValueError, match="must be resected"):
        ResectionPlanData(
            labeled_volume,
            planes,
            RegionClassificationVolume(solved),
            VolumeGeometry(),
            safety,
        )


def test_complete_plan_requires_safety_mask_to_include_tumor(labeled_volume):
    planes = SelectedCuttingPlanes(np.asarray([[1.0, 0.0, 0.0, -6.5]]))
    solved = classify_regions(labeled_volume, planes)
    protected = np.zeros(labeled_volume.shape, dtype=bool)
    safety = TumorSafetyMarginData(labeled_volume, protected, protected, 1.0)

    with pytest.raises(ValueError, match="every tumor voxel"):
        ResectionPlanData(
            labeled_volume,
            planes,
            solved,
            VolumeGeometry(),
            safety,
        )
