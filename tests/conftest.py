from __future__ import annotations

import numpy as np
import pytest

from bone_cutting_plane_visualization import (
    LabeledVolume,
    ResectionPlanData,
    SelectedCuttingPlanes,
    TumorSafetyMarginData,
    VolumeGeometry,
    classify_regions,
)


@pytest.fixture
def labeled_volume() -> LabeledVolume:
    volume = np.zeros((9, 10, 11), dtype=np.int8)
    volume[1:8, 2:8, 2:9] = -1
    volume[3:6, 4:7, 4:7] = 1
    return LabeledVolume(volume)


@pytest.fixture
def cutting_planes() -> SelectedCuttingPlanes:
    return SelectedCuttingPlanes(
        np.asarray(
            [
                [1.0, 0.0, 0.0, -4.5],
                [0.0, 1.0, 0.0, -8.5],
            ]
        )
    )


@pytest.fixture
def plan(labeled_volume: LabeledVolume, cutting_planes: SelectedCuttingPlanes):
    return ResectionPlanData(
        labeled_volume,
        cutting_planes,
        classify_regions(labeled_volume, cutting_planes),
        VolumeGeometry(spacing=(2.0, 2.0, 2.0), origin=(10.0, 20.0, 30.0)),
    )


@pytest.fixture
def safety_plan(labeled_volume: LabeledVolume):
    planes = SelectedCuttingPlanes(np.asarray([[1.0, 0.0, 0.0, -6.5]]))
    classification = classify_regions(labeled_volume, planes)
    protected = labeled_volume.tumor_mask.copy()
    protected[6, 2:8, 2:9] = True
    danger = protected & labeled_volume.normal_bone_mask
    planning = labeled_volume.to_numpy(copy=True)
    planning[protected] = 1
    safety = TumorSafetyMarginData(LabeledVolume(planning), protected, danger, 5.0)
    return ResectionPlanData(
        labeled_volume,
        planes,
        classification,
        VolumeGeometry(spacing=(1.5, 1.5, 1.5)),
        safety,
    )
