from __future__ import annotations

import numpy as np

from bone_cutting_plane_visualization import VolumeGeometry
from bone_cutting_plane_visualization.compatibility import (
    plan_from_arrays,
    visualize_solution,
)


def test_array_adapter_builds_typed_plan(labeled_volume):
    equations = np.asarray([[1.0, 0.0, 0.0, -4.5]])
    plan = plan_from_arrays(
        labeled_volume.data,
        equations,
        geometry=VolumeGeometry(spacing=(2.0, 2.0, 2.0)),
    )

    assert plan.labeled_volume.shape == labeled_volume.shape
    assert plan.cutting_planes.count == 1
    assert plan.geometry.spacing == (2.0, 2.0, 2.0)
    assert np.any(plan.classification.retained_mask)
    assert np.any(plan.classification.resected_mask)


def test_array_adapter_accepts_injected_region_solver(labeled_volume):
    solved = labeled_volume.to_numpy(copy=True)
    solved[labeled_volume.normal_bone_mask] = 2

    def region_solver(data, equations):
        np.testing.assert_array_equal(data, labeled_volume.data)
        assert equations.shape == (1, 4)
        return solved

    plan = plan_from_arrays(
        labeled_volume.data,
        np.asarray([[1.0, 0.0, 0.0, -4.5]]),
        region_solver=region_solver,
    )
    np.testing.assert_array_equal(plan.classification.data, solved)


def test_legacy_solution_can_compute_statistics_without_rendering(labeled_volume, capsys):
    result = visualize_solution(
        labeled_volume.data,
        np.asarray([[1.0, 0.0, 0.0, -4.5]]),
        False,
    )

    assert result is None
    assert "keep_rate:" in capsys.readouterr().out
