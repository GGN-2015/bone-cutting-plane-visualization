from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

from bone_cutting_plane_visualization import (
    BoneTumorVisualizer,
    VolumeGeometry,
    patient_orientation_cube,
    plane_box_intersection,
)


def test_prepares_bone_and_tumor_without_loading_vtk(labeled_volume):
    model = BoneTumorVisualizer().prepare_bone_and_tumor(labeled_volume)

    assert model.title == "Normal bone and tumor"
    assert [layer.name for layer in model.surfaces] == ["Normal bone", "Tumor"]
    assert int(model.surfaces[0].mask.sum()) == 267
    assert int(model.surfaces[1].mask.sum()) == 27


def test_convex_hull_uses_physical_coordinates(labeled_volume):
    geometry = VolumeGeometry(spacing=(2.0, 3.0, 4.0), origin=(10.0, 20.0, 30.0))
    model = BoneTumorVisualizer(geometry).prepare_tumor_convex_hull(labeled_volume)
    hull = model.meshes[0]

    assert hull.name == "Tumor convex hull"
    assert hull.show_edges
    assert hull.faces.shape[1] == 3
    assert np.all(hull.points[:, 0] >= 16.0)
    assert np.all(hull.points[:, 1] >= 32.0)
    assert np.all(hull.points[:, 2] >= 46.0)


def test_plane_box_intersection_is_ordered_and_bounded():
    equation = np.asarray([1.0, 1.0, 0.0, -7.5])
    points = plane_box_intersection(equation, (9, 10, 11))

    assert len(points) >= 4
    np.testing.assert_allclose(points @ equation[:3] + equation[3], 0.0, atol=1e-9)
    assert np.all(points >= 0.0)
    assert np.all(points <= np.asarray([8.0, 9.0, 10.0]))


def test_resection_rejects_plane_outside_volume(plan):
    from dataclasses import replace

    from bone_cutting_plane_visualization import SelectedCuttingPlanes

    outside = replace(
        plan,
        cutting_planes=SelectedCuttingPlanes(np.asarray([[1.0, 0.0, 0.0, -100.0]])),
    )
    with pytest.raises(ValueError, match="does not intersect"):
        BoneTumorVisualizer().prepare_resection(outside)


def test_patient_orientation_cube_is_small_and_outside_negative_volume_bounds():
    geometry = VolumeGeometry(
        spacing=(2.0, 3.0, 4.0),
        origin=(10.0, 20.0, 30.0),
        direction=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    shape = (9, 10, 11)
    layout = patient_orientation_cube(shape, geometry)
    index_corners = np.asarray(
        [
            (x, y, z)
            for x in (-0.5, shape[0] - 0.5)
            for y in (-0.5, shape[1] - 0.5)
            for z in (-0.5, shape[2] - 0.5)
        ]
    )
    volume_min = geometry.index_to_world(index_corners).min(axis=0)
    cube_max = np.asarray([layout.bounds[1], layout.bounds[3], layout.bounds[5]])

    cube_min = np.asarray([layout.bounds[0], layout.bounds[2], layout.bounds[4]])
    assert np.all(cube_max[:2] < volume_min[:2])
    assert cube_min[2] >= geometry.index_to_world(index_corners).min(axis=0)[2]
    assert cube_max[2] <= geometry.index_to_world(index_corners).max(axis=0)[2]
    assert layout.size < 0.15 * 44.0
    assert dict(layout.face_labels) == {
        "L": (1.0, 0.0, 0.0),
        "R": (-1.0, 0.0, 0.0),
        "P": (0.0, 1.0, 0.0),
        "A": (0.0, -1.0, 0.0),
        "S": (0.0, 0.0, 1.0),
        "i": (0.0, 0.0, -1.0),
    }


def test_prepares_complete_resection_scene(plan):
    model = BoneTumorVisualizer().prepare_resection(plan)

    assert [layer.name for layer in model.surfaces] == [
        "Retained normal bone",
        "Resected normal bone",
        "Tumor",
    ]
    assert [polygon.name for polygon in model.polygons] == [
        "Cutting plane 1",
        "Cutting plane 2",
    ]
    assert model.geometry == plan.geometry
    assert model.keep_rate == plan.keep_rate
    assert f"{plan.keep_rate:.1%}" in model.title


def test_safety_margin_is_visually_distinct(safety_plan):
    model = BoneTumorVisualizer().prepare_resection(safety_plan)

    assert [layer.name for layer in model.surfaces] == [
        "Retained normal bone",
        "Resected normal bone",
        "Safety-margin bone (<= 5 mm)",
        "Tumor",
    ]
    safety_layer = model.surfaces[2]
    assert safety_layer.color == (0.98, 0.74, 0.18)
    np.testing.assert_array_equal(safety_layer.mask, safety_plan.safety_margin.danger_bone_mask)
    assert "safety margin: 5 mm" in model.title


def test_boundary_view_preserves_cutting_plane_labels(plan):
    model = BoneTumorVisualizer().prepare_resection_boundary(plan)

    assert model.title.startswith("Resection boundary")
    assert len(model.polygons) == plan.cutting_planes.count
    assert any(layer.name == "Cut boundary 1" for layer in model.surfaces)


@pytest.mark.skipif(importlib.util.find_spec("vtk") is None, reason="VTK is not installed")
def test_vtk_scene_builds_without_opengl_render(labeled_volume):
    visualizer = BoneTumorVisualizer()
    scene = visualizer.build_scene(
        visualizer.prepare_bone_and_tumor(labeled_volume),
        offscreen=True,
        window_size=(480, 360),
    )
    assert "Normal bone" in scene.actors
    assert "Tumor" in scene.actors
    assert "Legend" in scene.actors
    assert "Patient orientation cube" in scene.actors
    assert all(f"Orientation {label}" in scene.actors for label in "LRPASi")
    cube_property = scene.actors["Patient orientation cube"].GetProperty()
    assert cube_property.GetColor() == (0.0, 0.0, 0.0)
    assert cube_property.GetEdgeColor() == (1.0, 1.0, 1.0)
    assert cube_property.GetEdgeVisibility() == 1


@pytest.mark.skipif(
    importlib.util.find_spec("vtk") is None or os.environ.get("BCV_TEST_VTK_RENDER") != "1",
    reason="VTK rendering is not enabled",
)
def test_vtk_offscreen_plan_render_writes_nonempty_png(plan, tmp_path):
    screenshot = tmp_path / "plan.png"
    scene = BoneTumorVisualizer().show_resection(
        plan,
        interactive=False,
        offscreen=True,
        screenshot=screenshot,
        window_size=(480, 360),
    )

    assert "Cutting plane 1" in scene.actors
    assert screenshot.stat().st_size > 10_000
    pixels = np.asarray(Image.open(screenshot).convert("RGB"))
    assert pixels.shape == (360, 480, 3)
    background = np.asarray([19, 19, 22], dtype=np.uint8)
    assert np.count_nonzero(np.any(np.abs(pixels.astype(int) - background) > 8, axis=2)) > 1_000
    assert np.count_nonzero(np.ptp(pixels, axis=2) > 12) > 500
