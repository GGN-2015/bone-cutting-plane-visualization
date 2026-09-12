"""Array-based adapters for callers migrating to the typed public API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .classification import classify_regions
from .data import (
    LabeledVolume,
    RegionClassificationVolume,
    ResectionPlanData,
    SelectedCuttingPlanes,
    TumorSafetyMarginData,
    VolumeGeometry,
)
from .visualization import (
    BOUNDARY_COLOR_MAP,
    BOUNDARY_LABEL_NAMES,
    DEFAULT_COLOR_MAP,
    DEFAULT_LABEL_NAMES,
    BoneTumorVisualizer,
    VisualizationScene,
)

RegionSolver = Callable[[np.ndarray, np.ndarray], np.ndarray]


def plan_from_arrays(
    volume: np.ndarray,
    equations: np.ndarray,
    *,
    solved_volume: np.ndarray | None = None,
    geometry: VolumeGeometry | None = None,
    region_solver: RegionSolver | None = None,
    danger_bone_mask: np.ndarray | None = None,
    safety_margin_mm: float = 0.0,
) -> ResectionPlanData:
    """Convert the former NumPy arguments into one validated plan object."""

    labeled = LabeledVolume(volume)
    planes = SelectedCuttingPlanes(equations)
    if solved_volume is not None and region_solver is not None:
        raise ValueError("provide solved_volume or region_solver, not both")
    if solved_volume is not None:
        classification = RegionClassificationVolume(solved_volume)
    elif region_solver is not None:
        classification = RegionClassificationVolume(region_solver(labeled.data, planes.equations))
    else:
        classification = classify_regions(labeled, planes)
    resolved_geometry = geometry if geometry is not None else VolumeGeometry()

    safety = None
    danger = (
        np.zeros(labeled.shape, dtype=bool)
        if danger_bone_mask is None
        else np.asarray(danger_bone_mask, dtype=bool)
    )
    if danger.shape != labeled.shape:
        raise ValueError("danger_bone_mask must have the same shape as volume")
    if safety_margin_mm > 0 or np.any(danger):
        protected = labeled.tumor_mask | danger
        planning = labeled.to_numpy(copy=True)
        planning[protected] = 1
        safety = TumorSafetyMarginData(LabeledVolume(planning), protected, danger, safety_margin_mm)
    return ResectionPlanData(labeled, planes, classification, resolved_geometry, safety)


def visualize_resection_arrays(
    volume: np.ndarray,
    equations: np.ndarray,
    *,
    solved_volume: np.ndarray | None = None,
    geometry: VolumeGeometry | None = None,
    region_solver: RegionSolver | None = None,
    show_retained: bool = True,
    danger_bone_mask: np.ndarray | None = None,
    safety_margin_mm: float = 0.0,
    **render_options: Any,
) -> VisualizationScene:
    plan = plan_from_arrays(
        volume,
        equations,
        solved_volume=solved_volume,
        geometry=geometry,
        region_solver=region_solver,
        danger_bone_mask=danger_bone_mask,
        safety_margin_mm=safety_margin_mm,
    )
    return BoneTumorVisualizer(plan.geometry).show_resection(
        plan, show_retained=show_retained, **render_options
    )


def visualize_volume(
    data: np.ndarray,
    color_map: Mapping[float, tuple[float, float, float, float]] | None = None,
    *,
    label_names: Mapping[float, str] | None = None,
    title: str = "Volume visualization",
    geometry: VolumeGeometry | None = None,
    interactive: bool = True,
    offscreen: bool = False,
    screenshot: str | Path | None = None,
    window_size: tuple[int, int] = (1100, 800),
) -> VisualizationScene:
    visualizer = BoneTumorVisualizer(geometry)
    model = visualizer.prepare_labeled_volume(
        data,
        color_map=DEFAULT_COLOR_MAP if color_map is None else color_map,
        label_names=DEFAULT_LABEL_NAMES if label_names is None else label_names,
        title=title,
    )
    return visualizer.render(
        model,
        interactive=interactive,
        offscreen=offscreen,
        screenshot=screenshot,
        window_size=window_size,
    )


def visualize_solution(
    data: np.ndarray,
    equations: np.ndarray,
    show_volume: bool,
    show_only_resected_bone: bool = False,
    show_only_boundary: bool = False,
    *,
    geometry: VolumeGeometry | None = None,
    interactive: bool = True,
    offscreen: bool = False,
    screenshot: str | Path | None = None,
    window_size: tuple[int, int] = (1100, 800),
) -> VisualizationScene | None:
    plan = plan_from_arrays(data, equations, geometry=geometry)
    retained = int(np.count_nonzero(plan.classification.retained_mask))
    resected = int(np.count_nonzero(plan.classification.resected_mask))
    print(f"- keep_rate: {plan.keep_rate * 100:7.3f}% ({retained}/{retained + resected})")
    if not show_volume:
        return None
    visualizer = BoneTumorVisualizer(plan.geometry)
    render_options = {
        "interactive": interactive,
        "offscreen": offscreen,
        "screenshot": screenshot,
        "window_size": window_size,
    }
    if show_only_boundary:
        boundary = visualizer.prepare_resection_boundary(plan)
        return visualizer.render(boundary, **render_options)
    return visualizer.show_resection(
        plan,
        show_retained=not show_only_resected_bone,
        **render_options,
    )


__all__ = [
    "BOUNDARY_COLOR_MAP",
    "BOUNDARY_LABEL_NAMES",
    "DEFAULT_COLOR_MAP",
    "DEFAULT_LABEL_NAMES",
    "plan_from_arrays",
    "visualize_resection_arrays",
    "visualize_solution",
    "visualize_volume",
]
