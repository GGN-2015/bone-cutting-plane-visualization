"""Standalone data contracts and VTK rendering for bone cutting-plane plans."""

from .classification import classify_regions
from .data import (
    LabeledVolume,
    RegionClassificationVolume,
    ResectionPlanData,
    SelectedCuttingPlanes,
    TumorSafetyMarginData,
    VolumeGeometry,
)
from .geometry import plane_box_intersection
from .ubd import (
    IncompleteResectionPlanError,
    load_labeled_volume,
    load_resection_plan,
    save_resection_plan,
)
from .visualization import (
    BoneTumorVisualizer,
    MeshLayer,
    PatientOrientationCube,
    PolygonLayer,
    SurfaceLayer,
    VisualizationModel,
    VisualizationScene,
    patient_orientation_cube,
    resection_boundary,
    visualize_bone_and_tumor,
    visualize_labeled_volume,
    visualize_resection,
    visualize_tumor_convex_hull,
)

__all__ = [
    "BoneTumorVisualizer",
    "IncompleteResectionPlanError",
    "LabeledVolume",
    "MeshLayer",
    "PatientOrientationCube",
    "PolygonLayer",
    "RegionClassificationVolume",
    "ResectionPlanData",
    "SelectedCuttingPlanes",
    "SurfaceLayer",
    "TumorSafetyMarginData",
    "VisualizationModel",
    "VisualizationScene",
    "VolumeGeometry",
    "classify_regions",
    "load_labeled_volume",
    "load_resection_plan",
    "patient_orientation_cube",
    "plane_box_intersection",
    "resection_boundary",
    "save_resection_plan",
    "visualize_bone_and_tumor",
    "visualize_labeled_volume",
    "visualize_resection",
    "visualize_tumor_convex_hull",
]
