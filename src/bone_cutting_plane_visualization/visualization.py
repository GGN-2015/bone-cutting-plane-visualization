"""High-level VTK visualization for bone-tumor resection planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import LabeledVolume, ResectionPlanData, SelectedCuttingPlanes, VolumeGeometry
from .geometry import clip_convex_polygon_to_halfspaces, plane_box_intersection

Color = tuple[float, float, float]
RgbaColor = tuple[float, float, float, float]

DEFAULT_COLOR_MAP = {
    -6.0: (0.87, 0.84, 0.76, 0.28),
    -5.0: (0.42, 0.82, 0.30, 0.58),
    -4.0: (0.92, 0.38, 0.64, 0.58),
    -3.0: (0.34, 0.58, 0.96, 0.58),
    -2.0: (0.98, 0.74, 0.18, 0.58),
    -1.0: (0.30, 0.78, 0.52, 0.42),
    0.0: (0.00, 0.00, 0.00, 0.0),
    1.0: (0.88, 0.12, 0.16, 0.96),
    2.0: (0.55, 0.40, 0.78, 0.58),
}

DEFAULT_LABEL_NAMES = {
    -6.0: "Bone interior",
    -5.0: "Bone cut by plane 5",
    -4.0: "Bone cut by plane 4",
    -3.0: "Bone cut by plane 3",
    -2.0: "Bone cut by plane 2",
    -1.0: "Normal bone",
    1.0: "Tumor",
    2.0: "Resected normal bone",
}

BOUNDARY_COLOR_MAP = {
    -6.0: (0.87, 0.84, 0.76, 0.16),
    -5.0: (0.42, 0.82, 0.30, 0.88),
    -4.0: (0.92, 0.38, 0.64, 0.88),
    -3.0: (0.34, 0.58, 0.96, 0.88),
    -2.0: (0.98, 0.74, 0.18, 0.88),
    -1.0: (0.12, 0.76, 0.84, 0.88),
    0.0: (0.00, 0.00, 0.00, 0.0),
    1.0: (0.88, 0.12, 0.16, 0.0),
    2.0: (0.55, 0.40, 0.78, 0.0),
}

BOUNDARY_LABEL_NAMES = {
    -6.0: "Retained bone context",
    -5.0: "Cut boundary 5",
    -4.0: "Cut boundary 4",
    -3.0: "Cut boundary 3",
    -2.0: "Cut boundary 2",
    -1.0: "Cut boundary 1",
}


@dataclass(frozen=True)
class SurfaceLayer:
    name: str
    mask: np.ndarray
    color: Color
    opacity: float


@dataclass(frozen=True)
class MeshLayer:
    name: str
    points: np.ndarray
    faces: np.ndarray
    color: Color
    opacity: float
    show_edges: bool = False


@dataclass(frozen=True)
class PolygonLayer:
    name: str
    points: np.ndarray
    color: Color
    opacity: float


@dataclass(frozen=True)
class PatientOrientationCube:
    """World-space layout for a patient LPS orientation cube."""

    center: tuple[float, float, float]
    size: float
    face_labels: tuple[tuple[str, tuple[float, float, float]], ...]

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        half = self.size / 2.0
        return (
            self.center[0] - half,
            self.center[0] + half,
            self.center[1] - half,
            self.center[1] + half,
            self.center[2] - half,
            self.center[2] + half,
        )


@dataclass(frozen=True)
class VisualizationModel:
    title: str
    volume_shape: tuple[int, int, int]
    geometry: VolumeGeometry
    surfaces: tuple[SurfaceLayer, ...]
    meshes: tuple[MeshLayer, ...] = ()
    polygons: tuple[PolygonLayer, ...] = ()
    keep_rate: float | None = None
    safety_margin_mm: float = 0.0


@dataclass
class VisualizationScene:
    model: VisualizationModel
    renderer: Any
    render_window: Any
    interactor: Any
    actors: dict[str, Any]
    vtk: Any

    def show(
        self,
        *,
        interactive: bool = True,
        screenshot: str | Path | None = None,
    ) -> VisualizationScene:
        self.render_window.Render()
        if screenshot is not None:
            screenshot_path = Path(screenshot).expanduser().resolve()
            if screenshot_path.suffix.lower() != ".png":
                raise ValueError("screenshot path must use the .png extension")
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            window_image = self.vtk.vtkWindowToImageFilter()
            window_image.SetInput(self.render_window)
            window_image.SetInputBufferTypeToRGB()
            window_image.ReadFrontBufferOff()
            window_image.Update()
            writer = self.vtk.vtkPNGWriter()
            writer.SetFileName(str(screenshot_path))
            writer.SetInputConnection(window_image.GetOutputPort())
            writer.Write()
        if interactive:
            self.interactor.Initialize()
            self.interactor.Start()
        return self


class BoneTumorVisualizer:
    """Prepare and render the main bone-tumor inspection views."""

    _BONE_COLOR = (0.87, 0.84, 0.76)
    _TUMOR_COLOR = (0.88, 0.12, 0.16)
    _HULL_COLOR = (0.98, 0.68, 0.12)
    _RETAINED_COLOR = (0.30, 0.78, 0.52)
    _RESECTED_COLOR = (0.55, 0.40, 0.78)
    _DANGER_BONE_COLOR = (0.98, 0.74, 0.18)
    _PLANE_COLORS = (
        (0.12, 0.76, 0.84),
        (0.98, 0.74, 0.18),
        (0.34, 0.58, 0.96),
        (0.92, 0.38, 0.64),
        (0.42, 0.82, 0.30),
    )

    def __init__(self, geometry: VolumeGeometry | None = None) -> None:
        self.geometry = geometry if geometry is not None else VolumeGeometry()

    def prepare_bone_and_tumor(self, volume: LabeledVolume) -> VisualizationModel:
        _require_labeled_volume(volume)
        _require_bone_and_tumor(volume)
        surfaces = (
            SurfaceLayer("Normal bone", volume.normal_bone_mask, self._BONE_COLOR, 0.34),
            SurfaceLayer("Tumor", volume.tumor_mask, self._TUMOR_COLOR, 0.96),
        )
        return VisualizationModel(
            title="Normal bone and tumor",
            volume_shape=volume.shape,
            geometry=self.geometry,
            surfaces=surfaces,
        )

    def prepare_labeled_volume(
        self,
        volume: np.ndarray,
        *,
        color_map: Mapping[float, RgbaColor],
        label_names: Mapping[float, str] | None = None,
        title: str = "Volume visualization",
        cutting_planes: SelectedCuttingPlanes | None = None,
    ) -> VisualizationModel:
        data = np.asarray(volume)
        if data.ndim != 3 or not np.all(np.isfinite(data)):
            raise ValueError("volume must be a finite three-dimensional array")
        names = label_names if label_names is not None else {}
        surfaces = []
        for label, rgba in color_map.items():
            color, opacity = _validate_rgba(label, rgba)
            mask = data == label
            if opacity <= 0.0 or not np.any(mask):
                continue
            surfaces.append(
                SurfaceLayer(names.get(label, _default_label_name(label)), mask, color, opacity)
            )
        if not surfaces:
            raise ValueError("volume contains no visible labels from color_map")
        polygons = (
            self._prepare_cutting_planes(cutting_planes, data.shape)
            if cutting_planes is not None
            else ()
        )
        return VisualizationModel(
            title=title,
            volume_shape=data.shape,
            geometry=self.geometry,
            surfaces=tuple(surfaces),
            polygons=polygons,
        )

    def prepare_tumor_convex_hull(self, volume: LabeledVolume) -> VisualizationModel:
        model = self.prepare_bone_and_tumor(volume)
        tumor_points = np.argwhere(volume.tumor_mask)
        if len(tumor_points) < 4:
            raise ValueError("at least four tumor voxels are required for a convex hull")

        from scipy.spatial import ConvexHull, QhullError

        try:
            hull = ConvexHull(tumor_points)
        except QhullError as exc:
            raise ValueError("tumor voxels do not form a three-dimensional convex hull") from exc
        used_indices = np.unique(hull.simplices)
        faces = np.searchsorted(used_indices, hull.simplices).astype(np.int64)
        points = self.geometry.index_to_world(tumor_points[used_indices])
        mesh = MeshLayer(
            "Tumor convex hull", points, faces, self._HULL_COLOR, 0.22, show_edges=True
        )
        return VisualizationModel(
            title="Normal bone, tumor, and tumor convex hull",
            volume_shape=model.volume_shape,
            geometry=self.geometry,
            surfaces=model.surfaces,
            meshes=(mesh,),
        )

    def prepare_resection(
        self,
        plan: ResectionPlanData,
        *,
        show_retained: bool = True,
    ) -> VisualizationModel:
        if not isinstance(plan, ResectionPlanData):
            raise TypeError("plan must be a ResectionPlanData")
        data = plan.labeled_volume
        solved = plan.classification
        danger = (
            plan.safety_margin.danger_bone_mask
            if plan.safety_margin is not None
            else np.zeros(data.shape, dtype=bool)
        )
        margin_mm = plan.safety_margin.margin_mm if plan.safety_margin is not None else 0.0
        retained = data.normal_bone_mask & solved.retained_mask & ~danger
        resected = data.normal_bone_mask & solved.resected_mask & ~danger
        surfaces = []
        if show_retained:
            surfaces.append(
                SurfaceLayer("Retained normal bone", retained, self._RETAINED_COLOR, 0.38)
            )
        surfaces.append(SurfaceLayer("Resected normal bone", resected, self._RESECTED_COLOR, 0.58))
        if np.any(danger):
            surfaces.append(
                SurfaceLayer(
                    f"Safety-margin bone (<= {margin_mm:g} mm)",
                    danger,
                    self._DANGER_BONE_COLOR,
                    0.92,
                )
            )
        surfaces.append(SurfaceLayer("Tumor", data.tumor_mask, self._TUMOR_COLOR, 0.96))

        polygons = self._prepare_cutting_planes(plan.cutting_planes, data.shape, plan.geometry)
        title = f"Resection plan - retained normal bone: {plan.keep_rate:.1%}"
        if margin_mm > 0:
            title += f" - safety margin: {margin_mm:g} mm"
        return VisualizationModel(
            title=title,
            volume_shape=data.shape,
            geometry=plan.geometry,
            surfaces=tuple(surfaces),
            polygons=polygons,
            keep_rate=plan.keep_rate,
            safety_margin_mm=margin_mm,
        )

    def prepare_resection_boundary(self, plan: ResectionPlanData) -> VisualizationModel:
        boundary = resection_boundary(plan.classification.data)
        return BoneTumorVisualizer(plan.geometry).prepare_labeled_volume(
            boundary,
            color_map=BOUNDARY_COLOR_MAP,
            label_names=BOUNDARY_LABEL_NAMES,
            title=f"Resection boundary - retained normal bone: {plan.keep_rate:.1%}",
            cutting_planes=plan.cutting_planes,
        )

    def show_bone_and_tumor(
        self, volume: LabeledVolume, **render_options: Any
    ) -> VisualizationScene:
        return self.render(self.prepare_bone_and_tumor(volume), **render_options)

    def show_tumor_convex_hull(
        self, volume: LabeledVolume, **render_options: Any
    ) -> VisualizationScene:
        return self.render(self.prepare_tumor_convex_hull(volume), **render_options)

    def show_resection(
        self,
        plan: ResectionPlanData,
        *,
        show_retained: bool = True,
        **render_options: Any,
    ) -> VisualizationScene:
        return self.render(
            self.prepare_resection(plan, show_retained=show_retained), **render_options
        )

    def show_resection_boundary(
        self, plan: ResectionPlanData, **render_options: Any
    ) -> VisualizationScene:
        return self.render(self.prepare_resection_boundary(plan), **render_options)

    def render(
        self,
        model: VisualizationModel,
        *,
        interactive: bool = True,
        offscreen: bool = False,
        screenshot: str | Path | None = None,
        window_size: tuple[int, int] = (1100, 800),
    ) -> VisualizationScene:
        scene = self.build_scene(model, offscreen=offscreen, window_size=window_size)
        return scene.show(interactive=interactive and not offscreen, screenshot=screenshot)

    def build_scene(
        self,
        model: VisualizationModel,
        *,
        offscreen: bool = False,
        window_size: tuple[int, int] = (1100, 800),
    ) -> VisualizationScene:
        vtk, numpy_support = _load_vtk()
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.075, 0.075, 0.085)
        renderer.SetUseDepthPeeling(True)
        renderer.SetMaximumNumberOfPeels(100)
        renderer.SetOcclusionRatio(0.0)

        render_window = vtk.vtkRenderWindow()
        render_window.SetWindowName(model.title)
        render_window.SetSize(*window_size)
        render_window.SetAlphaBitPlanes(1)
        render_window.SetMultiSamples(0)
        render_window.AddRenderer(renderer)
        if offscreen:
            render_window.SetOffScreenRendering(1)

        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        actors: dict[str, Any] = {}
        for layer in model.surfaces:
            actor = _surface_actor(vtk, numpy_support, layer, model.geometry)
            renderer.AddActor(actor)
            actors[layer.name] = actor
        for layer in model.meshes:
            actor = _mesh_actor(vtk, layer)
            renderer.AddActor(actor)
            actors[layer.name] = actor
        for layer in model.polygons:
            actor = _polygon_actor(vtk, layer)
            renderer.AddActor(actor)
            actors[layer.name] = actor

        orientation_actors = _add_patient_orientation_cube(vtk, renderer, model)
        actors.update(orientation_actors)

        title = vtk.vtkTextActor()
        title.SetInput(model.title)
        title.SetPosition(22, window_size[1] - 48)
        title.GetTextProperty().SetFontSize(22)
        title.GetTextProperty().SetColor(0.94, 0.94, 0.94)
        title.GetTextProperty().SetBold(False)
        renderer.AddViewProp(title)
        actors["Title"] = title

        legend_layers = (*model.surfaces, *model.meshes, *model.polygons)
        legend = _legend_actor(vtk, legend_layers)
        renderer.AddViewProp(legend)
        actors["Legend"] = legend

        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        camera.Azimuth(32.0)
        camera.Elevation(24.0)
        renderer.ResetCameraClippingRange()
        return VisualizationScene(
            model=model,
            renderer=renderer,
            render_window=render_window,
            interactor=interactor,
            actors=actors,
            vtk=vtk,
        )

    def _prepare_cutting_planes(
        self,
        cutting_planes: SelectedCuttingPlanes,
        volume_shape: tuple[int, int, int],
        geometry: VolumeGeometry | None = None,
    ) -> tuple[PolygonLayer, ...]:
        geometry = geometry if geometry is not None else self.geometry
        polygons = []
        for index, equation in enumerate(cutting_planes.equations):
            points = plane_box_intersection(equation, volume_shape)
            if len(points) < 3:
                raise ValueError(f"cutting plane {index + 1} does not intersect the volume")
            points = clip_convex_polygon_to_halfspaces(
                points,
                cutting_planes.equations,
            )
            if len(points) < 3:
                continue
            polygons.append(
                PolygonLayer(
                    f"Cutting plane {index + 1}",
                    geometry.index_to_world(points),
                    self._PLANE_COLORS[index % len(self._PLANE_COLORS)],
                    0.28,
                )
            )
        return tuple(polygons)


def resection_boundary(classification: np.ndarray) -> np.ndarray:
    """Keep retained-bone labels only near tumor or resected bone."""

    from scipy.ndimage import binary_dilation

    data = np.asarray(classification)
    if data.ndim != 3:
        raise ValueError("classification must be a three-dimensional array")
    result = data.copy()
    result[(result == 1) | (result == 2)] = 0
    adjacent_to_removed = binary_dilation(data >= 1, structure=np.ones((3, 3, 3), bool))
    result[(result <= -1) & ~adjacent_to_removed] = -6
    return result


def patient_orientation_cube(
    volume_shape: tuple[int, int, int], geometry: VolumeGeometry
) -> PatientOrientationCube:
    """Place a small cube beyond the negative side of the volume's LPS bounds."""

    shape = np.asarray(volume_shape, dtype=np.int64)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError("volume_shape must contain three positive dimensions")
    lower = np.full(3, -0.5, dtype=np.float64)
    upper = shape.astype(np.float64) - 0.5
    index_corners = np.asarray(
        [
            (x, y, z)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    world_corners = geometry.index_to_world(index_corners)
    world_min = world_corners.min(axis=0)
    world_max = world_corners.max(axis=0)
    world_mid = (world_min + world_max) / 2.0
    longest_extent = float(np.max(world_max - world_min))
    size = max(longest_extent * 0.14, min(geometry.spacing) * 1.5)
    gap = max(longest_extent * 0.25, size * 0.8)
    center = world_mid.copy()
    center[:2] = world_min[:2] - gap - size / 2.0
    return PatientOrientationCube(
        center=tuple(float(value) for value in center),
        size=float(size),
        face_labels=(
            ("L", (1.0, 0.0, 0.0)),
            ("R", (-1.0, 0.0, 0.0)),
            ("P", (0.0, 1.0, 0.0)),
            ("A", (0.0, -1.0, 0.0)),
            ("S", (0.0, 0.0, 1.0)),
            ("i", (0.0, 0.0, -1.0)),
        ),
    )


def visualize_bone_and_tumor(
    volume: LabeledVolume,
    *,
    geometry: VolumeGeometry | None = None,
    **render_options: Any,
) -> VisualizationScene:
    return BoneTumorVisualizer(geometry).show_bone_and_tumor(volume, **render_options)


def visualize_labeled_volume(
    volume: np.ndarray,
    *,
    color_map: Mapping[float, RgbaColor],
    label_names: Mapping[float, str] | None = None,
    title: str = "Volume visualization",
    cutting_planes: SelectedCuttingPlanes | None = None,
    geometry: VolumeGeometry | None = None,
    **render_options: Any,
) -> VisualizationScene:
    visualizer = BoneTumorVisualizer(geometry)
    model = visualizer.prepare_labeled_volume(
        volume,
        color_map=color_map,
        label_names=label_names,
        title=title,
        cutting_planes=cutting_planes,
    )
    return visualizer.render(model, **render_options)


def visualize_tumor_convex_hull(
    volume: LabeledVolume,
    *,
    geometry: VolumeGeometry | None = None,
    **render_options: Any,
) -> VisualizationScene:
    return BoneTumorVisualizer(geometry).show_tumor_convex_hull(volume, **render_options)


def visualize_resection(
    plan: ResectionPlanData,
    *,
    show_retained: bool = True,
    boundary_only: bool = False,
    **render_options: Any,
) -> VisualizationScene:
    visualizer = BoneTumorVisualizer(plan.geometry)
    if boundary_only:
        return visualizer.show_resection_boundary(plan, **render_options)
    return visualizer.show_resection(plan, show_retained=show_retained, **render_options)


def _require_labeled_volume(volume: LabeledVolume) -> None:
    if not isinstance(volume, LabeledVolume):
        raise TypeError("volume must be a LabeledVolume")


def _require_bone_and_tumor(volume: LabeledVolume) -> None:
    if not np.any(volume.normal_bone_mask):
        raise ValueError("labeled volume contains no normal bone voxels")
    if not np.any(volume.tumor_mask):
        raise ValueError("labeled volume contains no tumor voxels")


def _validate_rgba(label: float, rgba: RgbaColor) -> tuple[Color, float]:
    values = np.asarray(rgba, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"color for label {label} must contain four finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"color for label {label} must be within [0, 1]")
    return (float(values[0]), float(values[1]), float(values[2])), float(values[3])


def _default_label_name(label: float) -> str:
    return DEFAULT_LABEL_NAMES.get(float(label), f"Label {label:g}")


def _load_vtk() -> tuple[Any, Any]:
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        raise RuntimeError(
            "VTK is required for visualization. Install it with "
            "'python -m pip install bone-cutting-plane-visualization[viz]'."
        ) from exc
    return vtk, numpy_support


def _surface_actor(vtk: Any, numpy_support: Any, layer: SurfaceLayer, geometry: VolumeGeometry):
    padded = np.pad(np.asarray(layer.mask, dtype=np.uint8), 1)
    image = vtk.vtkImageData()
    image.SetDimensions(*padded.shape)
    image.SetSpacing(*geometry.spacing)
    shifted_origin = geometry.index_to_world(np.asarray([[-1.0, -1.0, -1.0]]))[0]
    image.SetOrigin(*shifted_origin)
    direction = vtk.vtkMatrix3x3()
    for row in range(3):
        for column in range(3):
            direction.SetElement(row, column, geometry.direction[row][column])
    image.SetDirectionMatrix(direction)
    scalars = numpy_support.numpy_to_vtk(
        padded.ravel(order="F"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    image.GetPointData().SetScalars(scalars)
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(image)
    contour.SetValue(0, 1)
    contour.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(contour.GetOutputPort())
    normals.SetFeatureAngle(60.0)
    normals.ConsistencyOn()
    normals.SplittingOff()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*layer.color)
    actor.GetProperty().SetOpacity(layer.opacity)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetSpecular(0.15)
    actor.GetProperty().SetSpecularPower(20.0)
    return actor


def _add_patient_orientation_cube(
    vtk: Any,
    renderer: Any,
    model: VisualizationModel,
) -> dict[str, Any]:
    layout = patient_orientation_cube(model.volume_shape, model.geometry)
    cube = vtk.vtkCubeSource()
    cube.SetCenter(*layout.center)
    cube.SetXLength(layout.size)
    cube.SetYLength(layout.size)
    cube.SetZLength(layout.size)
    cube.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(mapper)
    cube_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
    cube_actor.GetProperty().SetOpacity(1.0)
    cube_actor.GetProperty().SetRepresentationToSurface()
    cube_actor.GetProperty().EdgeVisibilityOn()
    cube_actor.GetProperty().SetEdgeColor(1.0, 1.0, 1.0)
    cube_actor.GetProperty().SetLineWidth(2.0)
    cube_actor.GetProperty().LightingOff()
    cube_actor.PickableOff()
    renderer.AddActor(cube_actor)
    actors: dict[str, Any] = {"Patient orientation cube": cube_actor}

    orientations = {
        (1.0, 0.0, 0.0): (0.0, 90.0, 0.0),
        (-1.0, 0.0, 0.0): (0.0, -90.0, 0.0),
        (0.0, 1.0, 0.0): (-90.0, 0.0, 0.0),
        (0.0, -1.0, 0.0): (90.0, 0.0, 0.0),
        (0.0, 0.0, 1.0): (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0): (0.0, 180.0, 0.0),
    }
    center = np.asarray(layout.center, dtype=np.float64)
    for label, normal_tuple in layout.face_labels:
        source = vtk.vtkVectorText()
        source.SetText(label)
        source.Update()
        bounds = source.GetOutput().GetBounds()
        width = float(bounds[1] - bounds[0])
        height = float(bounds[3] - bounds[2])
        scale = layout.size * 0.46 / max(width, height)
        normal = np.asarray(normal_tuple, dtype=np.float64)

        label_mapper = vtk.vtkPolyDataMapper()
        label_mapper.SetInputConnection(source.GetOutputPort())
        label_actor = vtk.vtkActor()
        label_actor.SetMapper(label_mapper)
        label_actor.SetOrigin(
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            0.0,
        )
        label_actor.SetScale(scale)
        label_actor.SetOrientation(*orientations[normal_tuple])
        label_actor.SetPosition(*(center + normal * layout.size * 0.505))
        label_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        label_actor.GetProperty().LightingOff()
        label_actor.PickableOff()
        renderer.AddActor(label_actor)
        actors[f"Orientation {label}"] = label_actor
    return actors


def _mesh_actor(vtk: Any, layer: MeshLayer):
    poly_data = _poly_data(vtk, layer.points, layer.faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*layer.color)
    actor.GetProperty().SetOpacity(layer.opacity)
    if layer.show_edges:
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(*layer.color)
        actor.GetProperty().SetLineWidth(1.6)
    return actor


def _polygon_actor(vtk: Any, layer: PolygonLayer):
    face = np.arange(len(layer.points), dtype=np.int64)[None, :]
    poly_data = _poly_data(vtk, layer.points, face)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*layer.color)
    actor.GetProperty().SetOpacity(layer.opacity)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(*layer.color)
    actor.GetProperty().SetLineWidth(2.2)
    actor.GetProperty().LightingOff()
    return actor


def _poly_data(vtk: Any, points_array: np.ndarray, faces: np.ndarray):
    points = vtk.vtkPoints()
    for point in np.asarray(points_array, dtype=np.float64):
        points.InsertNextPoint(*point)
    cells = vtk.vtkCellArray()
    for face in np.asarray(faces, dtype=np.int64):
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(face))
        for index, point_id in enumerate(face):
            polygon.GetPointIds().SetId(index, int(point_id))
        cells.InsertNextCell(polygon)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(cells)
    return poly_data


def _legend_actor(vtk: Any, layers: tuple[Any, ...]):
    legend = vtk.vtkLegendBoxActor()
    legend.SetNumberOfEntries(len(layers))
    legend.SetPosition(0.015, 0.02)
    legend.SetPosition2(0.29, min(0.055 * len(layers) + 0.035, 0.42))
    legend.UseBackgroundOn()
    legend.SetBackgroundColor(0.12, 0.12, 0.13)
    legend.SetBackgroundOpacity(0.82)
    legend.BorderOff()
    legend.GetEntryTextProperty().SetColor(0.94, 0.94, 0.94)
    legend.GetEntryTextProperty().SetFontSize(15)
    glyph = vtk.vtkCubeSource()
    glyph.Update()
    for index, layer in enumerate(layers):
        legend.SetEntry(index, glyph.GetOutput(), layer.name, layer.color)
    return legend
