# bone-cutting-plane-visualization

Standalone data contracts and VTK visualization for bone-tumor cutting-plane
plans. The package is independent of any planning or optimization repository.

All persisted inputs use the `.ubd.npz` format from
[`ct-mri-dicom-nii-reader`](https://pypi.org/project/ct-mri-dicom-nii-reader/).
Volumes are interpreted in `(Left, Posterior, Superior)` axis order. Standard
UBD files can be used for the bone/tumor and convex-hull views; a compatible,
namespaced extension stores a complete resection plan in one UBD archive.

## Installation

```powershell
python -m pip install -e ".[all]"
```

The mandatory runtime dependency on `ct-mri-dicom-nii-reader` provides the UBD
loader and `BodyData` contract. VTK remains optional so data inspection and
scene preparation work on headless systems:

```powershell
python -m pip install -e .
python -m pip install -e ".[viz]"
```

## Data contracts

The public plan consists of five explicit classes:

- `LabeledVolume`: `-1` normal bone, `0` other, `1` tumor.
- `SelectedCuttingPlanes`: ordered index-space rows `[a, b, c, d]`.
- `RegionClassificationVolume`: negative retained bone and `2` resected bone.
- `VolumeGeometry`: spacing, LPS origin, and index-to-LPS direction.
- `TumorSafetyMarginData`: optional planning volume, protected mask, danger-bone
  mask, and physical margin.

`ResectionPlanData` combines them and validates shapes, labels, plane references,
and safety-margin clearance.

```python
import numpy as np

from bone_cutting_plane_visualization import (
    ResectionPlanData,
    SelectedCuttingPlanes,
    classify_regions,
    load_labeled_volume,
    save_resection_plan,
)

volume, geometry = load_labeled_volume("case-labels.ubd.npz")
planes = SelectedCuttingPlanes(
    np.asarray([[1.0, 0.0, 0.0, -42.5]], dtype=np.float64)
)
classification = classify_regions(volume, planes)
plan = ResectionPlanData(volume, planes, classification, geometry)
save_resection_plan(plan, "case-plan.ubd.npz")
```

Arrays are copied, normalized to stable dtypes, and exposed read-only.

Callers migrating from NumPy-based APIs can use `compatibility.plan_from_arrays`,
`compatibility.visualize_resection_arrays`, `compatibility.visualize_volume`,
and `compatibility.visualize_solution`. These adapters depend only on this package;
they never import a planning repository.

## UBD plan extension

Every plan archive retains the standard UBD v1 fields:

```text
format_version, body_data, image_type, mmpd
```

It adds the following `bcv_` fields:

```text
bcv_schema_version
bcv_cutting_planes
bcv_region_classification
bcv_spacing_mm
bcv_origin_lps_mm
bcv_direction_index_to_lps
bcv_has_safety_margin
bcv_safety_planning_volume       # present when enabled
bcv_safety_protected_mask        # present when enabled
bcv_safety_danger_bone_mask      # present when enabled
bcv_safety_margin_mm             # present when enabled
```

Unknown fields are ignored by `UnifiedBodyDataLoader`, so a plan archive remains
readable as an ordinary mask UBD file. The package verifies that compatibility
after every save. Because UBD stores one scalar `mmpd`, persisted plans require
isotropic spacing. Direct in-memory visualization still supports arbitrary valid
`VolumeGeometry` values.

## Visualization

```python
from bone_cutting_plane_visualization import load_resection_plan, visualize_resection

plan = load_resection_plan("case-plan.ubd.npz")
visualize_resection(plan)
```

The final view includes retained normal bone, resected normal bone, tumor,
optional safety-margin bone, bounded cutting-plane polygons, legend, retention
rate, and the same camera and visual styling as the source implementation. Every
VTK scene also includes a compact black orientation cube with white edges and
`L/R`, `P/A`, and `S/i` face labels. It is scaled from the physical volume bounds
and placed beyond the negative X/Y side of the LPS bounds, outside the clinical
geometry and business legend.

The CLI accepts only `.ubd.npz` inputs:

```powershell
bone-cutting-plane-viz view case-labels.ubd.npz
bone-cutting-plane-viz hull case-labels.ubd.npz
bone-cutting-plane-viz plan case-plan.ubd.npz
bone-cutting-plane-viz plan case-plan.ubd.npz --view resected
bone-cutting-plane-viz plan case-plan.ubd.npz --offscreen --screenshot plan.png
bone-cutting-plane-viz inspect case-plan.ubd.npz
```

## Development

```powershell
python -m pytest
ruff check src tests
```
