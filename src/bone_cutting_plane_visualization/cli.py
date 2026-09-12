"""Command-line entry points for UBD-backed visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ubd import IncompleteResectionPlanError, load_labeled_volume, load_resection_plan
from .visualization import (
    BoneTumorVisualizer,
    visualize_bone_and_tumor,
    visualize_resection,
    visualize_tumor_convex_hull,
)


def _render_options(args: argparse.Namespace) -> dict:
    return {
        "interactive": not args.offscreen,
        "offscreen": args.offscreen,
        "screenshot": args.screenshot,
        "window_size": tuple(args.window_size),
    }


def _view(args: argparse.Namespace) -> int:
    volume, geometry = load_labeled_volume(args.ubd)
    visualize_bone_and_tumor(volume, geometry=geometry, **_render_options(args))
    return 0


def _hull(args: argparse.Namespace) -> int:
    volume, geometry = load_labeled_volume(args.ubd)
    visualize_tumor_convex_hull(volume, geometry=geometry, **_render_options(args))
    return 0


def _plan(args: argparse.Namespace) -> int:
    plan = load_resection_plan(args.ubd)
    visualize_resection(
        plan,
        show_retained=args.view != "resected",
        boundary_only=args.view == "boundary",
        **_render_options(args),
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    try:
        plan = load_resection_plan(args.ubd)
    except IncompleteResectionPlanError:
        volume, geometry = load_labeled_volume(args.ubd)
        print(f"kind=labeled-volume shape={volume.shape} mmpd={geometry.spacing[0]:g}")
        return 0
    print(
        f"kind=resection-plan shape={plan.labeled_volume.shape} "
        f"planes={plan.cutting_planes.count} keep_rate={plan.keep_rate:.6f} "
        f"mmpd={plan.geometry.spacing[0]:g}"
    )
    return 0


def _add_render_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--screenshot", type=Path, help="also save the rendered view as PNG")
    parser.add_argument(
        "--offscreen", action="store_true", help="render without opening an interactive window"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=(1100, 800),
        metavar=("WIDTH", "HEIGHT"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bone-cutting-plane-viz")
    subparsers = parser.add_subparsers(dest="command", required=True)
    view = subparsers.add_parser("view", help="visualize bone and tumor from .ubd.npz")
    view.add_argument("ubd", type=Path)
    _add_render_arguments(view)
    view.set_defaults(handler=_view)
    hull = subparsers.add_parser("hull", help="visualize bone, tumor, and tumor hull")
    hull.add_argument("ubd", type=Path)
    _add_render_arguments(hull)
    hull.set_defaults(handler=_hull)
    plan = subparsers.add_parser("plan", help="visualize a complete resection-plan .ubd.npz")
    plan.add_argument("ubd", type=Path)
    plan.add_argument("--view", choices=("kept", "resected", "boundary"), default="kept")
    _add_render_arguments(plan)
    plan.set_defaults(handler=_plan)
    inspect = subparsers.add_parser("inspect", help="inspect a visualization .ubd.npz")
    inspect.add_argument("ubd", type=Path)
    inspect.set_defaults(handler=_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "window_size") and any(value <= 0 for value in args.window_size):
        raise ValueError("window dimensions must be positive")
    return int(args.handler(args))


__all__ = ["BoneTumorVisualizer", "build_parser", "main"]
