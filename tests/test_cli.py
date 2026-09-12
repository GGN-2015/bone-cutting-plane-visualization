from __future__ import annotations

from bone_cutting_plane_visualization import save_resection_plan
from bone_cutting_plane_visualization.cli import build_parser, main


def test_parser_exposes_only_ubd_visualization_workflows():
    parser = build_parser()
    assert parser.parse_args(["view", "case.ubd.npz"]).command == "view"
    assert parser.parse_args(["hull", "case.ubd.npz"]).command == "hull"
    assert parser.parse_args(["plan", "case.ubd.npz"]).command == "plan"
    assert parser.parse_args(["inspect", "case.ubd.npz"]).command == "inspect"


def test_inspect_reports_complete_plan(plan, tmp_path, capsys):
    path = save_resection_plan(plan, tmp_path / "plan.ubd.npz")

    assert main(["inspect", str(path)]) == 0
    output = capsys.readouterr().out
    assert "kind=resection-plan" in output
    assert "planes=2" in output
    assert "mmpd=2" in output


def test_source_tree_does_not_reference_old_project():
    from pathlib import Path

    source_root = Path(__file__).parents[1] / "src"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))
    assert "pbtr_py_interface" not in source
    assert "prts-pytorch" not in source
