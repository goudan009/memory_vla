"""Check relocated Gym entry points without importing simulator dependencies."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_gym_entry_points_resolve_in_shared_evaluation():
    count = 0
    for source in (ROOT / "evaluation" / "sim").rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.keyword) or node.arg != "entry_point":
                continue
            assert isinstance(node.value, ast.Constant)
            module, symbol = node.value.value.split(":")
            assert module.startswith("evaluation.sim.")
            path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
            tree = ast.parse(path.read_text())
            assert any(isinstance(item, ast.ClassDef) and item.name == symbol for item in tree.body)
            count += 1
    assert count > 0
    assert (ROOT / "evaluation/sim/BEHAVIOR/available_tasks.yaml").is_file()
