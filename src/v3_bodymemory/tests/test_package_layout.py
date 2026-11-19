"""Check imports and runtime resources after renaming the upstream package."""

import ast
from importlib import resources
from pathlib import Path

import v3_bodymemory


PACKAGE = Path(v3_bodymemory.__file__).parent


def test_internal_imports_resolve_within_v3():
    for source in PACKAGE.rglob("*.py"):
        if ".venv" in source.relative_to(PACKAGE).parts:
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name != "gr00t" and not name.startswith("gr00t."), source
                if name.startswith("v3_bodymemory."):
                    path = PACKAGE.joinpath(*name.split(".")[1:])
                    assert path.is_dir() or path.with_suffix(".py").is_file(), (source, name)


def test_runtime_resources_are_available_as_package_files():
    package_files = resources.files(v3_bodymemory)
    for relative in (
        "configs/deepspeed/zero2_config.json",
        "configs/deepspeed/zero3_config.json",
        "model/modules/nvidia/Eagle-Block2A-2B-v2/config.json",
        "model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py",
        "model/modules/nvidia/Eagle-Block2A-2B-v2/tokenizer_config.json",
        "model/modules/nvidia/Eagle-Block2A-2B-v2/vocab.json",
        "model/modules/nvidia/Eagle-Block2A-2B-v2/merges.txt",
    ):
        assert package_files.joinpath(relative).is_file(), relative
