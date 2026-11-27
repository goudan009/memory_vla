"""Use the existing B1K task-name language contract in training and serving."""

import json
from pathlib import Path
import re


def build_b1k_instruction_mapping(task_files):
    mapping = {}
    for path in task_files:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            name = row["task_name"]
            canonical = name.replace("_", " ").capitalize() + "."
            for alias in (row["task"], name, canonical, canonical[:-1]):
                if alias in mapping and mapping[alias] != canonical:
                    raise ValueError(f"Ambiguous B1K task instruction: {alias}")
                mapping[alias] = canonical
    return mapping


def canonical_language(text, mapping, formalize=True):
    language = mapping.get(text.strip(), text)
    return re.sub(r"[^\w\s]", "", language.lower()) if formalize else language
