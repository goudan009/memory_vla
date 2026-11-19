"""Training demonstrator text and rollout task names share one protocol."""

import json

from v3_bodymemory.model.gr00t_n1d6.language import (
    build_b1k_instruction_mapping, canonical_language,
)


def test_b1k_aliases_share_canonical_text(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"task_index": 21, "task_name": "collecting_childrens_toys",
                               "task": "Place the toys inside the bookcase."}))
    mapping = build_b1k_instruction_mapping([path])
    texts = ("Place the toys inside the bookcase.", "collecting_childrens_toys",
             "Collecting childrens toys.")
    assert {canonical_language(text, mapping) for text in texts} == {"collecting childrens toys"}
    # Unrelated instructions retain the existing processor formalization.
    assert canonical_language("Open the Door!", mapping) == "open the door"
