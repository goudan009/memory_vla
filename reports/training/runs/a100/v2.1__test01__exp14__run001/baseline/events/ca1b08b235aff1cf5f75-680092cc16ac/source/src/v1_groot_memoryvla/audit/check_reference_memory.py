"""Execute unmodified reference bank classes without Prismatic dependencies.

Uses reduced feature width (32) to check memory semantics and gradients, not
GR00T action loss or trained weights. Also checks index-preserving scatter on
the actual token masks emitted by trace_b1k_inputs.py.
"""

import ast
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
REFERENCE = ROOT / "src/references_code/memoryvla/vla/memory_vla.py"


def main():
    torch.manual_seed(7)
    names = {"TimestepEmbedder", "CrossTransformerBlock", "BottleneckSE",
             "GateFusion", "CogMemBank", "PerMemBank"}
    parsed = ast.parse(REFERENCE.read_text())
    selected = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name in names], type_ignores=[])
    namespace = dict(torch=torch, nn=nn, F=F, math=math, np=np, Optional=Optional)
    exec(compile(selected, str(REFERENCE), "exec"), namespace)
    results = {"source": str(REFERENCE), "feature_width": 32, "checks": {}}
    # Execute original SE on three separate camera grids at the actual
    # Eagle widths; concatenating three grids violates its square assertion.
    se = namespace["BottleneckSE"](2048, 512, 2048)
    with torch.no_grad():
        camera_output = se(torch.randn(3, 81, 2048))
    rejected = []
    for token_count, width in ((243, 2048), (972, 1152)):
        try:
            se(torch.randn(1, token_count, width))
        except AssertionError:
            rejected.append(token_count)
    assert rejected == [243, 972]
    results["bottleneck_se"] = {"input_per_camera": [3, 81, 2048],
                                "output_per_camera": list(camera_output.shape),
                                "rejected_concatenated_token_counts": rejected,
                                "weights": "random; dimensional compatibility only"}
    for kind, count in (("CogMemBank", 1), ("PerMemBank", 4)):
        bank = namespace[kind](dataloader_type="group", group_size=16, token_size=32,
                               mem_length=16, retrieval_layers=2, fusion_type="gate",
                               consolidate_type="tome", update_fused=True)
        bank.train()
        tokens = torch.randn(16, count, 32, requires_grad=True)
        lengths = []
        hook = bank.retrieval_blocks[0].register_forward_pre_hook(
            lambda module, inputs: lengths.append(inputs[1].shape[1]))
        output = bank.process_batch(tokens, np.zeros(16, dtype=int), np.arange(16))
        hook.remove()
        assert lengths == [i * count for i in range(1, 16)]
        assert len(bank.bank[0]) == 16
        assert torch.allclose(bank.bank[0][-1][1], output[-1].detach())
        assert all(not v.requires_grad and v.grad_fn is None for _, v in bank.bank[0])
        # A differentiable surrogate head isolates the bank gradient contract.
        head = nn.Linear(32, 3)
        head(output[-1]).square().sum().backward()
        gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                     for name, parameter in bank.named_parameters()}
        assert all(value is not None for value in gradients.values())
        for name in ("retrieval_blocks.0.q_proj.weight", "retrieval_blocks.1.k_proj.weight",
                     "retrieval_blocks.1.v_proj.weight", "gate_fusion_blocks.proj.weight",
                     "timestep_encoder.mlp.0.weight"):
            assert gradients[name] > 1e-9
        assert tokens.grad[-1].norm() > 0
        assert tokens.grad[:-1].abs().sum() == 0  # detached history: no temporal BPTT
        bank.eval()
        lengths = []
        hook = bank.retrieval_blocks[0].register_forward_pre_hook(
            lambda module, inputs: lengths.append(inputs[1].shape[1]))
        bank.process_batch(torch.randn(1, count, 32), [0], [16])
        hook.remove()
        assert lengths == [16 * count] and len(bank.bank[0]) == 16
        assert len(bank.bank[0]) == 16
        bank.process_batch(torch.randn(1, count, 32), [1], [0])
        assert len(bank.bank[1]) == 1 and len(bank.bank[0]) == 16
        bank.clear_episode(1)
        assert 1 not in bank.bank and 0 in bank.bank
        bank.train()
        bank.process_batch(torch.randn(32, count, 32), [4] * 32, list(range(32)))
        assert len(bank.bank[4]) == 16  # same episode, two groups still reset
        # Verify overflow merger exactly, including choice and retained timestamp.
        bank.reset()
        vectors = torch.randn(17, count, 32)
        vectors[6] = vectors[5].clone()  # unique maximum similarity pair
        for i in range(17):
            bank._memory_consolidate(8, vectors[i], i)
        assert len(bank.bank[8]) == 16
        assert [t for t, _ in bank.bank[8]] == list(range(6)) + list(range(7, 17))
        assert torch.equal(bank.bank[8][5][1], vectors[5])
        results["checks"][kind] = {"output_shape": list(output.shape),
            "history_lengths_first_group": [i * count for i in range(1, 16)],
            "query_17_reads": 16, "post_write_17_capacity": 16,
            "merge_test_pair": [5, 6], "retained_timestamp": 5,
            "stored_fused_detached": True, "current_input_gradient": float(tokens.grad[-1].norm()),
            "past_input_gradient": float(tokens.grad[:-1].abs().sum()),
            "parameter_gradient_norms": gradients, "group_reset_and_episode_isolation": True}

    input_report = json.loads((ARTIFACTS / "input_audit.json").read_text())
    records = input_report["processor"]["tokens"]["padded"]
    scatter_checks = []
    for b in (0, 1):
        row = [r for r in records if r["batch"] == b]
        image_mask = torch.tensor([bool(r["image_mask"]) for r in row])
        attention_mask = torch.tensor([bool(r["attention_mask"]) for r in row])
        eos = [r["position"] for r in row if r["attention_mask"] and r["token"] == "<|im_end|>"][-1]
        original = torch.randn(len(row), 32)
        updated = original.clone()
        per = torch.randn(int(image_mask.sum()), 32, requires_grad=True)
        cog = torch.randn(32, requires_grad=True)
        updated[image_mask] = per
        updated[eos] = cog
        untouched = ~image_mask
        untouched[eos] = False
        assert torch.equal(updated[untouched], original[untouched])
        assert updated.shape == original.shape
        updated[image_mask | (torch.arange(len(row)) == eos)].sum().backward()
        assert per.grad is not None and cog.grad is not None
        scatter_checks.append({"batch": b, "sequence_length": len(row),
                               "image_tokens": int(image_mask.sum()), "cognitive_position": eos,
                               "image_attention_count": int((image_mask & attention_mask).sum()),
                               "text_attention_count": int((~image_mask & attention_mask).sum())})
    results["scatter"] = {"synthetic_features_on_real_masks": True, "rows": scatter_checks,
                           "shape_mask_and_untouched_positions_preserved": True}
    (ARTIFACTS / "reference_memory_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
