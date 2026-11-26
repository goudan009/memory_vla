"""Bounded GPU plumbing check with real observations/actions and TEST state labels.

This is not Task40 training or physical-state validation. The frozen Eagle
encodes actual images once; its outputs are cached for the three short steps.
Run through the workbench's v2_training_smoke job, not as an unmanaged GPU job.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.experiment.task21_diagnostic import load_model, head_hash
from v1_groot_memoryvla.experiment.first_training import cache_sequence
from v1_groot_memoryvla.data.dataset.history_dataset import HistoryDataset
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from .model.gr00t_n1d6 import Gr00tN1d6
from .model.state import PHYSICAL_LABELS
from .training import parameter_manifest


FEATURE_KEYS = ("backbone_features", "visual_features", "image_mask", "backbone_attention_mask")


class CachedBackbone(nn.Module):
    """Use the recorded frozen-backbone features without another image encoding."""

    def __init__(self, device='cuda'):
        super().__init__()
        # HF prepare_input derives dtype/device from the first parameter. Keep
        # the original frozen Eagle's BF16 contract when reusing its features.
        self.anchor = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)

    def prepare_input(self, inputs):
        assert "state_targets" not in inputs and "state_known" not in inputs
        return BatchFeature({key: inputs[key] for key in (*FEATURE_KEYS, "input_ids")})

    def forward(self, inputs):
        return BatchFeature({key: inputs[key] for key in FEATURE_KEYS})


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_groups(dataset):
    """Select the first window of two distinct episodes, never two adjacent windows."""
    indices = [i for i, (_, _, start) in enumerate(dataset.groups) if start == 0][:2]
    if len(indices) != 2:
        raise ValueError("Smoke requires two distinct official episodes")
    return indices


def encode_real_groups(model, checkpoint, path):
    processor = Gr00tN1d6Processor.from_pretrained(checkpoint,
        transformers_loading_kwargs={'trust_remote_code':True, 'local_files_only':True})
    processor.eval()
    dataset = HistoryDataset([path], processor, group_size=3, video_backend='pyav')
    model.backbone.cuda()
    groups = []
    for index in select_groups(dataset):
        seq = cache_sequence(model, processor, dataset, index, microbatch=1)
        assert len(seq['timesteps']) == 3
        rows = [dict(episode=seq['episode_key'], timestep=seq['timesteps'][i],
            frame_index=seq['frame_indices'][i],
            features={k:v[i:i+1] for k,v in seq['features'].items()},
            input_ids=seq['input_ids'][i:i+1], actions={k:v[i:i+1] for k,v in seq['actions'].items()})
            for i in range(3)]
        groups.append(rows)
        print(json.dumps({'encoded_episode':seq['episode_key'],'frames':seq['frame_indices']}),flush=True)
    model.backbone.cpu()
    torch.cuda.empty_cache()
    return groups


def smoke_inputs(row, *, sequence_start=False, sequence_end=False):
    # Deliberately artificial values, unrelated to episode/time or physical truth.
    return dict(row["features"], input_ids=row["input_ids"], **row["actions"],
        state_targets=torch.tensor([[0., 1., float("nan")]], device=row['input_ids'].device),
        state_known=torch.tensor([[True, True, False]], device=row['input_ids'].device),
        memory_context=dict(mode="train_sequence", sequence_start=sequence_start, sequence_end=sequence_end,
                            episode_keys=[row["episode"]], timesteps=[row["timestep"]],
                            memory_diagnostics=True))


def selected_parameters(model):
    bank = model.memory.cognitive
    return {
        "state_head": bank.state_head.proj.weight,
        **{f"state_query_{i}": block.state_query.weight for i, block in enumerate(bank.retrieval_blocks)},
        "state_gate": bank.gate_fusion_blocks.state_gate.weight,
        "query": bank.retrieval_blocks[0].q_proj.weight,
        "fusion": bank.gate_fusion_blocks.proj.weight,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Engineering smoke is limited to 1–100 optimizer steps")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.manual_seed(42)
    start = time.monotonic()
    old = load_model(args.checkpoint, True, 42)
    groups = encode_real_groups(old, args.checkpoint, args.dataset)
    encoding_peak = torch.cuda.max_memory_allocated()/2**30
    encoding_reserved = torch.cuda.max_memory_reserved()/2**30
    torch.cuda.reset_peak_memory_stats()
    model = Gr00tN1d6.from_v1_model(old, state_labels=PHYSICAL_LABELS, state_loss_weight=.3)
    model.backbone = CachedBackbone()
    del old
    gc.collect()
    model.train()
    expert_before = head_hash(model)
    frozen_before = {n: p.detach().clone() for n, p in model.memory.named_parameters() if not p.requires_grad}
    tracked = selected_parameters(model)
    initial = {n: p.detach().clone() for n, p in tracked.items()}
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(n.startswith("memory.cognitive.") for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01, foreach=False)
    rows = []
    for step in range(args.steps):
        tick = time.monotonic()
        model.memory.reset()
        optimizer.zero_grad(set_to_none=True)
        samples = groups[step % len(groups)]
        # Build two genuinely past slots using this checkpoint's own memory.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i, row in enumerate(samples[:2]):
                model(smoke_inputs(row, sequence_start=i == 0))
        assert len(model.memory.cognitive.bank[samples[-1]["episode"]]) == 2
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(smoke_inputs(samples[-1], sequence_end=True))
        action_grads = torch.autograd.grad(outputs["action_loss"],
            [p for n, p in tracked.items() if n != "state_head"], retain_graph=True)
        action_norms = {n: float(g.float().norm()) for n, g in
                        zip([n for n in tracked if n != "state_head"], action_grads)}
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in action_grads), action_norms
        state_grad, = torch.autograd.grad(outputs['state_loss'], tracked['state_head'], retain_graph=True)
        assert torch.isfinite(state_grad).all() and state_grad.abs().sum() > 0
        outputs["loss"].backward()
        assert tracked["state_head"].grad is not None and tracked["state_head"].grad.abs().sum() > 0
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in params)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        torch.cuda.synchronize()
        record = dict(step=step+1, loss=float(outputs["loss"].detach()),
            action_loss=float(outputs["action_loss"].detach()), state_loss=float(outputs["state_loss"].detach()),
            gradient_norm=float(norm), action_gradients=action_norms,
            state_head_gradient=float(tracked["state_head"].grad.float().norm()),
            state_loss_head_gradient=float(state_grad.float().norm()),
            seconds=time.monotonic()-tick)
        rows.append(record)
        print(json.dumps(record), flush=True)
    updates = {n: float((p.detach()-initial[n]).float().abs().max()) for n, p in tracked.items()}
    assert all(value > 0 for value in updates.values()), updates
    assert expert_before == head_hash(model), "Frozen action expert changed"
    assert all(torch.equal(p, frozen_before[n]) for n, p in model.memory.named_parameters() if not p.requires_grad)

    # Verify saved neural memory weights + runtime restore under a fixed action RNG.
    model.eval()
    model.memory.reset()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, row in enumerate(groups[0][:2]):
            model(smoke_inputs(row, sequence_start=i == 0))
    runtime = model.memory.runtime_state_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.parent / f"v2__test02__exp01__run001__step{args.steps:06d}"
    checkpoint.mkdir()
    torch.save(model.memory.state_dict(), checkpoint/"memory.pt")
    model.config.save_pretrained(checkpoint)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(9876)
        before = model(smoke_inputs(groups[0][-1], sequence_end=True))
        model.memory.load_state_dict(torch.load(checkpoint/"memory.pt", map_location="cuda", weights_only=True))
        model.memory.load_runtime_state_dict(runtime)
        torch.manual_seed(9876)
        after = model(smoke_inputs(groups[0][-1], sequence_end=True))
    differences = {key: float((before[key]-after[key]).float().abs().max())
                   for key in ("loss", "action_loss_unreduced", "state_logits", "backbone_features")}
    assert all(value == 0 for value in differences.values()), differences
    report = dict(passed=True, category="engineering_smoke", steps=args.steps,
        scope="Real Task40 images encoded by this checkpoint, real actions, ARTIFICIAL state test labels; no scientific training",
        checkpoint=str(args.checkpoint.resolve()), dataset=str(args.dataset.resolve()),
        checkpoint_sha256={p.name:file_hash(p) for p in sorted(args.checkpoint.glob('*.safetensors'))},
        feature_source="Fresh frozen backbone forward from this exact checkpoint and its processor",
        dataset_meta_sha256=file_hash(args.dataset/'meta/info.json'), labels="TEST ONLY [0,1,unknown]; NOT physical truth",
        state_loss_weight=.3, state_loss_weight_scope="arbitrary plumbing fixture, not calibrated for training",
        seed=42, optimizer="AdamW", learning_rate=1e-5, weight_decay=.01,
        world_size=1, microbatch=1, grad_accum_steps=1, effective_batch=1,
        seen_loss_samples=args.steps, prefix_decisions_per_step=2,
        cached_episode_times=[[(r["episode"],r["timestep"]) for r in group] for group in groups],
        encoded_frame_indices=[[r['frame_index'] for r in group] for group in groups],
        frozen_action_head_unchanged=True, frozen_perception_and_scale_unchanged=True,
        parameter_manifest=parameter_manifest(model), updates=updates, training=rows,
        reload_max_abs=differences, temporary_memory_checkpoint=str(checkpoint),
        reload_scope="memory neural weights and runtime, frozen expert retained; not a full HF checkpoint",
        elapsed_seconds=time.monotonic()-start,
        encoding_peak_cuda_gib=encoding_peak,
        training_peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_cuda_gib=max(encoding_peak,torch.cuda.max_memory_allocated()/2**30),
        peak_reserved_gib=max(encoding_reserved,torch.cuda.max_memory_reserved()/2**30))
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("parameter_manifest","training")}), flush=True)


if __name__ == "__main__":
    main()
