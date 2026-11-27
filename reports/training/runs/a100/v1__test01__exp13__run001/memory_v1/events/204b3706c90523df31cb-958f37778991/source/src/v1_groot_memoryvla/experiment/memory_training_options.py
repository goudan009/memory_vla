"""Options shared by the public launcher, worker and checkpoint validator."""

import argparse


TRAINING_PROTOCOL = "memoryvla_direct_tail_qknorm_v3"
OPENVLA_PROTOCOL = "memoryvla_openvla_b1k_v1"
EXP13_PROTOCOL = "memoryvla_openvla_b1k_group16_raw_v2"


def training_protocol(args):
    if getattr(args, "memory_conditioning", "residual") != "openvla":
        if getattr(args, "history_sampling", "contiguous") != "contiguous":
            raise ValueError("Random ordered sampling requires openvla conditioning")
        return TRAINING_PROTOCOL
    return EXP13_PROTOCOL if getattr(args, "history_sampling", "contiguous") == "random_ordered" else OPENVLA_PROTOCOL


def add_memory_training_options(parser):
    parser.add_argument("--history-sampling", choices=("contiguous", "random_ordered"), default="contiguous",
                        help="Exp13: sample 16 distinct same-Episode stride8 decision frames, then sort")
    parser.add_argument("--optimizer-eps", type=float, default=1e-8,
                        help="AdamW denominator epsilon; saved and checked on resume")
    parser.add_argument("--memory-cognitive-residual-projection", action=argparse.BooleanOptionalAction, default=False,
                        help="Zero-initialized learned cognitive residual projection instead of a scalar")
    parser.add_argument("--memory-cognitive-residual-scope", choices=("eos", "text"), default="eos",
                        help="Apply cognitive residual to EOS or all valid text tokens")
    parser.add_argument("--memory-cognitive-qk-norm", choices=("none", "layernorm"), default="layernorm",
                        help="Normalize cognitive query/key inputs to prevent GR00T EOS retrieval saturation")
    parser.add_argument("--memory-conditioning", choices=("direct", "residual", "openvla"), default="residual",
                        help="Residual preserves GR00T features; direct replaces its perception/text conditions")
    parser.add_argument("--memory-residual-readout", choices=("delta", "value"), default="delta",
                        help="Residual source: fused-current delta or historical attention values only")
    parser.add_argument("--memory-length", type=int, default=64,
                        help="Entries per stream (e.g. 16, 64 or 256); independent of 32-decision windows")
    parser.add_argument("--image-augmentation", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the checkpoint processor's training transforms; validation always uses eval")
    parser.set_defaults(training_protocol=TRAINING_PROTOCOL)


def memory_training_argv(args):
    return ["--history-sampling", getattr(args, "history_sampling", "contiguous"),
            "--optimizer-eps", str(getattr(args, "optimizer_eps", 1e-8)),
            "--memory-cognitive-residual-projection" if getattr(args, "memory_cognitive_residual_projection", False)
            else "--no-memory-cognitive-residual-projection",
            "--memory-cognitive-residual-scope", getattr(args, "memory_cognitive_residual_scope", "eos"),
            "--memory-conditioning", getattr(args, "memory_conditioning", "residual"),
            "--memory-residual-readout", getattr(args, "memory_residual_readout", "delta"),
            "--memory-cognitive-qk-norm", getattr(args, "memory_cognitive_qk_norm", "layernorm"),
            "--memory-length", str(getattr(args, "memory_length", 64)),
            "--image-augmentation" if getattr(args, "image_augmentation", True) else "--no-image-augmentation"]
