"""Options shared by the public launcher, worker and checkpoint validator."""

import argparse


TRAINING_PROTOCOL = "memoryvla_direct_tail_v2"


def add_memory_training_options(parser):
    parser.add_argument("--memory-conditioning", choices=("direct", "residual"), default="direct",
                        help="Direct fused perception/EOS conditions; residual is the earlier ablation")
    parser.add_argument("--memory-length", type=int, default=64,
                        help="Entries per stream (e.g. 16, 64 or 256); independent of 32-decision windows")
    parser.add_argument("--image-augmentation", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the checkpoint processor's training transforms; validation always uses eval")
    parser.set_defaults(training_protocol=TRAINING_PROTOCOL)


def memory_training_argv(args):
    return ["--memory-conditioning", getattr(args, "memory_conditioning", "direct"),
            "--memory-length", str(getattr(args, "memory_length", 64)),
            "--image-augmentation" if getattr(args, "image_augmentation", True) else "--no-image-augmentation"]
