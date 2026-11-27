"""Serializable fine-tuning contract; standard library only (also used by web)."""
import argparse
import math

COMPONENTS = ('llm', 'vision', 'vision_projector', 'action_dit', 'action_projector', 'action_vlln', 'memory')
MODES = ('action_full', 'full', 'custom', 'lora')
LORA_COMPONENTS = ('llm', 'vision', 'vision_projector', 'action_dit')


def resolve_finetuning(values):
    values = vars(values) if not isinstance(values, dict) else values
    mode = values.get('fine_tuning_mode', 'action_full')
    if mode not in MODES:
        raise ValueError('微调方式须为 action_full / full / custom / lora')
    targets = values.get('lora_targets', 'llm,action_dit')
    targets = targets.split(',') if isinstance(targets, str) else targets
    if not isinstance(targets, (list, tuple)) or not targets or any(t not in LORA_COMPONENTS for t in targets) or len(set(targets)) != len(targets):
        raise ValueError('LoRA目标须从llm、vision、vision_projector、action_dit中选择且不重复')
    freeze = {}
    for name in COMPONENTS:
        default = False if mode == 'full' else name in ('llm', 'vision', 'vision_projector')
        if mode == 'lora' and name in targets:
            default = False
        value = values.get('freeze_' + name)
        if value is not None and type(value) is not bool:
            raise ValueError('freeze_' + name + '须为开关')
        freeze[name] = default if value is None else value
    rank, alpha, dropout = (values.get(k, d) for k, d in
                            [('lora_rank', 16), ('lora_alpha', 32.), ('lora_dropout', 0.)])
    if type(rank) is not int or not 1 <= rank <= 256:
        raise ValueError('LoRA rank须为1–256的整数')
    for label, number, valid in [('alpha', alpha, lambda v: v > 0), ('dropout', dropout, lambda v: 0 <= v < 1)]:
        if type(number) not in (float, int) or not math.isfinite(number) or not valid(number):
            raise ValueError('LoRA ' + label + '超出范围')
    layers = values.get('llm_last_n_layers', 0)
    if type(layers) is not int or layers < 0:
        raise ValueError('LLM末层数须为非负整数，0表示全部所选层')
    checkpointing = values.get('gradient_checkpointing', False)
    if type(checkpointing) is not bool:
        raise ValueError('gradient_checkpointing须为开关')
    scopes = [t for t in targets if not freeze[t]] if mode == 'lora' else []
    if mode == 'lora' and not scopes:
        raise ValueError('LoRA目标已全部冻结，请至少解冻一个LoRA目标组件')
    fusion = values.get('fusion_variant')
    if fusion and (mode != 'custom' or not all(freeze.values())):
        raise ValueError('Exp10 requires custom mode with all base components frozen')
    if (all(freeze.values()) and not fusion) or (values.get('arm') == 'baseline' and all(freeze[n] for n in COMPONENTS[:-1])):
        raise ValueError('没有可训练组件')
    return dict(fine_tuning_mode=mode, **{'freeze_' + k: v for k, v in freeze.items()},
                lora_targets=','.join(targets), lora_rank=rank, lora_alpha=float(alpha),
                lora_dropout=float(dropout), llm_last_n_layers=layers,
                gradient_checkpointing=checkpointing)


def add_finetuning_options(parser):
    parser.add_argument('--fsdp-accumulation', choices=('shard_each', 'retain'), default='shard_each',
                        help='Engineering policy: retain full parameters/gradients until the final microbatch')
    parser.add_argument('--fine-tuning-mode', choices=MODES, default='action_full',
                        help='action_full: frozen Eagle; full: all components; custom: component switches; lora: low-rank adapters')
    for name in COMPONENTS:
        parser.add_argument('--freeze-' + name.replace('_', '-'), action=argparse.BooleanOptionalAction,
                            default=None, help='Override mode default for ' + name)
    parser.add_argument('--lora-targets', default='llm,action_dit', help='Comma-separated component names; all Linear layers in those components')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=float, default=32.)
    parser.add_argument('--lora-dropout', type=float, default=0.)
    parser.add_argument('--llm-last-n-layers', type=int, default=0, help='Restrict LLM full/LoRA training to last N retained Eagle layers; 0 = all')
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=False,
                        help='Checkpoint trainable LLM/vision activations using non-reentrant recomputation')
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.)


def finetuning_argv(args):
    values = resolve_finetuning(args)
    result = []
    for key, value in values.items():
        flag = key.replace('_', '-')
        if type(value) is bool:
            result.append(('--' if value else '--no-') + flag)
        else:
            result.extend(['--' + flag, str(value)])
    result += ['--weight-decay', str(getattr(args, 'weight_decay', .01)),
               '--max-grad-norm', str(getattr(args, 'max_grad_norm', 1.))]
    return result
