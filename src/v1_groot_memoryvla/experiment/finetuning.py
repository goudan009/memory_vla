"""Apply component freezing and LoRA before building DDP and the optimizer."""
import torch
from torch import nn

from .finetuning_options import COMPONENTS, resolve_finetuning
from v1_groot_memoryvla.model.modules.lora import LoRALinear, install_lora


def components(model):
    head = model.action_head
    projector = [head.state_encoder, head.action_encoder, head.action_decoder]
    if head.config.add_pos_embed:
        projector.append(head.position_embedding)
    if head.state_dropout_prob > 0:
        projector.append(head.mask_token)
    if getattr(head, "openvla_memory", False):
        projector.extend((head.memory_cognition_projection, head.memory_perception_projection,
                          head.memory_unconditional_cognition))
    return dict(llm=[model.backbone.model.language_model], vision=[model.backbone.model.vision_model],
                vision_projector=[model.backbone.model.mlp1], action_dit=[head.model],
                action_projector=projector, action_vlln=[head.vlln], memory=[model.memory] if model.memory is not None else [])


def apply_finetuning(model, args):
    options = resolve_finetuning(args)
    groups = components(model)
    model.requires_grad_(False)
    selected = set(options['lora_targets'].split(',')) if options['fine_tuning_mode'] == 'lora' else set()
    layers = model.backbone.model.language_model.model.layers
    last_n = options['llm_last_n_layers']
    if last_n > len(layers):
        raise ValueError(f'LLM仅保留{len(layers)}层，不能选择最后{last_n}层')
    selected_groups = dict(groups)
    if last_n:
        selected_groups['llm'] = list(layers[-last_n:])
    module_names = {id(m): n for n, m in model.named_modules()}
    paths = []
    for group, modules in selected_groups.items():
        if options['freeze_' + group]:
            continue
        if group in selected:
            count = 0
            for root in modules:
                if not isinstance(root, nn.Module):
                    continue
                prefix = module_names[id(root)]
                for name, module in root.named_modules():
                    if isinstance(module, nn.Linear) and name != 'lm_head':
                        paths.append(prefix + ('.' + name if name else ''))
                        count += 1
            if not count:
                raise ValueError('LoRA组件没有Linear层: ' + group)
        else:
            for module in modules:
                module.requires_grad_(True)
    # Existing LoRA targets must remain identifiable when continuing from weights.
    saved = getattr(model.config, 'finetuning_lora', None)
    existing = list(saved.get('modules', [])) if saved else []
    all_paths = sorted(set(existing + paths))
    if all_paths:
        if saved and paths and (saved['rank'] != options['lora_rank'] or saved['alpha'] != options['lora_alpha'] or saved['dropout'] != options['lora_dropout']):
            raise ValueError('已有LoRA权重的rank/alpha/dropout与请求不一致')
        spec = dict(modules=all_paths, rank=saved['rank'] if saved else options['lora_rank'],
                    alpha=saved['alpha'] if saved else options['lora_alpha'],
                    dropout=saved['dropout'] if saved else options['lora_dropout'])
        install_lora(model, spec)
        model.config.finetuning_lora = spec
        for path in all_paths:
            module = model.get_submodule(path)
            # Frozen scopes stay entirely frozen. Full mode can update base W too.
            owner = next(g for g, roots in groups.items() if any(
                path == module_names[id(root)] or path.startswith(module_names[id(root)] + '.')
                for root in roots if isinstance(root, nn.Module)))
            active = not options['freeze_' + owner] and (owner != 'llm' or not last_n or any(
                path.startswith(module_names[id(layer)] + '.') for layer in layers[-last_n:]))
            module.lora_A.requires_grad_(active)
            module.lora_B.requires_grad_(active)
            module.weight.requires_grad_(active and owner not in selected)
            if module.bias is not None:
                module.bias.requires_grad_(active and owner not in selected)
    # Keep optimizer master weights FP32; frozen backbone remains BF16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    for group, modules in groups.items():
        model.config.__dict__['freeze_' + group] = options['freeze_' + group]
    model.backbone.tune_llm = any(p.requires_grad for p in model.backbone.model.language_model.parameters())
    model.backbone.tune_visual = any(p.requires_grad for p in model.backbone.model.vision_model.parameters())
    model.action_head.tune_diffusion_model = any(p.requires_grad for p in model.action_head.model.parameters())
    model.action_head.tune_projector = not options['freeze_action_projector']
    model.action_head.tune_vlln = not options['freeze_action_vlln']
    model.config.tune_llm = model.backbone.tune_llm
    model.config.tune_visual = model.backbone.tune_visual
    model.config.tune_diffusion_model = model.action_head.tune_diffusion_model
    model.config.tune_projector = model.action_head.tune_projector
    model.config.tune_vlln = model.action_head.tune_vlln
    model.config.fine_tuning_options = options
    if getattr(args, "fusion_variant", None):
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Exp10 requires all original parameter groups frozen")
        fusion = (model.memory.cognitive_fusion if args.fusion_variant == "gated_eos"
                  else model.action_head.cognitive_fusion)
        fusion.requires_grad_(True).float()
    if options['gradient_checkpointing'] and not getattr(model.action_head, 'openvla_memory', False):
        for group in ('llm', 'vision'):
            module = groups[group][0]
            if any(p.requires_grad for p in module.parameters()):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                module.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise ValueError('没有可训练参数')
    summary = dict(options, trainable_parameters=trainable,
        total_parameters=sum(p.numel() for p in model.parameters()),
        components={name: dict(total=sum(p.numel() for m in modules for p in (m.parameters() if isinstance(m, nn.Module) else [m])),
                              trainable=sum(p.numel() for m in modules for p in (m.parameters() if isinstance(m, nn.Module) else [m]) if p.requires_grad))
                    for name, modules in groups.items()}, lora_modules=all_paths,
        frozen_eagle=not any(p.requires_grad for p in model.backbone.parameters()))
    model.finetuning_summary = summary
    if getattr(args, "fusion_variant", None):
        summary["fusion_parameters"] = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    return summary
