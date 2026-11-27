"""Sampled fixed-input history interventions; preserve the actual stream and RNG."""
from copy import deepcopy

import torch


def _snapshot(memory):
    return [(bank, {key:list(rows) for key,rows in bank.bank.items()},
             bank.last_written.copy(),bank.merge_count.copy())
            for bank in (memory.perceptual,memory.cognitive)]


def _restore(snapshot):
    for bank,rows,last,merges in snapshot:
        bank.bank={key:list(value) for key,value in rows.items()}
        bank.last_written=last.copy()
        bank.merge_count=merges.copy()


@torch.no_grad()
def action_controls(memory, backbone, input_ids, context, predict):
    before=_snapshot(memory)
    cpu_rng=torch.get_rng_state()
    device=backbone.backbone_features.device
    cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    cuda_rng=torch.cuda.get_rng_state(device) if cuda_devices else None
    actual=predict(memory(backbone,input_ids,context))
    after=_snapshot(memory)
    # These are non-persistent observation attributes, never checkpoint tensors.
    observations=[(module,name,deepcopy(getattr(module,name))) for module in memory.modules()
                  for name in ('last_diagnostics','last_history_control','last_attention','last_gate_mean')
                  if hasattr(module,name)]
    original_mode=context.get('history_mode','normal')
    conditions=[mode for mode in ('normal','empty','other_episode') if mode!=original_mode]
    paired=len(context['episode_keys'])>1 and len({tuple(k) if isinstance(k,list) else k for k in context['episode_keys']})==len(context['episode_keys'])
    if not paired:
        conditions=[mode for mode in conditions if mode!='other_episode']
    report=dict(actual_history_mode=original_mode,conditions=conditions,comparisons=[],
                actual_normalized_first8=actual['action_pred'][:,:8].detach().float().cpu().tolist(),
                current_input='同一份backbone特征、input_ids和动作头输入',
                randomness='各条件恢复同一动作噪声状态；正式输出后的RNG与Memory恢复',
                interpretation='归一化动作输出变化，不是成功率、动作准确率或正确历史优势；含模型输出的全部维度')
    if not paired:
        report['other_episode_unavailable']='requires_distinct_paired_environments'
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            for mode in conditions:
                _restore(before)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng,device)
                alternative=predict(memory(backbone,input_ids,dict(context,history_mode=mode,memory_diagnostics=False)))
                difference=actual['action_pred'].float()-alternative['action_pred'].float()
                report['comparisons'].append(dict(history_mode=mode,
                    normalized_first8=alternative['action_pred'][:,:8].detach().float().cpu().tolist(),
                    action_shape=list(difference.shape),
                    normalized_action_shift_rms=float(difference.square().mean().sqrt()),
                    normalized_action_shift_max_abs=float(difference.abs().max()),
                    normalized_first8_shift_rms=float(difference[:,:8].square().mean().sqrt())))
    finally:
        _restore(after)
        for module,name,value in observations:
            setattr(module,name,value)
    actual['memory_action_controls']=report
    return actual
