"""Read-only verification of the completed Task-2 evidence and backups."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/task2_memory_training_20261113'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(4)
    result = {}
    configs = {a: read(OUT / a / 'config.json') for a in ('A_joint', 'B_frozen')}
    assert configs['A_joint']['initialization_sha256'] == configs['B_frozen']['initialization_sha256']
    assert (OUT / 'A_joint/order.json').read_bytes() == (OUT / 'B_frozen/order.json').read_bytes()
    result['identical_initialization_and_training_order'] = True
    for arm in configs:
        rows = [json.loads(line) for line in (OUT / arm / 'training.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == list(range(1, 101))
        gradient_checks = []
        for step in (1, 2, 10, 50, 100):
            audit = read(OUT / arm / f'gradients_step{step:03d}.json')['parameters']
            memory = [v for n, v in audit.items() if n.startswith('memory.')]
            assert len(memory) == 75 and all(v['requires_grad'] and not v['grad_none'] and v['grad_norm'] > 0 for v in memory)
            if arm == 'B_frozen':
                assert all(not v['requires_grad'] and v['grad_none'] for n, v in audit.items() if n.startswith('action_head.'))
            gradient_checks.append(step)
        state = torch.load(OUT / arm / 'final_training_state.pt', map_location='cpu', weights_only=False, mmap=True)
        steps = sorted({int(v['step']) for v in state['optimizer']['state'].values()})
        assert steps == [100] and state['completed_steps'] == 100
        assert len(state['optimizer']['state']) == (548 if arm == 'A_joint' else 75)
        result[arm] = dict(optimizer_states=len(state['optimizer']['state']), actual_saved_optimizer_steps=steps,
            all_75_memory_tensors_have_nonzero_grad_at_steps=gradient_checks,
            final_checkpoint_sha256_matches=sha(OUT / arm / 'final_training_state.pt') == read(OUT / arm / 'complete.json')['final_parameters_sha256'])
        assert result[arm]['final_checkpoint_sha256_matches']
        if arm == 'B_frozen':
            checkpoint = OUT / 'backup/official_checkpoint'
            weight_map = read(checkpoint / 'model.safetensors.index.json')['weight_map']
            checked = 0
            for shard in sorted(set(weight_map.values())):
                with safe_open(checkpoint / shard, framework='pt', device='cpu') as source:
                    for name in source.keys():
                        if name.startswith('action_head.'):
                            assert torch.equal(source.get_tensor(name).float(), state['action_head'][name.removeprefix('action_head.')])
                            checked += 1
            assert checked == 473
            result[arm]['original_head_tensors_bitwise_equal_to_specialist'] = checked
        del state
    manifest = read(OUT / 'backup/manifest.json')['files']
    unchanged = []
    for relative, details in manifest.items():
        if relative.startswith(('src/', 'configs/', 'train/')):
            assert sha(ROOT / relative) == details['sha256']
            unchanged.append(relative)
        assert sha(OUT / 'backup' / relative) == details['sha256']
    result['backup_manifest_verified_files'] = len(manifest)
    result['original_source_and_configs_unchanged_files'] = len(unchanged)
    existing = OUT / 'backup/existing_run20k'
    files = {str(p.relative_to(existing)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(existing.rglob('*')) if p.is_file()}
    (OUT / 'backup/existing_run20k_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    result['existing_run20k_backed_up_files'] = len(files)
    result['existing_run20k_model_shards'] = {n: v['sha256'] for n, v in files.items() if n.endswith('.safetensors')}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
