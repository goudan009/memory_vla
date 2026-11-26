"""Certify the audited recorded predicate subset, not geometric replay labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT/'Output/test_02/experiment_01'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    split = json.loads((EXPERIMENT/'evidence/data_split.json').read_text())
    if report.get('passed') is not True or report.get('label_names') != ['started']:
        raise ValueError('Official full-record deserialization did not pass')
    episodes = {e['episode']:e for e in report['episodes']}
    assert len(episodes) == 200 and set(episodes) == set(split['train']+split['validation'])
    # These three original-camera transition panels were inspected by the Agent.
    # ep400010 also agrees at every frame with two full physical playbacks.
    for eid in (400010,400020,403000):
        name = 'label_alignment_400010' if eid == 400010 else f'started_alignment_{eid}'
        assert (EXPERIMENT/f'evidence/{name}.png').is_file()
        assert episodes[eid]['positive_valid']>0 and episodes[eid]['negative_valid']>0
    assert episodes[400010]['full_replay_comparison']['all_frame_disagreements'] == 0
    counts = {}
    for name,ids in (('train',split['train']),('validation',split['validation'])):
        count = dict(positive=0,negative=0,unknown=0,episodes=len(ids))
        for eid in ids:
            item = episodes[eid]
            path = Path(item['artifact'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['artifact_sha256']
            with np.load(path,allow_pickle=False) as data:
                y,known = data['labels'],data['known']
                assert y.shape == known.shape == (item['frame_count'],1)
                assert known.dtype == np.bool_ and np.isin(y,[0.,1.]).all()
                assert np.isfinite(y).all()
                count['positive'] += int(y[known].sum())
                count['negative'] += int((1-y[known]).sum())
                count['unknown'] += int((~known).sum())
            item['labels_certified'] = True
        assert count['positive']>0 and count['negative']>0
        counts[name] = count
    report.update(labels_certified=True,counts=counts,
        physical_definition='Recorded microwave_hjjxmi_0 non_kin.ToggledOn.value via initialized official og.sim.deserialize',
        source='Official behavior-1k/2025-challenge-rawdata task-0040; revision b3c9a714b8730701c480e77173d8aa4a8f19e1c0',
        disabled_fields=['door_open','bag_inside'],
        validation='3 original RGB before/after panels; ep400010 full physical replay agrees all frames; official schema consumes every raw vector; episode identity and length checked',
        unknown_policy='Mask +/-8 frames around a recorded toggle change; no boundary shifting or missing-as-false',
        scope='Certified recorded started only, not certification of geometric predicates or training performance')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,labels_certified=True,counts=counts)),flush=True)


if __name__ == '__main__':
    main()
