"""Validate the bundled evidence, build a SHA256 manifest and verify the tarball.

No simulator, network, external dataset scan or model execution. --check-only is read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
FIELDS=('door_open','started','cooked_popcorn_exists','bag_grasped','bag_inside')
EXPECTED={'door_open':(398237,249317,1,400),'started':(21412,622743,3400,200),
          'cooked_popcorn_exists':(8012,639543,0,194),'bag_grasped':(0,0,647555,None),
          'bag_inside':(0,0,647555,None)}


def read(name):return json.loads((OUT/name).read_text())
def digest_fileobj(f):
    h=hashlib.sha256()
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def sha(p):
    with p.open('rb') as f:return digest_fileobj(f)
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def files():
    return sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json')


def validate():
    required=('report.md','state_inventory.json','task40_labels.parquet','label_statistics.csv',
        'joint_topology.json','state_action_mapping.json','fk_validation.json','model_interfaces.md',
        'robot_description/r1pro_kinematics.usda','robot_description/topology.svg',
        'scripts/README.md','workflow/plan-v001.json','progress.md')
    for name in required:assert (OUT/name).is_file(),name
    for p in files():
        assert not p.is_symlink(),p
        if p.suffix=='.json':json.loads(p.read_text())
    labels=pd.read_parquet(OUT/'task40_labels.parquet')
    stats=pd.read_csv(OUT/'label_statistics.csv',dtype={'episode_index':str})
    assert len(labels)==647555 and labels.episode_index.nunique()==200
    for k in FIELDS:
        assert np.array_equal(labels[k].notna(),labels[k+'_valid']),k
        value=labels[k].fillna(False).to_numpy(dtype=bool);valid=labels[k+'_valid'].to_numpy()
        counts=(int((value&valid).sum()),int((~value&valid).sum()),int((~valid).sum()))
        assert counts==EXPECTED[k][:3],(k,counts)
        row=stats[(stats.episode_index=='ALL_200')&(stats.state==k)].iloc[0]
        assert tuple(int(row[c]) for c in ('positive','negative','unknown'))==counts
        recorded=row.recorded_switches_before_mask
        assert (pd.isna(recorded) if EXPECTED[k][3] is None else recorded==EXPECTED[k][3])
    decode=read('samples/cpu_decode_report.json')
    assert decode['passed'] and len(decode['episodes'])==200
    assert decode['official_compared_episodes']==37 and decode['official_compared_frames']==127312
    assert sum(x['frames'] for x in decode['episodes'])==len(labels)
    for item in decode['episodes']:
        p=OUT/'samples/decoded'/Path(item['artifact']).name
        assert sha(p)==item['sha256'],p
    cert=read('samples/historical_started_certification.json')
    certs={x['episode']:x for x in cert['episodes']};radius=cert['boundary_unknown_radius_frames']
    threshold=decode['open_joints'][0]['threshold']
    for eid,group in labels.groupby('episode_index',sort=True):
        T=len(group);np.testing.assert_array_equal(group.frame_index,np.arange(T))
        assert np.max(np.abs(group.timestamp_s-np.arange(T)/30))<1e-4
        with np.load(OUT/'samples/decoded'/f'episode_{eid:08d}.npz') as z:
            assert all(len(z[k])==T for k in z.files)
            np.testing.assert_array_equal(group.started_recorded_value,z['started'])
            transitions=(np.flatnonzero(z['started'][1:]!=z['started'][:-1])+1).tolist()
            assert transitions==certs[eid]['transition_frames']
            known=np.ones(T,dtype=bool)
            for t in transitions:known[max(0,t-radius):min(T,t+radius+1)]=False
            np.testing.assert_array_equal(group.started_valid,known)
            assert int((known&z['started']).sum())==certs[eid]['positive_valid']
            assert int((known&~z['started']).sum())==certs[eid]['negative_valid']
            assert int((~known).sum())==certs[eid]['unknown']
            q=z['microwave_joint'];door_known=np.isfinite(q)&(np.abs(q-threshold)>1e-5)
            np.testing.assert_array_equal(group.door_open_valid,door_known)
            np.testing.assert_array_equal(group.loc[door_known,'door_open'].to_numpy(dtype=bool),(q>threshold)[door_known])
            np.testing.assert_array_equal(group.cooked_popcorn_exists.to_numpy(dtype=bool),z['cooked_particle_count']>0)
            assert group.bag_inside.isna().all() and group.bag_grasped.isna().all()
        rows=stats[stats.episode_index==str(eid)]
        assert len(rows)==5
        for k in FIELDS:
            row=rows[rows.state==k].iloc[0];v=group[k];valid=v.notna();yv=v.fillna(False).to_numpy(dtype=bool)
            assert row.frames==T and row.positive==int(v.sum()) and row.negative==int((~v).sum()) and row.unknown==int(v.isna().sum())
            mask=valid.to_numpy();assert row.adjacent_valid_switches==int(np.sum(mask[1:]&mask[:-1]&(yv[1:]!=yv[:-1])))
    mapping=read('state_action_mapping.json');top=read('joint_topology.json')
    assert [d['index'] for d in mapping['state_dimensions']]==list(range(82))
    assert [d['action_index'] for d in mapping['action_channels']]==list(range(23))
    assert [d['processor_padded_column'] for d in mapping['action_channels']]==list(range(23))
    assert len(mapping['topology_prior']['edges'])==25
    assert len(top['joints'])==43 and len(top['links'])==44 and top['counts']['nonfixed']==28
    links={l['name'] for l in top['links']};children=[j['child'] for j in top['joints']]
    assert len(set(children))==43 and set(children)<links
    assert all(j['parent'] in links and j['child'] in links for j in top['joints'])
    fk=read('fk_validation.json');assert fk['passed'] and fk['comparisons']==300 and len(fk['errors'])==300
    assert fk['summary']['position_error_m']['max']<2e-6 and fk['summary']['rotation_error_rad']['max']<1.4e-6
    inv=read('state_inventory.json');assert len(inv['entries'])==87
    assert sum(e['category']=='object_state' for e in inv['entries'])==55
    video=read('samples/video_alignment_metadata.json');assert len(video['videos'])==600 and len(video['keyframes'])==72
    for item in video['keyframes']:
        assert (OUT/item['image']).is_file()
        assert abs(item['timestamp_s']-item['frame']/30)<1e-6
    model=read('samples/model_interface_evidence.json');assert model['sample_count']==40 and model['eos_equals_existing_capture']
    assert len(model['sources'])==35
    for s in model['sources']:assert s['byte_equal_git_blob'] and sha(OUT/s['bundle'])==s['sha256']
    for job in read('samples/workbench/job_summary.json'):
        assert job['status']['status']=='stopped' and not job['original_process_alive']
    return dict(passed=True,episodes=200,frames=len(labels),nullable_labels_checked=5,state_dimensions=82,
        action_channels=23,joints=43,links=44,inventory_entries=87,object_state_classes=55,
        fk_comparisons=300,official_reference_episodes=37,official_reference_frames=127312,
        video_metadata_records=600,keyframes=72,git_exact_sources=35,
        limits='Validates bundled values/provenance, not a new physics replay, independent official deserialization or model forward. Inside and grasp remain unknown.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    result=validate()
    if a.check_only:print(json.dumps(result));return
    write('samples/package_validation.json',result)
    records=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files()]
    write('MANIFEST.json',dict(algorithm='sha256',scope='All bundled files except MANIFEST.json itself and __pycache__',files=records))
    archive=OUT.parent/(OUT.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for rel in [r['path'] for r in records]+['MANIFEST.json']:
            tar.add(OUT/rel,arcname=OUT.name+'/'+rel,recursive=False)
    expected={OUT.name+'/'+r['path']:r['sha256'] for r in records}
    expected[OUT.name+'/MANIFEST.json']=sha(OUT/'MANIFEST.json')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(expected)
        for member in members:
            assert member.isfile() and member.name in expected
            with tar.extractfile(member) as f:assert digest_fileobj(f)==expected.pop(member.name),member.name
        assert not expected
    digest=sha(archive)
    archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(validation=result,archive=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(records)+1)))


if __name__=='__main__':main()
