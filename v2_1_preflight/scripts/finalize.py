"""Archive read-only workbench/source evidence and enrich delivery semantics.

Does not launch a simulator, model or training process. Run before package.py.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'v2_1_preflight'
OG=Path('/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson')
JOBS=['20261126-200257-15a1de03','20261126-201130-d9b4bb46']


def read(name):return json.loads((OUT/name).read_text())
def write(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args,cwd=ROOT):
    r=subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return dict(argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)


def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    jobs=[]
    for job in JOBS:
        src=ROOT/'outputs/benchmarks/web'/job;dest=OUT/'samples/workbench'/job;dest.mkdir(parents=True,exist_ok=True)
        for name in ('plan.json','state.json','process.json','preflight_extract.json','run.log','worker.log'):
            shutil.copy2(src/name,dest/name)
        state=json.loads((src/'state.json').read_text());assert state['status']=='stopped',state
        process=json.loads((src/'process.json').read_text())
        pid=process['pid'];same_process=False
        stat=Path(f'/proc/{pid}/stat')
        if stat.exists():same_process=stat.read_text().rsplit(')',1)[1].split()[19]==str(process['start'])
        assert not same_process,(job,pid,'still running')
        jobs.append(dict(job_id=job,status=state,original_process_alive=same_process,
            reason='Intentional stop of read-only extraction; CPU decoder completed all200 after reference agreement. Not a completed200 simulator job.'))
    request=urllib.request.Request('http://127.0.0.1:8765/api/services')
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=45) as response:
        services=json.load(response)
    write('samples/workbench/final_services.json',dict(queried_at_utc=now,response=services))
    write('samples/workbench/job_summary.json',jobs)
    versions=dict(created_at_utc=now,hostname=command(['hostname']),
        current_git=command(['git','rev-parse','HEAD']),source_commit='0d28bc506c3def15b2f76bdbdc0a3ccb1b4567a8',
        historical_v2_commit='b710fc4df586216b4b6a1a3776f09c725300cd98',
        omnigibson_git=command(['git','rev-parse','HEAD'],OG),
        raw_dataset_revision='b3c9a714b8730701c480e77173d8aa4a8f19e1c0',
        design_sources=[dict(path=p,sha256=sha(ROOT/p)) for p in ('docs/v2/2.1.html','docs/v2/2.1.md','docs/v2/2.1-protocol.md','docs/v2/notation.md')],
        local_gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv']),
        local_gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv']))
    write('samples/version_manifest.json',versions)
    mapping=read('state_action_mapping.json')
    mapping['source_evidence']=dict(
        packing='samples/source_evidence/current_v1/src/v1_groot_memoryvla/data/convert_b1k.py:STATE_SLICES',
        normalization='samples/source_evidence/current_v1/src/v1_groot_memoryvla/data/utils.py:57,101',
        relative_and_groups='samples/source_evidence/current_v1/src/v1_groot_memoryvla/data/state_action/state_action_processor.py:apply_action/unapply_action',
        processor='samples/source_evidence/current_v1/src/v1_groot_memoryvla/model/gr00t_n1d6/processing_gr00t_n1d6.py',
        recording_controller='samples/actual_recording_config.json:robots[0].controller_config',
        controller_implementations=['samples/source_evidence/omnigibson/controllers/'+n+'.py' for n in ('controller_base','joint_controller','multi_finger_gripper_controller','holonomic_base_joint_controller')],
        physical_asset='joint_topology.json; robot_description/asset_provenance.json',
        measured_ranges='samples/dataset_audit.json:ranges; 200episodes647555frames')
    write('state_action_mapping.json',mapping)
    inv=read('state_inventory.json');ranges=read('samples/decoded_ranges.json')
    for entry in inv['entries']:
        if entry['field'] in ('Open','Joint','ToggledOn','Pose','Temperature'):
            entry['verification_scope']='All200 only for extracted microwave joint/Open/ToggledOn, bag+microwave pose/temperature; not all object instances or derived geometric predicates'
            key={'Open':'microwave_joint','Joint':'microwave_joint','ToggledOn':'started','Pose':'bag_pose','Temperature':'temperature'}[entry['field']]
            entry.update(observed_range=ranges[key],verification='all200_recorded_source_decoded',per_frame_evidence='samples/decoded/*.npz')
            example=dict(source='samples/decoded/episode_00400010.npz',field=key,frame=0,value=ranges[key]['first_frame_example'])
            entry['actual_examples']=[x for x in entry['actual_examples'] if x.get('source')!=example['source']]+[example]
        if entry['field'] in ('robot.joint_pos','robot.joint_vel','joint_effort'):
            start,end={'robot.joint_pos':(0,28),'robot.joint_vel':(84,112),'joint_effort':(112,140)}[entry['field']]
            raw=read('samples/dataset_audit.json')['ranges']['raw_state256']
            entry['observed_range']={k:v[start:end] for k,v in raw.items()}
        if entry['field'] in ('ParticleModifier','ParticleApplier','ParticleRemover','ParticleSource','ParticleSink'):
            entry.update(api='obj.states[State].dump_state(serialized=False) / update; inherited IntrinsicObjectState has no get_value',
                dtype='intrinsic state/configuration dict, not a Boolean getter',shape=None,unit=None,
                concrete_value_implementation=False)
        if entry['field']=='sensor.rgb':entry.update(shape=['H','W',3],dtype='uint8 RGB (video-decoded; simulator raw camera may carry RGBA)',observed_range=[0,255])
        if entry['field'] in ('goal_status','task_success','q_score.final','task_progress'):
            entry['actual_examples']=None;entry['unavailable_reason']='No original per-frame predicate/metric values recorded; runtime source semantics verified only'
    inv['entries']=[e for e in inv['entries'] if e['field']!='cooked_popcorn_exists']
    inv['entries'].append(dict(field='cooked_popcorn_exists',category='particle_system',meaning='Current serialized cooked__popcorn particle count >0; not full task success or bag Contains predicate',
        api='env.scene.get_system("cooked__popcorn",force_init=False).n_particles; decoded serialized MacroPhysicalParticleSystem count',
        source='samples/source_evidence/omnigibson/systems/macro_particle_system.py:1519; scripts/offline_decode.py',
        dtype='bool',shape=[],unit='1',coordinate_frame='system identity',official_data=dict(parquet=False,hdf5=True),realtime_only=False,
        strict_demo_alignment='state[t] before action[t], final stateT excluded',verification='all200_recorded_source_decoded',
        observed_range=[False,True],actual_examples=[dict(episode=400010,frame=3795,count=0,value=False),dict(episode=400010,frame=3827,count=256,value=True)],
        particle_count_range=ranges['cooked_particle_count']))
    inv['coverage_limit']='Source enumeration covers object_states classes, robot physics, task/evaluation and vision modalities. Abstract/mixin entries are infrastructure. This is not a claim that every Isaac sensor/physics API was dynamically exercised.'
    inv['entry_count']=len(inv['entries']);write('state_inventory.json',inv)
    joint_source=OG/'omnigibson/prims/joint_prim.py'
    shutil.copy2(joint_source,OUT/'samples/source_evidence/omnigibson/prims/joint_prim.py')
    topology=read('joint_topology.json')
    topology['limit_interpretation']='Limits and max velocity are authored USD attributes converted to SI. None means not authored, not zero. Runtime JointPrim.max_velocity may substitute defaults for unspecified/inf values; controllers may impose narrower limits. Native angular values are degrees, runtime q is radians.'
    write('joint_topology.json',topology)
    historical=Path('/nvme3/projects/memory_vla-v2-v3-backup-oEWaKh/Output/test_02/experiment_01/evidence')
    asset_lines=[]
    for p in historical.glob('4090-label-audit-attempt*.json'):
        d=json.loads(p.read_text())
        lines=[line for line in d.get('log','').splitlines() if 'popcorn_bag_73 was expected to have USD file hash' in line]
        if lines:asset_lines.append(dict(source=str(p),sha256=sha(p),matching_log_lines=lines))
    write('samples/historical_asset_mismatch.json',asset_lines)
    schema=dict(index_convention='zero-based; episode_index is original episode ID, not contiguous ordinal',rows=647555,episodes=200,
        alignment='state[t] = obs/RGB[t] before action[t]. reward/terminated/truncated[t] are action[t] outcomes; state[T] has no corresponding RGB/action and is excluded.',
        labels={
            'door_open':dict(semantics='Official Open predicate on microwave joint j_link_0, q>0.07853977382183075rad',unknown='abs(q-threshold)<=1e-5rad or nonfinite',source='object_states/open_state.py:_compute_joint_threshold/_get_value'),
            'started':dict(semantics='Current ToggledOn.value, not ever-started',unknown='Inherited certified +/-8 frames inclusive around every toggle switch; 17 frames/episode',source='object_states/toggle.py:_deserialize/_get_value',recorded_value='started_recorded_value remains available at masked boundary'),
            'bag_grasped':dict(semantics='Robot is_grasping(candidate_obj=bag)',unknown='All frames: serialized per-frame AG constraint unavailable'),
            'bag_inside':dict(semantics='bag.states[Inside].get_value(microwave)',unknown='All frames: bag collision asset differs from original; recorded poses alone insufficient for certified AABB/meta-volume'),
            'cooked_popcorn_exists':dict(semantics='cooked__popcorn serialized particle count >0',unknown='None in these records',not_equivalent_to='full Real AND Contains task success')},
        validity='Each nullable Boolean has a corresponding *_valid bool. Unknown is null, never false. Statistics must filter valid.',
        explicit_flags='*_explicitly_recorded false means official sleeping-object omission and carry prior recorded/initial state; does not automatically mean invalid.',
        numeric_fields=dict(microwave_door_joint_rad='float32 rad',microwave_temperature_c='float32 degreesC',bag_temperature_c='float32 degreesC',cooked_popcorn_particle_count='int32',raw_popcorn_particle_count='int32'),
        statistics='recorded_switches_before_mask uses unmasked recorded values; adjacent_valid_switches counts only adjacent valid pairs; unknown switch count is null, not0.',
        historical_note='samples/official_decode_report.json is a compatibility copy of cpu_decode_report.json, NOT200 official simulator episodes; workbench logs retain37 official episodes.')
    write('samples/label_schema.json',schema)
    write('samples/visual_review.json',dict(review='Manual visual inspection of original-head-camera panels, 2026-11-26',
        panels=['certified_transition_panel_400010.png','certified_transition_panel_400020.png'],
        observations='Door changes agree with q and Open. Button red-to-green agrees with current ToggledOn. Cooked particles appear in aperture after particle count changes.',
        limitation='No visual certification of Inside or grasp. Keyframes +/-16 around raw transitions do not prove exact subframe rendering latency.',
        historical_provisional='label_alignment_400010.png and corresponding JSON contain provisional Inside labels, labels_certified=false; excluded from task40_labels.parquet'))
    print(json.dumps(dict(jobs_stopped=len(jobs),inventory_entries=len(inv['entries']),archived=True)))


if __name__=='__main__':main()
