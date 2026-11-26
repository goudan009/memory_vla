"""Discover all actual object state classes, preserve definitions and evidence grades."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'v2_1_preflight'
OG=Path('/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson')


def main():
    scene=json.loads((OUT/'samples/scene_initial_state.json').read_text())
    report_path=ROOT/'outputs/benchmarks/web/20261126-201130-d9b4bb46/preflight_extract.json'
    if not report_path.exists():report_path=ROOT/'outputs/benchmarks/web/20261126-200257-15a1de03/preflight_extract.json'
    actual=json.loads(report_path.read_text());schemas=actual['schemas']
    boolean={'Open','ToggledOn','Inside','OnTop','Under','NextTo','Touching','Contains','Filled','Covered','Saturated','Cooked','Burnt','Frozen','Heated','OnFire','AttachedTo','Overlaid','Draped','Folded','Unfolded','IsGrasping'}
    numeric={'Temperature':('float32',[], 'degC'),'MaxTemperature':('float32',[],'degC'),'AABB':('tuple[tensor,tensor]',[[3],[3]],'m'),
        'Pose':('tuple[tensor,tensor]',[[3],[4]],'m,quaternion_xyzw'),'Joint':('tensor',['n_joints'],'rad or m by joint'),
        'ContactBodies':('set[RigidPrim]',None,None),'HorizontalAdjacency':('namedtuple[set]',None,None),
        'VerticalAdjacency':('namedtuple[set]',None,None),'ContainedParticles':('dict',None,'particle IDs/counts')}
    entries=[]
    classes=[]
    for p in sorted((OG/'object_states').glob('*.py')):
        classes.extend(c for c in ast.parse(p.read_text()).body if isinstance(c,ast.ClassDef))
    state_names={c.name for c in classes if any('State' in ast.unparse(b) for b in c.bases) or any(isinstance(n,ast.FunctionDef) and n.name=='_get_value' for n in c.body)}
    while True:
        expanded=state_names|{c.name for c in classes if any(ast.unparse(b) in state_names for b in c.bases)}
        if expanded==state_names:break
        state_names=expanded
    # Include every class with a state base / mixin or defined value method. Mark
    # infrastructure explicitly so this file does not claim abstract classes callable.
    for p in sorted((OG/'object_states').glob('*.py')):
        source=p.read_text();tree=ast.parse(source)
        for c in tree.body:
            if not isinstance(c,ast.ClassDef):continue
            bases=[ast.unparse(x) for x in c.bases]
            methods={n.name:n for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            if c.name not in state_names:continue
            relative=any('Relative' in b for b in bases)
            concrete='_get_value' in methods
            stateful=[s['name'] for s in schemas if c.name in s['stateful']]
            attached=[s['name'] for s in schemas if c.name in s['states']]
            n=methods.get('_get_value');signature=ast.unparse(n.args) if n else None
            dtype,shape,unit=('bool',[],'1') if c.name in boolean or 'BooleanStateMixin' in bases else numeric.get(c.name,('source-defined; see signature/snippet',None,None))
            sample=[]
            for name,s in scene['state']['registry']['object_registry'].items():
                if c.name in s.get('non_kin',{}):sample.append(dict(object=name,value=s['non_kin'][c.name],scope='original initial snapshot, not per-frame validation'))
            extraction=('obj.states[object_states.%s].get_value(%s)'%(c.name,'other' if relative or (signature and 'other' in signature) else ''))
            snippet=ast.get_source_segment(source,n) if n else None
            entries.append(dict(field=c.name,category='object_state',concrete_value_implementation=concrete,bases=bases,
                source=str(p),line=c.lineno,api=extraction,signature=signature,meaning=ast.get_docstring(c),
                dtype=dtype,shape=shape,unit=unit,coordinate_frame='world/scene world for geometric states; relation to explicit other object for relative predicates; see implementation',
                initialized_Task40_objects=attached,stateful_Task40_objects=stateful,
                official_data=dict(parquet=False,hdf5='serialized stateful values only' if stateful else 'derived geometry/interface, not direct saved value'),
                realtime_only=not bool(stateful),strict_demo_alignment='recorded state[t] if serialized; derived state requires asset/schema/cache alignment validation',
                verification='initialized_interface_and_initial_record' if sample else 'initialized_interface_only' if attached else 'source_definition_only',
                actual_examples=sample[:3],observed_range=None,value_implementation=snippet))
    def add(field,category,api,source,dtype,shape,unit,frame,saved,alignment,example=None,verified='source_definition_only'):
        entries.append(dict(field=field,category=category,api=api,source=str(OG/source),dtype=dtype,shape=shape,unit=unit,
            coordinate_frame=frame,official_data=saved,realtime_only=not bool(saved),strict_demo_alignment=alignment,
            actual_examples=example,verification=verified,observed_range=None))
    robot=scene['state']['registry']['object_registry']['robot_r1']
    for field,n,unit in [('joint_pos',28,'rad for revolute / m for prismatic'),('joint_vel',28,'rad/s or m/s')]:
        add('robot.'+field,'robot',f'robot.get_joint_{"positions" if field=="joint_pos" else "velocities"}()', 'robots/robot_base.py','float32',[n],unit,'joint-local',True,'raw256 and state[t]',robot[field],'all200_raw256_verified')
    for field,n,unit in [('pos',3,'m'),('ori',4,'quaternion_xyzw'),('lin_vel',3,'m/s'),('ang_vel',3,'rad/s')]:
        examples=[dict(object=name,value=s['root_link'].get(field)) for name,s in scene['state']['registry']['object_registry'].items() if name in ('robot_r1','microwave_hjjxmi_0','popcorn_bag_73')]
        add('object.root.'+field,'object_physics',{'pos':'obj.get_position_orientation()[0]','ori':'obj.get_position_orientation()[1]', 'lin_vel':'obj.get_linear_velocity()','ang_vel':'obj.get_angular_velocity()'}[field],
            'prims/rigid_prim.py','float32',[n],unit,'world (scene pose transform for serialized values)',True,'state[t], carry skipped sleeping object unchanged; stabilized skipped velocities require care',examples,'original_snapshot_verified')
    for field,api,dtype,shape in [('link_pose','link.get_position_orientation()','tuple[tensor,tensor]',[[3],[4]]),('link_velocity','link.get_linear_velocity(), link.get_angular_velocity()','tensor',[6]),('joint_effort','robot.get_joint_efforts()','tensor',[28])]:
        add(field,'robot',api,'utils/usd_utils.py',dtype,shape,'m/rad/s/N/Nm by field','world/joint',field=='joint_effort','raw256 qeffort112:140; link poses FK reproducible, velocities live')
    add('bag_grasped','interaction','robot.is_grasping(arm="left/right",candidate_obj=bag)','robots/manipulation_robot.py','bool',[],'1','object identity',False,
        'unknown in this delivery: no per-frame assisted constraint serialization, grasp keys excluded from256; do not use finger closure as truth')
    add('contact_impulse','interaction','ContactBodies / Touching; RigidContactAPI','utils/usd_utils.py','tensor/dict',['contacts',3],'N*s/physics-step depending API','world',False,'requires contact collection at original physics step, not recovered from q alone')
    add('goal_status','task','env.task._termination_conditions["predicate"].goal_status; info["goal_status"]','termination_conditions/predicate_goal.py','dict',{ 'satisfied':'list[int]','unsatisfied':'list[int]'},'1','BDDL goal',False,'step result after action; not direct observation[t] field')
    add('task_success','task','env.task.success; evaluate_goal_conditions(env.task.activity_goal_conditions)','tasks/behavior_task.py','bool',[],'1','BDDL goal',False,'runtime predicate caches and transitions required; raw terminated is not interchangeable with success')
    add('q_score.final','evaluation','TaskMetric.start_callback/end_callback/gather_results','metrics/task_metric.py','float',[],'[0,1]','goal predicate options',False,'episode metric, baseline initial predicates and final state needed')
    add('task_progress','evaluation','info["task_progress"]; CHALLENGE_TASKS_PROGRESS_APPROXIMATION[activity](env)','learning/wrappers/task_progress_wrapper.py','dict[str,bool]',['task-specific'],'1','task predicate / near distance',False,'approximation checks, not Q score or exclusive stage')
    for name in ('reward','terminated','truncated'):
        add('raw.'+name,'recording','hdf5["data"][matched_demo][field][t]','envs/data_wrapper.py','float32' if name=='reward' else 'bool',[],'task-dependent' if name=='reward' else '1','step outcome',True,'action[t] outcome; state[t]/RGB[t] are before action[t]',verified='all200_exported')
    source=(OG/'sensors/vision_sensor.py').read_text();tree=ast.parse(source)
    modalities=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALL_MODALITIES' for t in node.targets):modalities=ast.literal_eval(node.value)
    for name in modalities:
        add('sensor.'+name,'sensor',f'robot.sensors[name].get_obs()[0]["{name}"]','sensors/vision_sensor.py',
            'uint8' if name=='rgb' else 'source-defined tensor/dict',['H','W',3] if name=='rgb' else None,
            'pixel0..255' if name=='rgb' else 'see modality implementation','camera/image/world by modality',name=='rgb',
            'RGB video PTS=t/30; only3 RGB cameras saved in investigated converted schema',verified='all200_RGB_metadata' if name=='rgb' else 'source_definition_only')
    tasksource=OG/'learning/utils/task_progress_utils.py'
    tasktext=tasksource.read_text();tasktree=ast.parse(tasktext)
    tasknames=[]
    for n in ast.walk(tasktree):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHALLENGE_TASKS_PROGRESS_APPROXIMATION' for t in n.targets):tasknames=[ast.literal_eval(k) for k in n.value.keys]
    target=OUT/'samples/source_evidence';target.mkdir(parents=True,exist_ok=True)
    paths=list((OG/'object_states').glob('*.py'))+[OG/x for x in ['metrics/task_metric.py','termination_conditions/predicate_goal.py','tasks/behavior_task.py','learning/utils/task_progress_utils.py','learning/wrappers/task_progress_wrapper.py','learning/utils/eval_utils.py','robots/r1.py','robots/r1pro.py','robots/holonomic_base_robot.py','robots/manipulation_robot.py','utils/registry_utils.py','scenes/scene_base.py','prims/entity_prim.py','controllers/multi_finger_gripper_controller.py','controllers/holonomic_base_joint_controller.py']]
    hashes=[]
    for p in paths:
        rel=p.relative_to(OG);dest=target/'omnigibson'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        hashes.append(dict(path=str(p),bundle=str(dest.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    bddl=OG.parent.parent/'bddl/bddl/activity_definitions/make_microwave_popcorn/problem0.bddl'
    shutil.copy2(bddl,target/'task40_problem0.bddl')
    inventory=dict(scope='all discovered object state classes plus robot/object/task/evaluation/sensor fields; source-only entries never imply verified values',
        statuses=['all200_raw256_verified','original_snapshot_verified','initialized_interface_and_initial_record','initialized_interface_only','source_definition_only'],
        task40_object_bindings=scene['metadata']['task']['inst_to_name'],entries=entries,
        initialized_objects=schemas,other_tasks=dict(generic_api='object_scope; obj.states[State].get_value(other); BehaviorTask goal_status; common R1Pro proprio',
            progress_supported_tasks=tasknames,count=len(tasknames),verification='source dictionary enumerated, other task runtime not launched'),
        sources=hashes)
    (OUT/'state_inventory.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(entries=len(entries),object_state_entries=sum(x['category']=='object_state' for x in entries),other_tasks=len(tasknames))))


if __name__=='__main__':main()
