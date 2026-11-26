"""Extract actual USD physics tree, retain URDF differences, test FK against demos.

Run with b1k2025-sim/bin/python; imports pxr only, does not start Isaac.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'v2_1_preflight'
ASSET=ROOT/'assets/simulation/omnigibson-robot-assets/models/r1pro'


def write(name,value):
    (OUT/name).parent.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def finite(x):
    if x is None:return None
    x=float(x)
    return x if np.isfinite(x) else ('+inf' if x>0 else '-inf')


def transform(pos,quat):
    t=np.eye(4);t[:3,:3]=Rotation.from_quat(quat).as_matrix();t[:3,3]=pos
    return t


def fk(joints,q,root='base_footprint_x'):
    poses={root:np.eye(4)};pending=list(joints)
    while pending:
        ready=[j for j in pending if j['parent'] in poses]
        if not ready:raise ValueError([j['name'] for j in pending])
        for j in ready:
            motion=np.eye(4);axis=np.eye(3)['XYZ'.index(j['axis'] or 'X')]
            if j['type']=='PhysicsRevoluteJoint':motion[:3,:3]=Rotation.from_rotvec(axis*q.get(j['name'],0)).as_matrix()
            if j['type']=='PhysicsPrismaticJoint':motion[:3,3]=axis*q.get(j['name'],0)
            poses[j['child']]=poses[j['parent']]@transform(j['pos0'],j['quat0_xyzw'])@motion@np.linalg.inv(transform(j['pos1'],j['quat1_xyzw']))
            pending.remove(j)
    return poses


def main():
    dest=OUT/'robot_description';dest.mkdir(parents=True,exist_ok=True)
    path=ASSET/'usd/r1pro.usda';stage=Usd.Stage.Open(str(path))
    joints=[];links=[]
    def quat(q):return list(q.GetImaginary())+[q.GetReal()]
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):links.append(dict(name=p.GetName(),path=str(p.GetPath())))
        if not p.IsA(UsdPhysics.Joint):continue
        j=UsdPhysics.Joint(p)
        targets=lambda rel:[str(x).split('/')[-1] for x in rel.GetTargets()]
        a,b=targets(j.GetBody0Rel()),targets(j.GetBody1Rel())
        def attr(n):return p.GetAttribute(n).Get() if p.HasAttribute(n) else None
        joints.append(dict(name=p.GetName(),path=str(p.GetPath()),type=p.GetTypeName(),
            parent=a[0] if a else 'world',child=b[0] if b else 'world',axis=attr('physics:axis'),
            pos0=list(j.GetLocalPos0Attr().Get()),pos1=list(j.GetLocalPos1Attr().Get()),
            quat0_xyzw=quat(j.GetLocalRot0Attr().Get()),quat1_xyzw=quat(j.GetLocalRot1Attr().Get()),
            lower_native=finite(attr('physics:lowerLimit')),upper_native=finite(attr('physics:upperLimit')),
            native_unit='degree' if p.IsA(UsdPhysics.RevoluteJoint) else 'metre' if p.IsA(UsdPhysics.PrismaticJoint) else None,
            max_velocity_native=finite(attr('physxJoint:maxJointVelocity')),
            effort_limit=finite(attr('drive:angular:physics:maxForce') if p.IsA(UsdPhysics.RevoluteJoint) else attr('drive:linear:physics:maxForce')),
            runtime_virtual=p.GetName().startswith('base_footprint_')))
    for j in joints:
        j['limits_SI']=[np.deg2rad(x) if j['native_unit']=='degree' and isinstance(x,(float,int)) else x
                        for x in [j['lower_native'],j['upper_native']]]
        j['velocity_limit_SI']=np.deg2rad(j['max_velocity_native']) if j['native_unit']=='degree' and isinstance(j['max_velocity_native'],(float,int)) else j['max_velocity_native']
    provenance=dict(actual_usd=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    md5=hashlib.md5(path.read_bytes()).hexdigest(),size=path.stat().st_size,
                    meters_per_unit=stage.GetMetadata('metersPerUnit'),up_axis=stage.GetMetadata('upAxis'))
    urdf_diffs=[]
    for name in ['r1pro.urdf','r1pro_with_meta_links.urdf','r1pro_source.urdf']:
        src=ASSET/'urdf'/name
        if src.exists():
            shutil.copy2(src,dest/name)
            u={j.attrib['name']:j.attrib for j in ET.parse(src).getroot().findall('joint')}
            for j in joints:
                if j['name'] in u and ('Fixed' in j['type']) != (u[j['name']]['type']=='fixed'):
                    urdf_diffs.append(dict(urdf=name,joint=j['name'],usd_type=j['type'],urdf_type=u[j['name']]['type']))
    # A small asset derived from original physics joint prims, without copyrighted mesh payload.
    lite=Usd.Stage.CreateInMemory()
    lite.SetMetadata('metersPerUnit',stage.GetMetadata('metersPerUnit'));lite.SetMetadata('upAxis',stage.GetMetadata('upAxis'))
    lite.DefinePrim('/r1pro','Xform')
    for l in links:
        p=lite.DefinePrim(l['path'],'Xform');UsdPhysics.RigidBodyAPI.Apply(p)
    for j in joints:
        p=stage.GetPrimAtPath(j['path']);target=lite.DefinePrim(j['path'],j['type'])
        for a in p.GetAttributes():
            if a.GetName().startswith(('physics:','physxJoint:','drive:')) and a.HasAuthoredValue():
                target.CreateAttribute(a.GetName(),a.GetTypeName()).Set(a.Get())
        for r in p.GetRelationships():
            if r.GetName().startswith('physics:'):target.CreateRelationship(r.GetName()).SetTargets(r.GetTargets())
    lite.GetRootLayer().Export(str(dest/'r1pro_kinematics.usda'))
    robot=dict(provenance=provenance,links=links,joints=joints,counts=dict(links=len(links),joints=len(joints),
        nonfixed=sum('Fixed' not in j['type'] for j in joints)),urdf_differences=urdf_diffs,
        runtime_root_fixed_joint='world -> base_footprint_x: runtime rootJoint, initialized root pose; holonomic_base_robot.py',
        coupling='Each smooth gripper command drives two independent prismatic joints equally; no URDF mimic coupling assumed')
    write('joint_topology.json',robot);write('robot_description/asset_provenance.json',provenance)
    graph='digraph R1Pro { rankdir=LR;\n'+''.join('"%s" -> "%s" [label="%s\\n%s %s"];\n'%(j['parent'],j['child'],j['name'],j['type'].replace('Physics','').replace('Joint',''),j['axis'] or '') for j in joints)+'}\n'
    (dest/'topology.dot').write_text(graph)
    try:subprocess.run(['dot','-Tsvg',str(dest/'topology.dot'),'-o',str(dest/'topology.svg')],check=True)
    except FileNotFoundError:pass
    (dest/'topology.mmd').write_text('flowchart LR\n'+''.join('  %s -->|%s| %s\n'%(j['parent'],j['name'],j['child']) for j in joints))
    # Stored eef poses come from actual PhysX link transforms, independently of this FK.
    names=['base_footprint_'+c+'_joint' for c in ('x','y','z','rx','ry','rz')]+['torso_joint'+str(i) for i in range(1,5)]+[f'{side}_arm_joint{i}' for i in range(1,8) for side in ('left','right')]+[f'{side}_gripper_finger_joint{i}' for side in ('left','right') for i in (1,2)]
    results=[];samples=[]
    rawdir=ROOT/'assets/datasets/b1k2025/data/task-0040'
    for ep in (400010,400020,400030,401000,402500,403000):
        p=rawdir/f'episode_{ep:08d}.parquet'
        if not p.exists():p=next((ROOT/'assets/datasets/b1k2025/data').glob(f'**/episode_{ep:08d}.parquet'))
        frame=pd.read_parquet(p,columns=['observation.state']);state=np.stack(frame['observation.state'])
        for t in np.unique(np.linspace(0,len(state)-1,25).astype(int)):
            row=state[t];qs=dict(zip(names,map(float,row[:28])));poses=fk(joints,qs)
            for side,pos_i,quat_i in [('left',186,189),('right',225,228)]:
                # ViewAPI.get_position_orientation explicitly substitutes the moving
                # base footprint link for the articulation root. R1 names it base_link.
                pred=np.linalg.inv(poses['base_link'])@poses[side+'_eef_link'];truth=transform(row[pos_i:pos_i+3],row[quat_i:quat_i+4])
                pe=float(np.linalg.norm(pred[:3,3]-truth[:3,3]));re=float(Rotation.from_matrix(pred[:3,:3].T@truth[:3,:3]).magnitude())
                results.append(dict(episode=ep,frame=int(t),side=side,position_error_m=pe,rotation_error_rad=re))
            if t in (0,len(state)-1):samples.append(dict(episode=ep,frame=int(t),joint_qpos=row[:28].tolist(),
                reference='base_link',links={k:dict(position_m=v[:3,3].tolist(),quaternion_xyzw=Rotation.from_matrix(v[:3,:3]).as_quat().tolist()) for k,a in poses.items() for v in [np.linalg.inv(poses['base_link'])@a]},
                recorded_eef_left=row[186:193].tolist(),recorded_eef_right=row[225:232].tolist()))
    summary={k:dict(max=max(r[k] for r in results),mean=float(np.mean([r[k] for r in results]))) for k in ('position_error_m','rotation_error_rad')}
    write('fk_validation.json',dict(method='Tchild=Tparent*T_local0*motion(q)*inverse(T_local1); USD anchors, SI q',
        reference='Moving base_link (R1 base_footprint_link_name); ViewAPI.get_position_orientation substitutes it for root, usd_utils.py:941',
        root_world='world_T_base_link from robot_pos and atan2(robot_ori_sin,robot_ori_cos) Euler XYZ; world_T_root requires inverse virtual-base FK',
        original_state_dim=256,requires='raw28 for full tree including virtual chain; packed82 torso4+arms14+fingers4 suffice for every descendant of base_link',
        comparisons=len(results),episodes=sorted(set(r['episode'] for r in results)),summary=summary,errors=results,
        passed=summary['position_error_m']['max']<0.005 and summary['rotation_error_rad']['max']<0.01,
        threshold='Diagnostic 5mm/0.01rad, actual max reported; no model performance conclusion',
        sim_ground_truth='Official observed eef poses computed from PhysX get_link_transforms during original recordings; no newly replayed observation',
        limitations='All links numerically calculable, independent recorded truth only for two eef links; recorded82 eef duplicates FK given packed torso/arms. Initial root-relative diagnostic failed (max4.48m); source tracing resolved frame mismatch.'))
    write('samples/fk_samples.json',samples)
    print(json.dumps(dict(joints=len(joints),links=len(links),fk=summary)))


if __name__=='__main__':main()
