"""Compare CPU reconstruction to actual OG predicate methods without Isaac.

Methods are compiled unchanged from local source. Lightweight objects supply
derived geometry/recorded poses: this is a source adapter check, NOT an original
runtime-state or replay certification.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
import torch as th
from label_api import transform_points

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'v2_1_preflight/label_extension'
OG=Path('/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson')


def actual_method(path,cls,name,namespace):
    source=path.read_text();tree=ast.parse(source)
    node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]


def main():
    th.set_num_threads(1)
    namespace=dict(th=th,AABB='AABB',PrimType=SimpleNamespace(CLOTH='cloth'),
        macros=SimpleNamespace(object_states=SimpleNamespace(contains=SimpleNamespace(CONTAINER_META_LINK_TYPES=['fillable','openfillable']))))
    paths=[OG/'object_states/inside.py',OG/'prims/geom_prim.py',OG/'prims/rigid_prim.py']
    inside=actual_method(paths[0],'Inside','_get_value',namespace)
    checklocal=actual_method(paths[1],'GeomPrim','check_local_points_in_volume',namespace)
    checkworld=actual_method(paths[1],'GeomPrim','check_points_in_volume',namespace)
    checklink=actual_method(paths[2],'RigidPrim','check_points_in_volume',namespace)
    data=pd.read_parquet(OUT/'task40_labels.parquet',columns=['episode_index','frame_index','inside_geometry_margin_m','bag_inside','bag_inside_quality'])
    episodes=json.loads((OUT/'episode_evidence.json').read_text());geo=np.load(OUT/'predicate_geometry.npz');results=[]
    geometry=SimpleNamespace(_mesh_type='Mesh',delaunay_triangulation=Delaunay(geo['microwave_container_hull']))
    geometry.check_local_points_in_volume=lambda points:checklocal(geometry,points)
    geometry.check_points_in_volume=lambda points:checkworld(geometry,points)
    link=SimpleNamespace(is_meta_link=True,meta_link_type='fillable',visual_meshes={'mesh':geometry})
    link.check_points_in_volume=lambda points:checklink(link,points)
    for item in episodes:
        eid=item['episode'];g=data[data.episode_index==eid].reset_index(drop=True)
        with np.load(ROOT/'v2_1_preflight/samples/decoded'/f'episode_{eid:08d}.npz') as z:bag=z['bag_pose'];micro=z['microwave_pose']
        # Three frames per episode, including the closest geometric boundary.
        selected=sorted(set([0,len(g)-1,int(np.abs(g.inside_geometry_margin_m).argmin())]))
        bs=item['asset_args']['bag']['scale'];ms=item['asset_args']['microwave']['scale']
        bp=transform_points(geo['bag_collision_hull'],bag[selected],bs).astype(np.float32)
        mp=transform_points(geo['microwave_base_collision_hull'],micro[selected],ms).astype(np.float32)
        for i,t in enumerate(selected):
            lower=th.tensor(bp[i].min(0));upper=th.tensor(bp[i].max(0))
            outerlower=th.tensor(mp[i].min(0));outerupper=th.tensor(mp[i].max(0))
            state=SimpleNamespace(obj=SimpleNamespace(states={'AABB':SimpleNamespace(get_value=lambda:(lower,upper))}))
            other=SimpleNamespace(prim_type='rigid',states={'AABB':SimpleNamespace(get_value=lambda:(outerlower,outerupper))},links={'fillable':link})
            matrix=np.eye(4);matrix[:3,:3]=Rotation.from_quat(micro[t,3:7]).as_matrix()@np.diag(ms);matrix[:3,3]=micro[t,:3]
            geometry.scaled_transform=th.tensor(matrix,dtype=th.float32)
            actual=bool(inside(state,other));candidate=bool(g.inside_geometry_margin_m.iloc[t]>=0)
            results.append(dict(episode=eid,frame=t,source_method_result=actual,cpu_candidate=candidate,
                margin_m=float(g.inside_geometry_margin_m.iloc[t]),quality=int(g.bag_inside_quality.iloc[t]),equal=actual==candidate))
    valid=[r for r in results if r['quality']>0]
    assert all(r['equal'] for r in valid)
    report=dict(passed=True,kind='unchanged_source_methods_with_geometry_adapters_not_live_simulation',
        comparisons=len(results),matches=sum(r['equal'] for r in results),valid_comparisons=len(valid),valid_matches=sum(r['equal'] for r in valid),
        source_files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        limitation='Does not establish original asset identity, PhysX pose agreement, contacts or rendering consistency; no simulator started.',results=results)
    (OUT/'inside_source_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','source_files')}))


if __name__=='__main__':main()
