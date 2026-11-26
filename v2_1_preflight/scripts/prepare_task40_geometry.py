"""CPU-only asset comparison and minimal derived predicate geometry; no Isaac import.

Encrypted originals and decrypted artwork remain outside the public repository.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile
from cryptography.fernet import Fernet
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'v2_1_preflight/label_extension'
CACHE=Path('/nvme3/datasets/task40_asset_recovery')
REV='015e37a26823a110fedb6a0c3d943f573de00ffe'


def hull(points):
    points=np.unique(np.asarray(points),axis=0)
    return points[ConvexHull(points).vertices]


def extract(path):
    stage=Usd.Stage.Open(str(path));cache=UsdGeom.XformCache()
    collisions={};containers=[];signature=[]
    for p in stage.Traverse():
        rel=str(p.GetPath()).split('/',2)[-1]
        if p.IsA(UsdGeom.Mesh):
            points=np.asarray(p.GetAttribute('points').Get(),dtype=np.float64)
            transform=np.asarray(cache.GetLocalToWorldTransform(p)).T
            points=points@transform[:3,:3].T+transform[:3,3]
            collision=p.HasAPI(UsdPhysics.CollisionAPI)
            signature.append((rel,'mesh',collision,points.tolist(),list(p.GetAttribute('faceVertexIndices').Get())))
            if collision:collisions.setdefault(rel.split('/')[0],[]).append(points)
            elif rel.startswith('meta__') and ('_fillable_' in rel or '_openfillable_' in rel):containers.append(points)
        if p.IsA(UsdPhysics.Joint):
            attrs={a.GetName():str(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValue() and a.GetName().startswith('physics:')}
            signature.append((rel,p.GetTypeName(),attrs))
    collision_hulls={k:hull(np.concatenate(v)) for k,v in collisions.items()}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    return collision_hulls,[hull(v) for v in containers],digest


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    fernet=Fernet((ROOT/'assets/simulation/omnigibson.key').read_bytes())
    scene=json.loads((ROOT/'v2_1_preflight/samples/scene_initial_state.json').read_text())
    records=[];models={}
    for role,category,model,instance in [('bag','popcorn_bag','hdcpqg','popcorn_bag_73'),('microwave','microwave','hjjxmi','microwave_hjjxmi_0')]:
        original=CACHE/REV/'objects'/category/model/'usd'/f'{model}.usdz.encrypted'
        current=ROOT/'assets/simulation/behavior-1k-assets/objects'/category/model/'usd'/f'{model}.encrypted.usd'
        legacy_dir=CACHE/REV/'objects'/category/model/'usd/private_unpacked';legacy_dir.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fernet.decrypt(original.read_bytes()))) as archive:
            legacy=legacy_dir/f'{model}.usd';legacy.write_bytes(archive.read(f'{model}.usd'))
        private=CACHE/'current_private'/f'{model}.usd';private.parent.mkdir(parents=True,exist_ok=True)
        private.write_bytes(fernet.decrypt(current.read_bytes()))
        a,ac,ah=extract(legacy);b,bc,bh=extract(private)
        expected=scene['objects_info']['init_info'][instance]['args']['expected_file_hash']
        record=dict(role=role,category=category,model=model,expected_encrypted_usd_md5=expected,
            recovered_version=(CACHE/REV/'VERSION').read_text().strip(),historical_encrypted_usdz_md5=hashlib.md5(original.read_bytes()).hexdigest(),
            current_encrypted_usd_md5=hashlib.md5(current.read_bytes()).hexdigest(),
            expected_hash_found=False,historical_plaintext_usd_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
            current_plaintext_usd_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),
            historical_physics_geometry_sha256=ah,current_physics_geometry_sha256=bh,
            all_mesh_world_vertices_faces_and_joint_physics_equal=ah==bh,
            collision_hull_max_difference_m={k:float(np.max(np.abs(a[k]-b[k]))) if a[k].shape==b[k].shape else None for k in a},
            container_count=len(ac),private_historical_path=str(legacy),private_current_path=str(private))
        records.append(record);models[role]=(a,ac)
    bag,_=models['bag'];micro,volumes=models['microwave']
    assert len(volumes)==1
    # The volume is inside the convex hull of base-link collisions. Thus every
    # volume-positive point necessarily passes Inside's outer world-AABB gate,
    # regardless of door angle or rigid object pose. No door-FK approximation.
    planes=ConvexHull(micro['base_link']).equations
    violation=float(np.max(volumes[0]@planes[:,:3].T+planes[:,3]))
    assert violation<=1e-7,violation
    np.savez_compressed(OUT/'predicate_geometry.npz',bag_collision_hull=hull(np.concatenate(list(bag.values()))),
        microwave_base_collision_hull=micro['base_link'],microwave_container_hull=volumes[0])
    report=dict(assets=records,volume_within_base_collision_convex_hull_max_plane_violation_m=violation,
        outer_aabb_gate_redundancy_proved_for_recovered_geometry=True,
        geometry_source='official historical 3.7.0rc23 and local 3.7.2rc1; neither encrypted USD hash matches original recording',
        certification='blocked_original_asset_identity; derived Inside candidate must remain weak',
        key_or_decrypted_artwork_published=False,simulation_started=False)
    (OUT/'geometry_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
