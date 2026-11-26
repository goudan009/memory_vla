"""Build original-video transition samples, range evidence and delivery checks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'v2_1_preflight'
DATA=ROOT/'assets/datasets/b1k2025_groot_task40_v1_200'
OG=Path('/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson')


def write(name,x):
    (OUT/name).parent.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    labels=pd.read_parquet(OUT/'task40_labels.parquet')
    stats=pd.read_csv(OUT/'label_statistics.csv')
    decode=json.loads((OUT/'samples/cpu_decode_report.json').read_text())
    episodes=[];ranges={};examples={};video_meta=[];images=[]
    for eid,df in labels.groupby('episode_index',sort=True):
        with np.load(OUT/'samples/decoded'/f'episode_{eid:08d}.npz') as z:
            events={}
            for label,arr in [('door_open',z['microwave_joint']>decode['open_joints'][0]['threshold']),('started',z['started']),('cooked_popcorn_exists',z['cooked_particle_count']>0)]:
                frames=np.flatnonzero(arr[1:]!=arr[:-1])+1
                events[label]=[dict(frame=int(t),value=bool(arr[t])) for t in frames]
            for k in z.files:
                arr=z[k].reshape(len(df),-1);lo=arr.min(0);hi=arr.max(0)
                if k not in ranges:ranges[k]=[lo,hi];examples[k]=z[k][0].tolist()
                else:ranges[k]=[np.minimum(lo,ranges[k][0]),np.maximum(hi,ranges[k][1])]
            episodes.append(dict(episode=int(eid),events=events,
                door_reopened_after_close=any(x['value'] for x in events['door_open'][2:]),
                started_then_off=any(not x['value'] for x in events['started']),
                started_while_door_open_frames=int(np.sum(z['started']&(z['microwave_joint']>decode['open_joints'][0]['threshold']))),
                cooked_at_final_observation=bool(z['cooked_particle_count'][-1]>0)))
        for camera in ('head','left_wrist','right_wrist'):
            video=DATA/'videos/task-0040'/f'observation.images.rgb.{camera}_256_256'/f'episode_{eid:08d}.mp4'
            with av.open(str(video)) as c:
                stream=c.streams.video[0]
                assert stream.frames==len(df) and stream.average_rate==30,(eid,camera,stream.frames,len(df))
                video_meta.append(dict(episode=int(eid),camera=camera,path=str(video),frames=stream.frames,
                    width=stream.width,height=stream.height,average_rate=str(stream.average_rate),time_base=str(stream.time_base)))
        if eid not in (400010,400020,403000):continue
        wanted=set()
        for label,ev in events.items():
            for event in ev:
                for d in (-16,16):wanted.add(max(0,min(len(df)-1,event['frame']+d)))
        for camera in ('head','left_wrist','right_wrist'):
            # Original-resolution source, before the V1 256x256 conversion.
            video=ROOT/'assets/datasets/b1k2025/videos/task-0040'/f'observation.images.rgb.{camera}'/f'episode_{eid:08d}.mp4'
            frames={}
            with av.open(str(video)) as c:
                stream=c.streams.video[0]
                for index,frame in enumerate(c.decode(stream)):
                    if index in wanted:
                        assert abs(float(frame.pts*stream.time_base)-index/30)<1e-6
                        rgb=frame.to_image();target=OUT/'samples/keyframes'/f'{eid}_{index:05d}_{camera}.png';target.parent.mkdir(parents=True,exist_ok=True);rgb.save(target)
                        row=df.iloc[index]
                        info=dict(episode=int(eid),frame=index,camera=camera,pts=frame.pts,time_base=str(stream.time_base),timestamp_s=index/30,
                            source=str(video),image=str(target.relative_to(OUT)),source_size=[rgb.width,rgb.height],
                            door_open=None if pd.isna(row.door_open) else bool(row.door_open),started=None if pd.isna(row.started) else bool(row.started),
                            started_recorded_value=bool(row.started_recorded_value),door_joint_rad=float(row.microwave_door_joint_rad),
                            cooked_particle_count=int(row.cooked_popcorn_particle_count))
                        images.append(info);frames[index]=(rgb,info)
                    if index>=max(wanted):break
            if camera=='head':
                columns=4;thumb_w=360;thumb_h=250;panel=Image.new('RGB',(columns*thumb_w,((len(frames)+columns-1)//columns)*thumb_h),'white');draw=ImageDraw.Draw(panel)
                for k,(index,(rgb,info)) in enumerate(sorted(frames.items())):
                    x=(k%columns)*thumb_w;y=(k//columns)*thumb_h;rgb.thumbnail((thumb_w,200));panel.paste(rgb,(x,y))
                    draw.text((x+4,y+202),f'ep{eid} frame{index} q={info["door_joint_rad"]:.4f} rad',fill='black')
                    draw.text((x+4,y+218),f'Open={info["door_open"]} On={info["started"]} cookedN={info["cooked_particle_count"]}',fill='black')
                panel.save(OUT/'samples'/f'certified_transition_panel_{eid}.png')
    summary=dict(episodes=len(episodes),door_transition_count_distribution=pd.Series([len(x['events']['door_open']) for x in episodes]).value_counts().sort_index().to_dict(),
        started_transition_count_distribution=pd.Series([len(x['events']['started']) for x in episodes]).value_counts().sort_index().to_dict(),
        door_reopen_episodes=[x['episode'] for x in episodes if x['door_reopened_after_close']],
        started_then_off_episodes=[x['episode'] for x in episodes if x['started_then_off']],
        cooked_absent_final_observation=[x['episode'] for x in episodes if not x['cooked_at_final_observation']],
        caution='Only Open/ToggledOn/particle existence order is known. Grasp and Inside order/repetitions remain unknown. FrameT extra terminal state excluded.')
    write('samples/transition_analysis.json',dict(summary=summary,episodes=episodes))
    write('samples/video_alignment_metadata.json',dict(videos=video_meta,keyframes=images,
        interpretation='PTS exact at saved source frames; visual spot-check supports door/button changes, images do not certify Inside or grasp'))
    write('samples/decoded_ranges.json',{k:dict(min=v[0].tolist(),max=v[1].tolist(),first_frame_example=examples[k]) for k,v in ranges.items()})
    # Expand portable source evidence for field semantics and decoder grammar.
    paths=['envs/data_wrapper.py','robots/robot_base.py','prims/rigid_prim.py','prims/rigid_dynamic_prim.py','prims/xform_prim.py',
        'objects/stateful_object.py','objects/controllable_object.py','controllers/controller_base.py','controllers/joint_controller.py',
        'systems/system_base.py','systems/macro_particle_system.py','utils/python_utils.py','utils/usd_utils.py','utils/processing_utils.py','sensors/vision_sensor.py']
    extra=[]
    for rel in paths:
        src=OG/rel;dest=OUT/'samples/source_evidence/omnigibson'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        extra.append(dict(source=str(src),bundle=str(dest.relative_to(OUT)),sha256=sha(src)))
    write('samples/additional_source_provenance.json',extra)
    inv=json.loads((OUT/'state_inventory.json').read_text())
    for entry in inv['entries']:
        field=entry['field']
        if field in ('Open','ToggledOn','Temperature','Pose','Joint'):
            key={'Open':'microwave_joint','ToggledOn':'started','Temperature':'temperature','Pose':'bag_pose','Joint':'microwave_joint'}[field]
            entry.update(observed_range=dict(min=ranges[key][0].tolist(),max=ranges[key][1].tolist()),
                per_frame_evidence='samples/decoded/*.npz; decoded_ranges.json',verification='all200_recorded_source_decoded')
            entry['actual_examples'].append(dict(source='samples/decoded/episode_00400010.npz',field=key,frame=0,value=examples[key]))
        if field.startswith('robot.joint_') or field=='joint_effort':
            a=json.loads((OUT/'samples/dataset_audit.json').read_text())['ranges']['raw_state256'];b,e=(0,28) if field.endswith('pos') else (84,112) if field.endswith('vel') else (112,140)
            entry['observed_range']=dict(min=a['min'][b:e],max=a['max'][b:e])
    inv['additional_sources']=extra
    write('state_inventory.json',inv)
    # Exact structural and validity gates; no network training tests.
    assert len(labels)==647555 and labels.episode_index.nunique()==200
    for field in ('door_open','bag_grasped','bag_inside','started','cooked_popcorn_exists'):
        assert np.array_equal(labels[field].notna(),labels[field+'_valid'])
        assert (stats.loc[stats.state==field,['positive','negative','unknown']].sum(axis=1)==stats.loc[stats.state==field,'frames']).all()
    assert labels.bag_grasped.isna().all() and labels.bag_inside.isna().all()
    mapping=json.loads((OUT/'state_action_mapping.json').read_text());top=json.loads((OUT/'joint_topology.json').read_text())
    assert [x['index'] for x in mapping['state_dimensions']]==list(range(82))
    assert [x['action_index'] for x in mapping['action_channels']]==list(range(23))
    assert len(top['joints'])==43 and len(top['links'])==44
    assert json.loads((OUT/'fk_validation.json').read_text())['passed']
    for item in decode['episodes']:assert sha(Path(item['artifact']))==item['sha256']
    write('samples/delivery_validation.json',dict(passed=True,episodes=200,frames=len(labels),state_dimensions=82,action_channels=23,joints=43,links=44,
        video_metadata_checked=len(video_meta),keyframe_pts_checked=len(images),official_comparison_frames=decode['official_compared_frames'],transition_summary=summary))
    print(json.dumps(dict(passed=True,summary=summary,keyframes=len(images))))


if __name__=='__main__':main()
