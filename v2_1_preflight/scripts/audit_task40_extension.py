"""Mask/history/statistics checks plus original multiview transition samples."""
import json
from pathlib import Path
import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from label_api import Label, conjunction, disjunction, ever
from build_task40_extension import statistics

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'v2_1_preflight/label_extension'


def algebra_checks():
    # Unknown past cannot certify "never"; a later certified positive can certify ever.
    x=Label([False,False,False,True,False,True],[1,0,1,2,1,1])
    h=ever(x)
    np.testing.assert_array_equal(h.value,[0,0,0,1,1,1])
    np.testing.assert_array_equal(h.quality,[1,0,0,2,2,1])
    for cut in range(1,len(x.value)+1):
        prefix=ever(Label(x.value[:cut],x.quality[:cut]))
        np.testing.assert_array_equal(prefix.value,h.value[:cut]);np.testing.assert_array_equal(prefix.quality,h.quality[:cut])
    unknown=Label([0],[0]);false=Label([0],[1]);true=Label([1],[1]);weaktrue=Label([1],[2])
    assert conjunction(unknown,false).quality[0]==1 and not conjunction(unknown,false).value[0]
    assert conjunction(unknown,true).quality[0]==0
    assert conjunction(weaktrue,true).quality[0]==2
    assert disjunction(unknown,true).quality[0]==1 and disjunction(unknown,true).value[0]
    assert disjunction(unknown,false).quality[0]==0


def main():
    algebra_checks()
    frame=pd.read_parquet(OUT/'task40_labels.parquet')
    summary=json.loads((OUT/'summary.json').read_text());states=summary['states']
    stats=pd.read_csv(OUT/'label_statistics.csv',dtype={'episode_index':str})
    evidence=json.loads((OUT/'episode_evidence.json').read_text());episodes={e['episode']:e for e in evidence}
    assert len(frame)==647555 and frame.episode_index.nunique()==200
    for eid,g in frame.groupby('episode_index',sort=True):
        np.testing.assert_array_equal(g.frame_index,np.arange(len(g)))
        assert np.max(np.abs(g.timestamp_s-np.arange(len(g))/30))<1e-4
        for name in states:
            q=g[name+'_quality'].to_numpy();v=g[name].fillna(False).to_numpy(bool)
            assert np.array_equal(g[name].notna(),q>0)
            assert np.array_equal(g[name+'_valid'],q>0)
            assert np.array_equal(g[name+'_certified_valid'],q==1)
            assert np.array_equal(g[name+'_weak_valid'],q==2)
            row=stats[(stats.episode_index==str(eid))&(stats.state==name)].iloc[0]
            actual=statistics(eid,name,Label(v,q))
            for field,value in actual.items():
                if field not in ('episode_index','state'):assert row[field]==value,(eid,name,field)
            assert row.positive+row.negative+row.unknown==len(g)
        for source,target in [('door_open','door_ever_opened'),('bag_inside','bag_ever_inside'),
                ('bag_grasped','bag_ever_grasped'),('started','microwave_ever_started'),('cooked_popcorn_exists','cooked_popcorn_ever_observed')]:
            history=ever(Label(g[source].fillna(False),g[source+'_quality']))
            np.testing.assert_array_equal(history.quality,g[target+'_quality'])
            known=history.quality>0;np.testing.assert_array_equal(history.value[known],g[target].fillna(False).to_numpy(bool)[known])
        assert not g.bag_inside_certified_valid.any() and not g.bag_grasped_certified_valid.any()
    old=pd.read_parquet(ROOT/'v2_1_preflight/task40_labels.parquet')
    for name in ('door_open','started','cooked_popcorn_exists'):
        pd.testing.assert_series_equal(frame[name],old[name]);pd.testing.assert_series_equal(frame[name+'_certified_valid'],old[name+'_valid'],check_names=False)
    regression=frame[frame.episode_index==400450].set_index('frame_index')
    assert bool(regression.loc[2069,'bag_grasped']) and regression.loc[2069,'bag_grasped_quality']==2
    assert not bool(regression.loc[2609,'bag_grasped']) and regression.loc[2609,'bag_grasped_quality']==2
    masks=pd.read_parquet(OUT/'task40_masks.parquet')
    pd.testing.assert_frame_equal(masks,frame[masks.columns])
    images=[];panels=[]
    selected_episodes=(400010,400020,403000,400450,400640)
    for eid in selected_episodes:
        g=frame[frame.episode_index==eid].reset_index(drop=True);e=episodes[eid];wanted={0}
        for name in ('bag_inside','bag_grasped'):
            for change in e['events'][name]:
                t=change['first_new_frame']
                for delta in (-20,20):wanted.add(max(0,min(len(g)-1,t+delta)))
        positive=np.flatnonzero(g.bag_grasped.fillna(False).to_numpy(bool))
        if len(positive):wanted.add(int(positive[len(positive)//2]))
        else:
            # Include an unsuccessful inference case; do not tune labels from it.
            wanted.add(int(g.grasp_right_bag_motion_m.fillna(0).argmax()))
            wanted.add(int(g.grasp_right_distance_m.argmin()))
        wanted=sorted(wanted)
        if eid==400450:wanted=sorted(set(wanted)|{2069,2609,2649})
        panel=Image.new('RGB',(1200,len(wanted)*330),'white');draw=ImageDraw.Draw(panel)
        for ci,camera in enumerate(('head','left_wrist','right_wrist')):
            video=ROOT/'assets/datasets/b1k2025/videos/task-0040'/f'observation.images.rgb.{camera}'/f'episode_{eid:08d}.mp4'
            with av.open(str(video)) as container:
                stream=container.streams.video[0]
                assert stream.frames==len(g) and stream.average_rate==30
                for index,decoded in enumerate(container.decode(stream)):
                    if index in wanted:
                        assert abs(float(decoded.pts*stream.time_base)-index/30)<1e-6
                        rgb=decoded.to_image();relative=f'samples/keyframes/{eid}_{index:05d}_{camera}.png'
                        dest=OUT/relative;dest.parent.mkdir(parents=True,exist_ok=True);rgb.save(dest)
                        images.append(dict(episode=eid,frame=index,camera=camera,pts=decoded.pts,time_base=str(stream.time_base),
                            source=str(video),image=relative,source_size=[rgb.width,rgb.height],timestamp_s=index/30))
                        rgb.thumbnail((400,250));y=wanted.index(index)*330;x=ci*400;panel.paste(rgb,(x,y))
                        row=g.iloc[index]
                        draw.text((x+5,y+252),f'{camera} ep{eid} t={index} ({index/30:.3f}s)',fill='black')
                        draw.text((x+5,y+270),f'grasp={row.bag_grasped} q={row.bag_grasped_quality} inside={row.bag_inside} q={row.bag_inside_quality}',fill='black')
                        draw.text((x+5,y+288),f'dR={row.grasp_right_distance_m:.3f}m relErr={row.grasp_right_relative_error_m:.4f}m',fill='black')
                        draw.text((x+5,y+306),f'Inside margin={row.inside_geometry_margin_m:.4f}m; on={row.started}',fill='black')
                    if index>=wanted[-1]:break
        panel_name=f'samples/multiview_{eid}.png';panel.save(OUT/panel_name);panels.append(panel_name)
    report=dict(passed=True,episodes=200,frames=len(frame),states=len(states),quality_masks_checked=True,
        certified_original_labels_unchanged=True,weak_never_promoted_to_certified=True,
        causal_history_unknown_propagation_checked=True,original_video_pts_checked=len(images),
        videos_checked=3*len(selected_episodes),images=images,panels=panels,
        limitation='Multiview samples are spot checks; no whole-dataset manual visual certification or original-asset predicate runtime validation.')
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('images','panels')}))


if __name__=='__main__':main()
