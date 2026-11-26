"""Publish schema, separate masks and source references for the label extension."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'v2_1_preflight';OUT=BASE/'label_extension'
OG=Path('/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson/omnigibson')


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    # Standardize derived diagnostic scalars to source-observation precision.
    labels=pd.read_parquet(OUT/'task40_labels.parquet')
    floats=[n for n in labels if n.startswith('grasp_') and labels[n].dtype==np.float64]
    if floats:
        labels[floats]=labels[floats].astype(np.float32)
        temporary=OUT/'task40_labels.finalizing.parquet'
        labels.to_parquet(temporary,index=False,compression='zstd')
        temporary.replace(OUT/'task40_labels.parquet')
    definitions={
        'door_open':('当前Open门谓词：j_link_0 > 0.07853977382183075 rad；阈值±1e-5未知','recorded_joint_plus_og_open'),
        'started':('当前ToggledOn；沿用原认证±8帧未知边界；不是历史启动','recorded_toggled_on'),
        'cooked_popcorn_exists':('当前cooked__popcorn系统粒子数>0；不是完整任务成功','recorded_system_count'),
        'raw_popcorn_exists':('当前popcorn系统粒子数>0','recorded_system_count'),
        'bag_inside':('袋世界碰撞AABB中心位于微波炉fillable容器凸体；资产原始身份未认证；边界±2mm未知','geometry_weak'),
        'bag_grasped_left':('左EEF共同运动+相对位姿稳定+夹爪开度的因果弱抓取；含有证据的静止保持','grasp_motion_weak'),
        'bag_grasped_right':('右EEF共同运动+相对位姿稳定+夹爪开度的因果弱抓取；含有证据的静止保持','grasp_motion_weak'),
        'bag_grasped':('左右弱抓取的三值OR；不等于辅助抓取约束真值','bag_grasped_left OR bag_grasped_right'),
        'door_closed':('当前NOT door_open；不另设闭门物理阈值','NOT door_open'),
        'door_ever_opened':('从本episode frame0到当前曾观察到门打开；不追溯采集前历史','EVER door_open'),
        'bag_ever_inside':('从frame0到当前曾有Inside弱正例；不会升级为认证','EVER bag_inside'),
        'bag_ever_grasped':('从frame0到当前曾有抓取弱正例','EVER bag_grasped'),
        'microwave_ever_started':('从frame0到当前曾有有效开机真值','EVER started'),
        'cooked_popcorn_ever_observed':('从frame0到当前曾观察到熟粒子','EVER cooked_popcorn_exists'),
        'door_reclosed_after_open':('本episode曾开门且当前门关闭；当前复合状态','door_ever_opened AND door_closed'),
        'door_ever_reclosed_after_open':('曾满足开门后关闭；单调历史事件','EVER door_reclosed_after_open'),
        'bag_inside_with_door_closed':('当前Inside候选且门关闭','bag_inside AND door_closed'),
        'bag_released_inside_candidate':('当前Inside候选且弱抓取为false；不能等同真实释放/支撑','bag_inside AND NOT bag_grasped'),
        'microwave_on_with_closed_door':('当前开关为on且门关闭；不等于热源实际运行','started AND door_closed'),
        'microwave_on_with_open_door':('当前开关为on且门打开；本数据所有帧可认证为false','started AND door_open'),
        'bag_inside_while_on':('当前Inside候选且开关为on','bag_inside AND started')}
    assert set(definitions)==set(summary['states'])
    schema=dict(version='task40_label_extension_v002',primary_key=['episode_index','frame_index'],
        frames=summary['frames'],episodes=summary['episodes'],timestamp='timestamp_s = frame_index / 30; observation/state before action',
        value_dtype='nullable boolean',quality_dtype='uint8',quality_codes={'0':'unknown','1':'certified recorded/source predicate','2':'weak inference'},
        mask_semantics={'_valid':'quality>0; includes weak; NOT a strict certification mask',
            '_certified_valid':'quality==1; strict certified targets only','_weak_valid':'quality==2; explicit weak targets only'},
        unknown='Value is null; quality=0; all three masks false. Never fill null with false without preserving mask.',
        derived_logic='Three-valued logic, quality follows sufficient evidence. Certified false conjunct can certify a false composite despite weak/unknown other terms.',
        history='Causal OR of prefix; known false only if entire prefix known false. Certified true witness overrides earlier weak/unknown. No across-episode carry.',
        states=[dict(name=n,meaning=definitions[n][0],implementation=definitions[n][1],
            columns=[n,n+'_quality',n+'_valid',n+'_certified_valid',n+'_weak_valid']) for n in summary['states']],
        preserved_numerical_fields=['microwave_door_joint_rad','microwave_temperature_c','bag_temperature_c','cooked_popcorn_particle_count','raw_popcorn_particle_count'],
        extra_evidence_fields={'inside_geometry_margin_m':'signed inward distance in metres; abs<=0.002 unknown',
            'grasp_{left,right}_distance_m':'EEF origin to bag root world distance, metres',
            'grasp_{left,right}_relative_error_m':'max relative-position deviation from current frame over past 10-frame window',
            'grasp_{left,right}_relative_error_rad':'max relative-orientation deviation over past window',
            'grasp_{left,right}_{bag,eef}_motion_m':'endpoint displacement over past window',
            'grasp_{left,right}_reason':{'0':'insufficient evidence','1':'co-motion witness','2':'hold witnessed relative transform','3':'far weak negative','4':'both fingers open weak negative'}},
        not_certified_or_unavailable=['original asset exact identity','AG runtime attachment/grasp','contact/support/OnTop countertop','full BDDL Contains/Real success','actual heat-source activity'],
        raw_outcomes='raw_reward/raw_terminated/raw_truncated[t] are POST-action t outcomes; not observation-t physical predicate labels.',
        reusable_api='../scripts/label_api.py; per-task role/asset/coordinate adapters and config remain explicit')
    (OUT/'label_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
    masks=['episode_index','frame_index']+[n+s for n in summary['states'] for s in ('_quality','_valid','_certified_valid','_weak_valid')]
    labels[masks].to_parquet(OUT/'task40_masks.parquet',index=False,compression='zstd')
    sources=[]
    for relative in ['object_states/inside.py','object_states/contains.py','object_states/open_state.py','object_states/toggle.py',
            'robots/manipulation_robot.py','prims/geom_prim.py','prims/rigid_prim.py','objects/usd_object.py','utils/asset_utils.py']:
        source=OG/relative
        dest=OUT/'source_evidence'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        sources.append(dict(original=str(source),bundle=str(dest.relative_to(OUT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    episodes=json.loads((OUT/'episode_evidence.json').read_text())
    stats=pd.read_csv(OUT/'label_statistics.csv',dtype={'episode_index':str})
    report=dict(sources=sources,inside_candidate_change_distribution=pd.Series([len(e['events']['bag_inside']) for e in episodes]).value_counts().sort_index().to_dict(),
        grasp_candidate_change_distribution=pd.Series([len(e['events']['bag_grasped']) for e in episodes]).value_counts().sort_index().to_dict(),
        no_positive_grasp_episodes=stats[(stats.state=='bag_grasped')&(stats.episode_index!='ALL_200')&(stats.positive==0)].episode_index.tolist(),
        same_models_scales_and_expected_hashes_all200=True,
        expected_hashes={role:sorted({e['asset_args'][role]['expected_file_hash'] for e in episodes}) for role in ('bag','microwave')})
    for role in ('bag','microwave'):
        assert len(report['expected_hashes'][role])==1
        assert len({tuple(e['asset_args'][role]['scale']) for e in episodes})==1
    (OUT/'source_and_order_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(states=len(definitions),masks_bytes=(OUT/'task40_masks.parquet').stat().st_size,sources=len(sources))))


if __name__=='__main__':main()
