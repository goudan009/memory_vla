"""Summarize only sealed bounded diagnostic runs; not a task success benchmark."""
import argparse
import csv
import json
from pathlib import Path

from analyze_baseline_action_trace import analyze


def summarize(directory, mapping):
    trials=[]
    videos=[]
    for n in (4,8,16):
        for h in (8,16):
            name=f'N{n}_H{h}'
            jobfile=directory/(name+'_job.json')
            if not jobfile.exists():
                trials.append(dict(variant=name,status='NOT_STARTED'))
                continue
            job=json.loads(jobfile.read_text())
            root=Path(job['output'])
            resultfile=root/'results.json'
            if not resultfile.exists():
                trials.append(dict(variant=name,status='RUNNING',job_id=job['id'],output=str(root)))
                continue
            result=json.loads(resultfile.read_text())
            status=result['status']
            trial=dict(variant=name,inference_steps=n,execution_steps=h,status=status,
                job_id=job['id'],output=str(root),category='bounded_fault_diagnostic',
                completed_episodes=result.get('completed_episodes'),
                total_action_steps=result.get('total_action_steps'),
                elapsed_wall_s=result.get('elapsed_wall_s'),
                configuration=result.get('configuration'),
                per_gpu_performance={gpu:row.get('performance') for gpu,row in result.get('per_gpu',{}).items()},groups={})
            if status not in ('complete','completed'):
                trials.append(trial)
                continue
            stats=analyze(root,mapping)
            (directory/(name+'_trace_analysis.json')).write_text(json.dumps(stats,indent=2)+'\n')
            trial['chunks']=stats['chunks']
            for group,v in stats['groups'].items():
                trial['groups'][group]=dict(interior_mean=v['interior']['mean'],
                    boundary_mean=v['boundary']['mean'],boundary_p95=v['boundary']['p95'],
                    boundary_max=v['boundary']['max'],boundary_ratio=v['boundary_to_interior_mean_ratio'],
                    all_transition_mean=v['all_executed_transition_abs_delta_mean'],
                    boundary_variation_fraction=v['boundary_fraction_of_total_variation'],
                    out_of_limit_command_elements=v['limits'],command_elements=v['elements'],
                    tracking=v['tracking'])
            for video in sorted((root/'videos').rglob('*.mp4')):
                videos.append(dict(variant=name,job_id=job['id'],path=str(video)))
            trials.append(trial)
    summary=dict(category='bounded_fault_diagnostic',paired_noise_stream=False,
        task_success_rate_claim=False,trials=trials)
    (directory/'matrix_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    flat=[]
    for trial in trials:
        for group,values in trial.get('groups',{}).items():
            flat.append(dict(variant=trial['variant'],job_id=trial['job_id'],status=trial['status'],
                group=group,**{k:v for k,v in values.items() if k!='tracking'},
                tracking_mean=(values['tracking'] or {}).get('mean'),
                tracking_p95=(values['tracking'] or {}).get('p95')))
    for name,table in [('matrix_summary.csv',flat),('video_index.csv',videos)]:
        if table:
            with (directory/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps([dict(variant=t['variant'],status=t['status'],job=t.get('job_id')) for t in trials]))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--directory',type=Path,default=Path('Output/test_02/experiment_01/baseline_diagnostic'))
    p.add_argument('--mapping',type=Path,default=Path('v2_1_preflight/state_action_mapping.json'))
    args=p.parse_args();summarize(args.directory,args.mapping)
