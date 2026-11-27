"""plan-v005 continuation client for the existing workbench and systemd timer.

No private GPU queue or direct model/simulator launch. This file lives with the
experiment workflow, outside the immutable training source manifest.
"""
import fcntl
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import shlex
import subprocess
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from native_archive import ROOT, EXPERIMENT, EVIDENCE, RUN, TRAINING, archive, ssh, write

PORT = 5567
SOURCE = '/nvme3/projects/memory_vla/deployments/' + RUN
CHECKPOINT = TRAINING + '/' + RUN + '/run/checkpoints/' + RUN + '__step006000'
DIGEST = '53a2c0ab6c747c0264988ca0b6c53e12c433cbc7354d6c542bfb95158d818615'
TAG = RUN + '__step006000__official_eval'
ACTIVE = {'queued','waiting_for_gpus','submitted','initializing','training','validating','running'}
LIVE_SERVICES = {'ready','loading','starting','stopping'}
OPENER = build_opener(ProxyHandler({}))


def now():
    return datetime.now().astimezone()


def api(method, route, body=None):
    request = Request('http://127.0.0.1:8765' + route, method=method,
                      data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type':'application/json'})
    with OPENER.open(request, timeout=55) as response:
        data = json.load(response)
    if data.get('error'):
        raise ValueError(data['error'])
    return data


def save(state):
    write(EVIDENCE / 'watchdog_state.json', state)


def mutate(state, name, route, request):
    item = state.setdefault('actions', {}).get(name)
    if item:
        if item['status'] == 'done':
            return item['result']
        raise RuntimeError('Prior HTTP mutation uncertain; inspect jobs before retry: ' + name)
    item = dict(status='sending', request=request, at=now().isoformat())
    state['actions'][name] = item
    save(state)
    result = api('POST', route, request)
    item.update(status='done',result=result)
    write(EVIDENCE / (name + '-result.json'), result)
    save(state)
    return result


def evaluation_request():
    return dict(device='local',version='V_2_1',group='baseline',model_name='groot',
                mode='dual2',task=40,preset='official-eval',seed=0,seed_mode='sequence',
                policy_host='172.28.62.17',policy_port=PORT,run_tag=TAG,
                episodes_per_gpu=5,repetitions=1,instances_gpu0=list(range(5)),
                instances_gpu1=list(range(5,10)),action_steps=8,
                record_video=True,video_steps_per_render=24,wait_for_gpu=True)


def check_plan(plan):
    expected = dict(mode='dual2',preset='official-eval',task_id=40,base_seed=0,
                    seed_mode='sequence',episodes_per_gpu=5,repetitions=1,
                    n_envs_per_gpu=2,action_chunk=8,record_video=True,
                    video_steps_per_render=24,video_fps=20,run_tag=TAG,
                    version='V_2_1',group='baseline',
                    instances={'0':list(range(5)),'1':list(range(5,10))},
                    runner_max_episode_steps=999999999)
    for key,value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f'Locked evaluation protocol mismatch: {key}={plan.get(key)!r}')
    identity = plan.get('policy_identity') or {}
    for key,value in dict(checkpoint=CHECKPOINT,source_dir=SOURCE,source_digest=DIGEST,
                          training_id=RUN,binding='observed_process').items():
        if identity.get(key) != value:
            raise ValueError('Observed policy identity mismatch: ' + key)


def validate_result(output):
    result = json.loads((output / 'results.json').read_text())
    check_plan(result['configuration'])
    rows = [json.loads(line) for line in (output/'episodes.jsonl').read_text().splitlines() if line.strip()]
    # Native BEHAVIOR partitions each GPU's five instances contiguously, 2+3.
    # Native sequence seed is base + env_idx + local episode_number, not instance_id.
    expected = {(gpu,env,instance,env+number)
                for gpu in (0,1) for env,ids in [(0,range(gpu*5,gpu*5+2)),
                                               (1,range(gpu*5+2,gpu*5+5))]
                for number,instance in enumerate(ids)}
    observed = {(row.get('gpu_index'),row.get('env_idx'),row.get('instance_id'),row.get('seed')) for row in rows}
    if len(rows) != 10 or observed != expected or any(row.get('valid') is not True for row in rows):
        raise ValueError('Need exactly ten valid native dual2 episodes with original instance/seed assignment')
    for row in rows:
        native = row.get('native_termination') or {}
        conditions = native.get('termination_conditions') or {}
        if (row.get('evaluation_mode') != 'official-eval' or row.get('task_id') != 40
                or not (row.get('terminated') or row.get('truncated'))
                or type(native.get('success')) is not bool
                or not any(isinstance(value,dict) and value.get('done') for value in conditions.values())):
            raise ValueError('Episode lacks actual native terminal condition')
        if any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) for k in ('q_score','task_progress')):
            raise ValueError('Episode native metrics missing/nonfinite')
        video = (output/row.get('video_file','')).resolve()
        if not video.is_relative_to(output.resolve()) or video.suffix != '.mp4' or not video.is_file() or video.stat().st_size == 0:
            raise ValueError('Episode video missing/not closed')
    successes = sum(row['native_termination']['success'] for row in rows)
    p=successes/10;z=1.959963984540054;den=1+z*z/10
    center=(p+z*z/20)/den;half=z*math.sqrt(p*(1-p)/10+z*z/400)/den
    return dict(status='DONE',run=RUN,checkpoint=CHECKPOINT,checkpoint_source_digest=DIGEST,
                valid_episodes=10,successful_episodes=successes,success_rate=p,
                wilson95=[center-half,center+half],episodes=rows,output=str(output),
                mean_q_score=sum(r['q_score'] for r in rows)/10,
                mean_task_progress=sum(r['task_progress'] for r in rows)/10,
                configuration=result['configuration'])


def verified_checkpoint():
    code = '''
import hashlib,json
from pathlib import Path
path=Path(CHECKPOINT)
marker=json.loads((path/'COMPLETE.json').read_text())
if marker['step']!=6000 or marker['run_id']!=RUN or marker['source_digest']!=DIGEST or marker['world_size']!=4:
    raise RuntimeError('Checkpoint identity contract mismatch')
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
for name,digest in marker['files'].items():
    if sha(path/name)!=digest:raise RuntimeError('Checkpoint hash mismatch: '+name)
print(json.dumps(dict(checkpoint=str(path),verified=True,complete=marker)))
'''
    return ssh('\n'.join(f'{k}={v!r}' for k,v in [('CHECKPOINT',CHECKPOINT),('RUN',RUN),('DIGEST',DIGEST)])+'\n'+code,timeout=160)


def matches_service(job):
    return (str(job.get('checkpoint_resolved') or job.get('checkpoint')) == CHECKPOINT
            and job.get('source_dir') == SOURCE and job.get('history_mode','normal') == 'normal'
            and str(job.get('policy_seed',job.get('seed'))) == '42'
            and str(job.get('gpu_index')) == '0' and int(job.get('port',0)) == PORT
            and '--inference-steps 4' in job.get('command',''))


def observe_command(job):
    # The workbench rediscovers processes after its own restart; those records
    # need not retain the original start response's command field.
    code = '''
import json, shlex
from pathlib import Path
path=Path('/proc')/str(PID)
fields=(path/'stat').read_text().rsplit(')',1)[1].split()
if fields[0] in ('Z','X') or int(fields[19])!=START:
    raise RuntimeError('Service process identity changed')
argv=(path/'cmdline').read_bytes().decode().strip('\\0').split('\\0')
print(json.dumps(dict(command=shlex.join(argv))))
'''
    return dict(job,**ssh(f'PID={job["pid"]!r}\nSTART={job["start_ticks"]!r}\n'+code,timeout=20))


def eval_tick(state):
    jobs=api('GET','/api/launch/jobs')['jobs']
    found=[j for j in jobs if TAG in shlex.split(j.get('command',''))]
    outputs=[p.parent for p in (ROOT/'evaluation/V_2_1/baseline').glob('task40_eval*/results.json')
             if json.loads(p.read_text()).get('configuration',{}).get('run_tag') == TAG]
    if len(found)>1 or len(outputs)>1:
        raise RuntimeError('Duplicate current evaluation identity; no additional launch')
    job=found[0] if found else None
    if job:
        state['evaluation_job']=job
    output=outputs[0] if outputs else Path(job['output']) if job and job.get('output') else None
    if output and (output/'results.json').exists():
        result=json.loads((output/'results.json').read_text())
        state['evaluation_output']=str(output)
        state['completed_episodes']=result.get('completed_episodes',0)
        if result.get('status') in ('paused','pausing','interrupted'):
            state.update(status='user_paused',reason='Preserve actual evaluation pause');return
        if result.get('status') in ('complete','completed'):
            evidence=validate_result(output)
            write(EVIDENCE/'closed_loop_result.json',evidence)
            state.update(status='cleanup_inference',closed_loop_result=str(EVIDENCE/'closed_loop_result.json'))
            cleanup(state,jobs)
            return
        if result.get('status') == 'failed':
            raise RuntimeError('Evaluation failed; preserve real episodes and logs')
    if job:
        if job.get('state') in ('failed','stopped'):
            raise RuntimeError('Evaluation controller '+job['state'])
        state['status']='evaluating';return
    if output:
        raise RuntimeError('Evaluation output exists without workbench job; inspect before allocation')
    services=api('GET','/api/inference/jobs?device=a100')['jobs']
    on_port=[observe_command(j) for j in services if int(j.get('port',0))==PORT and j.get('state') in LIVE_SERVICES]
    if len(on_port)>1 or on_port and not matches_service(on_port[0]):
        raise RuntimeError('Current inference port is owned by a different configuration')
    if not on_port:
        if state.get('actions',{}).get('start-inference'):
            raise RuntimeError('Inference process disappeared; inspect load failure before retry')
        resources=api('GET','/api/services')
        write(EVIDENCE/'inference-resources.json',resources)
        remote=resources.get('remote',{})
        gpu=next((g for g in remote.get('gpus',[]) if g.get('index')==0),{})
        if not remote.get('ok') or gpu.get('total_mb',0)-gpu.get('used_mb',0)<24576:
            state['status']='waiting_for_inference_capacity';return
        request=dict(device='a100',checkpoint=CHECKPOINT,gpu_index=0,port=PORT,seed=42,
                     source_dir=SOURCE,python_path='/nvme3/projects/memory_vla/envs/n1d6-p0/bin/python',
                     history_mode='normal',inference_steps=4)
        preview=api('POST','/api/inference/preview',request)
        if preview.get('checkpoint')!=CHECKPOINT or preview.get('source_dir')!=SOURCE:
            raise RuntimeError('Inference preview checkpoint/source mismatch')
        write(EVIDENCE/'inference-preview.json',preview)
        mutate(state,'start-inference','/api/inference/start',request)
        state['status']='waiting_for_inference';return
    service=on_port[0];state['inference_job']=service
    if service.get('state')!='ready':
        state['status']='waiting_for_inference';return
    if any(j.get('state') in ('running','queued','loading') and set(map(str,j.get('gpus',[]))) & {'0','1'} for j in jobs):
        state['status']='waiting_for_simulation_slot';return
    request=evaluation_request()
    preview=api('POST','/api/launch/preview',request)
    check_plan(preview['plan'])
    write(EVIDENCE/'evaluation-preview.json',preview)
    state['evaluation_job']=mutate(state,'start-evaluation','/api/launch/start',request)
    state['status']='evaluating'


def cleanup(state,jobs):
    if any(j.get('state') in ('running','queued','loading') and j.get('policy_port')==PORT for j in jobs):
        state['status']='waiting_for_inference_dependents';return
    services=api('GET','/api/inference/jobs?device=a100')['jobs']
    owned=state.get('actions',{}).get('start-inference',{}).get('result',{})
    live=next((j for j in services if j.get('pid')==owned.get('pid')
               and j.get('start_ticks')==state.get('inference_job',{}).get('start_ticks')
               and j.get('state') in LIVE_SERVICES),None)
    if live:
        live=observe_command(live)
        if not matches_service(live):raise RuntimeError('Inference ownership changed during cleanup')
        if live.get('state')!='stopping':
            request={k:live[k] for k in ('pid','port','checkpoint','start_ticks')}
            mutate(state,'stop-inference','/api/inference/stop',dict(request,device='a100'))
        state['status']='cleanup_inference';return
    state.update(status='completed',completed_at=now().isoformat(),valid_episodes=10)
    evidence=json.loads((EVIDENCE/'closed_loop_result.json').read_text())
    references=[]
    reference_paths=[*(ROOT/'evaluation/V_1/official').glob('task40_eval*/results.json'),
                     ROOT/'evaluation/V_2_1/baseline/task40_eval01/results.json']
    for path in reference_paths:
        if not path.is_file():continue
        data=json.loads(path.read_text());config=data.get('configuration',{})
        identity=config.get('policy_identity') or {}
        references.append(dict(path=str(path),status=data.get('status'),
            completed_episodes=data.get('completed_episodes'),reported_success_rate=data.get('success_rate'),
            checkpoint=identity.get('checkpoint'),mode=config.get('mode'),
            preset=config.get('preset'),seed=config.get('base_seed'),seed_mode=config.get('seed_mode'),
            action_chunk=config.get('action_chunk'),paired_comparison=False,
            limitation='Historical layout/seed or terminal record differs; performance reference only'))
    write(EVIDENCE/'historical-performance-references.json',references)
    (EVIDENCE/'report.md').write_text(
        '# plan-v005 Stage1 执行结果\n\n'
        f'Status: DONE。{RUN} 完成6000 optimizer steps，四卡严格暂停恢复门槛已通过。\n\n'
        f'闭环：{evidence["successful_episodes"]}/10 有效回合成功；95% Wilson区间 {evidence["wilson95"]}。'
        f'平均Q Score {evidence["mean_q_score"]:.6f}，Task Progress {evidence["mean_task_progress"]:.6f}。\n\n'
        f'checkpoint：`{CHECKPOINT}`。评估与视频：`{evidence["output"]}`。\n\n'
        '配置、逐回合原生判据和视频索引见closed_loop_result.json；训练hash验收见checkpoint-verified.json。'
        '历史Stage1与官方B1K成绩及不同协议限制见historical-performance-references.json；'
        '旧Stage1记录为cancelled/10条，官方历史为单卡双环境及不同seed，不能视作本轮配对实验。'
        '未启动Stage2；不能仅由训练loss作改善结论。\n')


def tick(state):
    if state.get('status') in ('completed','blocked','user_paused'):return
    if state.get('checkpoint_verified'):
        eval_tick(state);return
    current=api('GET','/api/training/detail?device=a100&run='+RUN+'&arm=baseline')
    config=current['config'];observed=current['state']
    if (config.get('recipe')!='native_random_v2' or config.get('run_id')!=RUN
            or config.get('source_manifest',{}).get('digest')!=DIGEST
            or config.get('steps')!=6000 or config.get('gpus')!=[0,1,2,3]):
        raise RuntimeError('Formal workbench identity/config mismatch')
    state['observed']=observed
    write(EVIDENCE/'latest-detail.json',current)
    step=observed.get('step',0)
    if state.get('last_step')!=step:
        state.update(last_step=step,last_progress_at=now().isoformat())
    if any(not math.isfinite(row[key]) for row in current.get('metrics',[]) for key in ('fm','grad_norm','lr')
           if type(row.get(key)) in (int,float)):
        if observed.get('status') in ACTIVE:
            mutate(state,'nonfinite-pause','/api/training/stop',dict(device='a100',run=RUN,arm='baseline'))
        raise RuntimeError('Nonfinite training metrics; pause requested through workbench')
    status=observed.get('status')
    if status in ('paused','stopped'):
        state.update(status='user_paused',reason='Preserve formal training pause');return
    if status in ('failed','cancelled'):
        raise RuntimeError('Formal training '+status+'; preserve original evidence')
    if status=='completed':
        if step!=6000 or not any(c.get('step')==6000 and c.get('path')==CHECKPOINT for c in current.get('checkpoints',[])):
            raise RuntimeError('Formal completion missing exact step6000 checkpoint')
        record=verified_checkpoint()
        write(EVIDENCE/'checkpoint-verified.json',record)
        archive(RUN)
        state.update(checkpoint_verified=True,status='waiting_for_inference')
        return
    state['status']='training'
    resources=api('GET','/api/services')
    write(EVIDENCE/'latest-resources.json',resources)
    last=state.get('last_progress_at')
    if last and (now()-datetime.fromisoformat(last)).total_seconds()>600:
        state['attention']='No optimizer progress for >10min; inspect latest controller log and resources'
    else:state.pop('attention',None)


def progress(state):
    path=EXPERIMENT/'progress.md'
    text=path.read_text()
    start='<!-- native-supervisor:start -->';end='<!-- native-supervisor:end -->'
    block=(f'{start}\n核对时间：{state["checked_at"]}；持久巡检：`memory-vla-v2-1-training.timer`（120秒）。'
           f'当前状态 `{state["status"]}`；正式训练 {state.get("last_step",0)}/6000，'
           f'闭环 {state.get("completed_episodes",state.get("valid_episodes",0))}/10。'
           f'作业 `{RUN}`；全部动作和产物索引见 `evidence/native_random_v2/watchdog_state.json`。'
           f'下次检查：{state.get("next_check_at")}。\n{end}')
    if start in text:text=text[:text.index(start)]+block+text[text.index(end)+len(end):]
    else:text=text.replace('## plan-v005 Stage1重训（当前任务）','## plan-v005 Stage1重训（当前任务）\n\n'+block,1)
    if state.get('checkpoint_verified'):
        rows=text.splitlines()
        for index,row in enumerate(rows):
            if row.startswith('| Stage1正式6000步训练 |'):
                rows[index]=f'| Stage1正式6000步训练 | 已完成·6000/6000 | `{RUN}`；完整checkpoint及全部文件hash已核验，证据`evidence/native_random_v2/checkpoint-verified.json` |'
            if row.startswith('| Stage1统一闭环评估 |'):
                rows[index]=f'| Stage1统一闭环评估 | {state["status"]}·{state.get("completed_episodes",state.get("valid_episodes",0))}/10 | A100推理；双4090各2环境；确切step6000、积分4/执行8；实际job与视频路径见`evidence/native_random_v2/watchdog_state.json` |'
            if row.startswith('| 归档与基线验收 |') and state.get('status')=='completed':
                rows[index]='| 归档与基线验收 | 已完成 | `evidence/native_random_v2/report.md`、`closed_loop_result.json`、历史性能参考及视频索引已生成；本次推理已释放。未执行Stage2 |'
        text='\n'.join(rows)+'\n'
    temporary=path.with_suffix('.pending');temporary.write_text(text);temporary.replace(path)


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    with (EVIDENCE/'supervisor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        path=EVIDENCE/'watchdog_state.json'
        state=json.loads(path.read_text()) if path.exists() else dict(status='training',run=RUN,plan='plan-v005.json')
        try:
            tick(state)
            state.pop('query_error',None)
        except (URLError,TimeoutError) as error:
            state['query_error']=repr(error)  # read-only network outage: retry next timer tick
        except Exception as error:
            state.update(status='blocked',error=repr(error))
        state['checked_at']=now().isoformat()
        state['next_check_at']=None if state['status'] in ('completed','blocked','user_paused') else (now()+timedelta(seconds=120)).isoformat()
        save(state);progress(state)
        with (EVIDENCE/'watchdog_log.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=state['checked_at'],status=state['status'],
                step=state.get('last_step'),episodes=state.get('completed_episodes'),
                error=state.get('error'),query_error=state.get('query_error'),
                attention=state.get('attention'),next_check_at=state['next_check_at']))+'\n')
        print(json.dumps({k:v for k,v in state.items() if k not in ('actions','observed','inference_job','evaluation_job')}))
        if state['status']=='completed':
            subprocess.run(['systemctl','--user','stop','memory-vla-v2-1-training.timer'],check=True,timeout=15)


if __name__=='__main__':main()
