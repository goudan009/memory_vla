"""Short candidate windows for the existing VL service, with frame provenance."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import task_stage_diagnostics as stages
import vlm_diagnosis
from vlm_stage_protocol import PROMPT_VERSION


def candidate_windows(report, action_limit):
    centers = {0, min(action_limit, 480), action_limit}
    last = None
    for point in report['timeline']:
        if last is not None and point['facts'] != last:
            centers.add(point['action_steps'])
        last = point['facts']
    for event in report['failures']:
        centers.add(event['action_steps'])
    # When no state change exists, sample navigation and door-attempt windows too.
    if len(centers) <= 3:
        centers.update(int(action_limit*f) for f in (.25,.5,.75))
    windows = []
    for center in sorted(centers):
        start, end = max(0,center-160), min(action_limit,center+160)
        if (windows and start <= windows[-1]['action_end']
                and end-windows[-1]['action_start'] <= 640):
            windows[-1]['action_end'] = max(end, windows[-1]['action_end'])
        else:
            windows.append(dict(action_start=start,action_end=end))
    # Coverage is explicit; caller can analyze remaining windows in another pass.
    return windows


def _extract(video, frames):
    selection = '+'.join('eq(n\\,%d)' % frame for frame in frames)
    with tempfile.TemporaryDirectory(prefix='task40-clip-') as directory:
        result = subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',
            f"select='{selection}',scale=768:-2",'-vsync','0','-q:v','3',
            str(Path(directory)/'%04d.jpg')],capture_output=True,text=True,timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr[-1200:])
        images = [base64.b64encode(p.read_bytes()).decode() for p in sorted(Path(directory).glob('*.jpg'))]
    if len(images) != len(frames):
        raise ValueError('候选窗口帧数与录像不匹配；请核对录像时间映射')
    return images


def normalize(document, frames, report, episode):
    if document.get('_ok') is not True or not isinstance(document.get('events'),list):
        raise ValueError('VL服务未返回Task40结构化事件；需要既有服务的task40_stage模式')
    scale = 1 / report['time_mapping']['video_seconds_per_action_step']
    # Global frame indices have known video time but only approximate action steps.
    fps = scale / ((report.get('_steps_per_render')) or 24)
    captured = {row['frame_index']: row for row in report['time_mapping'].get('frames', [])}
    result = []
    for raw in document['events']:
        start,end = raw.get('frame_start'),raw.get('frame_end')
        evidence = raw.get('evidence_frames',[])
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(frames):
            raise ValueError('VL事件帧范围不正确')
        if not evidence or any(type(i) is not int or not 1 <= i <= len(frames) for i in evidence):
            raise ValueError('VL事件缺少有效证据帧')
        # Use video seconds separately; action coordinates are explicitly approximate.
        item = dict(raw)
        if item.get('stage_id') == 'S7' and item.get('outcome') == 'completed':
            evidence_fields = item.get('stage_evidence')
            evidence_fields = evidence_fields if isinstance(evidence_fields, dict) else {}
            accepted = (item.get('observed_behavior') == 'place_bag'
                        and evidence_fields.get('destination') == 'microwave_interior'
                        and evidence_fields.get('bag_inside_visible') is True
                        and evidence_fields.get('release_visible') is True)
            item['model_outcome'] = item['outcome']
            item['semantic_validation'] = dict(accepted=accepted,
                reason='模型需明确报告微波炉内部位置与可见释放；此检查不等于图像事实已核实',
                required=['destination=microwave_interior','bag_inside_visible=true','release_visible=true'])
            if not accepted:
                item['outcome'] = 'unknown'
                item['error_type'] = 'unknown'
        item.update(action_start=frames[start-1]/fps*scale,
                    action_end=min(episode.get('sim_steps',float('inf')),frames[end-1]/fps*scale),
                    time_start=frames[start-1]/fps,time_end=frames[end-1]/fps,
                    time_axis='video_seconds',action_time_approximate=True,
                    evidence_frames=[frames[i-1] for i in evidence])
        if captured:
            first, last = captured[frames[start-1]], captured[frames[end-1]]
            item.update(action_start=first['action_steps'], action_end=last['action_steps'],
                        time_start=first['frame_index']/first['video_fps'],
                        time_end=last['frame_index']/last['video_fps'],
                        sim_time_start_s=first['sim_time_s'], sim_time_end_s=last['sim_time_s'],
                        action_time_approximate=False)
        # Visual frame alignment cannot establish a simultaneous physical truth.
        result.extend(stages.validate_events([item],episode))
    return result


def diagnose(run, episode, video, endpoint=None, force=False):
    path = run/'stage_diagnostics'/'vl_events'/(str(episode['episode_index'])+'.json')
    if path.is_file() and not force:
        return json.loads(path.read_text())
    endpoint = endpoint or vlm_diagnosis.DEFAULT_ENDPOINT
    if not vlm_diagnosis.health(endpoint).get('ok'):
        raise RuntimeError('既有VL服务当前离线；物理阶段分析可继续使用')
    report = stages.analyze_episode(run,episode)
    config = json.loads((run/'results.json').read_text()).get('configuration',{})
    render = config.get('video_steps_per_render') or 24
    fps = config.get('video_fps') or 20
    report['_steps_per_render'] = render
    limit = episode.get('sim_steps') or episode.get('episode_length')
    windows = candidate_windows(report,limit)
    records, events = [], []
    for window in windows[:8]:
        first,last = int(window['action_start']/render),max(0,int(window['action_end']/render)-1)
        captured = report['time_mapping'].get('frames', [])
        if captured:
            within = [row['frame_index'] for row in captured
                      if window['action_start'] <= row['action_steps'] <= window['action_end']]
            if not within:
                continue
            first, last = within[0], within[-1]
        frames = sorted({round(first+(last-first)*i/11) for i in range(12)})
        raw = vlm_diagnosis._analyze(endpoint,dict(mode='task40_stage',task='40',
            episode=stages.episode_key(episode),group='task40_stages',
            label=f"动作步约{window['action_start']}–{window['action_end']}，录像约{first/fps:.2f}–{last/fps:.2f}秒",
            notes='只根据可见行为；采样不能确认的过程标unknown。',frames=_extract(video,frames)))
        try:
            normalized = normalize(raw,frames,report,episode)
            error = None
        except (ValueError, TypeError) as failure:
            normalized = []
            error = str(failure)
        events.extend(normalized)
        records.append(dict(**window,video_frames=frames,raw_output=raw,events=normalized,error=error))
    document = dict(schema_version=1,episode_id=stages.episode_key(episode),source='vl',
        prompt_version=PROMPT_VERSION, created_at=datetime.now(timezone.utc).isoformat(),
        model_endpoint=endpoint,events=events,windows=records,candidate_windows=len(windows),
        time_mapping=report['time_mapping'],
        analyzed_windows=len(records),failed_windows=sum(r['error'] is not None for r in records),
        unprocessed_windows=windows[8:],
        coverage='候选窗口抽样，不能以无事件证明未尝试；'+(
            '证据帧含实录动作步与仿真时间，帧间过程不可见' if report['time_mapping'].get('frames')
            else '动作步与视频映射为近似'))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        previous = path.read_bytes()
        history = path.parent/'history'/path.stem
        history.mkdir(parents=True,exist_ok=True)
        archive = history/(hashlib.sha256(previous).hexdigest()+'.json')
        if not archive.exists():
            archive.write_bytes(previous)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
    pending.replace(path)
    return document
