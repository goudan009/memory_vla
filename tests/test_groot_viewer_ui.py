"""Exercise the viewer's refresh behavior without starting a simulation."""

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node") or "/home/1024201092WYH/.nvm/versions/node/v22.23.1/bin/node"


def run_viewer(assertions):
    if not Path(NODE).exists():
        pytest.skip("Node.js is required for viewer behavior checks")
    page = (ROOT / "evaluation/scripts/result_viewer.html").read_text()
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    # Disable startup polling; tests invoke the real render functions directly.
    script = script.replace('$("refresh").onclick=refresh; refresh(); setInterval(refresh,5000);', "")
    harness = """
const assert = require('node:assert/strict');
const elements = {};
global.setInterval=()=>0;
global.window={addEventListener() {}};
global.checkpointNotes={bind() {}, brief() { return ''; }};
global.location=new URL('http://localhost/');
global.history={pushState(_state, _title, url) { global.location=new URL(url); }};
function element() {
  const queried = {};
  return {
    _innerHTML: '', textContent: '', value: '', hidden: false, src: '', children: [], options: [],
    set innerHTML(value) { this._innerHTML = value; this.children = []; },
    get innerHTML() { return this._innerHTML; },
    loads: 0, pauses: 0, scrollHeight: 0,
    load() { this.loads++; }, pause() { this.pauses++; },
    appendChild(node) { this.children.push(node); },
    querySelector(selector) { return queried[selector] ||= element(); },
    querySelectorAll() { return []; },
    addEventListener(name, handler) { this['on'+name] = handler; },
    setAttribute(name, value) { this[name] = value; },
    removeAttribute(name) { this[name] = ''; },
    getAttribute(name) { return this[name] || null; }
  };
}
global.document = {getElementById(id) { return elements[id] ||= element(); }, createElement: element};
function runCards(node=document.getElementById('runs')) {
  return node.children.flatMap(child=>child.className?.split(' ').includes('run-card')?[child]:runCards(child));
}
"""
    result = subprocess.run([NODE], input=harness + script + assertions, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    return page


def test_refresh_keeps_selected_video_and_playback_position():
    run_viewer("""
const run = {_folder: 'baseline/groot_task21_1', _videos: ['videos/gpu1/a.mp4', 'videos/gpu1/b.mp4']};
renderVideos(run);
const player = $('video-player'), select = $('video-select');
select.value = '1'; select.onchange();
player.currentTime = 17;
const loads = player.loads, pauses = player.pauses, src = player.src;
renderVideos({...run, _videos: ['videos/gpu1/0.mp4', ...run._videos]});
assert.equal(player.loads, loads);
assert.equal(player.pauses, pauses);
assert.equal(player.currentTime, 17);
assert.equal(player.src, src);
assert.equal(select.value, '2');
renderVideos({...run, _folder: 'experiment_1/groot_task21_1'});
assert.equal(player.loads, loads + 1);
assert.match(player.src, /experiment_1/);
renderVideos(null);
assert.equal(player.src, '');
assert.equal(player.hidden, true);
""")


def test_native_2026_form_and_artifacts_use_the_native_contract():
    run_viewer("""
$('launch-kind').value='n17';$('n17-instance').value='18';$('n17-port').value='8032';
$('n17-gpu').value='0';$('n17-notes').value='V1.2 / N1.7 / B1K2026: official radio, RTC off';
$('n17-video').checked=true;
const request=collectLaunchRequest();
assert.equal(request.version,'V_1_2');assert.equal(request.protocol,'b1k2026_websocket');
assert.equal(request.instance,18);assert.equal(request.rollouts,1);assert.equal(request.gpu_index,0);
assert.equal(request.task,'turning_on_radio');assert.ok(request.notes);assert.equal(request.record_video,true);
assert.equal(request.run_tag,undefined);assert.equal(request.seed_mode,undefined);
$('n17-instance').value='20';assert.throws(()=>collectLaunchRequest(),/0–19/);
$('n17-instance').value='18';$('n17-notes').value='';assert.throws(()=>collectLaunchRequest(),/备注/);
const card=renderNativeJob({notes:'native baseline',native_metrics:[{instance_id:319,steps:100,success:false,q_score:0.5}],
 artifacts:[{path:'json/radio.json',url:'/api/launch/artifact?run=radio&path=json/radio.json',label:'native rollout JSON'},
 {url:'javascript:alert(1)',label:'bad'}]});
assert.match(card,/未成功/);assert.match(card,/0.5/);assert.match(card,/native rollout JSON/);assert.ok(!card.includes('javascript:'));
""")


def test_recording_highlights_selected_episode_metrics_and_shortens_labels():
    run_viewer("""
const a='videos/a.mp4',b='videos/b.mp4';
const run={_folder:'recording',status:'complete',evaluation_mode:'official-eval',mean_task_progress:99,
  _videos:[a,b],_episodes:[{episode_index:1,task_progress:34.8,q_score:0,success:false,sim_steps:1200,video_file:a},
    {episode_index:2,task_progress:0,q_score:.5,success:true,action_steps:0,video_file:b}],
  _video_episodes:{[a]:{episode_index:1,seed:42}}};
runs=[run];selected=run;render();chooseEpisode(1);
assert.ok($('episode-metrics').innerHTML.includes('34.8%'));
assert.ok($('episode-metrics').innerHTML.includes('0.0000'));
assert.ok($('episode-metrics').innerHTML.includes('未成功'));
assert.ok(!$('episode-metrics').innerHTML.includes('99.0%'));
assert.ok(!mediaOptions(run)[0].label.includes('seed'));
assert.equal($('video-empty').hidden,true);
chooseEpisode(2);
assert.ok($('episode-metrics').innerHTML.includes('0.0%'));
assert.ok($('episode-metrics').innerHTML.includes('0.5000'));
assert.ok($('episode-metrics').innerHTML.includes('成功'));
assert.ok(!$('episode-metrics').innerHTML.includes('34.8%'));
""")


def test_official_b1k_metrics_and_groot_metrics_remain_separate():
    run_viewer("""
const run={_folder:'paired-metrics',evaluation_mode:'official-eval',status:'complete',
 completed_episodes:2,requested_episodes:10,success_rate:1,successful_episodes:2,
 mean_q_score:.25,q_score_episodes:2,mean_task_progress:90,
 b1k_metrics:{full_success_rate:0,success_episodes:2,missing_success_episodes:0},
 _videos:[],_episodes:[{episode_index:1,valid:true,success:true,q_score:.5,task_progress:100,
 native_termination:{success:false}},{episode_index:2,valid:true,success:true,q_score:1,task_progress:80}]};
runs=[run];selected=run;render();chooseEpisode(1);
assert.ok($('summary').innerHTML.includes('B1K 官方 Q-score'));
assert.ok($('summary').innerHTML.includes('25.0%'));
assert.ok($('summary').innerHTML.includes('0.2500'));
assert.ok($('summary').innerHTML.includes('B1K 完整任务成功率'));
assert.ok($('summary').innerHTML.includes('0.0%'));
assert.ok($('summary').innerHTML.includes('GR00T 累计任务进度'));
assert.ok($('summary').innerHTML.includes('90.0%'));
assert.ok($('summary').innerHTML.includes('100.0%'));
assert.ok($('episode-metrics').innerHTML.includes('B1K 终局成功'));
assert.equal(b1kResult(run,run._episodes[0]),'未成功');
assert.equal(episodeResult(run,run._episodes[0]),'成功');
assert.equal(b1kResult(run,run._episodes[1]),'未记录');
assert.ok($('details').innerHTML.includes('<th>B1K Q</th><th>B1K 终局成功</th>'));
chooseEpisode(2);assert.ok($('episode-metrics').innerHTML.includes('未记录'));
""")


def test_vlm_selection_health_and_manual_service_lifecycle():
    run_viewer("""
(async()=>{
await Promise.resolve();await Promise.resolve();
$('vlm-service-choice').value='a6000-qwen3-vl';
global.fetch=async()=>({json:async()=>({ok:true,state:'ready',active:'active',pid:123,gpu_mem_gb:16.3})});
$('vlm-service-message').textContent='服务已启动，正在加载模型；稍后刷新状态。';
await refreshVlmHealth();
assert.equal($('vlm-service-message').textContent,'诊断服务已就绪。');
assert.equal($('vlm-run').disabled,false);assert.equal($('vlm-start').disabled,true);
assert.equal($('vlm-stop').disabled,false);
$('vlm-service-choice').value='';renderVlmBody();assert.equal($('vlm-run').disabled,true);
assert.equal($('vlm-batch').disabled,true);
$('vlm-service-choice').value='a6000-qwen3-vl';
const calls=[];
global.fetch=async(url,options)=>{calls.push([url,options.method]);return {ok:true,json:async()=>({ok:false,state:'stopped',active:'inactive',pid:0})};};
await controlVlm('stop');assert.equal($('vlm-start').disabled,false);assert.equal($('vlm-run').disabled,true);
assert.deepEqual(calls,[['/api/vlm/service/stop','POST']]);
global.fetch=async(url,options)=>({ok:true,json:async()=>({ok:false,state:'loading',active:'active',pid:124})});
await controlVlm('start');assert.ok($('vlm-service-state').textContent.includes('加载中'));
assert.equal($('vlm-start').disabled,true);assert.equal($('vlm-stop').disabled,false);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_vlm_delayed_diagnosis_never_overwrites_another_episode():
    run_viewer("""
(async()=>{
await Promise.resolve();await Promise.resolve();
const a='videos/a.mp4',b='videos/b.mp4';
const run={_folder:'recording',status:'complete',_videos:[a,b],_episodes:[
  {episode_index:1,video_file:a},{episode_index:2,video_file:b}]};
runs=[run];selected=run;render();chooseEpisode(1);
await Promise.resolve();await Promise.resolve();
vlmHealth={ok:true};$('vlm-service-choice').value='a6000-qwen3-vl';
let finish,payload;
global.fetch=(url,options)=>{if(options?.method==='POST'){
  payload=JSON.parse(options.body);return new Promise(resolve=>finish=resolve);}
  return Promise.resolve({json:async()=>({diagnosis:null})});};
const pending=diagnoseVlm();assert.equal(payload.service_id,'a6000-qwen3-vl');
chooseEpisode(2);await Promise.resolve();await Promise.resolve();
finish({status:200,json:async()=>({_ok:true,summary:'first episode'})});await pending;
assert.equal(vlmData,null);assert.ok(!$('vlm-body').innerHTML.includes('first episode'));
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_human_review_drafts_survive_refresh_switching_and_missing_video():
    run_viewer("""
const run={_folder:'V_1/memory/run',status:'complete',_videos:[],_human_reviews:{},
  _episodes:[{episode_index:1,instance_id:5,seed:42},{episode_index:2,instance_id:6,seed:43}]};
runs=[run];selected=run;render();chooseEpisode(1);
assert.equal($('human-review-box').hidden,false);
assert.equal($('video-player').hidden,true);
humanInput('execution_quality').value='0';humanInput('target_touched').value='false';
humanInput('notes').value='需要人工判断 <script>';editHumanReview();
assert.equal($('human-review-status').textContent,'有未保存的修改');
render();assert.equal(humanInput('execution_quality').value,'0');
chooseEpisode(2);assert.equal(humanInput('notes').value,'');
chooseEpisode(1);assert.equal(humanInput('notes').value,'需要人工判断 <script>');
assert.equal(humanValues().execution_quality,0);assert.equal(humanValues().target_touched,false);
const other={...run,_folder:'V_1/baseline/other'};runs.push(other);selected=other;render();chooseEpisode(1);
assert.equal(humanInput('notes').value,'');
selected=run;render();chooseEpisode(1);assert.equal(humanInput('notes').value,'需要人工判断 <script>');
""")


def test_supplement_reasons_are_stage_scoped_and_start_empty():
    run_viewer("""
reviewStageOptions=[{id:'stage',label:'階段一',supplements:[],causes:[
  {id:'D',label:'D. 决策错误',notes:[{id:'D1',label:'D1 初始决策错误'}]},
  {id:'M',label:'M. 操作执行错误',notes:[{id:'M1',label:'M1 抓取失败'}]}]},
  {id:'stage2',label:'阶段二',supplements:[{id:'S2',label:'只属于第二阶段'}],causes:[]}];
$('review-stage').value='stage';
renderReviewSecondary();
assert.equal($('review-secondary-specific').innerHTML,'<option value="">不添加补充原因</option>');
$('review-stage').value='stage2';renderReviewSecondary();
assert.ok($('review-secondary-specific').innerHTML.includes('S2'));
assert.ok(!$('review-secondary-specific').innerHTML.includes('D1'));
""")


def test_evaluation_run_cards_show_annotation_completion():
    run_viewer("""
const run={_folder:'baseline/groot_task40_1',task_id:40,version:'baseline',model_name:'groot',
  status:'complete',completed_episodes:2,requested_episodes:2,success_rate:0,successful_episodes:0,
  mean_task_progress:35,mean_q_score:0,_episodes:[
    {episode_index:1,human_reviewed:true,human_review_needs_update:false},
    {episode_index:2,human_reviewed:true,human_review_needs_update:false}]};
runs=[run];selected=run;render();
assert.ok(runCards()[0].innerHTML.includes('2/2 已标注'));
assert.ok(runCards()[0].innerHTML.includes('全部标注完成'));
run._episodes[1].human_reviewed=false;run._episodes[1].human_review_needs_update=true;
render();assert.ok(runCards()[0].innerHTML.includes('1/2 已标注 · 1 需更新'));
assert.ok(runCards()[0].innerHTML.includes('尚未全部完成'));
""")


def test_human_review_async_save_is_bound_to_original_episode_and_delete_clears_record():
    run_viewer("""
(async()=>{
const run={_folder:'V_1/memory/run',status:'complete',_videos:[],_human_reviews:{},
  _episodes:[{episode_index:1},{episode_index:2}]};
runs=[run];selected=run;render();chooseEpisode(1);
humanInput('execution_quality').value='75';humanInput('notes').value='<script>不是代码</script>';editHumanReview();
let finish,submitted;
global.fetch=(url,options)=>{assert.equal(url,'/api/human-review');submitted=JSON.parse(options.body);
  return new Promise(resolve=>{finish=resolve;});};
const pending=writeHumanReview();
assert.equal(submitted.episode_index,1);assert.equal(submitted.review.execution_quality,75);
chooseEpisode(2);humanInput('notes').value='第二集未保存';editHumanReview();
const saved={...submitted.review,updated_at:'2026-11-13T08:00:00Z',source:'human'};
finish({ok:true,json:async()=>({review:saved})});await pending;
assert.equal(humanInput('notes').value,'第二集未保存');
assert.equal($('human-review-status').textContent,'有未保存的修改');
chooseEpisode(1);assert.equal(humanInput('execution_quality').value,'75');
assert.equal($('human-review-delete').disabled,false);
assert.ok($('details').innerHTML.includes('75 / 100'));
assert.ok($('details').innerHTML.includes('&lt;script&gt;不是代码&lt;/script&gt;'));
global.fetch=async(url,options)=>{assert.equal(options.method,'DELETE');
  assert.equal(JSON.parse(options.body).episode_index,1);return {ok:true,json:async()=>({review:null,deleted:true})};};
await writeHumanReview(true);
assert.equal(humanInput('notes').value,'');assert.equal($('human-review-delete').disabled,true);
assert.ok(!('episode:1' in run._human_reviews));
chooseEpisode(2);assert.equal(humanInput('notes').value,'第二集未保存');
global.fetch=async()=>({ok:false,json:async()=>({error:'磁盘不可用'})});
await writeHumanReview();assert.equal(humanInput('notes').value,'第二集未保存');
assert.equal($('human-review-status').textContent,'保存失败：磁盘不可用');
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_video_labels_use_explicit_episode_identity_without_guessing_legacy_uuids():
    run_viewer("""
const named = 'videos/gpu1_resume2/episode_005_instance7_seed3_gpu1_env1_s0.mp4';
const legacy = 'videos/gpu0/3712ec9f-cf8b-4b7e-a0cc-84736a9effb5_s0.mp4';
const run = {_folder: 'V_1/baseline/groot_task21_1', _videos: [named, legacy],
  _video_episodes: {[named]: {episode_index: 5, instance_id: 7, seed: 3, gpu_index: 1,
    env_idx: 1, success: false, valid: true}}};
renderVideos(run);
const options = $('video-select').innerHTML;
assert.ok(options.includes('第 5 集'));
assert.ok(options.includes('实例 7'));
assert.ok(options.includes('seed 3'));
assert.ok(options.includes('GPU 1 / env 1'));
assert.ok(options.includes('未关联回合'));
assert.ok(!options.includes('cf8b-4b7e-a0cc-84736a9effb5'));
""")


def test_episode_selection_links_video_and_cameras_without_subscribing_to_live_frames():
    run_viewer("""
const a='videos/gpu1/a.mp4', b='videos/gpu1/b.mp4';
const episodes=[{episode_index:1,instance_id:0,seed:0,video_file:a},
  {episode_index:2,instance_id:1,seed:1,video_file:b},
  {episode_index:3,instance_id:2,seed:2}];
const run={_folder:'V_1/baseline/run',status:'running',_episodes:episodes,_videos:[a,b],
  _video_episodes:{[a]:episodes[0],[b]:episodes[1]}};
runs=[run];selected=run;render();
assert.equal($('episode-select').value,'live');
assert.equal($('video-player').hidden,true);
chooseEpisode(2);
assert.ok($('video-player').src.includes(encodeURIComponent(b)));
assert.equal($('episode-select').value,'episode:2');
assert.ok($('preview-title').textContent.includes('第 2 集'));
assert.equal(previewEnabled,false);
$('preview-toggle').checked=true;togglePreview();
assert.equal(previewEnabled,false); // Replay never requests live observations.
assert.ok($('preview-environments').innerHTML.includes('replay-head'));
const loads=$('video-player').loads;
render();assert.equal($('video-player').loads,loads);
chooseEpisode(1);
assert.equal(replayEnabled,true);
assert.equal($('preview-toggle').checked,true);
assert.ok($('video-player').src.includes(encodeURIComponent(a)));
chooseEpisode(3);
assert.equal($('video-player').hidden,true);
assert.ok($('video-empty').textContent.includes('第 3 集'));
assert.equal($('preview-environments').hidden,true);
assert.equal($('preview-toggle').disabled,true);
$('episode-select').value='live';$('episode-select').onchange();
assert.equal($('preview-title').textContent,'实时相机预览');
assert.equal($('preview-toggle').disabled,false);
""")


def test_replay_crops_the_real_recording_order_and_excludes_caption_footer():
    run_viewer("""
const file='videos/gpu1/a.mp4', episode={episode_index:1,video_file:file};
const run={_folder:'run',status:'complete',_episodes:[episode],_videos:[file],
  _video_episodes:{[file]:episode}};
runs=[run];selected=run;render();
$('preview-toggle').checked=true;togglePreview();
const draws={};
for(const name of ['head','left_wrist','right_wrist']) {
  $('replay-'+name).getContext=()=>({drawImage(...args){draws[name]=args;}});
}
const player=$('video-player');player.videoWidth=768;player.videoHeight=280;
player.readyState=2;player.paused=true;player.onloadeddata();
assert.deepEqual(draws.head.slice(1,5),[512,0,256,256]);
assert.deepEqual(draws.left_wrist.slice(1,5),[0,0,256,256]);
assert.deepEqual(draws.right_wrist.slice(1,5),[256,0,256,256]);
assert.equal($('preview-environments').hidden,false);
""")


def test_all_ten_episodes_are_visible_and_clicking_opens_media_tab():
    run_viewer("""
const run={_folder:'V_0/baseline/radio',version:'V_0',status:'complete',
  _episodes:Array.from({length:10},(_,i)=>({episode_index:i+1,instance_id:i,seed:i}))};
runs=[run];selected=run;render();
for(let i=1;i<=10;i++) assert.ok($('details').innerHTML.includes(`>第 ${i} 集</button>`));
assert.ok($('details').innerHTML.includes('全部 10 集'));
assert.equal($('pane-episodes').hidden,false);
assert.equal($('pane-media').hidden,true);
chooseEpisode(1);
assert.equal($('pane-episodes').hidden,true);
assert.equal($('pane-media').hidden,false);
assert.equal($('episode-select').value,'episode:1');
render();assert.equal($('pane-media').hidden,false);
selectDetailTab('episodes');
assert.equal($('pane-media').hidden,true);
assert.equal($('pane-episodes').hidden,false);
""")


def test_version_groups_sort_numerically_and_preserve_manual_collapse_on_refresh():
    run_viewer("""
runs=[{_folder:'V_2/baseline/a',version:'V_2',group:'baseline'},
      {_folder:'V_10/baseline/b',version:'V_10',group:'baseline'},
      {_folder:'V_2/experment_1/c',version:'V_2',group:'experment_1'}];
selected=runs[0];render();
const groups=$('runs').children;
assert.equal(groups.length,2);
assert.equal(groups[0]['data-version'],'V_10');
assert.equal(groups[1]['data-version'],'V_2');
assert.equal(groups[1].open,true);
assert.equal(runCards().length,3);
groups[1].open=false;groups[1].ontoggle();render();
assert.equal($('runs').children[1].open,false);
$('run-search').value='experment_1';render();
assert.equal($('runs').children.length,1);
assert.equal($('runs').children[0].open,true);
assert.equal(runCards().length,1);
$('run-search').value='';render();
assert.equal($('runs').children[1].open,false);
""")


def test_launch_form_only_sends_debug_step_limit_in_debug_and_preserves_resume_protocol():
    run_viewer("""
const values={kind:'new',version:'V_2',group:'experment_1',task:'22',mode:'single2',
  preset:'official-eval',seed:'0','seed-mode':'sequence',host:'172.28.62.17',port:'5561',
  model:'grootmeory','video-steps':'24',gpu:'1','debug-steps':'256'};
for(const [key,value] of Object.entries(values)) $('launch-'+key).value=value;
$('launch-video').checked=true;
updateLaunchForm();
let payload=collectLaunchRequest();
assert.equal(payload.mode,'single2');assert.equal(payload.gpu_index,1);
assert.equal(payload.policy_port,5561);assert.equal(payload.record_video,true);
assert.equal(payload.group,'experment_1');assert.equal(payload.version,'V_2');
assert.ok(!('debug_max_steps' in payload));
assert.equal($('launch-debug-steps').disabled,true);
$('launch-preset').value='fast-debug';$('launch-mode').value='dual2';updateLaunchForm();
payload=collectLaunchRequest();
assert.equal(payload.debug_max_steps,256);assert.ok(!('gpu_index' in payload));
assert.equal($('launch-gpu').disabled,true);
$('launch-kind').value='resume';$('launch-resume-run').value='V_1/baseline/old';
updateLaunchForm();payload=collectLaunchRequest();
assert.deepEqual(payload,{resume_from:'V_1/baseline/old'});
assert.equal($('launch-new').disabled,true);
$('launch-resume-port').value='5556';
assert.deepEqual(collectLaunchRequest(),{resume_from:'V_1/baseline/old',policy_port:5556});
""")


def test_result_tasks_keep_baseline_memory_comparisons_together_and_preserve_collapse():
    run_viewer("""
const make=(task,method,id)=>({_folder:`V_1/${method}/task${task}_eval${id}`,version:'V_1',
  task_id:task,group:method,status:'complete'});
const memory21=make(21,'memory','02'), baseline0=make(0,'baseline','01'),
  baseline21=make(21,'baseline','01'), memory0=make(0,'memory','01'), oldMemory21=make(21,'memory','01');
runs=[memory21,baseline0,baseline21,memory0,oldMemory21];selected=memory21;render();
const taskGroups=()=> $('runs').children[0].children[1].children;
const methods=group=>group.children[1].children;
assert.equal(taskGroups().length,2);
assert.equal(taskGroups()[0]['data-result-task'],JSON.stringify(['V_1','0']));
assert.equal(taskGroups()[1]['data-result-task'],JSON.stringify(['V_1','21']));
assert.equal(taskGroups()[0].open,false);assert.equal(taskGroups()[1].open,true);
assert.deepEqual(methods(taskGroups()[1]).map(m=>m['data-result-method']),['baseline','memory']);
assert.equal(runCards(taskGroups()[1]).length,3);
assert.equal(runCards(methods(taskGroups()[1])[1])[0].title,memory21._folder);
assert.equal(runCards(methods(taskGroups()[1])[1])[1].title,oldMemory21._folder);
taskGroups()[1].open=false;taskGroups()[1].ontoggle();render();
assert.equal(taskGroups()[1].open,false);
$('run-search').value='task21';render();
assert.equal(taskGroups().length,1);assert.equal(taskGroups()[0].open,true);
taskGroups()[0].ontoggle(); // Search expansion must not overwrite a manual collapse.
$('run-search').value='';render();assert.equal(taskGroups()[1].open,false);
runCards(taskGroups()[0])[1].onclick();assert.equal(selected,memory0);
assert.ok($('detail-title').textContent.includes('task0_eval01'));
""")


def test_gpu_overview_shows_both_hosts_and_marks_unknown_without_zero_usage():
    page=run_viewer("""
renderGpuUsage({remote:{ok:true,checked_at:'10:00:00',gpus:[
  {index:0,name:'A100',total_mb:81920,used_mb:74240,utilization:99},
  {index:1,name:'A100',total_mb:81920,used_mb:1024,utilization:0}]},
  local:{ok:true,gpus:[{index:1,name:'4090',total_mb:49152,used_mb:12288,utilization:80}]}});
assert.ok($('remote-gpus').innerHTML.includes('72.5 / 80 GiB'));
assert.ok($('remote-gpus').innerHTML.includes('GPU 利用率 99%'));
assert.ok($('local-gpus').innerHTML.includes('12.0 / 48 GiB'));
assert.ok($('remote-gpus').innerHTML.includes('gpu-reading busy'));
assert.equal($('remote-gpu-time').textContent,'更新 10:00:00');
renderGpuUsage({remote:{ok:false},local:{ok:true,gpus_error:'timeout'}});
assert.ok($('remote-gpus').innerHTML.includes('连接失败，显存未知'));
assert.ok(!$('remote-gpus').innerHTML.includes('0.0 /'));
assert.ok($('local-gpus').innerHTML.includes('显存读取失败'));
""")
    assert page.index('id="gpu-overview"') < page.index('id="page-training"')


def test_optional_device_gpu_memory_and_offline_display():
    run_viewer("""
renderServices({a6000:{ok:true,gpus:[{index:0,name:'RTX 6000 Ada',total_mb:49140,used_mb:1024,utilization:1}],processes:[{pid:7,kind:'training',label:'V_2_baseline_task0_train01',gpu:'0'}]}});
assert.ok($('a6000-gpus').innerHTML.includes('1.0 / 48 GiB'));
assert.ok($('a6000-services').innerHTML.includes('V_2_baseline_task0_train01'));
renderServices({a6000:{ok:false,last_seen:'2026-11-12 20:00:00'}});
assert.ok($('a6000-gpus').innerHTML.includes('离线'));
assert.ok($('a6000-gpus').innerHTML.includes('2026-11-12'));
assert.ok(!$('a6000-gpus').innerHTML.includes('0.0 /'));
assert.ok($('a6000-services').textContent.includes('自动重连'));
""")


def test_evaluation_actions_pause_resume_and_delete_states():
    run_viewer("""
const run={_folder:'V_2/memory/task21_eval01',status:'running',evaluation_mode:'official-eval'};
let actions=evaluationButtons(run);
assert.ok(actions.includes('data-evaluation-action="pause"'));
assert.ok(actions.includes('data-evaluation-action="delete" data-evaluation-run="V_2/memory/task21_eval01" disabled'));
evaluationStopping.add(run._folder);actions=evaluationButtons(run);
assert.ok(actions.includes('暂停中…'));
evaluationStopping.clear();
actions=evaluationButtons({...run,status:'interrupted'});
assert.ok(actions.includes('续跑评估'));assert.ok(actions.includes('删除评估'));
assert.ok(!actions.includes(' disabled'));
assert.ok(!evaluationButtons({...run,status:'interrupted',evaluation_mode:'fast-debug'}).includes('续跑评估'));
let calls=0;global.window={confirm:()=>false};launchAPI=async()=>{calls++;};
void evaluationAction('delete',run._folder);assert.equal(calls,0);
""")


def test_copy_launch_template_and_detected_policy_use_the_actual_endpoint():
    run_viewer("""
runs=[{_folder:'V_2/memory/run',version:'V_2',group:'memory',task_id:22,
  evaluation_mode:'official-eval',model_name:'grootmeory',configuration:{
    mode:'single2',base_seed:7,seed_mode:'fixed',episodes_per_gpu:10,
    instances:{'0':[0,1,2,3,4,5,6,7,8,9]},policy_server:'172.28.62.17:5561',
    video_steps_per_render:48,record_video:true}}];
$('launch-template').value='V_2/memory/run';copyLaunchTemplate();
assert.equal($('launch-version').value,'V_2');assert.equal($('launch-group').value,'memory');
assert.equal($('launch-port').value,'5561');assert.equal($('launch-gpu').value,'0');
assert.equal($('launch-seed').value,'7');assert.equal($('launch-video-steps').value,'48');
renderLaunchServices({remote:{host:'172.28.62.17',processes:[{kind:'inference',
  model:'/training/memory/checkpoints/step-10000',configured_port:'5563',listening_ports:[5563]}]}});
assert.ok($('launch-service').innerHTML.includes('memory / step-10000'));
$('launch-service').value='172.28.62.17:5563';$('launch-service').onchange();
assert.equal($('launch-port').value,'5563');
""")


def test_policy_selection_survives_service_reordering_and_updates_connection_details():
    run_viewer("""
$('launch-kind').value='new';$('launch-mode').value='single2';
$('launch-preset').value='official-eval';$('launch-task').value='22';
$('launch-host').value='172.28.62.17';$('launch-port').value='5556';
$('launch-version').value='V_2';$('launch-group').value='memory';
$('launch-model').value='grootmeory';$('launch-video').checked=true;
$('launch-video-steps').value='24';
const first={kind:'inference',configured_port:'5556',listening_ports:[5556],model:'/models/baseline'};
const second={kind:'inference',configured_port:'5561',listening_ports:[5561],model:'/models/memory'};
renderLaunchServices({remote:{host:'172.28.62.17',processes:[first,second]}});
$('launch-service').value='172.28.62.17:5561';$('launch-service').onchange();
renderLaunchServices({remote:{host:'172.28.62.17',processes:[second,first]}});
assert.equal($('launch-service').value,'172.28.62.17:5561');
assert.equal($('launch-port').value,'5561');
assert.ok($('launch-policy-details').textContent.includes('/models/memory'));
assert.ok($('launch-policy-summary').textContent.includes('172.28.62.17:5561'));
$('launch-service').value='';$('launch-service').onchange();
renderLaunchServices({remote:{host:'172.28.62.17',processes:[first,second]}});
assert.equal($('launch-service').value,'');
assert.equal($('launch-endpoint').open,true);
""")


def test_inference_form_keeps_policy_seed_separate_and_requires_explicit_gpu():
    page=run_viewer("""
$('inference-gpu').value='';
assert.throws(()=>collectInferenceRequest(),/GPU/);
$('inference-gpu').value='2';$('inference-checkpoint').value='/data/model with spaces';
$('inference-port').value='5557';$('inference-source').value='/data/source';
$('inference-python').value='/data/env/bin/python';$('inference-seed').value='42';
$('launch-seed').value='0';
const payload=collectInferenceRequest();
assert.equal(payload.checkpoint,'/data/model with spaces');
assert.equal(payload.seed,42);assert.equal(payload.gpu_index,2);
assert.equal($('launch-seed').value,'0');
""")
    assert 'id="page-tab-inference"' not in page
    assert 'id="inference-panel"' in page
    assert '<option value="24">16 倍速（默认）</option>' in page


def test_combined_launch_keeps_training_checkpoint_link_and_resume_configuration():
    page=run_viewer("""
global.window={};
loadLaunchOptions=async()=>{};refreshLaunchJobs=async()=>{};
refreshInferenceJobs=async()=>{};loadInferenceOptions=async()=>{};
$('launch-kind').value='resume';$('launch-mode').value='single1';
selectPage('inference');
assert.equal(activePage,'launch');assert.equal($('page-launch').hidden,false);
assert.equal($('inference-panel').open,true);assert.equal($('launch-kind').value,'new');
$('launch-kind').value='resume';updateLaunchForm();
assert.equal($('model-config').hidden,true);
assert.equal($('launch-host').disabled,true);assert.equal($('launch-port').disabled,true);
$('launch-kind').value='new';updateLaunchForm();
assert.equal($('model-config').hidden,false);assert.equal($('launch-host').disabled,false);
""")
    assert 'id="page-inference"' not in page
    assert 'id="page-tab-training"' in page
    assert 'id="launch-host"' in page
    assert 'id="launch-summary"' not in page
    assert '准备开始' not in page


def test_new_policy_blocks_eval_until_ready_without_resetting_task_or_seed():
    run_viewer("""
(async()=>{
  $('launch-kind').value='new';$('launch-mode').value='single2';
  $('launch-task').value='22';$('launch-seed').value='7';
  $('launch-host').value='172.28.62.17';$('launch-port').value='5557';
  launchServiceChoice='172.28.62.17:5557';
  pendingInference={pid:42,port:5557,host:'172.28.62.17',checkpoint:'/models/memory',gpu_index:2,state:'loading'};
  updateLaunchAvailability();
  assert.equal($('launch-submit').disabled,true);
  assert.ok($('launch-submit').textContent.includes('等待'));
  let calls=0;launchAPI=async()=>{calls++;throw Error('Must not submit while model is loading');};
  await submitLaunch(true);assert.equal(calls,0);
  $('inference-panel').open=true;
  inferenceAPI=async()=>({jobs:[{...pendingInference,state:'ready'}]});
  await refreshInferenceJobs();
  assert.equal(pendingInference.state,'ready');
  assert.equal($('launch-submit').disabled,false);assert.equal($('inference-panel').open,false);
  assert.equal($('launch-port').value,'5557');
  assert.equal($('launch-task').value,'22');assert.equal($('launch-seed').value,'7');
  pendingInference.state='failed';updateLaunchAvailability();
  assert.equal($('launch-submit').disabled,true);
  renderLaunchServices({remote:{host:'172.28.62.17',processes:[{kind:'inference',model:'/models/official',configured_port:'5556',listening_ports:[5556]}]}});
  $('launch-service').value='172.28.62.17:5556';$('launch-service').onchange();
  assert.equal(pendingInference,null);assert.equal($('launch-submit').disabled,false);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_inference_version_filters_weights_and_carries_recorded_source():
    run_viewer("""
inferenceOptions={source_dir:'/default',checkpoints:[
  {path:'/weights/V_1_task0_bas/baseline/checkpoints/step-1000',version:'V_1',label:'V_1 baseline',source_dir:'/snapshot1'},
  {path:'/weights/V_2_task0_mem/memory_v1/checkpoints/step-2000',version:'V_2',label:'V_2 memory',source_dir:'/snapshot2'}
]};
$('inference-version').value='V_2';renderInferenceWeights(true);
assert.ok($('inference-weight').innerHTML.includes('V_2 memory'));
assert.ok(!$('inference-weight').innerHTML.includes('V_1 baseline'));
assert.equal($('inference-checkpoint').value,inferenceOptions.checkpoints[1].path);
assert.equal($('inference-source').value,'/snapshot2');
""")


def test_stopping_policy_requires_confirmation_sends_exact_identity_and_blocks_eval():
    run_viewer("""
(async()=>{
  const job={pid:42,port:5557,checkpoint:'/weights/V_2_task0_mem/memory_v1/checkpoints/step-2000',
    start_ticks:123,version:'V_2',label:'V_2 memory',host:'172.28.62.17',gpu_index:1,state:'ready'};
  inferenceJobs=[job];let calls=[];
  global.window={confirm:()=>false};await stopInference(42);assert.equal(calls.length,0);
  $('launch-kind').value='new';$('launch-host').value=job.host;$('launch-port').value='5557';
  global.window.confirm=()=>true;
  inferenceAPI=async(path,payload)=>{
    calls.push([path,payload]);
    return path==='stop'?{...job,state:'stopping'}:{jobs:[{...job,state:'stopped'}]};
  };
  refreshServices=async()=>{};
  await stopInference(42);
  assert.deepEqual(calls[0],['stop',{pid:42,port:5557,checkpoint:job.checkpoint,start_ticks:123}]);
  assert.equal($('launch-submit').disabled,true);
  assert.equal(pendingInference.state,'stopped');
  assert.ok($('inference-manage-message').textContent.includes('已关闭'));
  assert.equal($('inference-history').hidden,true);
  assert.ok(!$('inference-jobs').innerHTML.includes('V_2'));
  assert.ok(!$('inference-jobs').innerHTML.includes('data-stop-policy'));
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_policy_lists_group_baseline_and_memory_and_exclude_stopped_services():
    run_viewer("""
const common={version:'V_2',host:'172.28.62.17',gpu_index:1,start_ticks:123};
const baseline={...common,pid:41,port:5556,checkpoint:'/weights/V_2_task0_bas/baseline/checkpoints/step-1000',state:'ready'};
const memory={...common,pid:42,port:5557,checkpoint:'/weights/V_2_task0_mem/memory_v1/checkpoints/step-2000',state:'ready'};
const stopped={...memory,pid:43,port:5558,state:'stopped'};
inferenceJobs=[baseline,memory,stopped];renderInferenceJobs();
assert.ok($('inference-jobs').innerHTML.includes('data-policy-version="V_2"'));
assert.ok($('inference-jobs').innerHTML.includes('<h3>Baseline</h3>'));
assert.ok($('inference-jobs').innerHTML.includes('<h3>Memory</h3>'));
assert.ok(!$('inference-jobs').innerHTML.includes('5558'));
inferenceOptions={checkpoints:[{path:baseline.checkpoint,label:'bas'}, {path:memory.checkpoint,label:'mem'}]};
renderInferenceWeights();
assert.ok($('inference-weight').innerHTML.includes('optgroup label="Baseline"'));
assert.ok($('inference-weight').innerHTML.includes('optgroup label="Memory"'));
pendingInference=stopped;
renderLaunchServices({remote:{host:common.host,processes:[
  {kind:'inference',pid:43,configured_port:5558,model:stopped.checkpoint,listening_ports:[5558]}
]}});
assert.ok(!$('launch-service').innerHTML.includes('5558'));
assert.ok($('launch-service').innerHTML.includes('5556'));
assert.ok($('launch-service').innerHTML.includes('5557'));
assert.ok($('launch-service').innerHTML.includes('optgroup label="V_2 / Baseline"'));
assert.ok($('launch-service').innerHTML.includes('optgroup label="V_2 / Memory"'));
""")


def test_preview_defaults_off_polls_only_when_enabled_and_stops_on_run_change():
    run_viewer("""
(async()=>{
  const calls=[], timers=new Map();let nextTimer=0;
  global.setTimeout=(callback)=>{timers.set(++nextTimer,callback);return nextTimer;};
  global.clearTimeout=id=>timers.delete(id);
  global.fetch=async (url,options)=>{
    calls.push([url,options]);
    return {ok:true,json:async()=>({supported:true,server_time:100,environments:[{
      gpu_index:1,env_idx:0,instance_id:4,seed:7,episode_number:2,action_steps:80,
      captured_at:100,status:'running',cameras:[{name:'head',label:'头部',src:'data:image/jpeg;base64,YWJj'}]
    }]})};
  };
  const run={_folder:'V_1/baseline/run',status:'running'};
  selected=run;renderPreviewControl(run);
  assert.equal(calls.length,0);assert.equal(previewEnabled,false);
  $('preview-toggle').checked=true;togglePreview();
  await new Promise(setImmediate);
  assert.equal(JSON.parse(calls[0][1].body).enabled,true);
  assert.ok(calls[1][0].includes('/api/preview?run='));
  assert.ok($('preview-environments').innerHTML.includes('GPU 1 / env 0'));
  assert.ok($('preview-environments').innerHTML.includes('实例 4'));
  assert.equal(timers.size,1);
  renderPreviewControl({...run});
  assert.equal(previewEnabled,true); // Five-second results refresh preserves preview.
  renderPreviewControl({_folder:'V_1/experment_1/run',status:'running'});
  assert.equal(previewEnabled,false);assert.equal(timers.size,0);
  assert.equal($('preview-environments').hidden,true);
  assert.equal(JSON.parse(calls.at(-1)[1].body).enabled,false);
  assert.equal(JSON.parse(calls.at(-1)[1].body).run,run._folder);
  renderPreviewControl({...run,status:'complete'});
  assert.equal($('preview-toggle').disabled,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_preview_ignores_response_after_switching_off_and_labels_stale_frames():
    run_viewer("""
(async()=>{
  let finishRead;
  global.fetch=async (url,options)=>url==='/api/preview'?{ok:true}:
    {ok:true,json:()=>new Promise(resolve=>{finishRead=resolve;})};
  selected={_folder:'V_1/baseline/run',status:'running'};
  renderPreviewControl(selected);$('preview-toggle').checked=true;togglePreview();
  await new Promise(setImmediate);
  $('preview-toggle').checked=false;togglePreview();
  finishRead({supported:true,environments:[{gpu_index:99}]});
  await new Promise(setImmediate);
  assert.equal($('preview-environments').hidden,true);
  assert.equal($('preview-environments').innerHTML,'');
  assert.ok($('preview-status').textContent.includes('默认关闭'));
  renderPreviewFrames({supported:true,server_time:120,environments:[{
    gpu_index:0,env_idx:1,instance_id:5,seed:1,status:'running',captured_at:100,
    cameras:[{label:'头部',src:'data:image/jpeg;base64,YWJj'}]
  }]});
  assert.ok($('preview-environments').innerHTML.includes('画面暂未更新'));
  assert.ok($('preview-environments').innerHTML.includes('20秒前'));
  renderPreviewFrames({supported:false,environments:[]});
  assert.ok($('preview-status').textContent.includes('下次启动或续跑'));
  assert.equal($('preview-environments').hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


@pytest.mark.parametrize("run,expected", [
    ({"configuration": {"record_video": False}}, "未开启录像"),
    ({"status": "running", "configuration": {"record_video": True}}, "封存"),
    ({"status": "complete", "evaluation_mode": "fast-debug", "configuration": {"record_video": True}}, "截断"),
    ({"status": "complete", "configuration": {"record_video": True}}, "未保留"),
])
def test_no_video_message_explains_recording_state(run, expected):
    run_viewer(f"""
renderVideos({json.dumps(run)});
assert.ok($('video-empty').textContent.includes({json.dumps(expected)}));
assert.equal($('video-player').hidden, true);
assert.equal($('video-select').hidden, true);
""")


def test_finished_run_hides_stale_environment_and_marks_debug_episodes():
    page = run_viewer("""
const run = {_folder: 'baseline/groot_task21_1', status: 'complete', evaluation_mode: 'fast-debug',
  per_gpu: {'1': {environments: {runtime_env0: {instance_id: 0, seed: 0}}}},
  _episodes: [{episode_index: 1, instance_id: 0, seed: 0, valid: true, success: false}]};
renderDetails(run);
assert.ok(!$('details').innerHTML.includes('当前环境'));
assert.ok($('details').innerHTML.includes('调试'));
assert.ok(!$('details').innerHTML.includes('>失败<'));
renderDetails({...run, status: 'running'});
assert.ok($('details').innerHTML.includes('当前环境'));
""")
    assert '<details class="foldout"' in page
    assert '<details class="foldout" open' not in page


def test_failed_run_displays_error_and_instance_without_opening_logs():
    run_viewer("""
renderDetails({status: 'failed', failure_reason: {
  message: 'PhysX 环境已不可恢复；停止评估。', gpu_index: 0,
  env_idx: 1, instance_id: 5, seed: 1}});
assert.ok($('details').innerHTML.includes('PhysX 环境已不可恢复'));
assert.ok($('details').innerHTML.includes('GPU 0 / env 1 · 实例 5 · seed 1'));
assert.ok(!$('details').innerHTML.includes('当前环境'));
""")


def test_service_refresh_updates_counts_ports_and_failed_remote_check():
    run_viewer("""
renderServices({local: {ok: true, processes: [
  {kind: 'evaluation', pid: 11, gpu: '1', n_envs: 2, env_name: 'task21',
   task_id: 21, run_directory: '/project/evaluation/V_1/baseline/groot_task21_2',
   policy_host: '172.28.62.17', policy_port: '5556'}]},
  remote: {ok: true, processes: [
    {kind: 'inference', pid: 13, gpu: '0', configured_port: '5556', listening_ports: [5556], model: '/models/base'},
    {kind: 'inference', pid: 14, configured_port: '5562', listening_ports: [], model: '<memory>'}]}});
assert.equal($('local-service-count').textContent, '1 个评估进程 · 2 个环境');
assert.ok($('local-services').innerHTML.includes('172.28.62.17:5556'));
assert.ok($('local-services').innerHTML.includes('GPU 1'));
assert.ok($('local-services').innerHTML.includes('Task21'));
assert.ok($('local-services').innerHTML.includes('V_1/baseline/groot_task21_2'));
assert.equal($('remote-service-count').textContent, '2 个推理进程 · 1 个已监听');
assert.ok($('remote-services').innerHTML.includes('监听端口：5556'));
assert.ok($('remote-services').innerHTML.includes('未监听 · 配置端口 5562'));
assert.ok($('remote-services').innerHTML.includes('&lt;memory&gt;'));
renderServices({local: {ok: true, processes: []}, remote: {ok: false, error: 'SSH 检查失败或超时'}});
assert.equal($('local-service-count').textContent, '0 个评估进程 · 0 个环境');
assert.equal($('remote-service-count').textContent, '无法检查');
assert.ok($('remote-services').textContent.includes('SSH'));
""")


def test_inference_cards_distinguish_models_with_the_same_checkpoint_step():
    run_viewer("""
renderServices({remote: {ok: true, processes: [
  {kind: 'inference', pid: 10, configured_port: '5560', listening_ports: [5560],
   model: '/training/task21_s42/baseline/checkpoints/step-010000'},
  {kind: 'inference', pid: 11, configured_port: '5561', listening_ports: [5561],
   model: '/training/task21_s42/memory_v1/checkpoints/step-010000'}]}});
const cards = $('remote-services').innerHTML;
assert.ok(cards.includes('baseline/checkpoints/step-010000</strong>'));
assert.ok(cards.includes('memory_v1/checkpoints/step-010000</strong>'));
assert.ok(cards.includes('title="/training/task21_s42/baseline/checkpoints/step-010000"'));
""")


def test_run_filter_selection_and_refresh_keep_video_and_correct_summary():
    run_viewer("""
(async()=>{
  const baseline = {_folder: 'V_1/baseline/groot_task21_1', version: 'V_1', group: 'baseline',
    task_id: 21, status: 'running', evaluation_mode: 'official-eval', completed_episodes: 2,
    requested_episodes: 10, mean_q_score: .25, q_score_episodes: 2, success_rate: 0,
    _videos: ['videos/gpu0/one.mp4'], configuration: {run_tag: 'control'}};
  const memory = {...baseline, _folder: 'V_1/experment_1/groot_task21_1', group: 'experment_1',
    mean_q_score: .5, configuration: {run_tag: 'memory_v1'}};
  const finished = {...baseline, _folder: 'V_0/baseline/groot_task22_1', version: 'V_0', task_id: 22,
    status: 'complete', completed_episodes: 10, _videos: []};
  runs = [baseline, memory, finished]; render();
  assert.equal(runCards().length, 3);
  assert.equal(selected, baseline);
  runCards()[1].onclick();
  assert.equal(selected, memory);
  assert.ok($('summary').innerHTML.includes('0.5000'));
  assert.equal($('detail-path').textContent, 'V_1 / experment_1');
  const player = $('video-player'); player.currentTime = 17;
  const loads = player.loads, pauses = player.pauses;
  global.fetch = async url => ({ok: true, json: async()=>url==='/api/services'?{}:
    {updated_at: '12:00:00', runs: [baseline, {...memory, completed_episodes: 3}, finished]}});
  await refresh();
  assert.equal(selected._folder, memory._folder);
  assert.equal(player.loads, loads); assert.equal(player.pauses, pauses);
  assert.equal(player.currentTime, 17);
  assert.ok($('summary').innerHTML.includes('3 / 10'));
  $('run-search').value = 'MEMORY_V1'; render();
  assert.equal(runCards().length, 1);
  assert.equal(selected.group, 'experment_1');
  $('run-search').value = ''; $('status-filter').value = 'complete'; render();
  assert.equal(selected.task_id, 22);
  assert.equal($('run-count').textContent, '1 / 3 条');
  assert.ok(!$('details').innerHTML.includes('当前环境'));
  $('run-search').value = '不存在'; render();
  assert.equal(selected, null); assert.equal($('empty').hidden, false);
  assert.equal($('video-player').hidden, true);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")
