"""The default benchmark must expose real 2026 rollouts, independently of 2025."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation/scripts'))
import view_results as viewer


def test_native_benchmark_jobs_do_not_mix_legacy_or_query_unrelated_devices():
    class Local:
        def snapshot(self):
            return [dict(id='radio', kind='b1k2026_websocket', state='complete'),
                    dict(id='legacy', version='V_2_1', state='complete')]

    class UnrelatedRemote:
        def snapshot(self):
            raise AssertionError('2026 local results must not wait for a legacy device')

    class Handler(viewer.Handler):
        launch_manager = Local()
        evaluation_devices = {'a6000': UnrelatedRemote()}

    with viewer.ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f'http://127.0.0.1:{server.server_port}/api/launch/jobs?benchmark=b1k2026') as response:
                data = json.load(response)
            assert [job['id'] for job in data['jobs']] == ['radio']
            assert data['device_errors'] == {}
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_benchmark_switch_shows_native_video_and_preserves_playback():
    node = shutil.which('node') or '/home/1024201092WYH/.nvm/versions/node/v22.23.1/bin/node'
    harness = r"""
const assert=require('node:assert/strict');
const elements={};
function element(){return {value:'',hidden:false,textContent:'',innerHTML:'',dataset:{},options:[],
  children:[],loads:0,pauses:0,src:'',currentTime:0,
  append(node){this.children.push(node)},appendChild(node){this.append(node)},before(){},
  querySelector(selector){return get(selector)},querySelectorAll(){return []},
  setAttribute(k,v){this[k]=v},removeAttribute(k){this[k]=''},
  load(){this.loads++},pause(){this.pauses++}};}
function get(id){return elements[id]||=(element());}
global.document={createElement:element,getElementById:get,querySelector:get,querySelectorAll:()=>[],head:element(),body:element()};
global.location=new URL('http://workbench.test/');
global.history={replaceState(_s,_t,url){global.location=new URL(url)}};
global.window={__benchmarkHTML:'',refresh:async()=>{}};
global.setInterval=()=>0;global.stopPreview=()=>{};
global.updateLaunchForm=()=>{};
global.selectPage=page=>window.benchmarkWorkbench.route(page);
const video='/api/launch/artifact?run=public_test_instance_18_attempt_02&path=videos%2Fradio.mp4';
const latest={id:'attempt2',kind:'b1k2026_websocket',state:'complete',started_at:'2026-11-28 14:27:32',
  folder:'V_1_2/official/public_test_instance_18_attempt_02',task_id:'turning_on_radio',
  notes:'V1.2 / N1.7 / B1K2026: task prompt',language:{text_prompt:'Turn on the radio'},
  native_metrics:[{instance_id:319,steps:3225,success:false,q_score:0}],
  artifacts:[{path:'videos/radio.mp4',url:video,label:'native rollout video'},
             {path:'json/radio.json',url:'/api/launch/artifact?run=radio&path=json%2Fradio.json',label:'native rollout JSON'}]};
let source=[{...latest,id:'startup',state:'failed',native_metrics:[],artifacts:[]},latest,
  {...latest,id:'attempt1',started_at:'2026-11-28 13:29:55',notes:'official default'},
  {id:'old2025',kind:'legacy',state:'complete'}];
const calls=[];
global.fetch=async url=>{calls.push(url);return {ok:true,json:async()=>({jobs:source})}};
get('n17-result-filter').value='rollouts';
get('launch-kind').options=[{value:'new'},{value:'n17'},{value:'resume'}];
"""
    assertions = r"""
(async()=>{
  const work=window.benchmarkWorkbench;
  assert.equal(work.current(),'b1k2026');
  await work.refresh();
  assert.equal(get('page-results').hidden,true,'legacy results must be hidden by default');
  assert.equal(get('n17-results').hidden,false);
  assert.match(get('n17-result-list').innerHTML,/attempt2/);
  assert.ok(!get('n17-result-list').innerHTML.includes('old2025'));
  assert.ok(!get('n17-result-list').innerHTML.includes('data-native-job="startup"'));
  assert.match(get('n17-result-status').textContent,/回合完成/);
  assert.match(get('n17-result-metrics').innerHTML,/未成功/);
  assert.match(get('n17-result-metrics').innerHTML,/3225/);
  assert.match(get('n17-result-metrics').innerHTML,/0/);
  assert.equal(get('n17-result-video').src,new URL(video,location.origin).href);
  get('n17-result-video').currentTime=25;
  const loads=get('n17-result-video').loads;
  await work.refresh();
  assert.equal(get('n17-result-video').loads,loads);
  assert.equal(get('n17-result-video').currentTime,25);
  work.route('training');
  assert.equal(get('page-training').hidden,true);
  assert.equal(get('n17-training').hidden,false);
  work.route('launch');
  assert.equal(get('page-launch').hidden,false);
  assert.equal(get('launch-kind').value,'n17');
  assert.ok(get('launch-kind').options.find(o=>o.value==='new').disabled);
  work.route('results');
  get('benchmark-choice').value='b1k';get('benchmark-choice').onchange();
  assert.equal(work.current(),'b1k');
  assert.equal(get('n17-results').hidden,true);
  assert.equal(get('page-results').hidden,false);
  assert.ok(get('n17-result-video').pauses>0);
  work.route('launch');
  assert.equal(get('launch-kind').value,'new');
  get('benchmark-choice').value='b1k2026';get('benchmark-choice').onchange();
  work.route('results');await work.refresh();
  assert.equal(get('n17-results').hidden,false);
  assert.ok(calls.every(url=>url==='/api/launch/jobs?benchmark=b1k2026'));
  source=[];await work.refresh();
  assert.equal(get('n17-result-video').src,'');
  assert.match(get('n17-result-list').innerHTML,/暂无/);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node], input=harness + (ROOT / 'evaluation/scripts/benchmark_workbench.js').read_text() + assertions,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_2026_refresh_does_not_read_legacy_result_summaries():
    from test_groot_viewer_ui import run_viewer

    run_viewer(r"""
(async()=>{
  let nativeRefreshes=0;
  window.benchmarkWorkbench={current:()=> 'b1k2026',isNative:()=>false,
    refresh:async()=>{nativeRefreshes++}};
  const calls=[];
  global.fetch=async url=>{calls.push(url);return {ok:true,json:async()=>({})}};
  await refresh();
  assert.equal(nativeRefreshes,1);
  assert.ok(!calls.some(url=>url==='/api/results'));
})().catch(error=>{console.error(error);process.exitCode=1});
""")
