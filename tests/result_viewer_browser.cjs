const {chromium}=require('../outputs/browser-tools/node_modules/playwright');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const {join}=require('node:path');

async function checkTask40Collection(page){
  page.setDefaultTimeout(10000);
  const base='http://workbench.test';
  const policy={task40_diagnostics:true,state:'ready',gpu_index:2,host:'172.28.62.17',port:5690,
    pid:1234,checkpoint:'/models/memory_v1/step-004000',version:'V_2',label:'MemoryVLA'};
  const gpu=index=>({index,name:'RTX 4090',total_mb:49140,used_mb:1000});
  const source='/deployments/task40_exp07_diagnostic_test';
  const output='Output/test_01/experiment_09/runs/abc123';
  const artifactUrl='/api/launch/artifact?run=abc123&path=videos%2Fepisode.mp4';
  const requests=[];
  let jobs=[], showDual=false, showTrace=false;
  const trace={...policy,port:5693,pid:1237,task40_trace:true};
  const forwardResults='/api/launch/artifact?run=def456&path=results.json';
  const evaluationFile=name=>readFileSync(join(__dirname,'../evaluation/scripts',name),'utf8');
  const trainingFile=name=>readFileSync(join(__dirname,'../train',name),'utf8');
  await page.route('**/*',async route=>{
    const request=route.request(), url=new URL(request.url());
    assert.equal(url.origin,base,'test should only request mocked workbench routes');
    const device=url.searchParams.get('device');
    const json=async data=>route.fulfill({json:data});
    switch(url.pathname){
      case '/':
        return route.fulfill({contentType:'text/html',path:join(__dirname,'../evaluation/scripts/result_viewer.html')});
      case '/benchmarks.js':
        return route.fulfill({contentType:'application/javascript',body:[
          'window.__benchmarkHTML = '+JSON.stringify(evaluationFile('benchmark_workbench.html'))+';',
          evaluationFile('benchmark_workbench.js'),evaluationFile('libero_workflow.js'),
        ].join('\n')});
      case '/training.js':
        return route.fulfill({contentType:'application/javascript',body:
          'window.__grootTrainingPageHTML = '+JSON.stringify(trainingFile('dashboard.html'))+';\n'+trainingFile('dashboard.js')});
      case '/api/results': return json({runs:[],updated_at:'12:00:00'});
      case '/api/services': return json({local:{},remote:{}});
      case '/api/benchmarks/options': return json({});
      case '/api/launch/options': return json({ready:true,tasks:[{id:40,name:'microwave'}],versions:{},
        gpus:device==='local'?[gpu(0),gpu(1)]:[gpu(0)],
        ...(showDual&&device==='local'?{task40_collect:{sim_slots:{0:2}}}:{})});
      case '/api/inference/options': return json({host:policy.host,gpus:[gpu(2)],sources:[source],source_dir:'/deployments/regular',
        python_path:'/env/bin/python',checkpoints:[{path:policy.checkpoint,label:policy.label}]});
      case '/api/inference/jobs': return json({jobs:device==='a100'?[policy,
        {...policy,port:5691,pid:1235,state:'loading'}, {...policy,port:5692,pid:1236,task40_diagnostics:false},
        {...trace,port:5694,pid:1238,state:'loading'}, ...(showTrace?[trace]:[])]:[]});
      case '/api/launch/jobs': return json({jobs});
      case '/api/inference/preview':
        requests.push([url.pathname,request.postDataJSON()]);
        return json({log:'/logs/inference.log',command:'check-inference'});
      case '/api/launch/preview':
      case '/api/launch/start': {
        const body=request.postDataJSON();
        requests.push([url.pathname,body]);
        if(body.task40_forward){
          const plan={kind:'task40_forward',batch:body.batch,output:output.replace('abc123','def456'),policy_identity:trace};
          if(url.pathname.endsWith('/preview')) return json({command:'check-forward',plan});
          const job={...plan,id:'forward1',state:'running',pid:2001,policy_port:trace.port,
            progress:{stage:'frozen_forward',completed:1,index:2,total:3,condition:'normal'},
            log_tail:'Forward worker started',forward_log_tail:'Evaluating normal history'};
          jobs=[job,...jobs];
          return json({...job,command:'run-forward'});
        }
        if(url.pathname.endsWith('/preview')) return json({command:'check-collection',plan:{kind:'task40_collect',output,policy_identity:policy}});
        jobs=[{id:'collect1',kind:'task40_collect',gpus:[0],state:'running',seed:body.seed,pid:2000,output,
          policy_port:policy.port,policy_identity:policy,progress:{stage:'closed_loop',trial:'episode_i4_s76100',decisions:12},
          log_tail:'Worker started',sim_log_tail:'Isaac ready',artifacts:[{label:'轨迹录像',path:'videos/episode.mp4',url:artifactUrl},
            {label:'外部链接',url:'https://example.com/video.mp4'}]}];
        return json({...jobs[0],command:'run-collection'});
      }
      default: return route.fulfill({status:404,body:'Not found'});
    }
  });
  await page.goto(base+'/?page=launch');
  await page.locator('#launch-kind').selectOption('collect');
  await page.locator('#collect-policy option[value="172.28.62.17:5690"]').waitFor({state:'attached'});
  assert.equal(await page.locator('#collect-policy option').count(),2,'only ready Task40 diagnostics are selectable');
  assert.equal(await page.locator('#collect-slots-field').isVisible(),false,'dual needs backend validation');
  await page.locator('#collect-seed').fill('76100');
  await page.locator('#collect-instances').fill('4,4');
  await page.locator('#launch-check').click();
  await page.getByText(/1–10 个不同的实例编号/).first().waitFor();
  assert.equal(requests.length,0,'invalid instances should not reach the server');
  await page.locator('#collect-instances').fill('4，7');
  await page.locator('#launch-check').click();
  await page.getByText(/配置检查通过/).waitFor();
  const expected={device:'local',task40_collect:true,sim_gpu:0,seed:76100,cycles:300,instances:[4,7],policy_host:policy.host,policy_port:policy.port};
  assert.deepEqual(requests[0],['/api/launch/preview',expected]);
  assert.match(await page.locator('#launch-plan').textContent(),/step-004000/);

  await page.locator('#inference-panel > summary').click();
  await page.locator('#inference-gpu option[value="2"]').waitFor({state:'attached'});
  await page.locator('#inference-panel details').getByText('推理高级设置 · 源码、环境与种子').click();
  await page.locator('#inference-task40').check();
  await page.locator('#inference-check').click();
  await page.getByText('检查通过，尚未启动推理。').waitFor();
  assert.deepEqual(requests[1],['/api/inference/preview',{device:'a100',checkpoint:policy.checkpoint,gpu_index:2,port:5557,
    source_dir:source,python_path:'/env/bin/python',seed:42,history_mode:'normal',task40_diagnostics:true}]);

  showDual=true;
  await page.locator('#launch-device').selectOption('a6000');
  await page.getByText('该设备尚未开放 Task40 轨迹采集。').waitFor();
  assert.equal(await page.locator('#launch-submit').isEnabled(),false);
  await page.locator('#launch-device').selectOption('local');
  await page.locator('#collect-slots-field').waitFor();
  await page.locator('#collect-slots').selectOption('2');
  await page.locator('#collect-gpu').selectOption('1');
  assert.equal(await page.locator('#collect-slots-field').isVisible(),false);
  await page.locator('#collect-gpu').selectOption('0');
  assert.equal(await page.locator('#collect-slots').inputValue(),'1');
  await page.locator('#collect-slots').selectOption('2');
  await page.setViewportSize({width:390,height:844});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Task40 collection overflows on mobile');
  await page.locator('#launch-submit').click();
  const job=page.locator('[data-launch-job="collect1"]');
  await job.getByText(/闭环执行.*决策 12/).waitFor();
  assert.deepEqual(requests[2],['/api/launch/start',{...expected,sim_slots:2}]);
  assert.match(await job.textContent(),/step-004000/);
  assert.equal(await job.getByRole('link',{name:'轨迹录像'}).getAttribute('href'),base+artifactUrl);
  assert.equal(await job.getByRole('link',{name:'外部链接'}).count(),0);
  await job.getByText('仿真日志').click();
  await job.getByText('Isaac ready').waitFor();
  jobs=[{...jobs[0],state:'failed',returncode:1,sim_log_tail:'PhysX error',artifact_sync_error:'connection lost'}];
  await page.evaluate(()=>refreshLaunchJobs());
  await job.getByText('PhysX error').waitFor();
  await job.getByText(/产物同步失败/).waitFor();
  await job.getByText('启动日志 · 退出码 1').waitFor();
  await job.getByRole('link',{name:'轨迹录像'}).waitFor();

  await page.locator('#launch-device').selectOption('a6000');
  await page.getByText('该设备尚未开放 Task40 轨迹采集。').waitFor();
  await page.locator('#inference-device').selectOption('a6000');
  await page.locator('#launch-kind').selectOption('forward');
  assert.equal(await page.locator('#launch-device-field').isVisible(),false);
  assert.equal(await page.locator('#collect-gpu').isVisible(),false);
  assert.equal(await page.locator('#launch-submit').isEnabled(),false,'forward needs a ready trace service');
  showTrace=true;
  await page.evaluate(()=>refreshCollectionPolicies());
  await page.locator('#collect-policy option[value="172.28.62.17:5693"]').waitFor({state:'attached'});
  assert.equal(await page.locator('#collect-policy option').count(),2,'forward should show only the ready A100 trace endpoint');
  await page.locator('#forward-batch').fill('/tmp/batch.json');
  await page.locator('#launch-check').click();
  await page.getByText(/批次配置须填写 Experiment 09/).waitFor();
  assert.equal(requests.length,3,'invalid forward batch should not reach the server');
  const batch='Output/test_01/experiment_09/configs/forward_batch.json';
  const expectedForward={device:'local',task40_forward:true,batch,policy_host:trace.host,policy_port:trace.port};
  await page.locator('#forward-batch').fill(batch);
  await page.locator('#launch-check').click();
  await page.getByText(/配置检查通过/).waitFor();
  assert.deepEqual(requests[3],['/api/launch/preview',expectedForward]);
  assert.match(await page.locator('#launch-plan').textContent(),/forward_batch\.json.*本地控制器/);
  await page.locator('#launch-submit').click();
  const forward=page.locator('[data-launch-job="forward1"]');
  await forward.getByText(/冻结前向.*完成 1\/3.*当前 2\/3.*normal/).waitFor();
  assert.deepEqual(requests[4],['/api/launch/start',expectedForward]);
  await forward.getByText('前向日志',{exact:true}).click();
  await forward.getByText('Evaluating normal history').waitFor();
  jobs[0]={...jobs[0],state:'complete',result_ready:true,progress:{stage:'completed',completed:3,total:3},
    artifacts:[{path:'results.json',label:'轨迹结果',url:forwardResults}]};
  await page.evaluate(()=>refreshLaunchJobs());
  await forward.getByText(/进度 已完成.*3\/3/).waitFor();
  assert.equal(await forward.getByRole('link',{name:'前向结果'}).getAttribute('href'),base+forwardResults);
  assert.equal(await forward.getByRole('button',{name:/评估结果/}).count(),0);
  console.log('Task40 collection and frozen forward previews, launch, service availability, progress, logs, and artifacts passed.');
}

(async()=>{
  const base=process.argv[2]||'http://127.0.0.1:8765';
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  try{
    if(process.argv.includes('--task40')){
      await checkTask40Collection(page);
      assert.deepEqual(errors,[]);
      return;
    }
    const api=await page.request.get(base+'/api/results');
    const data=await api.json();
    const run=data.runs.find(item=>item._episodes?.length);
    assert.ok(run,'Need at least one run with episode metrics');
    let delayResults=false,delayedRequests=0;
    await page.route('**/api/results',route=>{
      if(delayResults){delayedRequests++;setTimeout(()=>route.fulfill({json:{...data,runs:[run]}}),300);}
      else void route.fulfill({json:{...data,runs:[run]}});
    });
    const response=await page.goto(base,{waitUntil:'networkidle',timeout:30000});
    assert.equal(response.status(),200);
    await page.waitForFunction(()=>document.getElementById('caption').textContent.includes('次评估'));
    assert.equal(await page.locator('#page-tab-ai').getAttribute('href'),'/ai');
    assert.equal(await page.locator('.page-nav').count(),1);
    assert.equal(await page.locator('.page-nav a[href*="8772"]').count(),0);
    assert.equal(await page.locator('.page-tabs[role="tablist"] [role="tab"]').count(),4);
    await page.waitForFunction(()=>document.querySelectorAll('#episode-table tbody tr').length>0);
    await page.screenshot({path:'/tmp/memory-vla-result-viewer-desktop.png'});
    delayResults=true;
    await page.evaluate(()=>{refresh();refresh();refresh();});
    await page.waitForTimeout(400);
    assert.equal(delayedRequests,1,'overlapping refresh calls should share one results request');
    for(const width of [1440,1024,780,390]){
      await page.setViewportSize({width,height:width<600?844:1000});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`page overflow at ${width}px`);
    }
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.locator('#episode-table td[data-label="Q"]').first().count());
    assert.ok(await page.locator('#episode-table td[data-label="人工标注"]').first().count());
    await page.screenshot({path:'/tmp/memory-vla-result-viewer-mobile.png'});
    await page.goto(base+'/ai',{waitUntil:'networkidle',timeout:30000});
    assert.match(await page.title(),/研究协作/);
    assert.deepEqual(errors,[]);
    console.log('Result viewer navigation, API refresh, responsive layouts, and same-origin AI route passed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
