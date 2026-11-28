// Real Chromium on production templates; every network request is a local fixture.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.join(__dirname,'..'), evaluation=name=>fs.readFileSync(path.join(root,'evaluation/scripts',name),'utf8'), training=name=>fs.readFileSync(path.join(root,'train',name),'utf8');
const base='http://workbench.test', output=process.env.SCREENSHOT_DIR||'workbench-screenshots';
fs.mkdirSync(output,{recursive:true});
const gpu=index=>({index,name:'NVIDIA A100',total_mb:81920,used_mb:12000,utilization:20});
const checkpoint='/nvme3/models/GR00T-N1.6-3B';
const task={task:40,task_name:'microwave_popcorn',dataset:'/nvme3/datasets/task40',ready:true,training_profile:{total_windows:2000,train_windows:1800,total_episodes:200,validation_percent:10,split_seed:42}};
const resources=[{id:'model',kind:'weights',host:'4090',name:'GR00T-N1.6-3B',path:checkpoint,size_gb:6.5},{id:'data',kind:'dataset',host:'4090',name:'BEHAVIOR1K Task40 demonstrations',path:'/nvme3/datasets/task40',size_gb:24}];
const hosts=[{host:'4090',label:'4090'},{host:'A100',label:'A100'},{host:'A6000',label:'RTX 6000 Ada'}];
const node=(id,title,after)=>({id,title,after,acceptance:'核对结果和保存实验记录'});
const plan={version:'v001',author:'Agent',created_at:'2026-11-28',goal:'验证任务状态记忆对阶段决策的影响',reason:'初始方案',nodes:[node('prepare','准备数据',[]),node('train','模型训练',['prepare']),node('eval','闭环评估',['train']),node('report','分析结果',['eval'])],mainline:['prepare','train','eval','report']};
const agent={experiments:['experiment_01','experiment_02'],detail:{id:'experiment_01',name:'experiment_01',path:'Output/test_01/experiment_01',workflow:{versions:[plan],errors:[]},gates:{},task:{file:'task.md',markdown:'# 实验方案\n\n> 摘要：任务状态监督。'},progress:{file:'progress.md',markdown:'# 执行进度\n\n> 当前动作：训练模型。\n> 下一步：完成评估。\n\n| 阶段 | 状态 | 记录 |\n| --- | --- | --- |\n| 准备数据 | 已完成 | 数据已核验 |\n| 模型训练 | 进行中 | 训练 3000/6000 步 |'}},live:{jobs:[]}};
const run={_folder:'V_2_1/baseline/task40/run001',version:'V_2_1',group:'baseline',method:'baseline',task_id:40,task_name:'microwave_popcorn',status:'complete',evaluation_mode:'official-eval',requested_episodes:10,completed_episodes:10,recorded_episodes:10,successful_episodes:6,success_rate:0.6,mean_task_progress:70,q_score_episodes:10,mean_q_score:0.65,elapsed_wall_s:3600,configuration:{},episodes:[],_videos:[]};
async function routeAll(page){
 await page.route('**/*',async r=>{
  const u=new URL(r.request().url()), p=u.pathname, q=u.searchParams;
  assert.equal(u.origin,base);
  const json=data=>r.fulfill({json:data});
  const script=body=>r.fulfill({contentType:'application/javascript',body});
  if(r.request().method()!=='GET')return r.fulfill({status:400,json:{error:'Fixture: writes disabled'}});
  if(p==='/'||p==='/ai')return r.fulfill({contentType:'text/html',body:evaluation('result_viewer.html')});
  if(p==='/registry')return r.fulfill({contentType:'text/html',body:evaluation('registry_view.html')});
  if(p==='/benchmarks.js')return script('window.__benchmarkHTML='+JSON.stringify(evaluation('benchmark_workbench.html'))+';\n'+evaluation('benchmark_workbench.js')+'\n'+evaluation('libero_workflow.js'));
  if(p==='/training.js')return script('window.__grootTrainingPageHTML='+JSON.stringify(training('dashboard.html'))+';\n'+training('dashboard.js'));
  if(['/checkpoint_notes.js','/agent_experiments.js','/registry_storage.js','/resource_downloads.js','/task40_diagnostics.js'].includes(p))return script(evaluation(p.slice(1)));
  if(p==='/api/results')return json({runs:[run],updated_at:'12:00'});
  if(p==='/api/services')return json({local:{ok:true,gpus:[gpu(0),gpu(1)],processes:[]},remote:{ok:true,gpus:[gpu(0),gpu(1),gpu(2),gpu(3)],processes:[]},a6000:{ok:true,gpus:[gpu(0)],processes:[]}});
  if(p==='/api/training/options')return json({datasets:[task,...[0,21,34].map(i=>({...task,task:i,task_name:'training_task_'+i}))],gpus:[0,1,2,3].map(gpu),checkpoints:[{path:checkpoint,label:'GR00T-N1.6-3B',training_profile:{kind:'base'}}],model_versions:{V_1:{backend:'v1',description:'Baseline',prefix:'v1'}},sync:{}});
  if(p==='/api/training/jobs')return json({jobs:[],sync:{}});
  if(p==='/api/launch/options')return json({ready:true,tasks:[{id:40,name:'microwave_popcorn'}],gpus:[gpu(0),gpu(1)],versions:{V_1:['baseline'],V_2_1:['baseline']}});
  if(p==='/api/inference/options')return json({host:'172.28.62.17',gpus:[gpu(0),gpu(1)],sources:['/deployments/groot'],source_dir:'/deployments/groot',python_path:'/env/bin/python',checkpoints:[{path:checkpoint,label:'GR00T-N1.6-3B'}]});
  if(p==='/api/inference/jobs'||p==='/api/launch/jobs'||p==='/api/benchmarks/jobs')return json({jobs:[]});
  if(p==='/api/agent-experiments')return json(agent);
  if(p==='/api/registry')return json({items:resources,hosts,experiments:[],generated_at:'2026-11-28 12:00'});
  if(p==='/api/registry/annotations')return json({});
  if(p==='/api/registry/storage'){
   if(q.get('action')==='hosts')return json({hosts});
   if(q.get('action')==='disks')return json({disks:[{mount:'/nvme3',total:1000*1024**3,used:400*1024**3,free:600*1024**3,percent:40}]});
   const dir=q.get('path')||checkpoint;
   return json({type:'directory',path:dir,entries:[{type:'file',path:dir+'/config.json',name:'config.json',size:4096,modified:1760000000,can_delete:false},{type:'directory',path:dir+'/model',name:'model',size:null,modified:1760000000,can_delete:true}],total:2,offset:0,has_more:false,can_delete:true});
  }
  if(p==='/api/resources/downloads'){
   if(q.get('action')==='options')return json({recipes:[{id:'groot-base',host:'A100',repo:'nvidia/GR00T-N1.6-3B',target:checkpoint,kind:'weights'}]});
   return json({status:'running',target:checkpoint,total_bytes:6*1024**3,verified_bytes:3*1024**3,job_id:'fixture'});
  }
  if(p.startsWith('/api/'))return json({});
  return r.fulfill({status:404,body:'Not found'});
 });
}
async function bounds(page,name,selector){
 const result=await page.evaluate(sel=>{
   const root=document.querySelector(sel);
   if(!root)return {missing:sel};
   const r=root.getBoundingClientRect();
   return {viewport:innerWidth,document:document.documentElement.scrollWidth,root:Math.round(r.width),left:Math.round(r.left),right:Math.round(r.right)};
 },selector);
 assert.ok(!result.missing&&result.root>0,name+' missing '+JSON.stringify(result));
 assert.ok(result.document<=result.viewport+2,name+' document overflow '+JSON.stringify(result));
 assert.ok(result.left>=-2&&result.right<=result.viewport+2,name+' workspace overflow '+JSON.stringify(result));
}
(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 try{
  for(const width of [1440,1024,768,390,320]){
   const page=await browser.newPage({viewport:{width,height:900},serviceWorkers:'block'});
   const errors=[];page.on('pageerror',e=>errors.push(e.message));
   await routeAll(page);
   await page.goto(base+'/?page=training');
   await page.locator('#train-task-list input').first().waitFor({state:'attached'});
   for(let i=0;i<4;i++){
    await page.locator('[data-train-setup-step="'+i+'"]').click();
    await bounds(page,'training step '+i,'#train-config');
    if([1440,390].includes(width)&&[0,3].includes(i))await page.screenshot({path:output+'/training-'+i+'-'+width+'.png',fullPage:true});
   }
   await page.locator('#page-tab-launch').click();
   await page.locator('#launch-task option[value="40"]').waitFor({state:'attached'});
   await page.locator('#inference-panel > summary').click();
   await page.locator('#inference-gpu option[value="0"]').waitFor({state:'attached'});
   await bounds(page,'inference','#page-launch');
   if([1440,390].includes(width))await page.screenshot({path:output+'/inference-'+width+'.png',fullPage:true});
   await page.locator('#page-tab-results').click();
   await bounds(page,'results','#page-results');
   if([1440,390].includes(width))await page.screenshot({path:output+'/results-'+width+'.png',fullPage:true});
   await page.locator('#result-compare-panel > summary').click();
   await page.frameLocator('#result-compare-frame').locator('#view-exp').waitFor({state:'visible'});
   await bounds(page,'comparison','#page-results');
   await page.locator('#result-compare-panel > summary').click();
   await page.locator('#page-tab-registry').click();
   const frame=page.frameLocator('#registry-frame');
   await frame.locator('#resource-browser .storage-entry').first().waitFor({state:'visible'});
   await bounds(page,'resources','#page-registry');
   assert.ok(await frame.locator('body').evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'iframe overflow');
   if([1440,390].includes(width))await page.screenshot({path:output+'/resources-'+width+'.png',fullPage:true});
   await frame.locator('[data-v="downloads"]').click();
   await frame.locator('#resource-download-summary strong').first().waitFor({state:'visible'});
   assert.ok(await frame.locator('body').evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'downloads iframe overflow');
   if([1440,390].includes(width))await page.screenshot({path:output+'/downloads-'+width+'.png',fullPage:true});
   await frame.locator('[data-v="disks"]').click();
   await frame.locator('.storage-disk progress').first().waitFor({state:'visible'});
   await page.goto(base+'/ai?experiment=experiment_01');
   await page.locator('.workflow-node').first().waitFor({state:'visible'});
   await bounds(page,'experiments','#agent-board');
   if([1440,390].includes(width))await page.screenshot({path:output+'/experiments-'+width+'.png',fullPage:true});
   assert.deepEqual(errors,[],'JavaScript errors at '+width);
   console.log('Responsive workspace checks passed at '+width+'px');
   await page.close();
  }
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
