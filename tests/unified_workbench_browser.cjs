// Exercises the production shell with fixture API records, not a demo dataset.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const dir=path.join(__dirname,'../evaluation/scripts');
const read=name=>fs.readFileSync(path.join(dir,name),'utf8');
(async()=>{
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const checkpoint='/archive/step6000',job={run:'run1',arm:'baseline',version:'V_2_1',method:'baseline',task:40,training_id:'train1',train_number:1,status:'complete',parameter_summary:{learning_rate:0.00001},last:{loss:0.025}};
  const run={_folder:'V_2_1/baseline/eval1',version:'V_2_1',task_id:40,status:'complete',success_rate:.5,completed_episodes:2,policy_identity:{training_id:'train1',checkpoint,checkpoint_step:6000}};
  const native={id:'radio',kind:'b1k2026_websocket',version:'V_1_2',task_id:'turning_on_radio',official:true,state:'complete',policy_identity:{checkpoint:{path:'/official/radio'}},native_metrics:[{success:false}]};
  await page.route('**/*',async route=>{
   const u=new URL(route.request().url()),p=u.pathname;const json=data=>route.fulfill({json:data});
   if(p==='/workbench')return route.fulfill({contentType:'text/html',body:read('unified_workbench.html')});
   if(p==='/unified_workbench.js')return route.fulfill({contentType:'application/javascript',body:read('unified_workbench.js')});
   if(p==='/unified_workbench.css')return route.fulfill({contentType:'text/css',body:read('unified_workbench.css')});
   if(p==='/api/training/jobs')return json({jobs:u.searchParams.get('device')==='a100'?[job]:[]});
   if(p==='/api/training/detail')return json({checkpoints:[{path:checkpoint,step:6000}],loss_series:[[0,.3],[6000,.025]]});
   if(p==='/api/results')return json({runs:[run]});
   if(p==='/api/launch/jobs')return json({jobs:u.searchParams.has('benchmark')?[native]:[]});
   if(p==='/api/benchmarks/jobs')return json({jobs:[]});
   if(p==='/api/agent-experiments')return json({experiments:['experiment_01']});
   if(p==='/registry')return route.fulfill({contentType:'text/html',body:read('registry_view.html')});
   if(['/registry_storage.js','/resource_downloads.js'].includes(p))return route.fulfill({contentType:'application/javascript',body:read(p.slice(1))});
   if(p==='/api/registry')return json({items:[
      {id:'d1',kind:'dataset',host:'4090',name:'B1K-2026 Task0',path:'/data/task0',size_gb:12,n_parquet:200},
      {id:'d2',kind:'dataset',host:'A100',name:'B1K-2025 Task40',path:'/data/task40',size_gb:21,n_parquet:200},
      {id:'w1',kind:'weights',host:'4090',name:'GR00T N1.7 Radio',path:'/official/radio',size_gb:8,n_files:4}
    ],hosts:[{host:'4090',label:'4090'},{host:'A100',label:'A100'}],experiments:[],generated_at:'fixture'});
   if(p==='/api/registry/annotations')return json({});
   if(p==='/api/registry/storage'){
      const action=u.searchParams.get('action');
      if(action==='hosts')return json({hosts:[{host:'4090',label:'4090'},{host:'A100',label:'A100'}]});
      if(action==='browse'){const pth=u.searchParams.get('path');return json({type:'directory',path:pth,entries:[{name:'config.json',path:pth+'/config.json',type:'file',size:250,modified:0,can_delete:false}],offset:0,total:1,has_more:false});}
      if(action==='disks')return json({disks:[{mount:'/data',total:107374182400,used:53687091200,free:53687091200,percent:50}]});
   }
   if(p==='/api/resources/downloads')return json({recipes:[]});
   if(p==='/')return route.fulfill({contentType:'text/html',body:'<!doctype html><div id="selected"></div><script>window.unifiedWorkbenchPanel={select:async t=>document.getElementById("selected").textContent=JSON.stringify(t)}</script>'});
   return route.fulfill({status:404,body:''});
  });
  await page.goto('http://workbench.test/workbench');
  await page.locator('#tree .pick').filter({hasText:'V2.1'}).waitFor();
  await page.locator('#tree .pick').filter({hasText:'V2.1'}).click();
  await page.locator('#tree .pick').filter({hasText:'Train 01'}).click();
  await page.locator('.lineage-level').first().waitFor();
  assert.equal(await page.locator('.lineage-wire').count()>0,true);
  fs.mkdirSync('workbench-screenshots',{recursive:true});
  await page.screenshot({path:'workbench-screenshots/unified-lineage-a2.png'});
  await page.locator('#tree .pick').filter({hasText:'Step 6000'}).waitFor();
  await page.locator('#nav [data-page="training"]').click();
  await page.frameLocator('#panel').locator('#selected').filter({hasText:'run1'}).waitFor();
  assert.match(await page.locator('#tree .pick.active').innerText(),/Train 01/);
  await page.locator('#tree .pick').filter({hasText:'Step 6000'}).click();
  await page.locator('[data-action="evaluate"]').click();
  await page.frameLocator('#panel').locator('#selected').filter({hasText:checkpoint}).waitFor();
  await page.locator('#benchmarks [data-benchmark="b1k2026"]').click();
  assert.equal(await page.locator('#tree .pick').filter({hasText:'Train 01'}).count(),0);
  await page.locator('#tree .pick').filter({hasText:'V1.2'}).click();
  await page.locator('#tree .pick').filter({hasText:'官方 Radio 权重'}).click();
  await page.locator('#tree .pick').filter({hasText:'Radio ·'}).click();
  await page.locator('#nav [data-page="results"]').click();
  await page.frameLocator('#panel').locator('#selected').filter({hasText:'radio'}).waitFor();
  await page.locator('#nav [data-page="registry"]').click();
  await page.frameLocator('#panel').locator('[data-resource]').filter({hasText:'B1K-2026 Task0'}).waitFor();
  await page.locator('#inspector').filter({hasText:'B1K-2026 Task0'}).waitFor();
  await page.frameLocator('#panel').locator('#resource-browser').filter({hasText:'config.json'}).waitFor();
  assert.equal(await page.locator('#title').innerText(),'资源管理');
  assert.equal(await page.locator('#actions button').count(),0);
  assert.ok((await page.locator('#panel').getAttribute('src')).includes('/registry?embedded=1&section=datasets'));
  await page.locator('#tabs [data-tab="weights"]').click();
  await page.frameLocator('#panel').locator('[data-resource]').filter({hasText:'GR00T N1.7 Radio'}).waitFor();
  await page.locator('#inspector').filter({hasText:'GR00T N1.7 Radio'}).waitFor();
  assert.ok((await page.locator('#panel').getAttribute('src')).includes('section=weights'));
  await page.screenshot({path:'workbench-screenshots/unified-resources-a2.png'});
  for(const width of [1440,1024,768,390]){await page.setViewportSize({width,height:900});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'layout overflow at '+width);}
  assert.deepEqual(errors,[]);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
