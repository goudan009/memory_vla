// Verify experiment shell and explicit navigation; no GPU work is submitted.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const dir=path.join(__dirname,'../evaluation/scripts'),read=name=>fs.readFileSync(path.join(dir,name),'utf8');
(async()=>{const browser=await chromium.launch({headless:true});try{
 const page=await browser.newPage({viewport:{width:1440,height:900},serviceWorkers:'block'}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const checkpoint='/archive/step6000',job={run:'run1',arm:'baseline',version:'V_2_1',method:'baseline',task:40,training_id:'train1',train_number:1,status:'complete',parameter_summary:{learning_rate:1e-5},last:{loss:.025}};
 const job2={...job,run:'run2',training_id:'train2',train_number:2,method:'memory',arm:'memory'};
 const run={_folder:'V_2_1/baseline/eval1',version:'V_2_1',task_id:40,status:'complete',success_rate:.5,completed_episodes:2,policy_identity:{training_id:'train1',checkpoint,checkpoint_step:6000}};
 const native={id:'radio',kind:'b1k2026_websocket',version:'V_1_2',task_id:'turning_on_radio',official:true,state:'complete',policy_identity:{checkpoint:{path:'/official/radio'}},native_metrics:[{success:false}]};
 await page.route('**/*',async route=>{const u=new URL(route.request().url()),p=u.pathname,json=data=>route.fulfill({json:data});
  if(p==='/workbench')return route.fulfill({contentType:'text/html',body:read('unified_workbench.html')});
  if(p==='/unified_workbench.js')return route.fulfill({contentType:'application/javascript',body:read('unified_workbench.js')});
  if(p==='/unified_workbench.css')return route.fulfill({contentType:'text/css',body:read('unified_workbench.css')});
  if(p==='/api/training/jobs')return json({jobs:u.searchParams.get('device')==='a100'?[job,job2]:[]});
  if(p==='/api/training/detail')return json({checkpoints:u.searchParams.get('run')==='run1'?[{path:'/archive/step3000',step:3000},{path:checkpoint,step:6000}]:[{path:'/archive/run2/step6000',step:6000}]});
  if(p==='/api/results')return json({runs:[run]});
  if(p==='/api/launch/jobs')return json({jobs:u.searchParams.has('benchmark')?[native]:[]});
  if(p==='/api/benchmarks/jobs')return json({jobs:[]});
  if(p==='/api/agent-experiments')return json({experiments:['experiment_01']});
  if(p==='/api/services')return json({remote:{ok:true,gpus:[{index:0,name:'A100',total_mb:81920,used_mb:10240,utilization:99}]},local:{ok:true,gpus:[]},a6000:{ok:false,error:'offline'}});
  if(p==='/')return route.fulfill({contentType:'text/html',body:'<!doctype html><main>Original functional page fixture</main>'});
  return route.fulfill({status:404,body:''});});
 const base='http://workbench.test/workbench';
 await page.goto(base);
 await page.locator('#tree .pick').filter({hasText:'Train 01'}).waitFor();
 const palette=await page.evaluate(()=>({body:getComputedStyle(document.body).backgroundColor,active:getComputedStyle(document.querySelector('#nav button.active')).backgroundColor,selected:getComputedStyle(document.querySelector('.lineage-node.active')).backgroundColor,normal:getComputedStyle(document.querySelector('.lineage-node.kind-train:not(.active)')).backgroundColor}));
 assert.deepEqual(palette,{body:'rgb(242, 245, 249)',active:'rgb(237, 242, 247)',selected:'rgb(237, 242, 247)',normal:'rgb(255, 255, 255)'},'shared neutral palette');
 assert.equal(await page.locator('#search,.tree .folder,.tree .file,iframe').count(),0);
 await page.locator('#open-device-dialog').click();await page.locator('#device-content').filter({hasText:'10.0 / 80.0 GiB'}).waitFor();await page.locator('#close-device-dialog').click();
 await page.locator('.lineage-node').filter({hasText:'Train 01'}).click();
 assert.match(await page.locator('#node-actions').innerText(),/查看训练详情/);
 assert.match(await page.locator('#node-actions').innerText(),/最新已发布权重评估/);
 assert.equal(await page.locator('#inspector [data-action]').count(),0);
 assert.equal(await page.locator('#inspector-kind').innerText(),'训练');
 await page.locator('.lineage-node').filter({hasText:'Step 3000'}).waitFor();
 await page.locator('.lineage-node').filter({hasText:'Step 6000'}).click();
 assert.equal(await page.locator('#inspector-kind').innerText(),'权重');
 assert.ok(await page.locator('#node-actions [data-action="evaluate"]').isVisible());
 const action=page.locator('#node-actions [data-action="evaluate"]');
 assert.equal(await action.evaluate(el=>getComputedStyle(el).backgroundColor),'rgb(255, 255, 255)','default action is neutral');
 await action.hover();await page.mouse.down();
 assert.equal(await action.evaluate(el=>getComputedStyle(el).backgroundColor),'rgb(40, 95, 198)','blue only while pressing');
 await page.mouse.move(1,1);await page.mouse.up();

 assert.equal(await page.locator('#tree .pick[data-kind="checkpoint"]').first().locator('.tree-type').innerText(),'权重');
 assert.match(await page.locator('#inspector').innerText(),/所属训练/);
 await page.locator('.lineage-node').filter({hasText:'Train 01'}).click();
 async function checkEdges(){
  await page.waitForFunction(()=>document.querySelectorAll('.graph-edge').length>=3);
  const check=await page.evaluate(()=>{
   const graph=document.querySelector('#lineage-graph'),bounds=graph.getBoundingClientRect();
   const buttons=new Map([...graph.querySelectorAll('.lineage-node')].map(el=>[el.dataset.node,el]));
   const expected=[...buttons.keys()].filter(id=>[...graph.querySelectorAll('.graph-edge')].some(e=>e.dataset.child===id));
    const edges=[...graph.querySelectorAll('.graph-edge')];
    if(edges.length!==buttons.size-1||expected.length!==edges.length)return [false];
   return edges.map(edge=>{
    const parent=buttons.get(edge.dataset.parent),child=buttons.get(edge.dataset.child),a=parent.getBoundingClientRect(),b=child.getBoundingClientRect();
    const start=edge.getPointAtLength(0),end=edge.getPointAtLength(edge.getTotalLength()),c=JSON.parse(edge.dataset.child),p=JSON.parse(edge.dataset.parent);
    const correctParent=c[0]!=='checkpoint'||c[1]===edge.dataset.parent;
    return correctParent&&Math.abs(start.x-(a.left+a.width/2-bounds.left))<1.5&&Math.abs(start.y-(a.bottom-bounds.top))<1.5&&Math.abs(end.x-(b.left+b.width/2-bounds.left))<1.5&&Math.abs(end.y-(b.top-bounds.top))<1.5;
   });
  });
  assert.ok(check.length>=3&&check.every(Boolean),'edge endpoints and checkpoint parent identity');
 }
 await checkEdges();
 await page.setViewportSize({width:1180,height:900});await page.waitForTimeout(120);await checkEdges();
 await page.setViewportSize({width:1440,height:900});
 await page.locator('#node-actions [data-action="training"]').click();await page.waitForURL(u=>u.pathname==='/');
 let q=new URL(page.url()).searchParams;assert.equal(q.get('page'),'training');assert.equal(JSON.parse(q.get('lineage_target')).run,'run1');
 await page.goto(base);await page.locator('#tree .pick').filter({hasText:'Step 6000'}).waitFor();await page.locator('#tree .pick').filter({hasText:'Step 6000'}).click();
 await page.locator('#node-actions [data-action="evaluate"]').click();await page.waitForURL(u=>u.pathname==='/');
 q=new URL(page.url()).searchParams;assert.equal(JSON.parse(q.get('lineage_target')).path,checkpoint);
 await page.goto(base+'?benchmark=b1k2026');await page.locator('#tree .pick').filter({hasText:'官方 Radio 权重'}).waitFor();await page.locator('#tree .pick').filter({hasText:'官方 Radio 权重'}).click();
 await page.locator('.lineage-node').filter({hasText:'Radio ·'}).click();await page.locator('#node-actions [data-action="result"]').click();await page.waitForURL(u=>u.pathname==='/');
 q=new URL(page.url()).searchParams;assert.equal(q.get('benchmark'),'b1k2026');assert.equal(JSON.parse(q.get('lineage_target')).run,'radio');
 await page.goto(base);await page.locator('.lineage-node').first().waitFor();
 fs.mkdirSync('workbench-screenshots',{recursive:true});
 for(const width of [1440,1024,768,390]){await page.setViewportSize({width,height:900});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'overflow '+width);if(width===1440)await page.screenshot({path:'workbench-screenshots/experiment-final.png'});}
 assert.deepEqual(errors,[]);
 }finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
