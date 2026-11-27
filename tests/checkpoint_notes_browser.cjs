// Uses real workbench GETs; notes writes are mocked, all workload writes blocked.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const base=process.argv[2]||'http://127.0.0.1:8765', errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const jobs=await (await page.request.get(base+'/api/training/jobs?device=a100')).json();
 const job=jobs.jobs.find(j=>j.status==='completed'&&j.run==='v1__test01__exp13__run001');
 assert.ok(job,'completed checkpoint fixture');
 const detail=await (await page.request.get(base+'/api/training/detail?device=a100&run='+job.run+'&arm='+job.arm)).json();
 const [checkpoint,second]=detail.checkpoints;
 assert.ok(checkpoint&&second);
 const results=await (await page.request.get(base+'/api/results')).json();
 const result=results.runs.find(r=>(r.policy_identity||r.configuration?.policy_identity)?.checkpoint===checkpoint.path);
 assert.ok(result,'evaluation with the same checkpoint');
 let records={}, failSave=false, trainingActive=false;
 await page.route('**/api/**',async route=>{
  const request=route.request(),url=new URL(request.url());
  if(url.pathname==='/api/checkpoint-notes'){
   if(request.method()==='GET') return route.fulfill({json:{notes:records,devices:{a100:{label:'A100',host:'172.28.62.17'},a6000:{label:'RTX 6000 Ada',host:'100.100.2.30'}}}});
   if(failSave) return route.fulfill({status:500,json:{error:'测试保存失败'}});
   const body=request.postDataJSON(),key=JSON.stringify([body.device,body.checkpoint]);
   const entry={...body,notes:body.notes.trim(),updated_at:new Date().toISOString()};
   if(entry.notes)records[key]=entry;else delete records[key];
   return route.fulfill({json:entry});
  }
  if(request.method()!=='GET')return route.abort();
  if(url.pathname==='/api/results')return route.fulfill({json:{...results,runs:[result]}});
  if(url.pathname==='/api/training/jobs'&&trainingActive)return route.fulfill({json:{...jobs,
   jobs:jobs.jobs.map(row=>row.run===job.run?{...row,status:'training'}:row)}});
  return route.continue();
 });
 try{
  await page.goto(base+'/?page=training');
  const button=page.locator('[data-job][title="'+job.run+'"]');
  await button.waitFor({state:'attached'});
  await button.evaluate(el=>{for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
  await button.click();
  await page.locator('#train-monitor-body:not([hidden])').waitFor();
  await page.locator('#train-tab-controls').click();
  const input=page.locator('#train-saved-checkpoint-note textarea');
  const save=page.locator('#train-saved-checkpoint-note button');
  await input.waitFor();
  await page.waitForFunction(()=>!document.querySelector('#train-saved-checkpoint-note textarea').disabled);
  const text='验证集较好 <标签>\nTask40 待多回合评估';
  await input.fill(text);
  // A real detail refresh must not erase an unfinished note.
  const refreshed=page.waitForResponse(r=>r.url().includes('/api/training/detail'));
  await page.locator('#train-refresh').click();await refreshed;
  assert.equal(await input.inputValue(),text);
  await page.locator('#train-saved-checkpoint').selectOption(second.path);
  assert.equal(await input.inputValue(),'');
  await page.locator('#train-saved-checkpoint').selectOption(checkpoint.path);
  assert.equal(await input.inputValue(),text);
  failSave=true;await save.click();
  await page.waitForFunction(()=>document.querySelector('#train-saved-checkpoint-note [role=status]').textContent.includes('保存失败'));
  assert.equal(await input.inputValue(),text);
  failSave=false;await save.click();
  await page.waitForFunction(()=>document.querySelector('#train-saved-checkpoint-note [role=status]').textContent.includes('已保存'));
  assert.ok((await page.locator('#train-saved-checkpoint option:checked').textContent()).includes('备注：'));
  await page.locator('#train-use-checkpoint').click();
  const inference=page.locator('#inference-checkpoint-note textarea');
  await page.waitForFunction(()=>document.querySelector('#inference-checkpoint-note textarea')?.value.includes('Task40'));
  assert.equal(await inference.inputValue(),text);
  await page.locator('#inference-weight-path > summary').click();
  await page.locator('#inference-checkpoint').fill(second.path);
  assert.equal(await inference.inputValue(),'');
  await page.locator('#inference-checkpoint').fill(checkpoint.path);
  assert.equal(await inference.inputValue(),text);
  const endpoint=result.policy_identity.host+':'+result.policy_identity.port;
  await page.locator('#launch-service option[value="'+endpoint+'"]').waitFor({state:'attached'});
  await page.locator('#launch-service').selectOption(endpoint);
  assert.equal(await page.locator('#launch-checkpoint-note textarea').inputValue(),text);
  await page.locator('#page-tab-results').click();
  await page.locator('#result-note-panel > summary').click();
  const resultInput=page.locator('#result-checkpoint-note textarea');
  assert.equal(await resultInput.inputValue(),text);
  await resultInput.fill('已评估 50 回合');
  await page.locator('#result-checkpoint-note button').click();
  await page.waitForFunction(()=>document.getElementById('result-note-preview').textContent.includes('50 回合'));
  await page.locator('#page-tab-training').click();
  assert.equal(await input.inputValue(),'已评估 50 回合');
  trainingActive=true;
  // Notes are reloaded from the shared API when the browser reloads.
  await page.reload();
  await page.waitForFunction(()=>window.checkpointNotes?.note('a100',document.querySelector('#train-saved-checkpoint')?.value||'').includes('50 回合')||document.getElementById('result-note-preview').textContent.includes('50 回合'));
  await page.locator('#page-tab-results').click();
  await page.locator('#result-note-panel > summary').click();
  assert.equal(await resultInput.inputValue(),'已评估 50 回合');
  await page.locator('#page-tab-training').click();
  await button.waitFor({state:'attached'});
  await button.evaluate(el=>{for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
  await button.click();
  await page.locator('#train-monitor-body:not([hidden])').waitFor();
  await page.locator('#train-tab-controls').click();
  assert.equal(await input.inputValue(),'已评估 50 回合');
  assert.ok((await page.locator('#train-detail-status').textContent()).includes('训练中'));
  await page.locator('#page-tab-results').click();
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:1000});
   const box=await page.locator('#result-checkpoint-note').boundingBox();
   assert.ok(box.x>=0&&box.x+box.width<=width+1,'note editor fits viewport');
   if(process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/checkpoint-note-'+width+'.png',fullPage:true});
  }
  await resultInput.fill('');await page.locator('#result-checkpoint-note button').click();
  await page.waitForFunction(()=>document.getElementById('result-note-preview').textContent.includes('暂无备注'));
  assert.deepEqual(records,{});
  assert.deepEqual(errors,[]);
  console.log('Checkpoint notes: training drafts, edit/clear, failure/retry, inference/results sharing, reload and mobile layout passed.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
