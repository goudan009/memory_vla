/* Verify research decisions, run snapshots and the frontend-only execution boundary. */
const {chromium}=require('../outputs/browser-tools/node_modules/playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');

(async()=>{
  const base=process.argv[2]||'http://127.0.0.1:8772';
  const out=path.resolve(__dirname,'../outputs/shared/diagnostics/ai_human_console_v3');
  await fs.mkdir(out,{recursive:true});
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const context=await browser.newContext({viewport:{width:1600,height:1100},acceptDownloads:true});
  const page=await context.newPage(),errors=[],writes=[],apiCalls=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{
    if(!['GET','HEAD'].includes(r.method()))writes.push(r.method()+' '+r.url());
    if(new URL(r.url()).pathname.startsWith('/api/'))apiCalls.push(r.url());
  });
  const state=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('memory-vla.research-console.preview.v3')));
  const task=async(id='R-001')=>(await state()).tasks.find(t=>t.id===id);
  const select=id=>page.click('#task-list [data-task="'+id+'"]');
  const next=()=>page.click('#primary-action');
  const tab=name=>page.click('#tab-'+name);
  const demo=async name=>{await page.click('#demo-button');await page.click('[data-demo="'+name+'"]');};
  const shot=async name=>{await page.locator('#toast').evaluate(n=>n.hidden=true);await page.screenshot({path:path.join(out,name+'.png'),fullPage:true});};
  const editBudget=async steps=>{
    await page.click('[data-action="edit"]');
    await page.fill('#plan-form [name="steps"]',String(steps));
    await page.locator('#plan-form [type="submit"]').click();
  };
  try{
    await page.goto(base+'/ai',{waitUntil:'networkidle'});
    assert.equal(await page.locator('#project-task-table tbody tr').count(),3);
    assert.equal(await page.locator('#project-metrics>div').count(),4);
    assert.equal(await page.locator('#control-list [data-task]').count(),1);
    assert.ok(await page.locator('#project-workspace').isVisible());
    await shot('01-project-overview');
    await select('R-001');
    assert.equal(await page.locator('.milestone').count(),6);
    assert.equal(await page.locator('#panel-plan tbody tr').count(),6);
    await shot('02-research-plan');
    await next();assert.equal((await task()).phase,1);
    await page.click('[data-action="request-changes"]');assert.equal((await task()).review,'changes');
    await shot('03-code-review');
    await next();assert.equal((await task()).codeRound,2);
    await next();assert.equal((await task()).phase,2);
    await demo('resource');assert.equal((await task()).issue,'resource');
    assert.equal((await task()).runs.length,0);
    await next();await next();assert.equal((await task()).phase,3);
    assert.equal((await task()).runs[0].snapshot.steps,100);
    assert.equal(await page.locator('#panel-execution .data-table tbody tr').count(),6);
    await shot('04-execution-dependencies');
    await next();assert.equal((await task()).paused,true);
    await page.reload({waitUntil:'networkidle'});
    await select('R-001');assert.equal((await task()).runs[0].status,'paused');
    await next();await demo('fail');assert.equal((await task()).runs[0].status,'failed');
    await next();assert.equal((await task()).runs.length,2);
    await demo('fail');assert.equal((await task()).retries,1);
    assert.equal(await page.locator('#primary-action').getAttribute('data-action'),'edit');
    await shot('05-failure-control');
    await editBudget(120);
    let t=await task();assert.equal(t.revision,2);assert.equal(t.phase,0);
    assert.equal(t.runs[0].snapshot.steps,100);assert.equal(t.runs[1].status,'failed');
    await next();await next();await next();
    await editBudget(150);t=await task();
    assert.equal(t.revision,3);assert.equal(t.runs[2].status,'paused');
    assert.equal(t.runs[2].snapshot.steps,120);
    await next();await next();await next();
    await demo('advance');assert.equal((await task()).phase,4);
    // Once training has completed, zero-training-consumption failure must not be offered.
    await page.click('#demo-button');
    assert.equal(await page.locator('[data-demo="fail"]').count(),0);
    await page.keyboard.press('Escape');
    await demo('advance');assert.equal((await task()).phase,5);
    assert.ok((await page.locator('#panel-results').textContent()).includes('合成数字'));
    await shot('06-results-and-acceptance');
    await page.locator('#panel-results [data-artifact="identity"]').click();
    assert.ok((await page.locator('#artifact-body').textContent()).includes('r3'));
    await page.keyboard.press('Escape');
    await tab('records');
    await page.fill('#message','保留负结果；下一轮增加独立样本。 <script>bad()</script>');
    await page.locator('#message-form [type="submit"]').click();
    t=await task();assert.equal(t.notes.length,1);assert.equal(t.phase,5);assert.equal(t.revision,3);
    assert.equal(await page.locator('#decision-records script').count(),0);
    const pending=page.waitForEvent('download');await page.click('[data-action="export"]');
    const download=await pending,payload=JSON.parse(await fs.readFile(await download.path(),'utf8'));
    assert.equal(payload.demo,true);assert.equal(payload.executable,false);
    assert.equal(payload.data.runs[3].snapshot.steps,150);
    await next();assert.equal((await task()).phase,6);
    await next();assert.ok(await page.locator('#new-form').isVisible());
    await page.locator('#new-form [type="submit"]').click();
    let saved=await state();assert.equal(saved.tasks.find(t=>t.id===saved.selected).parent,'R-001');
    await select('R-002');assert.equal(await page.locator('.milestone.skipped').count(),2);
    await demo('advance');
    assert.ok((await page.locator('#panel-results').textContent()).includes('B1K Q-score'));
    assert.ok(!(await page.locator('#panel-results').textContent()).includes('动作 MAE'));
    assert.equal((await task('R-002')).runs[0].snapshot.steps,0);
    await page.click('#new-task');
    await page.fill('#new-question','LIBERO 对照 <img src=x onerror=bad()>');
    await page.fill('#new-criterion','固定实例和起点，比较成功率。');
    await page.selectOption('#new-protocol','libero');
    await page.click('#new-materials');await page.locator('[name="material"][value="findings"]').check();
    await page.locator('#materials-form [type="submit"]').click();
    await page.locator('#new-form [type="submit"]').click();
    saved=await state();const customId=saved.selected;
    assert.ok(saved.tasks.find(t=>t.id===customId).materials.includes('findings'));
    assert.equal(await page.locator('#task-header img').count(),0);
    await page.click('[data-action="edit"]');
    await page.fill('#plan-form [name="tasks"]','10');
    await page.locator('#plan-form [type="submit"]').click();
    assert.ok(await page.locator('#plan-error').isVisible());
    await page.fill('#plan-form [name="tasks"]','0, 1');
    await page.fill('#plan-form [name="steps"]','3001');
    await page.locator('#plan-form [type="submit"]').click();
    assert.ok((await page.locator('#plan-error').textContent()).includes('3000'));
    await page.fill('#plan-form [name="steps"]','100');
    await page.locator('#plan-form [type="submit"]').click();
    await next();await next();await next();await demo('advance');await demo('advance');
    assert.ok((await page.locator('#panel-results').textContent()).includes('LIBERO 成功率'));
    assert.ok(!(await page.locator('#panel-results').textContent()).includes('B1K Q-score'));
    // Search, keyboard tabs and all layouts must remain usable.
    await page.fill('#task-search','does-not-exist');assert.equal(await page.locator('.task-item').count(),0);
    await page.fill('#task-search','');await select('R-001');
    await page.locator('#tab-results').focus();await page.keyboard.press('Home');
    assert.equal(await page.locator('#tab-plan').getAttribute('aria-selected'),'true');
    for(const width of [1600,1280,1024,780,500,390]){
      await page.setViewportSize({width,height:width<600?844:1100});
      await page.click('#project-overview');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'overview overflow '+width);
      await select('R-001');
      for(const name of ['plan','code','execution','results','records']){
        await tab(name);
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),name+' overflow '+width);
      }
      if([1600,780,390].includes(width)){await tab('execution');await shot('execution-'+width);}
      await page.click('[data-action="edit"]');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'dialog overflow '+width);
      await page.keyboard.press('Escape');
      await page.click('#new-task');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'new form overflow '+width);
      if(width===390)await shot('07-new-task-mobile');
    }
    await page.setViewportSize({width:1600,height:1100});
    await page.click('#demo-button');await page.click('#reset-demo');
    assert.equal(await page.locator('#project-task-table tbody tr').count(),3);
    assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);assert.deepEqual(apiCalls,[]);
    const architecture=await context.newPage();
    await architecture.goto(base+'/ai/architecture',{waitUntil:'networkidle'});
    assert.equal(await architecture.locator('.task-card').count(),3);
    await fs.writeFile(path.join(out,'report.json'),JSON.stringify({passed:true,errors,writes,apiCalls,widths:[1600,1280,1024,780,500,390]},null,2));
    console.log('PASS: research gates, snapshots, failures, protocols, decisions, exports, preserved architecture and six viewport widths.');
  }catch(error){
    await shot('failure');await fs.writeFile(path.join(out,'report.json'),JSON.stringify({passed:false,error:error.stack,errors,writes,apiCalls},null,2));throw error;
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
