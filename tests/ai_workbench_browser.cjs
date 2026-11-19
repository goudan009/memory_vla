/* Exercise research handoffs and immutable snapshots using browser-only demo data. */
const {chromium} = require('../outputs/browser-tools/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8772';
  const out = path.resolve(__dirname, '../outputs/shared/diagnostics/ai_research_workbench_v2');
  await fs.mkdir(out, {recursive:true});
  const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
  const context = await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true});
  const page = await context.newPage();
  const errors = [], apiCalls = [], writes = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if(new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url());
    if(!['GET','HEAD'].includes(request.method())) writes.push(request.method()+' '+request.url());
  });
  const readState = () => page.evaluate(() => JSON.parse(localStorage.getItem('memory-vla.ai-workbench.preview.v2')));
  const task = async id => (await readState()).tasks.find(t=>t.id===id);
  const next = () => page.click('#advance-task');
  const view = name => page.click('[data-view="'+name+'"]');
  const tab = name => page.click('[data-tab="'+name+'"]');
  const shot = async name => {
    await page.locator('#toast').evaluate(node=>node.hidden=true);
    await page.screenshot({path:path.join(out,name+'.png'),fullPage:true});
  };
  try {
    await page.goto(base+'/ai/architecture', {waitUntil:'networkidle'});
    await page.waitForSelector('.task-card');
    assert.equal(await page.locator('.task-card').count(),3);
    assert.ok((await page.locator('#task-heading').textContent()).includes('这次 Memory 修改'));
    assert.equal(await page.locator('#stage-track li').count(),5);
    await shot('01-research-overview');
    await next();
    assert.equal((await task('R-001')).phase,1);
    await page.click('[data-action="request-changes"]');
    assert.equal((await task('R-001')).review,'changes');
    await shot('02-code-review-return');
    await next();
    assert.equal((await task('R-001')).codeRound,2);
    assert.equal((await task('R-001')).phase,1);
    await next();await next();
    assert.equal((await task('R-001')).phase,3);
    await page.click('[data-action="pause"]');
    await page.reload({waitUntil:'networkidle'});
    assert.ok((await page.locator('#task-heading').textContent()).includes('演示已暂停'));
    await next();
    await tab('work');
    await page.click('[data-action="fail"]');
    assert.equal((await task('R-001')).runs[0].status,'failed');
    await view('knowledge');
    assert.equal(await page.locator('.finding-card').count(),2);
    await view('research');await next();
    assert.equal((await task('R-001')).runs.length,2);
    await page.click('[data-action="fail"]');
    assert.ok((await page.locator('#advance-task').textContent()).includes('调整方案'));
    await shot('03-failure-and-budget');
    // Retry exhaustion cannot turn into another unrestricted run.
    await next();
    assert.ok(await page.locator('#task-dialog').isVisible());
    await page.click('#advanced-settings summary');
    await page.fill('[name="steps"]','120');
    await page.locator('#task-form [type="submit"]').click();
    let t=await task('R-001');
    assert.equal(t.revision,2);assert.equal(t.phase,0);
    assert.equal(t.runs[0].snapshot.steps,100);
    assert.equal(t.runs[1].snapshot.steps,100);
    assert.equal(t.runs[0].status,'failed');
    assert.equal(t.runs[1].status,'failed');
    await next();await next();await next();
    assert.equal((await task('R-001')).phase,3);
    // Revising an active run preserves its snapshot and pauses the old attempt.
    await page.click('[data-action="edit"]');
    await page.click('#advanced-settings summary');
    await page.fill('[name="steps"]','150');
    await page.locator('#task-form [type="submit"]').click();
    t=await task('R-001');
    assert.equal(t.revision,3);assert.equal(t.runs[2].status,'paused');
    assert.equal(t.runs[2].snapshot.steps,120);
    await next();await next();await next();await next();await next();
    t=await task('R-001');assert.equal(t.phase,5);
    assert.equal(t.runs[3].snapshot.steps,150);
    assert.ok((await page.locator('#panel-results').textContent()).includes('合成数据'));
    await shot('04-results-and-evidence');
    const downloadPromise=page.waitForEvent('download');
    await page.click('[data-action="export"]');
    const download=await downloadPromise;
    const exported=JSON.parse(await fs.readFile(await download.path(),'utf8'));
    assert.equal(exported.demo,true);assert.equal(exported.executable,false);
    assert.equal(exported.task.runs[3].snapshot.steps,150);
    // New default versions must not alter existing plans or historical runs.
    await view('standards');
    await page.fill('#standards-form [name="defaultSteps"]','500');
    await page.fill('#standards-form [name="maxSteps"]','200');
    await page.locator('#standards-form [type="submit"]').click();
    assert.ok(await page.locator('#standard-error').isVisible());
    await page.fill('#standards-form [name="defaultSteps"]','200');
    await page.locator('#standards-form [name="queueOnAvailable"]').uncheck();
    await page.locator('#standards-form [type="submit"]').click();
    assert.equal((await readState()).standards.revision,2);
    t=await task('R-001');
    assert.equal(t.standard.revision,1);
    assert.equal(t.runs[3].snapshot.standard.revision,1);
    await shot('05-shared-standards');
    await view('research');await next();
    assert.ok(await page.locator('#task-dialog').isVisible());
    assert.equal(await page.inputValue('[name="steps"]'),'150');
    await page.locator('#task-form [type="submit"]').click();
    let saved=await readState();t=saved.tasks.find(item=>item.id===saved.selected);
    assert.equal(t.parentId,'R-001');assert.equal(t.standard.revision,2);
    // Evaluation-only tasks skip code and training and retain the selected metric protocol.
    await page.click('#task-list [data-task="R-003"]');
    await next();assert.equal((await task('R-003')).phase,2);
    await next();await next();await next();
    assert.equal((await task('R-003')).phase,5);
    assert.ok((await page.locator('#panel-results').textContent()).includes('B1K Q-score'));
    assert.ok(!(await page.locator('#panel-results').textContent()).includes('动作 MAE'));
    assert.equal((await task('R-003')).runs[0].snapshot.steps,0);
    // The question is user-defined; text is escaped rather than interpreted as HTML.
    await page.click('#new-task-button');
    await page.fill('[name="question"]','LIBERO 检查 <img src=x onerror=alert(1)>');
    await page.fill('[name="criterion"]','比较同一套件和实例下的成功率。');
    await page.click('#advanced-settings summary');
    assert.equal(await page.inputValue('[name="steps"]'),'200');
    await page.selectOption('[name="protocol"]','libero');
    await page.fill('[name="tasks"]','10');
    await page.locator('#task-form [type="submit"]').click();
    assert.ok(await page.locator('#form-error').isVisible());
    await page.fill('[name="tasks"]','0, 1');
    await page.locator('#task-form [type="submit"]').click();
    assert.equal(await page.locator('#task-heading img').count(),0);
    for(let i=0;i<5;i++)await next();
    assert.ok((await page.locator('#panel-results').textContent()).includes('LIBERO 成功率'));
    assert.ok(!(await page.locator('#panel-results').textContent()).includes('B1K Q-score'));
    await page.click('#task-list [data-task="R-002"]');
    assert.ok((await page.locator('#panel-results').textContent()).includes('同任务错误历史'));
    await page.locator('#tab-results').focus();await page.keyboard.press('Home');
    assert.equal(await page.locator('#tab-plan').getAttribute('aria-selected'),'true');
    // All role cards provide useful handoff descriptions.
    await view('team');
    assert.equal(await page.locator('.role-card').count(),5);
    for(const role of ['planning','code','resources','execution','evaluation']) {
      await page.click('[data-role="'+role+'"]');
      assert.ok((await page.locator('#role-detail').textContent()).includes('接收什么'));
      assert.ok((await page.locator('#role-detail').textContent()).includes('交付什么'));
    }
    await page.click('[data-role="code"]');await shot('06-roles-and-architecture');
    await page.click('#delivery-plan summary');
    assert.ok((await page.locator('#delivery-plan').textContent()).includes('三类判断角色'));
    await view('knowledge');
    await page.fill('#knowledge-search','not-a-real-finding');
    assert.equal(await page.locator('.finding-card').count(),0);
    await page.fill('#knowledge-search','');
    await shot('07-shared-research-records');
    // Check every main page, every task tab and the form at narrow and wide sizes.
    for(const width of [1440,1024,780,500,390]) {
      await page.setViewportSize({width,height:width<600?844:1080});
      for(const name of ['research','team','standards','knowledge']) {
        await view(name);
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),name+' overflow at '+width);
      }
      await view('research');
      await page.click('#task-list [data-task="R-001"]');
      for(const name of ['plan','work','results']) {
        await tab(name);
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),name+' overflow at '+width);
      }
      if([1440,780,390].includes(width)){await tab('plan');await shot('research-'+width);}
      await page.click('#new-task-button');
      assert.ok(await page.locator('#task-dialog').isVisible());
      if(width===390)await shot('08-new-task-mobile');
      await page.keyboard.press('Escape');
    }
    await page.setViewportSize({width:1440,height:1080});
    await page.click('#reset-button');await page.click('#confirm-reset');
    await page.reload({waitUntil:'networkidle'});
    assert.equal(await page.locator('.task-card').count(),3);
    assert.equal((await readState()).standards.revision,1);
    await shot('01-research-overview');
    assert.deepEqual(apiCalls,[]);assert.deepEqual(writes,[]);assert.deepEqual(errors,[]);
    const result={passed:true,viewports:[1440,1024,780,500,390],apiCalls,writes,browserErrors:errors,checks:['five responsibilities','role handoff boundaries','delivery architecture','plan approval','independent review and revision','pause and reload','failure stop and bounded retries','immutable run and standard snapshots','plan revisions','archive and shared records','follow-up lineage','versioned defaults','evaluation-only flow','benchmark metric isolation','user-defined questions and escaping','plan export','keyboard tabs','responsive pages','local persistence and reset']};
    await fs.writeFile(path.join(out,'browser-checks.json'),JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify(result,null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
