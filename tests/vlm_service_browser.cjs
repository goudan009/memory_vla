/* Exercise the visible service controls without changing real GPU services. */
const {chromium}=require('../outputs/browser-tools/node_modules/playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');

(async()=>{
  const base=process.argv[2]||'http://127.0.0.1:8765';
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[],actions=[];
  page.on('pageerror',error=>errors.push(error.message));
  try{
    const data=await (await page.request.get(base+'/api/results')).json();
    const run=data.runs.find(r=>r._episodes?.length>=2&&r._videos?.length>=2&&r.status!=='running');
    assert.ok(run,'Need a completed evaluation with recordings');
    const [first,second]=run._episodes;
    let state='ready';
    const snapshot=()=>({ok:state==='ready',state,active:state==='stopped'?'inactive':'active',
      pid:state==='stopped'?0:123,gpu_mem_gb:16.3});
    await page.route('**/api/results',route=>route.fulfill({json:{runs:[run]}}));
    await page.route('**/api/vlm/service{,/**}',async route=>{
      const url=new URL(route.request().url());
      if(url.pathname.endsWith('/stop')){state='stopped';actions.push('stop');}
      if(url.pathname.endsWith('/start')){state='loading';actions.push('start');}
      if(url.pathname.endsWith('/logs')) return route.fulfill({json:{...snapshot(),logs:'[ready] loaded'}});
      await route.fulfill({json:snapshot()});
    });
    await page.goto(base,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>typeof selected!=='undefined'&&selected?._episodes?.length>=2);
    await page.evaluate(index=>chooseEpisode(index),first.episode_index);
    await page.waitForFunction(()=>document.getElementById('vlm-state').textContent.includes('在线'));
    const metrics=page.locator('#episode-metrics');
    assert.ok((await metrics.innerText()).includes(first.task_progress==null?'未记录':Number(first.task_progress).toFixed(1)+'%'));
    assert.ok(!(await page.locator('#episode-select').innerText()).includes('seed'));
    assert.equal(await page.locator('#video-location').isVisible(),false);
    await page.selectOption('#vlm-service-choice','');
    assert.ok(await page.locator('#vlm-run').isDisabled());
    await page.selectOption('#vlm-service-choice','a6000-qwen3-vl');
    assert.ok(await page.locator('#vlm-run').isEnabled());
    await page.click('#vlm-stop');
    await page.waitForFunction(()=>document.getElementById('vlm-state').textContent==='已关闭');
    assert.ok(await page.locator('#vlm-run').isDisabled());
    await page.click('#vlm-start');
    await page.waitForFunction(()=>document.getElementById('vlm-state').textContent.includes('加载中'));
    state='ready';await page.evaluate(()=>refreshVlmHealth());
    assert.ok(await page.locator('#vlm-run').isEnabled());
    await page.selectOption('#episode-select','episode:'+second.episode_index);
    assert.ok((await metrics.innerText()).includes(second.task_progress==null?'未记录':Number(second.task_progress).toFixed(1)+'%'));
    const output=path.resolve(__dirname,'../outputs/shared/diagnostics/vlm_service_20261113');
    fs.mkdirSync(output,{recursive:true});
    await page.locator('#pane-media').screenshot({path:path.join(output,'recording_metrics.png')});
    await page.setViewportSize({width:390,height:844});
    await metrics.screenshot({path:path.join(output,'episode_metrics_mobile.png')});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    await page.setViewportSize({width:1440,height:1000});
    await page.evaluate(()=>selectPage('launch'));
    await page.locator('#vlm-service-panel').waitFor({state:'visible'});
    await page.click('#vlm-service-logs');
    await page.waitForFunction(()=>document.getElementById('vlm-service-log-text').textContent.includes('[ready]'));
    await page.locator('#vlm-service-panel').screenshot({path:path.join(output,'service_controls.png')});
    assert.deepEqual(actions,['stop','start']);assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'browser_result.json'),JSON.stringify({passed:true,actions,
      real_service_actions_intercepted:true,episode_metrics:true,optional_diagnosis:true,page_errors:errors},null,2));
    console.log('VLM controls, selection, Episode metrics and mobile layout passed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
