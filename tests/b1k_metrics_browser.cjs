const {chromium}=require('../outputs/browser-tools/node_modules/playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');

(async()=>{
  const base=process.argv[2]||'http://127.0.0.1:8765';
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  try{
    const data=await(await page.request.get(base+'/api/results')).json();
    const run=data.runs.find(r=>r._folder==='V_0/baseline/groot_task21_1');
    assert.ok(run?.b1k_metrics?.metric_coverage_complete);
    assert.ok(run.mean_task_progress>30&&run.b1k_metrics.mean_q_score_percent<5);
    await page.route('**/api/results',route=>route.fulfill({json:{runs:[run]}}));
    await page.goto(base,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>typeof selected!=='undefined'&&selected?._episodes?.length===10);
    const summary=await page.locator('#summary').innerText();
    for(const label of ['B1K 官方 Q-score','B1K 完整任务成功率','GR00T 累计任务进度','GR00T 回合成功率']) assert.ok(summary.includes(label));
    assert.ok(summary.includes('3.3%')&&summary.includes('34.8%'));
    assert.ok((await page.locator('#details').innerText()).includes('B1K 终局成功'));
    const output=path.resolve(__dirname,'../outputs/shared/diagnostics/b1k_metrics_20261113');
    fs.mkdirSync(output,{recursive:true});
    await page.locator('#summary').screenshot({path:path.join(output,'official_and_groot_summary.png')});
    await page.evaluate(index=>chooseEpisode(index),run._episodes[0].episode_index);
    await page.locator('#episode-metrics').screenshot({path:path.join(output,'episode_metrics.png')});
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.locator('#episode-metrics').screenshot({path:path.join(output,'episode_metrics_mobile.png')});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'browser_result.json'),JSON.stringify({passed:true,
      run:run._folder,groot_progress:run.mean_task_progress,b1k_metrics:run.b1k_metrics,
      no_real_data_modified:true,page_errors:errors},null,2));
    console.log('Official B1K and GR00T metrics, Episode table, recording and mobile checks passed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
