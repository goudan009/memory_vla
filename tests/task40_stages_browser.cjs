/* Real trace API and video seeking; human annotations are intercepted here. */
const {chromium} = require('../outputs/browser-tools/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:8766';
  const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1440,height:1100}});
  const errors = [], reviews = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    const response = await page.request.get(base+'/api/results');
    const runs = (await response.json()).runs;
    const run = runs.find(r => r._folder === 'V_1/memory/task40_eval02');
    const other = runs.find(r => r._folder === 'V_1/memory/task40_eval03');
    const paired = runs.find(r => r._folder === 'V_1/official/task40_eval01');
    assert.ok(run && other && paired);
    const episode = run._episodes.find(e => e.b1k_success === true);
    assert.ok(episode);
    const pairing = await page.request.get(base+'/api/vlm/stages/compare?'+new URLSearchParams({run:run._folder,other:paired._folder}));
    const expectedPairs = (await pairing.json()).matched_n;
    assert.ok(expectedPairs > 0);
    await page.route('**/api/results', route => route.fulfill({json:{runs:[run,other,paired]}}));
    await page.route('**/api/vlm/health', route => route.fulfill({json:{ok:false,error:'测试期间不启停诊断服务'}}));
    await page.route('**/api/vlm/stages/review', route => {reviews.push(route.request().postDataJSON());return route.fulfill({json:{saved:true}});});
    await page.goto(base,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => typeof selected !== 'undefined' && selected?._folder === 'V_1/memory/task40_eval02');
    await page.evaluate(index => {chooseEpisode(index);selectDetailTab('media');},episode.episode_index);
    await page.locator('#task40-timeline [data-stage="S4"]').waitFor();
    await page.waitForFunction(() => document.getElementById('video-player').readyState >= 1);
    await page.click('#task40-timeline [data-stage="S4"]');
    assert.ok((await page.locator('#task40-evidence').innerText()).includes('880'));
    await page.waitForFunction(() => Math.abs(document.getElementById('video-player').currentTime-880/480)<.1);
    await page.waitForFunction(() => document.getElementById('replay-head')?.width === 256);
    await page.locator('#task40-review > summary').click();
    await page.selectOption('#task40-review-decision','unknown');
    await page.fill('#task40-review-notes','浏览器测试：证据不足');
    await page.click('#task40-save-review');
    await page.waitForFunction(() => document.getElementById('task40-review-message').textContent.includes('已保存'));
    assert.equal(reviews.length,1);assert.equal(reviews[0].decision,'unknown');assert.equal(reviews[0].episode_index,episode.episode_index);
    await page.locator('summary').filter({hasText:'同任务模型 / Checkpoint 比较'}).click();
    await page.selectOption('#task40-compare-run',other._folder);
    await page.click('#task40-compare');
    await page.waitForFunction(() => document.getElementById('task40-comparison').textContent.includes('有效配对'));
    assert.ok((await page.locator('#task40-comparison').innerText()).includes('有效配对 0'));
    await page.selectOption('#task40-compare-run',paired._folder);
    await page.click('#task40-compare');
    await page.waitForFunction(count => document.getElementById('task40-comparison').textContent.includes(`有效配对 ${count}`),expectedPairs);
    assert.ok((await page.locator('#task40-comparison').innerText()).includes('配对子集错误转换'));
    // Exercise indexed seek with an in-browser fixture only; never add synthetic
    // video metadata or human annotations to a real research run.
    const originalReport = await (await page.request.get(base+'/api/vlm/stages?'+new URLSearchParams({run:run._folder}))).json();
    const indexedReport = structuredClone(originalReport);
    const indexedEpisode = indexedReport.episodes.find(ep => ep.identity.episode_index === episode.episode_index);
    indexedEpisode.time_mapping = {approximate:false, frames:[
      {frame_index:0,action_steps:23,video_fps:20},
      {frame_index:1,action_steps:880,video_fps:20},
      {frame_index:2,action_steps:960,video_fps:20}
    ]};
    const indexedRoute = route => route.fulfill({json:indexedReport});
    const stageURL = url => url.pathname === '/api/vlm/stages';
    await page.route(stageURL,indexedRoute);
    await page.click('#task40-refresh');
    await page.locator('#task40-timeline [data-stage="S4"]').waitFor();
    await page.click('#task40-timeline [data-stage="S4"]');
    await page.waitForFunction(() => Math.abs(document.getElementById('video-player').currentTime-.05)<.02);
    assert.ok((await page.locator('#task40-evidence').innerText()).includes('实录视频帧'));
    await page.unroute(stageURL,indexedRoute);
    await page.click('#task40-refresh');
    await page.locator('#task40-timeline [data-stage="S4"]').waitFor();
    const out = path.resolve(__dirname,'../outputs/overnight_memory_task40_20261114/verification/task40_stages_browser');
    fs.mkdirSync(out,{recursive:true});
    await page.locator('#task40-stages').screenshot({path:path.join(out,'stages_desktop.png')});
    await page.setViewportSize({width:390,height:844});
    await page.locator('#task40-stages').screenshot({path:path.join(out,'stages_mobile.png')});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({passed:true,episode:episode.episode_index,
      real_trace_api:true,approximate_seek_action_step:880,three_camera_replay:true,comparison:true,matched_pairs:expectedPairs,
      synthetic_indexed_seek:true,review_intercepted:true,page_errors:errors},null,2));
    console.log('Task40 real traces, stage seeking, three cameras, comparison and review controls passed.');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exitCode=1;});
