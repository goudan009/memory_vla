/* Browser integration check. Optional RUN_BENCHMARK_SMOKE=1 runs LIBERO Task0, 3 steps. */
const {chromium} = require('../outputs/browser-tools/node_modules/playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  const errors=[], requests=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>requests.push(r.url()));
  const base=process.argv[2]||'http://127.0.0.1:8770';
  try {
    await page.goto(base,{waitUntil:'domcontentloaded'});
    await page.waitForSelector('#benchmark-choice');
    assert.equal(await page.locator('.page-nav button').count(),3);
    await page.selectOption('#benchmark-choice','libero');
    await page.waitForFunction(()=>document.querySelectorAll('#bench-job-list [data-run]').length>0);
    assert.equal(await page.locator('#page-results').isVisible(),false);
    await page.selectOption('#benchmark-choice','mikasa');
    await page.waitForFunction(()=>document.getElementById('bench-job-list').textContent.includes('暂无匹配记录'));
    await page.selectOption('#benchmark-choice','libero');
    await page.waitForFunction(()=>document.querySelectorAll('#bench-job-list [data-run]').length>0,null,{timeout:7000});
    // Explicitly exercise the existing periodic refresh while a different benchmark is selected.
    const offset=requests.length;
    await page.evaluate(async()=>await refresh());
    assert.ok(!requests.slice(offset).some(url=>url.includes('/api/results')),'LIBERO refresh fetched B1K results');
    await page.click('#page-tab-training');
    assert.ok(await page.locator('#bench-training').isVisible());
    assert.ok(await page.locator('#libero-training-card').isVisible());
    assert.equal(await page.locator('#bench-training-unavailable').isVisible(),false);
    await page.waitForFunction(()=>document.getElementById('libero-train-suite').options.length===5);
    await page.selectOption('#libero-train-tasks',['0','1','2']);
    assert.equal(await page.inputValue('#libero-train-steps'),'900');
    assert.ok(await page.locator('#libero-train-start').isDisabled());
    await page.selectOption('#libero-train-tasks',['0']);
    assert.equal(await page.inputValue('#libero-train-steps'),'300');
    await page.screenshot({path:path.join(__dirname,'../outputs/browser-tools/libero-training.png'),fullPage:true});
    assert.equal(await page.locator('#page-training').isVisible(),false);
    await page.click('#page-tab-launch');
    await page.waitForFunction(()=>document.getElementById('bench-suite').options.length===5);
    await page.selectOption('#bench-suite','libero_90');
    assert.equal(await page.locator('#bench-tasks option').count(),90);
    await page.selectOption('#bench-suite','libero_spatial');
    assert.equal(await page.locator('#bench-tasks option').count(),10);
    await page.selectOption('#libero-eval-kind','libero');
    await page.click('#bench-libero-preview');
    await page.waitForFunction(()=>document.getElementById('bench-libero-message').textContent.includes('显存满足'));
    await page.screenshot({path:path.join(__dirname,'../outputs/browser-tools/libero-launch.png'),fullPage:true});
    if(process.env.RUN_BENCHMARK_SMOKE==='1') {
      await page.fill('#bench-steps','3');
      await page.fill('#bench-notes','网页联调验收 · Spatial Task0 · 3 步');
      await page.click('#bench-libero-start');
      await page.waitForFunction(()=>document.getElementById('bench-detail-subtitle').textContent.includes('网页联调验收'));
      await page.waitForFunction(()=>['检查通过','检查失败','运行中断'].includes(document.getElementById('bench-detail-status').textContent),null,{timeout:60000});
      assert.equal(await page.locator('#bench-detail-status').textContent(),'检查通过');
      await page.click('[data-bench-tab=cameras]');
      await page.waitForFunction(()=>{
        const images=[...document.querySelectorAll('#bench-pane-cameras img')];
        return images.length===4&&images.every(i=>i.complete&&i.naturalWidth===128);
      });
      await page.screenshot({path:path.join(__dirname,'../outputs/browser-tools/libero-cameras.png'),fullPage:true});
      await page.click('[data-bench-tab=logs]');
      assert.ok((await page.locator('#bench-log').textContent()).includes('libero_spatial'));
    }
    if(process.env.RUN_LIBERO_WORKFLOW==='1'){
      async function completed(note){
        await page.waitForFunction(note=>document.getElementById('bench-detail-subtitle').textContent.includes(note),note);
        await page.waitForFunction(()=>['已完成','运行失败','运行中断'].includes(document.getElementById('bench-detail-status').textContent),null,{timeout:180000});
        assert.equal(await page.locator('#bench-detail-status').textContent(),'已完成',await page.locator('#bench-log').textContent());
      }
      if(process.env.LIBERO_TRAIN_RUN){
        await page.click('#page-tab-results');
        await page.click('[data-run="'+process.env.LIBERO_TRAIN_RUN+'"]');
        await page.waitForSelector('[data-libero-evaluate]');
      }else{
      await page.click('#page-tab-training');
      await page.fill('#libero-train-notes','LIBERO 网页验收 · 数据准备');
      await page.click('#libero-prepare-data');
      await completed('LIBERO 网页验收 · 数据准备');
      await page.click('#page-tab-training');
      await page.selectOption('#libero-train-arm','baseline');
      await page.selectOption('#libero-train-mode','lora');
      await page.fill('#libero-train-steps','1');
      await page.locator('#libero-training-form details summary').click();
      await page.fill('#libero-train-group','4');
      await page.fill('#libero-train-notes','LIBERO 网页验收 · Baseline LoRA 1 步');
      await page.click('#libero-train-preview');
      await page.waitForFunction(()=>document.getElementById('libero-training-message').textContent.includes('显存满足'));
      await page.click('#libero-train-start');
      await completed('LIBERO 网页验收 · Baseline LoRA 1 步');
      }
      await page.click('[data-bench-tab=metrics]');
      await page.click('[data-libero-evaluate]');
      await page.waitForFunction(()=>document.getElementById('libero-eval-kind').value==='libero_eval'&&!document.getElementById('bench-launch').hidden);
      await page.selectOption('#libero-eval-preset','full');
      assert.equal(await page.inputValue('#bench-steps'),'0');
      assert.equal(await page.inputValue('#libero-eval-trials'),'50');
      await page.fill('#libero-eval-trials','1');
      await page.fill('#bench-notes','LIBERO 网页验收 · 完整时长单回合');
      await page.click('#bench-libero-preview');
      await page.waitForFunction(()=>document.getElementById('bench-libero-message').textContent.includes('显存满足'));
      await page.click('#bench-libero-start');
      await completed('LIBERO 网页验收 · 完整时长单回合');
      assert.ok((await page.locator('#bench-pane-metrics').textContent()).includes('完整时长评估'));
      await page.screenshot({path:path.join(__dirname,'../outputs/browser-tools/libero-policy-result.png'),fullPage:true});
      await page.click('[data-bench-tab=cameras]');
      await page.waitForFunction(()=>{
        const video=document.querySelector('#libero-media-content video');
        const images=[...document.querySelectorAll('#libero-media-content img')];
        return video?.readyState>=1 && video.duration>0 && images.length===4 && images.every(i=>i.complete&&i.naturalWidth===256);
      });
      const video=await page.locator('#libero-media-content video').getAttribute('src');
      const partial=await page.request.get(base+video,{headers:{Range:'bytes=0-31'}});
      assert.equal(partial.status(),206);assert.equal((await partial.body()).length,32);
      await page.screenshot({path:path.join(__dirname,'../outputs/browser-tools/libero-policy-video.png'),fullPage:true});
    }
    await page.selectOption('#benchmark-choice','b1k');
    await page.click('#page-tab-training');
    await page.waitForSelector('#train-form');
    assert.ok(await page.locator('#bench-code-checks').isVisible());
    assert.equal(await page.locator('#bench-workspace').isVisible(),false);
    await page.locator('#bench-code-checks summary').click();
    if(process.env.RUN_NETWORK_SMOKE==='1') {
      await page.fill('#bench-code-notes','网页联调验收 · 网络回归与真实 EOS 检索');
      await page.click('#bench-code-start');
      await page.waitForFunction(()=>document.getElementById('bench-detail-subtitle').textContent.includes('网页联调验收 · 网络回归'));
      await page.waitForFunction(()=>['检查通过','检查失败','运行中断'].includes(document.getElementById('bench-detail-status').textContent),null,{timeout:120000});
      assert.equal(await page.locator('#bench-detail-status').textContent(),'检查通过');
      assert.ok((await page.locator('#bench-pane-metrics').textContent()).includes('真实 EOS 认知检索'));
      await page.screenshot({path:path.join(__dirname,'../outputs/browser-tools/network-results.png'),fullPage:true});
      await page.click('#page-tab-training');
    }
    await page.click('#bench-code-history');
    await page.waitForFunction(()=>document.getElementById('bench-job-list').querySelectorAll('[data-run]').length>0);
    assert.ok(!(await page.locator('#bench-job-list').textContent()).includes('LIBERO'));
    assert.ok((await page.locator('#bench-results-subtitle').textContent()).includes('开发快测'));
    await page.click('#bench-policy-results');
    assert.ok(await page.locator('#page-results').isVisible());
    await page.selectOption('#benchmark-choice','simpler');
    await page.click('#page-tab-launch');
    assert.ok(await page.locator('#bench-not-connected').isVisible());
    assert.equal(await page.locator('#bench-libero-form').isVisible(),false);
    await page.selectOption('#benchmark-choice','libero');
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'mobile horizontal overflow');
    await page.screenshot({path:path.join(__dirname,'../outputs/browser-tools/libero-mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('Browser checks passed: three shared tabs, benchmark isolation, switching, refresh, tasks, results, responsive layout'+
      (process.env.RUN_BENCHMARK_SMOKE==='1'?', real LIBERO launch + 4 camera frames':'')+
      (process.env.RUN_NETWORK_SMOKE==='1'?', real network regression + EOS retrieval':''));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
