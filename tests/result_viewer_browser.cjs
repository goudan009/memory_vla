const {chromium}=require('../outputs/browser-tools/node_modules/playwright');
const assert=require('node:assert/strict');

(async()=>{
  const base=process.argv[2]||'http://127.0.0.1:8765';
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  try{
    const api=await page.request.get(base+'/api/results');
    const data=await api.json();
    const run=data.runs.find(item=>item._episodes?.length);
    assert.ok(run,'Need at least one run with episode metrics');
    let delayResults=false,delayedRequests=0;
    await page.route('**/api/results',route=>{
      if(delayResults){delayedRequests++;setTimeout(()=>route.fulfill({json:{...data,runs:[run]}}),300);}
      else void route.fulfill({json:{...data,runs:[run]}});
    });
    const response=await page.goto(base,{waitUntil:'networkidle',timeout:30000});
    assert.equal(response.status(),200);
    await page.waitForFunction(()=>document.getElementById('caption').textContent.includes('次评估'));
    assert.equal(await page.locator('#page-tab-ai').getAttribute('href'),'/ai');
    assert.equal(await page.locator('.page-nav').count(),1);
    assert.equal(await page.locator('.page-nav a[href*="8772"]').count(),0);
    assert.equal(await page.locator('.page-tabs[role="tablist"] [role="tab"]').count(),3);
    await page.waitForFunction(()=>document.querySelectorAll('#episode-table tbody tr').length>0);
    await page.screenshot({path:'/tmp/memory-vla-result-viewer-desktop.png'});
    delayResults=true;
    await page.evaluate(()=>{refresh();refresh();refresh();});
    await page.waitForTimeout(400);
    assert.equal(delayedRequests,1,'overlapping refresh calls should share one results request');
    for(const width of [1440,1024,780,390]){
      await page.setViewportSize({width,height:width<600?844:1000});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`page overflow at ${width}px`);
    }
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.locator('#episode-table td[data-label="Q"]').first().count());
    assert.ok(await page.locator('#episode-table td[data-label="人工标注"]').first().count());
    await page.screenshot({path:'/tmp/memory-vla-result-viewer-mobile.png'});
    await page.goto(base+'/ai',{waitUntil:'networkidle',timeout:30000});
    assert.match(await page.title(),/研究协作/);
    assert.deepEqual(errors,[]);
    console.log('Result viewer navigation, API refresh, responsive layouts, and same-origin AI route passed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
