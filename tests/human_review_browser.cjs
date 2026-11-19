/* Real browser form check; review writes are intercepted, never label real data. */
const {chromium}=require('../outputs/browser-tools/node_modules/playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');

(async()=>{
  const base=process.argv[2]||'http://127.0.0.1:8765';
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  try{
    const response=await page.request.get(base+'/api/results');
    const data=await response.json();
    const run=data.runs.find(run=>run._episodes?.length>=2&&run.status!=='running');
    assert.ok(run,'A finished run with two Episodes is needed for the browser check');
    run._human_reviews={};
    const [first,second]=run._episodes;
    await page.route('**/api/results',route=>route.fulfill({json:{runs:[run],updated_at:'人工评价界面测试'}}));
    await page.route('**/api/vlm/service',route=>route.fulfill({json:{ok:false,state:'stopped',active:'inactive',pid:0}}));
    const writes=[];
    await page.route('**/api/human-review',async route=>{
      const req=route.request(),payload=req.postDataJSON();
      writes.push({method:req.method(),payload});
      const key='episode:'+payload.episode_index;
      if(req.method()==='DELETE'){
        delete run._human_reviews[key];
        await route.fulfill({json:{review:null,deleted:true}});
      }else{
        const review={...payload.review,key,source:'human',updated_at:new Date().toISOString()};
        run._human_reviews[key]=review;
        await route.fulfill({json:{review}});
      }
    });
    await page.goto(base,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>typeof selected!=='undefined'&&selected?._episodes?.length>=2);
    await page.evaluate(index=>chooseEpisode(index),first.episode_index);
    await page.locator('#human-review-box').waitFor({state:'visible'});
    await page.fill('#human-execution-quality','65');
    await page.fill('#human-visual-progress','35');
    await page.selectOption('#human-behavior-class','goal_directed_incomplete');
    await page.selectOption('#human-failure-stage','grasp');
    await page.selectOption('#human-target-touched','true');
    await page.selectOption('#human-stuck','false');
    await page.selectOption('#human-vlm-agreement','disagree');
    const notes='人工观察：已接触目标，但没有抓稳；VLM 的完成判断不准确。';
    await page.fill('#human-notes',notes);
    await page.evaluate(async()=>await refresh());
    assert.equal(await page.inputValue('#human-notes'),notes,'Periodic refresh overwrote draft');
    await page.selectOption('#episode-select','episode:'+second.episode_index);
    assert.equal(await page.inputValue('#human-notes'),'');
    await page.selectOption('#episode-select','episode:'+first.episode_index);
    assert.equal(await page.inputValue('#human-notes'),notes);
    await page.click('#human-review-save');
    await page.waitForFunction(()=>document.getElementById('human-review-status').textContent.startsWith('已保存'));
    assert.equal(writes[0].payload.episode_index,first.episode_index);
    assert.equal(writes[0].payload.review.execution_quality,65);
    assert.equal(writes[0].payload.review.stuck,false);
    await page.reload({waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>typeof selected!=='undefined'&&selected?._episodes?.length>=2);
    await page.evaluate(index=>chooseEpisode(index),first.episode_index);
    assert.equal(await page.inputValue('#human-notes'),notes);
    assert.equal(await page.inputValue('#human-execution-quality'),'65');
    const output=path.resolve(__dirname,'../outputs/shared/diagnostics/human_review_20261113');
    fs.mkdirSync(output,{recursive:true});
    await page.locator('#human-review-box').screenshot({path:path.join(output,'human_review_form.png')});
    await page.setViewportSize({width:390,height:844});
    await page.locator('#human-review-box').screenshot({path:path.join(output,'human_review_mobile.png')});
    await page.click('#human-review-delete');
    await page.waitForFunction(()=>document.getElementById('human-review-status').textContent==='评价已删除');
    assert.equal(await page.inputValue('#human-notes'),'');
    assert.ok(await page.locator('#human-review-delete').isDisabled());
    assert.equal(writes[1].method,'DELETE');
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'browser_result.json'),JSON.stringify({
      passed:true,review_writes_intercepted:true,run:run._folder,episode:first.episode_index,
      checks:['draft_refresh','episode_isolation','save','reload','delete','mobile_layout'],page_errors:errors},null,2));
    console.log('Human review browser checks passed; no real review data was changed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
