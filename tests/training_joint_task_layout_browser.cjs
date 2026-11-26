const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 await page.route('**/api/**',r=>r.request().method()==='GET'?r.continue():r.abort());
 try{
  await page.goto((process.argv[2]||'http://127.0.0.1:8765')+'/?page=training');
  const group=page.locator('[data-train-task*=":0_1_2_"]').first();
  await group.waitFor({state:'attached'});
  const version=group.locator('xpath=..').locator('xpath=..');
  if(!await version.evaluate(el=>el.open))await version.locator(':scope > summary').click();
  const summary=group.locator(':scope > summary');
  const baseline=await summary.evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth,text:el.textContent}));
  console.log('Joint task folder:',baseline);
  assert.ok(baseline.scroll<=baseline.width+1,'joint task folder text must fit its own frame');
  if(!await group.evaluate(el=>el.open))await summary.click();
  await group.locator('[data-job]').first().click();
  await page.locator('#train-monitor-body:not([hidden])').waitFor();
  const subtitle=await page.locator('#train-detail-subtitle').textContent();
  assert.match(subtitle,/50 个任务/);
  assert.ok(subtitle.length<80);
  await page.locator('#train-detail-tasks > summary').click();
  const tasks=await page.locator('#train-detail-task-list').textContent();
  assert.equal(tasks.match(/Task \d+/g).length,50);
  assert.match(tasks,/Task 49/);
  for(const width of [1440,1024,768,390,320]){
   await page.setViewportSize({width,height:1000});
   const overflow=await page.locator('#train-monitor').evaluate(root=>[root,...root.querySelectorAll('*')].filter(el=>{
    const s=getComputedStyle(el),r=el.getBoundingClientRect();
    return r.width&&r.height&&el.clientWidth&&s.overflowX==='visible'&&!['inline','contents'].includes(s.display)&&el.scrollWidth>el.clientWidth+2;
   }).map(el=>el.id||el.className));
   assert.deepEqual(overflow,[],`joint training detail at ${width}px`);
   if(width===1440&&process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/joint-training-desktop.png',fullPage:true});
   if(width===390&&process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/joint-training-mobile.png',fullPage:true});
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#train-job-search').fill('Task 49');
  await group.waitFor({state:'visible'});
  console.log('Existing 50-task training record fits its sidebar and detail frame at five widths.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
