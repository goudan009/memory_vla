// Exercise the real training page; preview/start writes are browser fixtures.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const base=process.argv[2]||'http://127.0.0.1:8765',errors=[],writes=[];
 page.on('pageerror',error=>errors.push(error.message));
 await page.route('**/api/**',async route=>{
  const req=route.request(),url=new URL(req.url());
  if(req.method()==='GET')return route.continue();
  writes.push({path:url.pathname,body:req.postDataJSON()});
  if(url.pathname==='/api/training/preview')return route.fulfill({json:{run_name:'fixture_test01_exp99',global_decisions:128}});
  throw Error('Unexpected write: '+url.pathname);
 });
 const step=n=>page.locator('[data-train-setup-step="'+n+'"]');
 const panel=n=>page.locator('#train-setup-'+n);
 const next=()=>page.locator('#train-next').click();
 try{
  await page.goto(base+'/?page=training');
  await page.locator('#train-task-list input').first().waitFor({state:'attached'});
  assert.ok(await panel(0).isVisible());assert.ok(await page.locator('#train-submit').isHidden());
  await next();assert.ok(await panel(0).isVisible(),'missing experiment identity blocks next');
  await page.locator('#train-test-id').fill('1');await page.locator('#train-experiment-id').fill('99');
  await next();assert.ok(await panel(1).isVisible());
  await page.locator('#train-clear-tasks').click();await next();
  assert.ok(await panel(1).isVisible(),'empty tasks block next');
  await page.locator('#train-select-ready').click();
  const count=await page.locator('#train-dataset option:checked').count();assert.ok(count>=50,'all joint tasks selected');
  assert.match(await page.locator('#train-step-summary-1').textContent(),new RegExp('已选 '+count+' 个'));
  await page.locator('.train-selected-tasks summary').click();
  await next();assert.ok(await panel(2).isVisible());
  await next();assert.ok(await panel(2).isVisible(),'missing GPU blocks next');
  await page.locator('#train-gpus input').first().check();
  await next();assert.ok(await panel(3).isVisible());
  assert.ok(await page.locator('#train-submit').isVisible());
  await page.locator('#train-notes').fill('四步表单保留输入');
  await page.locator('#train-enqueue').evaluate(el=>{el.checked=true;el.dispatchEvent(new Event('input',{bubbles:true}));});
  assert.match(await page.locator('#train-submit').textContent(),/队列/);
  await step(0).click();assert.equal(await page.locator('#train-experiment-id').inputValue(),'99');
  await page.locator('#train-fine-tuning').selectOption('lora');
  await step(3).click();assert.ok(await page.locator('#train-lora-rank').isVisible());
  await page.locator('#train-lora-rank').fill('0');
  await step(0).click();await page.locator('#train-fine-tuning').selectOption('action_full');
  assert.ok(await page.locator('#train-lora-rank').isDisabled(),'inactive LoRA settings do not block standard tuning');
  await step(3).click();assert.equal(await page.locator('#train-notes').inputValue(),'四步表单保留输入');
  // A bad value in a folded advanced section opens its exact step and fold.
  await page.locator('#train-seed').evaluate(el=>{el.value='-1';el.dispatchEvent(new Event('input',{bubbles:true}));});
  await page.locator('#train-check').click();assert.equal(writes.length,0);
  assert.ok(await page.locator('#train-seed').isVisible());
  await page.locator('#train-seed').fill('42');await page.locator('#train-advanced summary').click();
  // Automatic recommendation should have generated a valid budget after GPU selection.
  await page.locator('#train-check').click();
  await page.locator('#train-preview-result:not([hidden])').waitFor();
  assert.match(await page.locator('#train-preview-result').textContent(),/检查通过.*尚未启动/);
  assert.equal(writes.length,1);assert.equal(writes[0].body.tasks.length,count);
  assert.equal(writes[0].body.enqueue,true);assert.equal(writes[0].body.notes,'四步表单保留输入');
  assert.equal(writes[0].body.validation_enabled,true);
  for(const width of [1440,1024,768,390,320]){
   await page.setViewportSize({width,height:1000});
   for(const n of [0,1,2,3]){
    await step(n).click();
    const overflow=await page.locator('#train-config').evaluate(root=>[root,...root.querySelectorAll('*')].filter(el=>{
     const s=getComputedStyle(el),r=el.getBoundingClientRect();
     return r.width&&r.height&&el.clientWidth&&s.overflowX==='visible'&&!['inline','contents'].includes(s.display)&&el.scrollWidth>el.clientWidth+2;
    }).map(el=>el.id||el.className));
    assert.deepEqual(overflow,[],`step ${n} at ${width}px`);
   }
   if(process.env.SCREENSHOT_DIR&&[1440,390].includes(width))await page.screenshot({path:process.env.SCREENSHOT_DIR+'/training-start-'+width+'.png',fullPage:true});
  }
  // Changing any configuration invalidates the prior preview.
  await page.locator('#train-notes').fill('更新配置');assert.ok(await page.locator('#train-preview-result').isHidden());
  assert.deepEqual(errors,[]);
  console.log('Training four-step navigation, required-field routing, 50-task selection, retained inputs, preview payload and five responsive widths passed; all writes mocked.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
