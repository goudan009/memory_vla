// Read-only validation against the live workbench; never controls a real job.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const base=process.argv[2]||'http://127.0.0.1:8765',errors=[],writes=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/**',route=>{
  if(route.request().method()==='GET')return route.continue();
  writes.push(route.request().url());return route.abort();
 });
 async function select(run){
  const row=page.locator('[data-job][title="'+run+'"]');
  await row.waitFor({state:'attached'});
  await row.evaluate(el=>{for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
  await row.click();
  await page.waitForFunction(run=>document.getElementById('train-detail-title').textContent===run&&
   !document.getElementById('train-monitor-body').hidden,run);
 }
 function metric(label){return page.locator('#train-metrics .metric').filter({has:page.locator('label',{hasText:label})});}
 try{
  await page.goto(base+'/?page=training');
  await select('v2.1__test01__exp14__run001');
  assert.ok(Number(await metric('最新 Loss').locator('strong').textContent())>0);
  assert.ok(Number(await metric('峰值显存').locator('strong').textContent())>0);
  assert.equal(await page.locator('#train-curve-mode option[value="epochs"]').isDisabled(),false);
  await page.locator('#train-curve-mode').selectOption('epochs');
  await page.locator('#train-chart polyline[data-loss-series="trend"]').waitFor({state:'attached'});
  assert.match(await page.locator('#train-eta').textContent(),/训练已结束/);
  await select('v2_1__test02__exp01__run003');
  for(const label of ['最新 Loss','训练吞吐','峰值显存']){
   const value=await metric(label).locator('strong').textContent();
   assert.ok(Number(value)>0,`${label}: ${value}`);
  }
  assert.match(await metric('训练吞吐').locator('small').textContent(),/样本/);
  for(const id of ['train-elapsed','train-eta','train-step-time']){
   assert.notEqual(await page.locator('#'+id).textContent(),'—');
  }
  assert.equal(await page.locator('#train-curve-mode').inputValue(),'steps');
  assert.equal(await page.locator('#train-curve-mode option[value="epochs"]').isDisabled(),true);
  assert.match(await page.locator('#train-curve-source-label').textContent(),/记录点 Loss/);
  assert.match(await page.locator('#train-eta-note').textContent(),/日志区间估算/);
  await page.locator('#train-chart polyline[data-loss-series="trend"]').waitFor({state:'attached'});
  assert.match(await page.locator('#train-validation-info').textContent(),/训练内验证/);
  assert.ok(await page.locator('#train-epoch-table tbody tr').count()>0);
  assert.match(await page.locator('#train-epoch-table').textContent(),/本训练未记录 Epoch/);
  const validationValues=await page.locator('#train-epoch-table tbody tr td:nth-child(2)').allTextContents();
  assert.ok(validationValues.every(value=>Number(value)>0),validationValues.join(','));
  await page.locator('#train-curve-mode').selectOption('blocks');
  assert.match(await page.locator('#train-curve-source-label').textContent(),/50 步记录均值/);
  for(const width of [1440,768,390]){
   await page.setViewportSize({width,height:1000});
   assert.equal(await page.locator('#train-metrics').evaluate(el=>el.scrollWidth>el.clientWidth+2),false);
  }
  assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);
  console.log('Live V2_1 native/legacy loss, throughput, memory, timing, validation, curves and epoch switching passed; read-only.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
