// Read-only check against the workbench; never starts or stops a workload.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');

(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 await page.route('**/api/**',route=>route.request().method()==='GET'?route.continue():route.abort());
 try{
  await page.goto((process.argv[2]||'http://127.0.0.1:8765')+'/?page=launch');
  await page.locator('#inference-panel > summary').click();
  await page.locator('#inference-gpu option[value="3"]').waitFor({state:'attached'});
  assert.equal(await page.locator('#inference-gpu option').count(),5);
  await page.locator('#inference-gpu').selectOption('2');
  assert.equal(await page.locator('#inference-gpu').inputValue(),'2');
  const versions=await page.locator('#inference-version option').evaluateAll(options=>options.map(o=>o.value));
  assert.ok(versions.includes('V_1')&&versions.includes('V_2_1'));
  assert.ok(!versions.some(v=>['V_10','V_11','V2_1','v2_1','v2.1'].includes(v)));
  await page.locator('#inference-version').selectOption('V_2_1');
  assert.ok(await page.locator('#inference-weight option').count()>1);
  const legacy=await page.locator('#inference-weight option').evaluateAll(options=>options.map(o=>o.value).find(p=>p.includes('v2.1__')&&p.endsWith('step006000')));
  assert.ok(legacy);
  await page.locator('#inference-weight').selectOption(legacy);
  assert.match(await page.locator('#inference-source').inputValue(),/deployments\/web_live$/);
  assert.match(await page.locator('#inference-source-hint').textContent(),/baseline/);
  await page.locator('#launch-mode').selectOption('single2');
  await page.locator('#launch-repetitions').fill('5');
  assert.match(await page.locator('#launch-mode-hint').textContent(),/共 50 回合/);
  await page.locator('#launch-repetitions').fill('10');
  assert.match(await page.locator('#launch-mode-hint').textContent(),/共 100 回合/);
  await page.locator('#inference-reload').click();
  await page.waitForFunction(()=>!document.getElementById('inference-reload').disabled);
  assert.equal(await page.locator('#inference-gpu').inputValue(),'2');
  await page.locator('#page-tab-training').click();
  await page.locator('[data-train-version="V_2_1"]').waitFor();
  assert.equal(await page.locator('[data-train-version="V_10"], [data-train-version="V_11"], [data-train-version="v2_1"], [data-train-version="v2.1"]').count(),0);
  console.log('Canonical versions, compatible V_2_1 source, 50/100 episode UI, and preserved GPU selection passed.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
