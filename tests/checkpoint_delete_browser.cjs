// Real page/GET fixtures; every deletion and restore is intercepted in the browser.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const base=process.argv[2]||'http://127.0.0.1:8765',errors=[],writes=[];
 page.on('pageerror',error=>errors.push(error.message));
 const jobs=await (await page.request.get(base+'/api/training/jobs?device=a100')).json();
 const job=jobs.jobs.find(j=>j.run==='v1__test01__exp13__run001');
 assert.ok(job,'existing training fixture');
 const detail=await (await page.request.get(base+'/api/training/detail?device=a100&run='+job.run+'&arm='+job.arm)).json();
 const [first,second]=detail.checkpoints;
 assert.ok(first&&second,'at least two checkpoints');
 let removed=new Set(),blocked=false,failDelete=false;
 await page.route('**/api/**',async route=>{
  const request=route.request(),url=new URL(request.url());
  if(request.method()!=='GET'){
   const body=request.postDataJSON();writes.push({path:url.pathname,body});
   if(url.pathname==='/api/training/checkpoint-delete'){
    if(body.preview){
     if(blocked)return route.fulfill({status:400,json:{error:'该 checkpoint 正被推理服务使用（端口 5557）'}});
     return route.fulfill({json:{status:'preview',expected_checkpoint:'fixture-identity',warning:'删除最新断点后无法从该步续跑。'}});
    }
    assert.equal(body.confirm,true);assert.equal(body.expected_checkpoint,'fixture-identity');
    if(failDelete)return route.fulfill({status:400,json:{error:'文件仍被占用，请稍后重试'}});
    removed.add(body.checkpoint);
    return route.fulfill({json:{...body,status:body.permanent?'deleted':'trashed',token:'a'.repeat(32),trash_path:'/fixture/trash'}});
   }
   if(url.pathname==='/api/registry/storage'){
    assert.equal(body.action,'restore');assert.equal(body.host,'A100');
    removed.delete(body.path);return route.fulfill({json:{restored:body.path}});
   }
   throw Error('Unexpected write: '+url.pathname);
  }
  if(url.pathname==='/api/training/jobs')return route.fulfill({json:jobs});
  if(url.pathname==='/api/training/detail')return route.fulfill({json:{...detail,checkpoints:detail.checkpoints.filter(p=>!removed.has(p.path))}});
  return route.continue();
 });
 try{
  await page.goto(base+'/?page=training');
  const row=page.locator('[data-job][title="'+job.run+'"]');
  await row.waitFor({state:'attached'});
  await row.evaluate(el=>{for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
  await row.click();await page.locator('#train-monitor-body:not([hidden])').waitFor();
  await page.locator('#train-tab-controls').click();
  const select=page.locator('#train-saved-checkpoint'),open=page.locator('#train-checkpoint-delete');
  const modal=page.locator('#train-checkpoint-delete-dialog'),confirm=page.locator('#train-checkpoint-delete-confirm');
  await select.selectOption(first.path);await open.click();await expectEnabled(confirm);
  assert.equal(await page.locator('#train-checkpoint-delete-path').textContent(),first.path);
  assert.equal(await page.locator('#train-checkpoint-delete-permanent').isChecked(),false);
  await page.locator('#train-checkpoint-delete-close').click();
  assert.equal(removed.size,0);assert.equal(writes.filter(w=>w.body.confirm).length,0);
  blocked=true;await open.click();
  await page.waitForFunction(()=>document.getElementById('train-checkpoint-delete-error').textContent.includes('5557'));
  assert.ok(await confirm.isDisabled());await page.locator('#train-checkpoint-delete-close').click();
  blocked=false;await open.click();await expectEnabled(confirm);
  // Polling or selection changes must not silently change the confirmed target.
  await select.evaluate((el,path)=>{el.value=path;el.dispatchEvent(new Event('change'));},second.path);
  assert.equal(await page.locator('#train-checkpoint-delete-path').textContent(),first.path);
  failDelete=true;await confirm.click();
  await page.waitForFunction(()=>document.getElementById('train-checkpoint-delete-error').textContent.includes('稍后重试'));
  assert.ok(await modal.isVisible());assert.equal(removed.size,0);
  failDelete=false;await confirm.click();await modal.waitFor({state:'hidden'});
  await page.waitForFunction(path=>![...document.querySelector('#train-saved-checkpoint').options].some(o=>o.value===path),first.path);
  assert.ok(removed.has(first.path));assert.ok(!removed.has(second.path));
  assert.equal(await select.inputValue(),second.path);assert.ok(await row.isVisible());
  assert.ok((await page.locator('#train-checkpoint-delete-message').textContent()).includes('仍占磁盘'));
  await page.locator('#train-checkpoint-undo').click();
  await page.waitForFunction(path=>[...document.querySelector('#train-saved-checkpoint').options].some(o=>o.value===path),first.path);
  assert.equal(removed.size,0);
  await select.selectOption(second.path);await open.click();await expectEnabled(confirm);
  await page.locator('#train-checkpoint-delete-permanent').check();
  assert.match(await confirm.textContent(),/永久删除/);
  for(const width of [390,320]){
   await page.setViewportSize({width,height:900});
   const bounds=await modal.boundingBox();assert.ok(bounds.x>=0&&bounds.x+bounds.width<=width);
   assert.equal(await modal.evaluate(el=>el.scrollWidth>el.clientWidth),false);
  }
  await page.screenshot({path:'/nvme3/projects/memory_vla/outputs/workbench-layout-check.SnNt7j/checkpoint-delete-320.png'});
  await confirm.click();await modal.waitFor({state:'hidden'});
  assert.ok(removed.has(second.path));assert.ok(!removed.has(first.path));
  assert.ok(await page.locator('#train-checkpoint-undo').isHidden());
  assert.ok(writes.find(w=>w.body.confirm&&w.body.permanent&&w.body.checkpoint===second.path));
  assert.deepEqual(errors,[]);
  console.log('Single checkpoint: cancel, busy protection, pinned identity, retry, recycle, restore, permanent delete and 320/390px layout passed; all writes mocked.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
async function expectEnabled(locator){await locator.page().waitForFunction(()=>!document.getElementById('train-checkpoint-delete-confirm').disabled);}
