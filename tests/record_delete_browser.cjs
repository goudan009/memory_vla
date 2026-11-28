// Check real deletion dialogs; all mutations are intercepted and use fixture records.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const base=process.argv[2]||'http://127.0.0.1:8765',writes=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const response=await (await page.request.get(base+'/api/training/jobs?device=a100')).json();
 const original=response.jobs.find(j=>j.can_delete&&j.can_delete_files);
 assert.ok(original,'read-only training fixture');
 const detail=await (await page.request.get(base+'/api/training/detail?device=a100&run='+original.run+'&arm='+original.arm)).json();
 const job={...original,run:'fixture_delete_training',name:'fixture_delete_training',output:'/fixture/training'};
 let jobs=[job],runs=[{_folder:'V_1/baseline/fixture_delete_eval',version:'V_1',group:'baseline',
  status:'complete',task_id:0,evaluation_mode:'official-eval',requested_episodes:1,
  recorded_episodes:1,completed_episodes:1,_episodes:[],_videos:[],_logs:{}}];
 await page.route('**/api/**',route=>{
  const req=route.request(),url=new URL(req.url());
  if(req.method()!=='GET'){
   const body=req.postDataJSON();writes.push({path:url.pathname,body});
   if(url.pathname==='/api/training/delete'){
    assert.equal(body.run,job.run);assert.equal(body.delete_files,true);
    assert.equal(body.expected_output,job.output);jobs=[];
    return route.fulfill({json:{status:'deleted',files_preserved:false,output:job.output}});
   }
   if(url.pathname==='/api/launch/delete'){
    assert.equal(body.run,'V_1/baseline/fixture_delete_eval');runs=[];
    return route.fulfill({json:{state:'deleted',message:'已永久删除此评估，无法恢复'}});
   }
   throw Error('Unexpected write: '+url.pathname);
  }
  if(url.pathname==='/api/training/jobs')return route.fulfill({json:{...response,jobs}});
  if(url.pathname==='/api/training/detail')return route.fulfill({json:detail});
  if(url.pathname==='/api/results')return route.fulfill({json:{runs,updated_at:'fixture'}});
  if(url.pathname==='/api/launch/jobs')return route.fulfill({json:{jobs:[]}});
  return route.continue();
 });
 try{
  await page.goto(base+'/?page=training');
  const row=page.locator('[data-job][title="'+job.run+'"]');
  await row.waitFor({state:'attached'});
  await row.evaluate(el=>{for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
  await row.click();await page.locator('#train-more-actions > summary').click();await page.locator('#train-delete:not([disabled])').waitFor();
  await page.locator('#train-delete').click();
  const files=page.locator('#train-delete-files');assert.equal(await files.isChecked(),false);
  assert.match(await page.locator('#train-delete-description').textContent(),/保留权重和日志/);
  await files.check();
  assert.match(await page.locator('#train-delete-description').textContent(),/永久删除.*无法恢复/);
  assert.ok(!(await page.locator('#train-delete-description').textContent()).includes('.trash'));
  await page.locator('#train-delete-close').click();assert.equal(writes.length,0);
  await page.locator('#train-delete').click();await files.check();
  await page.locator('#train-delete-confirm').click();
  await page.waitForFunction(()=>!document.getElementById('train-delete-dialog').open);
  assert.equal(writes[0].path,'/api/training/delete');
  await page.locator('#page-tab-results').click();
  const remove=page.locator('#evaluation-actions [data-evaluation-action="delete"]');
  await remove.waitFor();
  page.once('dialog',async dialog=>{assert.match(dialog.message(),/永久删除.*无法恢复/);await dialog.dismiss();});
  await remove.click();assert.equal(writes.length,1);
  page.once('dialog',async dialog=>{await dialog.accept();});await remove.click();
  await page.waitForFunction(()=>document.getElementById('evaluation-action-message').textContent.includes('永久删除'));
  assert.equal(writes.at(-1).path,'/api/launch/delete');assert.deepEqual(errors,[]);
  console.log('Training optional permanent output deletion and evaluation permanent deletion dialogs passed; all writes mocked.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
