// PLAYWRIGHT_MODULE points to an installed playwright or playwright-core package.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const base=process.argv[2]||'http://127.0.0.1:8765';
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const writes=[];
 // Never mutate real resources or start jobs from a browser test.
 await page.route('**/api/**',r=>r.request().method()==='GET'?r.continue():r.abort());
 try{
  await page.goto(base+'/?page=registry');
  const frame=page.frameLocator('#registry-frame');
  await frame.getByRole('button',{name:'数据与权重',exact:true}).click();
  await frame.locator('#res-host').selectOption('4090');
  await frame.getByRole('button',{name:'磁盘空间',exact:true}).click();
  await frame.locator('#storage-disks .storage-disk:not(.stale)').filter({hasText:'实时查询'}).first().waitFor();
  assert.match(await frame.locator('#storage-capacity-summary').textContent(),/剩余/);
  await frame.getByRole('button',{name:'数据与权重',exact:true}).click();
  const name='b1k2025_groot_task0_v1_200';
  await frame.locator('#res-q').fill(name);
  await frame.locator('#res-rows [data-resource]').filter({hasText:name}).click();
  await frame.locator('.storage-entry [data-open-path]').filter({hasText:/meta/}).waitFor();
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/storage-directory.png',fullPage:true});
  await frame.locator('.storage-entry [data-open-path]').filter({hasText:/meta/}).click();
  await frame.locator('.storage-entry [data-open-path]').filter({hasText:'info.json'}).click();
  await frame.locator('.storage-preview').waitFor();
  assert.match(await frame.locator('.storage-preview').textContent(),/total_episodes/);
  // Filtering the same selected resource and switching tabs must retain the open file.
  await frame.locator('#res-q').fill('task0_v1_200');
  await frame.getByRole('button',{name:'历史任务对比',exact:true}).click();
  await frame.getByRole('button',{name:'数据与权重',exact:true}).click();
  assert.match(await frame.locator('.storage-preview').textContent(),/total_episodes/);
  assert.equal(await frame.locator('.resource-info').evaluate(el=>el.open),false);
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/storage-desktop.png',fullPage:true});
  const root='/nvme3/projects/memory_vla/assets/datasets/'+name;
  for(const width of [1440,1024,768,390,320]){
   await page.setViewportSize({width,height:1000});
   assert.ok(await frame.locator('body').evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`resource overflow at ${width}`);
   const overflow=await frame.locator('#view-res').evaluate(root=>[root,...root.querySelectorAll('*')].filter(el=>{
    const style=getComputedStyle(el),rect=el.getBoundingClientRect();
    return rect.width&&rect.height&&el.clientWidth&&style.overflowX==='visible'&&!['inline','contents'].includes(style.display)&&el.scrollWidth>el.clientWidth+2;
   }).map(el=>el.id||el.className));
   assert.deepEqual(overflow,[],`resource content bounds at ${width}`);
   if(width===390&&process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/storage-mobile.png',fullPage:true});
  }
  await page.setViewportSize({width:390,height:844});
  await frame.locator('#storage-register').click();
  assert.ok(await frame.locator('#storage-dialog-submit').evaluate(el=>{
   const button=el.getBoundingClientRect(),iframe=window.frameElement.getBoundingClientRect();
   return button.top+iframe.top>=0&&button.bottom+iframe.top<=window.parent.innerHeight;
  }));
  await frame.locator('#storage-dialog-cancel').click();
  await page.setViewportSize({width:1440,height:1000});
  let created=false,trashed=false,registered=null;
  await page.route('**/api/registry/storage*',async route=>{
   const request=route.request(),url=new URL(request.url());
   if(request.method()==='POST'){
    const data=request.postDataJSON();writes.push(data.action);
    if(data.action==='mkdir'){created=true;return route.fulfill({json:{path:root+'/browser-test'}});}
    if(data.action==='trash'){trashed=true;return route.fulfill({json:{path:data.path,token:'a'.repeat(32)}});}
    if(data.action==='restore'){trashed=false;return route.fulfill({json:{restored:data.path}});}
    if(data.action==='register'){
     registered={host:'4090',path:root,kind:'dataset',name:data.name,id:'browser-added',registered:true};
     return route.fulfill({json:{item:registered}});
    }
    return route.abort();
   }
   if(url.searchParams.get('action')!=='browse')return route.continue();
   const path=url.searchParams.get('path');
   const entries=created&&!trashed?[{name:'browser-test',path:root+'/browser-test',type:'directory',modified:1,size:null}]:[];
   return route.fulfill({json:{host:'4090',path,type:'directory',parent:root,can_delete:true,entries,total:entries.length,offset:0,has_more:false}});
  });
  await page.route('**/api/registry',async route=>{
   const response=await route.fetch();const data=await response.json();
   if(registered)data.items=[...data.items.filter(i=>i.path!==root),registered];
   await route.fulfill({json:data});
  });
  await frame.locator('[data-storage-parent]').click();
  await frame.locator('[data-storage-mkdir]').click();
  await frame.locator('#storage-name').fill('browser-test');
  await frame.locator('#storage-dialog-submit').click();
  await frame.locator('.storage-entry').filter({hasText:'browser-test'}).waitFor();
  await frame.locator('.storage-entry .storage-menu > summary').click();
  await frame.locator('.storage-delete').click();
  await frame.locator('#storage-dialog-submit').click();
  await frame.locator('#res-message').getByRole('button',{name:'恢复刚删除的项目'}).click();
  await frame.locator('.storage-entry').filter({hasText:'browser-test'}).waitFor();
  await frame.locator('#storage-register').click();
  await frame.locator('#storage-host').selectOption('4090');
  await frame.locator('#storage-path').fill(root);
  await frame.locator('#storage-name').fill('浏览器登记测试');
  await frame.locator('#storage-dialog-submit').click();
  await frame.locator('#resource-detail h3').filter({hasText:'浏览器登记测试'}).waitFor();
  assert.deepEqual(writes,['mkdir','trash','restore','register']);
  await page.goto(base+'/?page=training');
  await page.waitForFunction(()=>document.querySelector('#train-dataset')?.options.length===50);
  await page.locator('#train-dataset').evaluate(el=>{for(const o of el.options)o.selected=true;el.dispatchEvent(new Event('change'));});
  assert.match(await page.locator('#train-task-selection-summary').textContent(),/50/);
  assert.ok((await page.locator('#train-submit-summary').textContent()).length<160);
  await page.locator('.train-selected-tasks > summary').click();
  assert.equal(await page.locator('#train-task-selection button').count(),50);
  for(const width of [1440,1024,768,390,320]){
   await page.setViewportSize({width,height:1000});
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`training overflow at ${width}`);
   assert.ok(await page.locator('#train-task-selection').evaluate(el=>el.clientHeight<=140));
  }
  assert.deepEqual(errors,[]);
  console.log('Live disk query, nested directory/text preview, mocked create/delete/restore/register, and 50-task layouts passed.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
