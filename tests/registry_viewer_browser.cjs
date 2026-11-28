const {chromium}=require('../outputs/browser-tools/node_modules/playwright');
const assert=require('node:assert/strict');

(async()=>{
  const base=process.argv[2]||'http://127.0.0.1:8765';
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  try{
    const response=await page.goto(base+'/registry',{waitUntil:'networkidle',timeout:30000});
    assert.equal(response.status(),200);
    await page.waitForURL('**/?page=registry');
    const registry=page.frameLocator('#registry-frame');
    await registry.locator('#view-res').waitFor({state:'visible'});
    await registry.getByRole('button',{name:'历史任务对比'}).click();
    await registry.locator('#chains [aria-current="true"]').waitFor({state:'visible'});
    assert.deepEqual(await registry.locator('#nav button').allTextContents(),['数据与权重','历史任务对比']);
    let annotations;
    await page.route('**/api/registry/annotations',async route=>{
      const {key,hidden}=route.request().postDataJSON();
      if(!annotations)annotations=await (await page.request.get(base+'/api/registry/annotations')).json();
      if(hidden)annotations[key]={hidden};
      else delete annotations[key];
      await route.fulfill({json:annotations});
    });
    const nav=page.getByRole('tablist',{name:'工作台',exact:true});
    const registryTab=nav.getByRole('tab',{name:'资源管理'});
    assert.equal(await registryTab.getAttribute('aria-selected'),'true');
    await page.evaluate(()=>{window.workbenchTestMarker=true;});
    for(const [label,panel] of [['启动训练','training'],['推理评估','launch'],['评估结果','results']]){
      const tab=nav.getByRole('tab',{name:label});
      await tab.click();
      await page.locator('#page-'+panel).waitFor({state:'visible'});
      await registryTab.click();
      await registry.locator('#view-exp').waitFor({state:'visible'});
    }
    await page.goBack();
    await page.locator('#page-results').waitFor({state:'visible'});
    await page.goForward();
    await registry.locator('#view-exp').waitFor({state:'visible'});
    assert.ok(await page.evaluate(()=>window.workbenchTestMarker),'workbench tabs must preserve the outer document');
    const comparedRun=registry.locator('[data-compare-record]').first();
    const comparedName=await comparedRun.textContent();
    await comparedRun.click();
    const record=registry.locator('#chains [data-experiment][aria-current="true"]');
    const recordName=await record.locator('.record-name').textContent();
    assert.equal(recordName,comparedName);
    assert.equal(await record.locator('xpath=ancestor::details[@open]').count(),3);
    await record.press('Enter');
    assert.equal(await registry.locator('.detail-run').textContent(),recordName);
    assert.equal(await registry.locator('#experiment-detail [data-detail-edit]').count(),1);
    await registry.getByRole('button',{name:'编辑标记'}).click();
    assert.equal(await registry.locator('#annotation-run').textContent(),recordName);
    await registry.getByRole('button',{name:'取消'}).click();
    const resultFolder=await registry.locator('[data-open-results]').getAttribute('data-open-results');
    const resultExists=await page.evaluate(folder=>runs.some(run=>run._folder===folder),resultFolder);
    await registry.getByRole('button',{name:'打开评估结果'}).click();
    await page.locator('#page-results').waitFor({state:'visible'});
    if(resultExists)assert.equal(await page.evaluate(()=>selected._folder),resultFolder);
    await registryTab.click();
    await registry.locator('#ledger-task').fill('no-match-registry-test');
    assert.equal(await registry.locator('#ledger-count').textContent(),'0 条');
    await nav.getByRole('tab',{name:'评估结果'}).click();
    await registryTab.click();
    assert.equal(await registry.locator('#ledger-task').inputValue(),'no-match-registry-test');
    await registry.locator('#ledger-clear').click();
    assert.equal(await registry.locator('#ledger-task').inputValue(),'');
    const runToInspect=registry.locator('[data-compare-record]').first();
    const runName=await runToInspect.textContent();
    await runToInspect.click();
    assert.equal(await registry.locator('.detail-run').textContent(),runName);
    for(const width of [1440,1024,780,390]){
      await page.setViewportSize({width,height:width<600?844:1000});
      assert.ok(await registry.locator('#experiment-detail .detail-metric').filter({hasText:'任务进度'}).isVisible());
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`page overflow at ${width}px`);
      assert.ok(await registry.locator('body').evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`registry overflow at ${width}px`);
    }
    await page.screenshot({path:'/tmp/memory-vla-registry-mobile.png'});
    await page.setViewportSize({width:1440,height:1000});
    await page.screenshot({path:'/tmp/memory-vla-registry-desktop.png'});
    await registry.getByRole('button',{name:'返回任务对比'}).click();
    assert.ok(await runToInspect.isVisible());
    const taskSummary=registry.locator('#chains > details:not([open]) > summary').last();
    const taskLabel=await taskSummary.textContent();
    const taskFolder=registry.locator('#chains > details > summary').getByText(taskLabel,{exact:true}).locator('..');
    await taskFolder.locator(':scope > summary').click();
    const taskButton=taskFolder.locator('[data-compare-task]');
    await taskButton.click();
    assert.equal(await registry.locator('#experiment-detail h3').textContent(),taskLabel);
    const taskFilter=registry.locator('#ledger-task-filter');
    await taskFilter.selectOption(taskLabel.match(/^Task (\d+)/)[1]);
    const filteredCount=await registry.locator('[data-compare-record]').count();
    assert.ok(filteredCount>0);
    assert.equal(await registry.locator('#ledger-count').textContent(),`${filteredCount} 条`);
    await registry.locator('[data-compare-record]').first().click();
    await registry.getByRole('button',{name:'返回任务对比'}).click();
    assert.equal(await registry.locator('[data-compare-record]').count(),filteredCount);
    assert.equal(await registry.locator('#experiment-detail h3').textContent(),taskLabel);
    await registry.getByRole('button',{name:'数据与权重'}).click();
    assert.equal(await registry.locator('#view-res').isVisible(),true);
    const inventory=await (await page.request.get(base+'/api/registry')).json();
    const resourceTree=registry.getByRole('navigation',{name:'资源目录'});
    const resourceLinks=resourceTree.locator('[data-resource]:visible');
    const openResourceFolder=async item=>{
      const path=item.path.replace(/^[^/]+:\//,'/');
      const directoryNames=[item.host,...path.split('/').filter(Boolean).slice(0,-1)];
      let branch=resourceTree.locator('#res-rows');
      for(const name of directoryNames){
        const summary=branch.locator(':scope > details > summary').getByText(name,{exact:true});
        const folder=summary.locator('..');
        if(!await folder.evaluate(element=>element.open))await summary.click();
        assert.ok(await folder.evaluate(element=>element.open));
        branch=folder.locator(':scope > .tree-children');
      }
    };
    assert.equal(await resourceLinks.count(),0);
    await registry.locator('#res-kind').selectOption('dataset');
    const dataset=inventory.items.find(item=>item.host==='4090'&&item.kind==='dataset');
    await openResourceFolder(dataset);
    await resourceLinks.getByText(dataset.name,{exact:true}).click();
    assert.equal(await registry.locator('#resource-detail .mono').textContent(),dataset.path);
    const remoteDataset=inventory.items.find(item=>item.host==='A100'&&item.kind==='dataset');
    await openResourceFolder(remoteDataset);
    await resourceLinks.getByText(remoteDataset.name,{exact:true}).click();
    assert.equal(await registry.locator('#resource-detail .mono').textContent(),remoteDataset.path);
    assert.ok(await resourceLinks.getByText(dataset.name,{exact:true}).isVisible());
    await resourceTree.locator('#res-rows > details > summary').getByText('A100',{exact:true}).click();
    assert.equal(await resourceLinks.getByText(remoteDataset.name,{exact:true}).count(),0);
    await registry.locator('#res-kind').selectOption('weights');
    const weightItem=inventory.items.find(item=>item.host==='4090'&&item.kind==='weights');
    await openResourceFolder(weightItem);
    const weightName=weightItem.name;
    const weight=resourceLinks.filter({has:registry.getByText(weightName,{exact:true})});
    await weight.press('Enter');
    assert.equal(await registry.locator('#resource-detail h3').textContent(),weightName);
    assert.equal(await registry.locator('#resource-detail .mono').textContent(),weightItem.path);
    await registry.locator('#res-q').fill(weightName);
    await resourceTree.getByText('跨目录搜索',{exact:true}).waitFor();
    page.once('dialog',dialog=>dialog.accept());
    await registry.getByRole('button',{name:'从台账移除'}).click();
    await registry.getByText('已从台账移除，可在「已移除」中恢复。',{exact:true}).waitFor();
    await registry.locator('#res-state').selectOption('removed');
    assert.equal(await registry.locator('#resource-detail h3').textContent(),weightName);
    await registry.getByRole('button',{name:'恢复到清单'}).click();
    await registry.getByText('已恢复到资源清单。',{exact:true}).waitFor();
    await registry.locator('#res-state').selectOption('active');
    assert.equal(await registry.locator('#resource-detail h3').textContent(),weightName);
    await registry.locator('#res-q').fill('no-match-registry-test');
    assert.equal(await registry.locator('#res-count').textContent(),'0 项');
    await registry.locator('#res-q').clear();
    await weight.waitFor({state:'visible'});
    for(const [label,view] of [['历史任务对比','exp'],['数据与权重','res']]){
      await registry.getByRole('button',{name:label}).click();
      for(const width of [1440,780,390]){
        await page.setViewportSize({width,height:1000});
        const columns=await registry.locator(`#view-${view} .ledger-workspace`).evaluate(workspace=>{
          const [navigation,details]=workspace.children;
          return details.getBoundingClientRect().left>navigation.getBoundingClientRect().left?2:1;
        });
        assert.equal(columns,width>700?2:1,`${view} layout at ${width}px`);
        assert.ok(await registry.locator('body').evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`${view} overflow at ${width}px`);
      }
      await page.setViewportSize({width:1440,height:1000});
      await page.evaluate(()=>scrollTo(0,0));
      await page.screenshot({path:`/tmp/memory-vla-registry-${view}.png`});
    }
    assert.deepEqual(errors,[]);
    console.log('Registry navigation, view switching, filters, responsive layouts, and browser runtime passed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
