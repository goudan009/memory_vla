// Real UI, fixture result records; never starts, stops or annotates real workloads.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const base=process.argv[2]||'http://127.0.0.1:8765',errors=[],writes=[];
 const fixture={_folder:'V_1/baseline/fixture_resume',version:'V_1',group:'baseline',task_id:0,
  status:'cancelled',resume_allowed:false,evaluation_mode:'official-eval',requested_episodes:50,completed_episodes:2,
  recorded_episodes:2,invalid_episodes:0,successful_episodes:0,success_rate:0,
  configuration:{mode:'single2',preset:'official-eval',policy_server:'172.28.62.17:5556'},
  _episodes:[1,2].map(i=>({episode_index:i,instance_id:i,seed:42+i,gpu_index:0,env_idx:i-1,valid:true,
    notes:'',success:false,q_score:.2,task_progress:20,sim_steps:32})),_videos:[],_logs:{}};
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/**',route=>{
  const req=route.request(),url=new URL(req.url());
  if(req.method()!=='GET'){
   const body=req.postDataJSON();writes.push({path:url.pathname,body});
   if(url.pathname==='/api/launch/preview')return route.fulfill({json:{command:'fixture --dry-run',plan:{
    recorded_episodes:2,additional_episodes:body.additional_episodes??48,requested_episodes:2+(body.additional_episodes??48),output:'/fixture',instances:{0:[0,1]}}}});
   if(url.pathname==='/api/episode-notes'){
    assert.equal(body.run,fixture._folder);
    fixture._episodes.find(e=>e.episode_index===body.episode_index).notes=body.notes.trim();
    return route.fulfill({json:{...body,notes:body.notes.trim()}});
   }
   throw Error('Unexpected write: '+url.pathname);
  }
  if(url.pathname==='/api/results')return route.fulfill({json:{runs:[fixture],updated_at:'fixture'}});
  if(url.pathname==='/api/launch/jobs')return route.fulfill({json:{jobs:[]}});
  if(url.pathname==='/api/human-review')return route.fulfill({json:{review:null}});
  return route.continue();
 });
 try{
  await page.goto(base+'/?page=results');
  const resume=page.locator('#evaluation-actions [data-evaluation-action="resume"]');
  await resume.waitFor();assert.equal(await page.locator('[data-evaluation-action="finish"]').count(),0);
  await resume.click();assert.equal(writes.length,0,'opening resume must not schedule a job');
  assert.equal(await page.locator('#launch-resume-budget').inputValue(),'additional');
  await page.locator('#launch-resume-count').fill('7');
  assert.match(await page.locator('#launch-resume-summary').textContent(),/已有 2 个.*本次再跑 7 个.*累计 9 个/);
  await page.locator('#launch-check').click();
  await page.waitForFunction(()=>document.getElementById('launch-plan').textContent.includes('累计 9'));
  assert.equal(writes.at(-1).body.additional_episodes,7);
  await page.locator('#launch-resume-budget').selectOption('remaining');
  assert.match(await page.locator('#launch-resume-summary').textContent(),/本次再跑 48 个.*累计 50 个/);
  await page.locator('#launch-check').click();
  await page.waitForFunction(()=>document.getElementById('launch-plan').textContent.includes('累计 50'));
  assert.ok(!('additional_episodes' in writes.at(-1).body));
  // Each completed episode gets its own free text note, without failure labels.
  await page.locator('#page-tab-results').click();
  await page.locator('[onclick="chooseEpisodeNote(1)"]').click();
  const text=page.locator('#episode-note-text'),save=page.locator('#episode-note-save');
  await text.fill('第一次抓取滑落 <script>');await save.click();
  await page.waitForFunction(()=>document.getElementById('episode-note-message').textContent==='备注已保存');
  assert.equal(fixture._episodes[0].notes,'第一次抓取滑落 <script>');
  assert.equal(fixture._episodes[0].human_reviewed,undefined,'a note does not mark failure review complete');
  await page.locator('#episode-select').selectOption('episode:2');
  assert.equal(await text.inputValue(),'');await text.fill('第二回合草稿');
  await page.locator('#episode-select').selectOption('episode:1');
  assert.equal(await text.inputValue(),'第一次抓取滑落 <script>');
  await page.locator('#episode-select').selectOption('episode:2');assert.equal(await text.inputValue(),'第二回合草稿');
  await page.evaluate(()=>refresh());assert.equal(await text.inputValue(),'第二回合草稿','polling preserves unsaved draft');
  await save.click();await page.waitForFunction(()=>document.getElementById('episode-note-message').textContent==='备注已保存');
  await page.reload();await resume.waitFor();
  await page.locator('[onclick="chooseEpisodeNote(2)"]').click();
  assert.equal(await text.inputValue(),'第二回合草稿','saved note survives reload');
  for(const width of [1440,768,390,320]){
   await page.setViewportSize({width,height:1000});
   assert.equal(await page.locator('#episode-note-panel').evaluate(el=>el.scrollWidth>el.clientWidth+2),false);
   await page.locator('#page-tab-results').click();await page.locator('#tab-episodes').click();
   assert.match(await page.locator('#details').textContent(),/第一次抓取滑落 <script>/);
   await page.locator('[onclick="chooseEpisodeNote(2)"]').click();
  }
  await page.locator('#episode-note-clear').click();
  await page.waitForFunction(()=>document.getElementById('episode-note-message').textContent==='已清空备注');
  assert.equal(fixture._episodes[1].notes,'');assert.equal(fixture._episodes[0].notes,'第一次抓取滑落 <script>');
  fixture.status='complete';fixture.recorded_episodes=fixture.requested_episodes;
  await page.evaluate(()=>refresh());await page.locator('#evaluation-actions [data-evaluation-action="resume"]').click();
  assert.equal(await page.locator('#launch-resume-budget').inputValue(),'additional');
  assert.deepEqual(errors,[]);
  console.log('Paused/historical/completed evaluation continuation, custom budgets, independent episode notes, drafts, reload and mobile layout passed; all writes mocked.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
