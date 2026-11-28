/* One benchmark selector, the existing training / inference / results pages. */
const benchmarkEscape = value => String(value ?? '').replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const benchmarkNumber = value => value == null ? '—' : Number(value).toPrecision(4);
function benchmarkTable(headers, rows) {
  return '<div class="bench-table-scroll"><table class="bench-table"><thead><tr>' +
    headers.map(v=>'<th>'+benchmarkEscape(v)+'</th>').join('')+'</tr></thead><tbody>'+
    rows.map(row=>'<tr>'+row.map(v=>'<td>'+benchmarkEscape(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
}
function benchmarkMetrics(reports) {
  let html = '';
  const actions = reports.filter(r=>r.data?.after_mae_first8);
  if(actions.length) {
    html += '<div class="bench-note">这里的通过表示前后向与数值检查通过。固定小样本上的动作误差不等于闭环成功率；正常、空、错误历史需要配合对照解读。</div>';
    const labels = {baseline:'Baseline',direct_empty:'Direct · 关闭历史',memory:'Memory'};
    const rows = actions.flatMap(({data:d})=>Object.entries(d.after_mae_first8).map(([condition,value])=>[
      labels[d.arm] || d.arm, {normal:'正常历史',empty:'空历史',wrong:'错误历史',baseline:'原始条件'}[condition] || condition,
      benchmarkNumber(d.before_mae_first8?.[condition]),benchmarkNumber(value)]));
    html += '<h3>前 8 个动作 MAE · 越低越好</h3>'+benchmarkTable(['训练分支','诊断条件','训练前','训练后'],rows);
    const gradients=actions.filter(r=>r.data.arm==='memory').flatMap(r=>r.data.training || []).map(r=>r.cognitive_q_gradient).filter(v=>v!=null);
    if(gradients.length) html += '<p class="bench-subtle">认知 Q 最小梯度 '+benchmarkNumber(Math.min(...gradients))+' · '+gradients.length+' 个优化步有记录。</p>';
    const series=actions.map((r,i)=>({name:labels[r.data.arm]||r.data.arm, points:r.data.training||[],color:['#285fc6','#c57626','#4866b5'][i]}));
    if(series.some(s=>s.points.length)) {
      const all=series.flatMap(s=>s.points), maxStep=Math.max(1,...all.map(p=>p.step)), maxLoss=Math.max(.001,...all.map(p=>p.loss));
      html += '<h3>短训练 Loss</h3><svg class="bench-chart" viewBox="0 0 660 180" role="img" aria-label="各分支短训练 Loss 曲线">'+
        '<line x1="45" y1="150" x2="640" y2="150" stroke="#dbe3ed"/><text x="7" y="20" font-size="11" fill="#63748b">'+maxLoss.toPrecision(3)+'</text><text x="12" y="153" font-size="11" fill="#63748b">0</text>'+series.map(s=>'<polyline fill="none" stroke="'+s.color+'" stroke-width="1.6" points="'+s.points.map(p=>(45+595*p.step/maxStep).toFixed(1)+','+(150-135*p.loss/maxLoss).toFixed(1)).join(' ')+'"/>').join('')+
        '<text x="570" y="172" font-size="11" fill="#63748b">'+maxStep+' steps</text></svg><div class="bench-legend">'+series.map(s=>'<span style="color:'+s.color+'">● '+benchmarkEscape(s.name)+'</span>').join('')+'</div>';
    }
  }
  for(const {data:d} of reports) {
    if(d.category==='exploratory_validation') {
      html += '<div class="bench-note">V2 真实物理标签单轮验证训练；Loss 与权重重载只验收训练链路，不等于闭环效果。</div>';
      html += benchmarkTable(['优化步数','显存峰值 GiB','冻结参数','完整重载最大差异'],
        [[d.steps,benchmarkNumber(d.peak_cuda_gib),d.frozen_parameters_unchanged?'保持原值':'待核对',
          benchmarkNumber(Math.max(...Object.values(d.full_reload_max_abs||{})))]]);
    }
    if(d.label_names?.includes('started') && d.episodes) {
      html += '<div class="bench-note">官方ToggledOn记录：'+(d.labels_certified?'started字段已认证':'解析完成，字段认证待执行')+'；本阶段不更新模型。</div>';
      html += benchmarkTable(['已导出示教','有效正帧','有效负帧','unknown帧'],
        [[d.episodes.length,...['positive_valid','negative_valid','unknown'].map(k=>d.episodes.reduce((s,e)=>s+(e[k]||0),0))]]);
    }
    if(d.episodes?.some(e=>e.label_names)) {
      html += '<div class="bench-note">官方示教物理状态回放审计；标签尚未认证用于训练，模型更新步数为 0。</div>';
      html += benchmarkTable(['示教','观测数','状态一致性','Open / Inside / On 正帧'],
        d.episodes.map(e=>[e.episode,e.observation_count,e.passed?'通过':'未通过',(e.positive_counts||[]).join(' / ')]));
    }
    if(d.category==='engineering_smoke' && d.labels) {
      html += '<div class="bench-note">V2 训练链路自检：状态标签为人工测试值，仅验证梯度、更新与重载，不代表任务状态学习或闭环效果。</div>';
      html += benchmarkTable(['优化步数','显存峰值 GiB','冻结动作专家','重载最大差异'],
        [[d.steps,benchmarkNumber(d.peak_cuda_gib),d.frozen_action_head_unchanged?'保持原值':'待核对',
          benchmarkNumber(Math.max(...Object.values(d.reload_max_abs||{})))]]);
    }
    if(d.type==='unit') html += '<h3>代码回归</h3>'+benchmarkTable(['测试数','失败','错误','跳过'],[[d.tests,d.failures,d.errors,d.skipped]]);
    if(d.results?.layernorm) {
      html += '<h3>真实 EOS 认知检索</h3>'+benchmarkTable(['检索实现','查询次数','饱和比例','最早一帧权重','Q 梯度中位数'],
        Object.entries(d.results).map(([k,v])=>[k==='none'?'旧实现':'Q/K 归一化',v.queries,
          (v.saturated_fraction*100).toFixed(1)+'%',(v.oldest_probability_mean*100).toFixed(2)+'%',benchmarkNumber(v.q_gradient_median)]));
      html += '<p class="bench-subtle">检查多条真实序列是否持续锁定同一帧。通过不能代替任务效果验证。</p>';
    }
    if(d.suite && d.tasks && !d.type) {
      html += '<div class="bench-note">零动作环境检查：只验证 reset、相机与仿真 step。没有运行模型策略，不统计策略成功率。</div>';
      html += '<h3>'+benchmarkEscape(d.suite)+'</h3>'+benchmarkTable(['任务','执行步数','相机','初态 / seed'],
        d.tasks.map(t=>['Task'+t.task+' · '+t.language,t.steps,Object.entries(t.image_shapes).map(([k,v])=>k+' '+v.join('×')).join(' / '),String(d.initial_state_index)+' / '+d.seed]));
    }
  }
  if(typeof window!=='undefined' && window.liberoWorkflow) html += window.liberoWorkflow.metrics(reports);
  return html || '<div class="bench-empty">等待检查完成后生成指标。运行进度和日志会自动更新。</div>';
}
if(typeof module!=='undefined' && module.exports) module.exports={benchmarkMetrics,benchmarkTable};

(() => {
  if(typeof window==='undefined') return;
  const holder=document.createElement('div');
  holder.innerHTML=window.__benchmarkHTML;
  delete window.__benchmarkHTML;
  const style=holder.querySelector('style');document.head.appendChild(style);
  document.querySelector('.page-nav').append(holder.querySelector('#benchmark-selector'));
  const workspace=holder.querySelector('#bench-workspace'), resultSwitch=holder.querySelector('#bench-result-switch'), checks=holder.querySelector('#bench-code-checks');
  document.getElementById('page-results').before(resultSwitch,workspace);
  document.getElementById('page-training').before(checks);
  const el=id=>document.getElementById(id), escape=benchmarkEscape;
  const titles={b1k:'BEHAVIOR-1K',libero:'LIBERO',mikasa:'MIKASA-Robo',simpler:'SimplerEnv'};
  const status={starting:'启动中',running:'运行中',stopping:'停止中',stopped:'已停止',complete:'检查通过',failed:'检查失败',interrupted:'运行中断'};
  let benchmark='b1k', page='results', resultMode='policy', selectedId=null, detail=null;
  let options=null, jobs=[], busy=false, refreshing=false, tab='metrics', optionsAt=0;
  const selection={};
  const running=s=>['starting','running','stopping'].includes(s);
  const currentKind=()=>benchmark==='b1k'?'code':benchmark;
  const isNative=()=>benchmark==='b1k' && (page!=='results'||resultMode==='policy');
  function message(id,text,error=false){el(id).textContent=text;el(id).className='bench-message'+(error?' error':'');}
  async function api(action,body){
    const response=await fetch('/api/benchmarks/'+action,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{cache:'no-store'});
    const data=await response.json();if(!response.ok)throw Error(data.error || '请求失败');return data;
  }
  function route(nextPage){
    page=nextPage;
    if(page!=='results')el('bench-pane-cameras').querySelector('video')?.pause();
    const native=isNative();
    for(const name of ['training','launch','results']) el('page-'+name).hidden=!(native && name===page);
    workspace.hidden=native;
    for(const name of ['training','launch','results']) el('bench-'+name).hidden=native || name!==page;
    checks.hidden=!(benchmark==='b1k' && page==='training');
    resultSwitch.hidden=!(benchmark==='b1k' && page==='results');
    el('bench-policy-results').setAttribute('aria-pressed',String(resultMode==='policy'));
    el('bench-code-results').setAttribute('aria-pressed',String(resultMode==='code'));
    el('service-overview').hidden=false;
    el('gpu-overview').hidden=benchmark==='mikasa'||benchmark==='simpler';
    if(!native){stopPreview();el('video-player').pause();}
    if(benchmark==='b1k' && !native){
      el('bench-results-title').textContent='BEHAVIOR-1K · 开发快测';
      el('bench-results-subtitle').textContent='开发快测与正式策略评估分开记录。';
    }
    if(!native || !checks.hidden)void refresh();
    window.liberoWorkflow?.route(benchmark,page);
    return !native;
  }
  function changeBenchmark(){
    selection[currentKind()]=selectedId;
    benchmark=el('benchmark-choice').value;
    selectedId=selection[currentKind()]||null;detail=null;
    const available=benchmark==='b1k'?'训练、策略推理与闭环评估已接入':benchmark==='libero'?'示教准备、训练与策略评估已接入 · 本机单卡':'环境、训练与策略接口待接入';
    el('benchmark-capability').textContent=available;
    el('benchmark-choice').title=available;
    document.querySelector('.page-foot').textContent=benchmark==='b1k'?'GR00T N1.6 · B1K-2025 · 调试记录与正式评估分别标记':titles[benchmark]+' · '+available;
    el('benchmark-description').textContent=benchmark==='b1k'?'沿用现有数据、权重与评估记录；训练页提供开发快测。':benchmark==='libero'?'Franka 末端控制 · 固定初态 · 双相机；结果与 B1K 独立。':'保留相同三步流程，接入后在当前 benchmark 下操作。';
    for(const name of ['training','launch','results']) el('bench-'+name+'-title').textContent=titles[benchmark]+' · '+{training:'训练',launch:'推理评估',results:'评估结果'}[name];
    el('bench-training-description').textContent=benchmark==='libero'?'当前已接通 LIBERO 环境检查。LIBERO 示教数据、动作归一化和 Franka 机器人接口尚未接入训练器。':titles[benchmark]+' 的数据、机器人接口和训练配置尚未接入。';
    el('bench-training-next').textContent=benchmark==='libero'?'先检查环境，再准备示教与策略适配。':'可以切换到已接入的 benchmark 继续使用工作台。';
    el('bench-policy-title').textContent=benchmark==='libero'?'GR00T / Memory · LIBERO':'推理接口待接入';
    el('bench-policy-description').textContent=benchmark==='libero'?'选择 LIBERO 训练权重，自动加载策略并评估，结束后释放模型。':'当前没有可启动的策略与环境。';
    el('bench-libero-form').hidden=benchmark!=='libero';el('bench-not-connected').hidden=benchmark==='libero';
    el('bench-gpu-refresh').hidden=benchmark!=='libero';
    el('bench-results-subtitle').textContent=benchmark==='b1k'?'开发快测与正式策略评估分开记录。':benchmark==='libero'?'数据准备、训练、策略评估分别记录；短回合调试不计入完整时长成功率。':'该 benchmark 暂无已接入的运行记录。';
    el('bench-pane-cameras').querySelector('video')?.pause();
    stopPreview();el('video-player').pause();
    selectPage(page);
    renderList();renderDetail();
  }
  function updateGPUs(){
    for(const id of ['bench-gpu','bench-code-gpu']){
      const select=el(id), previous=select.value;
      select.innerHTML=(options?.gpus||[]).map(g=>'<option value="'+g.index+'">GPU '+g.index+' · 可用 '+(g.free_mb/1024).toFixed(1)+' / '+(g.total_mb/1024).toFixed(0)+' GiB</option>').join('') || '<option value="">GPU 不可用</option>';
      if((options?.gpus||[]).some(g=>String(g.index)===previous))select.value=previous;
      else if(options?.gpus.length)select.value=String([...options.gpus].sort((a,b)=>b.free_mb-a.free_mb)[0].index);
    }
  }
  function updateTasks(){
    const suite=options?.suites.find(s=>s.id===el('bench-suite').value);
    el('bench-tasks').innerHTML=(suite?.tasks||[]).map(t=>'<option value="'+t.id+'"'+(t.id===0?' selected':'')+'>Task'+t.id+' · '+escape(t.name)+'</option>').join('');
  }
  async function loadOptions(){
    options=await api('options');optionsAt=Date.now();updateGPUs();
    if(!el('bench-suite').options.length){el('bench-suite').innerHTML=options.suites.map(s=>'<option value="'+s.id+'">LIBERO '+escape(s.title)+'</option>').join('');updateTasks();}
    window.liberoWorkflow?.options(options);setButtons();
  }
  function setButtons(){
    const anyRunning=jobs.some(j=>!j.legacy&&running(j.status));
    for(const [id,kind] of [['bench-libero-start','libero'],['bench-code-start',el('bench-code-kind').value]]){
      const item=options?.catalog.find(c=>c.id===kind);
      el(id).disabled=busy||anyRunning||!item?.available||!options?.gpus.length;
      el(id).title=anyRunning?'已有快速检查运行中':!item?.available?'缺少依赖：'+(item?.missing||[]).join('、'):'';
    }
    el('bench-libero-preview').disabled=busy;
    window.liberoWorkflow?.buttons({busy,anyRunning,options});
  }
  async function refresh(){
    if(refreshing)return;refreshing=true;
    try{
      if(!options)await loadOptions();
      const previous=jobs; jobs=(await api('jobs')).jobs;
      if(Date.now()-optionsAt>20000 || previous.some(j=>running(j.status)&&jobs.some(n=>n.id===j.id&&!running(n.status))))await loadOptions();
      renderList();setButtons();
      if(selectedId){const requested=selectedId;const result=await api('detail?run='+encodeURIComponent(requested));if(requested===selectedId){detail=result;renderDetail();}}
    }catch(error){message(page==='training'?(benchmark==='libero'?'libero-training-message':'bench-code-message'):page==='launch'?'bench-libero-message':'bench-detail-message',error.message,true);}
    finally{refreshing=false;}
  }
  function renderList(){
    const filter=el('bench-search').value.toLowerCase();
    const visible=jobs.filter(j=>benchmark==='b1k'?['network','action','v2_training_smoke','v2_label_audit','v2_state_decode','v2_labels_export','v2_train'].includes(j.kind):benchmark==='libero'?j.kind.startsWith('libero'):false)
      .filter(j=>(j.title+' '+(j.config.notes||'')+' '+j.id).toLowerCase().includes(filter));
    if(!visible.some(j=>j.id===selectedId)){selectedId=visible[0]?.id||null;detail=null;}
    el('bench-run-count').textContent=visible.length+' 条';
    if(!isNative())el('caption').textContent=titles[benchmark]+' · 自动刷新 · '+visible.length+' 条运行记录';
    el('bench-job-list').innerHTML=visible.map(j=>'<button class="bench-job" type="button" data-run="'+j.id+'" aria-selected="'+String(j.id===selectedId)+'"><span class="bench-job-top"><strong>'+escape(j.title)+'</strong><span class="bench-status '+j.status+'">'+escape(jobStatus(j))+'</span></span><small>'+escape(j.created_at)+(j.legacy?' · 已有验收报告':j.kind==='libero_dataset'?' · 数据准备':' · GPU '+j.config.gpu)+'</small>'+(j.config.notes?'<small>'+escape(j.config.notes)+'</small>':'')+'</button>').join('') || '<div class="bench-empty">当前 benchmark 暂无匹配记录。<br>运行完成后会自动出现在这里。</div>';
  }
  function jobStatus(job){return job.kind.startsWith('libero_')?({complete:'已完成',failed:'运行失败'}[job.status]||status[job.status]||job.status):(status[job.status]||job.status);}
  function renderDetail(){
    if(!detail || detail.job.id!==selectedId){
      el('bench-detail-title').textContent='选择一次运行';el('bench-detail-subtitle').textContent='';el('bench-detail-status').textContent='';
      el('bench-pane-metrics').innerHTML='<div class="bench-empty">从左侧选择记录查看指标、截图与日志。</div>';
      delete el('bench-pane-metrics').dataset.signature;
      el('bench-pane-cameras').innerHTML='';el('bench-log').textContent='';el('bench-config').textContent='';el('bench-raw').textContent='';el('bench-stop').hidden=true;el('bench-progress').hidden=true;el('bench-progress-label').textContent='';return;
    }
    const j=detail.job;
    el('bench-detail-title').textContent=j.title;el('bench-detail-subtitle').textContent=j.created_at+' · '+(j.legacy?'已有验收报告':j.id)+(j.config.notes?' · '+j.config.notes:'');
    el('bench-detail-status').textContent=jobStatus(j);el('bench-detail-status').className='bench-status '+j.status;
    el('bench-stop').hidden=j.legacy||!running(j.status);el('bench-stop').disabled=j.status==='stopping';
    const fraction=j.steps&&j.step?j.step/j.steps:0;
    el('bench-progress').hidden=false;el('bench-progress').max=j.total||1;el('bench-progress').value=(j.completed||0)+fraction;
    el('bench-progress-label').textContent=(j.stage||'已完成阶段')+' · '+(j.completed||0)+' / '+(j.total||1)+(j.step?' · '+j.step+'/'+j.steps+' 步':'')+(j.elapsed_seconds!=null?' · '+Math.round(j.elapsed_seconds)+' 秒':'');
    message('bench-detail-message',j.error||'');
    const signature=j.id+JSON.stringify(detail.reports);
    if(el('bench-pane-metrics').dataset.signature!==signature){
      el('bench-pane-metrics').innerHTML=benchmarkMetrics(detail.reports);el('bench-pane-metrics').dataset.signature=signature;
      const frames=detail.reports.flatMap(r=>(r.data.tasks||[]).flatMap(t=>Object.entries(t.frames||{}).flatMap(([when,cameras])=>Object.entries(cameras).map(([camera,file])=>({task:t.task,when,camera,file})))));
      el('bench-pane-cameras').querySelector('video')?.pause();
      el('bench-pane-cameras').innerHTML=frames.length?'<div class="bench-camera-grid">'+frames.map(f=>'<figure><img loading="lazy" alt="Task'+f.task+' '+escape(f.camera)+' '+escape(f.when)+'" src="/api/benchmarks/image?run='+encodeURIComponent(j.id)+'&file='+encodeURIComponent(f.file)+'"><figcaption>Task'+f.task+' · '+(f.when==='start'?'初始':'结束')+' · '+(f.camera==='agentview'?'外部相机':'腕部相机')+'</figcaption></figure>').join('')+'</div>':'<div class="bench-empty">本记录没有相机截图。网页新启动的 LIBERO 检查会保存起止画面。</div>';
      el('bench-raw').textContent=JSON.stringify(detail.reports,null,2);
      window.liberoWorkflow?.detail(j,detail.reports);
    }
    const log=el('bench-log'), atBottom=log.scrollHeight-log.scrollTop-log.clientHeight<35;
    if(log.textContent!==detail.log){log.textContent=detail.log;if(atBottom)log.scrollTop=log.scrollHeight;}
    el('bench-config').textContent=JSON.stringify(detail.plan||j.config,null,2);
  }
  function showCodeResults(){resultMode='code';selectPage('results');el('bench-results-title').textContent='BEHAVIOR-1K · 开发快测';}
  async function launch(kind,preview=false){
    const isCode=kind!=='libero', target=isCode?'bench-code-message':'bench-libero-message';
    const payload=isCode?{kind,gpu:Number(el('bench-code-gpu').value),notes:el('bench-code-notes').value,
      ...(kind==='action'?{steps:Number(el('bench-code-steps').value),arm:el('bench-arm').value}:{})}:
      {kind,gpu:Number(el('bench-gpu').value),suite:el('bench-suite').value,tasks:[...el('bench-tasks').selectedOptions].map(o=>Number(o.value)),steps:Number(el('bench-steps').value),seed:Number(el('bench-seed').value),notes:el('bench-notes').value};
    return run(payload,target,preview);
  }
  async function run(payload,target,preview=false){
    const isCode=['network','action'].includes(payload.kind);
    busy=true;setButtons();message(target,preview?'正在检查配置…':'正在启动…');
    try{
      const result=await api(preview?'preview':'start',payload);
      if(preview){message(target,result.title+' · '+result.stages.length+' 个阶段'+(payload.kind==='libero_dataset'?' · 配置可用':' · GPU '+result.config.gpu+'\n'+(result.ready?'显存满足运行要求。':'可用显存不足，请选择其他 GPU。')),!result.ready);}
      else{selectedId=result.job.id;detail=result;selection[currentKind()]=selectedId;message(target,'已启动，进度与结果自动保存。');
        el('bench-last-run').textContent='最近启动：'+result.job.id;
        jobs=[result.job,...jobs];if(isCode)showCodeResults();else selectPage('results');renderList();renderDetail();}
    }catch(error){message(target,error.message,true);}
    finally{busy=false;setButtons();}
  }
  el('benchmark-choice').onchange=changeBenchmark;
  el('bench-to-launch').onclick=()=>selectPage('launch');el('bench-to-results').onclick=()=>selectPage('results');
  el('bench-run-new').onclick=()=>{selectPage(benchmark==='b1k'?'training':'launch');if(benchmark==='b1k')checks.open=true;};
  el('bench-switch-libero').onclick=()=>{el('benchmark-choice').value='libero';changeBenchmark();};
  el('bench-policy-results').onclick=()=>{resultMode='policy';selectPage('results');};
  el('bench-code-results').onclick=showCodeResults;el('bench-code-history').onclick=showCodeResults;
  el('bench-suite').onchange=updateTasks;
  el('bench-gpu-refresh').onclick=()=>loadOptions().catch(e=>message('bench-libero-message',e.message,true));
  el('bench-code-kind').onchange=()=>{const action=el('bench-code-kind').value==='action';el('bench-arm-field').hidden=!action;el('bench-code-steps-field').hidden=!action;setButtons();};
  el('bench-libero-form').onsubmit=e=>{e.preventDefault();void launch('libero');};
  el('bench-libero-preview').onclick=()=>void launch('libero',true);
  el('bench-code-form').onsubmit=e=>{e.preventDefault();void launch(el('bench-code-kind').value);};
  el('bench-search').oninput=()=>{renderList();void refresh();};
  el('bench-job-list').onclick=e=>{const button=e.target.closest('[data-run]');if(button){selectedId=button.dataset.run;detail=null;renderList();renderDetail();void refresh();}};
  document.querySelectorAll('[data-bench-tab]').forEach(button=>button.onclick=()=>{tab=button.dataset.benchTab;if(tab!=='cameras')el('bench-pane-cameras').querySelector('video')?.pause();document.querySelectorAll('[data-bench-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.benchTab===tab)));for(const name of ['metrics','cameras','logs'])el('bench-pane-'+name).hidden=name!==tab;});
  el('bench-stop').onclick=async()=>{if(!detail)return;el('bench-stop').disabled=true;try{await api('stop',{run:detail.job.id});await refresh();}catch(e){message('bench-detail-message',e.message,true);el('bench-stop').disabled=false;}};
  window.benchmarkWorkbench={route,isNative,refresh,run,loadOptions,options:()=>options,updateButtons:setButtons,trainingReady:()=>route(page)};
  route(page);
  setInterval(()=>{if(!workspace.hidden||(!checks.hidden&&checks.open))void refresh();},4000);
})();
