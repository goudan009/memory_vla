/* LIBERO uses the same three pages and durable job runner as the workbench. */
(() => {
  const el=id=>document.getElementById(id), esc=benchmarkEscape, table=benchmarkTable;
  const work=window.benchmarkWorkbench;
  let options=null, control={busy:false,anyRunning:false}, mediaSelection={};
  const tasks=id=>[...el(id).selectedOptions].map(o=>Number(o.value));
  const saved=id=>options?.libero.checkpoints.find(c=>c.path===el(id).value);
  function choices(id,items,empty=''){
    const select=el(id), prior=select.value;
    select.innerHTML=items.map(([value,label])=>'<option value="'+esc(value)+'">'+esc(label)+'</option>').join('') || '<option value="">'+esc(empty)+'</option>';
    if(items.some(([value])=>String(value)===prior))select.value=prior;
  }
  function trainTasks(){
    const suite=options?.suites.find(s=>s.id===el('libero-train-suite').value);
    el('libero-train-tasks').innerHTML=(suite?.tasks||[]).map(t=>'<option value="'+t.id+'"'+(t.id===0?' selected':'')+'>Task'+t.id+' · '+esc(t.name)+'</option>').join('');
    recommend(); work.updateButtons();
  }
  function dataState(){
    const selected=tasks('libero-train-tasks'), suite=el('libero-train-suite').value;
    const ready=(options?.libero.datasets||[]).filter(d=>d.suite===suite&&selected.includes(d.task)&&d.ready);
    el('libero-data-state').textContent=ready.length+'/'+selected.length+' 个任务已准备 · '+ready.reduce((n,d)=>n+d.episodes,0)+' 条示教';
    return {ready:!!selected.length&&ready.length===selected.length,selected};
  }
  function recommend(){
    const count=tasks('libero-train-tasks').length, preset=el('libero-train-preset').value;
    const checkpoint=saved('libero-train-checkpoint'), start=checkpoint?.step||0;
    if(preset!=='custom'){
      const budget=preset==='quick'?Math.min(3000,300*Math.max(count,1)):(['libero_10','libero_90'].includes(el('libero-train-suite').value)?40000:20000);
      el('libero-train-steps').value=preset==='quick'?start+budget:Math.max(start+1,budget);
    }
    el('libero-recommendation').textContent=preset==='quick'?count+' 个任务 · 新增 '+Math.min(3000,300*Math.max(count,1))+' 步，用于初步迭代':preset==='paper'?'步数仅参考论文；当前训练所选套件，Long10 / 90 尚未联合训练。':'使用手动设置的训练预算。';
  }
  function resume(){
    const c=saved('libero-train-checkpoint');
    if(c){
      el('libero-train-suite').value=c.suite; trainTasks();
      [...el('libero-train-tasks').options].forEach(o=>o.selected=c.tasks.includes(Number(o.value)));
      for(const [id,key] of [['arm','arm'],['mode','fine_tuning_mode'],['group','group_size'],['seed','seed'],['lr','learning_rate']])el('libero-train-'+id).value=c[key];
      el('libero-train-preset').value='quick';
    }
    for(const id of ['suite','tasks','arm','mode','group','seed','lr'])el('libero-train-'+id).disabled=!!c;
    recommend();work.updateButtons();
  }
  function evaluationPreset(){
    const full=el('libero-eval-preset').value==='full';
    el('bench-steps').value=full?0:32;el('libero-eval-trials').value=full?50:1;
  }
  function evaluationMode(){
    const policy=el('libero-eval-kind').value==='libero_eval';
    for(const name of ['checkpoint','preset','trials'])el('libero-eval-'+name+'-field').hidden=!policy;
    el('bench-steps').min=policy?0:1;
    el('bench-steps').max=policy?(options?.libero.horizons[el('bench-suite').value]||520):200;
    if(policy)evaluationPreset();else el('bench-steps').value=20;
    el('libero-steps-help').textContent=policy?'0 表示套件完整时长；缩短时长仅用于调试。':'只验证仿真 reset、相机与 step。';
    el('libero-eval-save-note').textContent=policy?'保存每回合录像、截图与结果':'保存每个任务的起止截图';
    el('bench-libero-start').textContent=policy?'启动推理评估':'启动环境检查';
    work.updateButtons();
  }
  function chooseEvaluationCheckpoint(){
    const c=saved('libero-eval-checkpoint');if(!c)return;
    el('bench-suite').value=c.suite;el('bench-suite').onchange();
    [...el('bench-tasks').options].forEach(o=>o.selected=c.tasks.includes(Number(o.value)));
  }
  function updateOptions(value){
    options=value;
    choices('libero-train-gpu',value.gpus.map(g=>[g.index,'GPU '+g.index+' · 可用 '+(g.free_mb/1024).toFixed(1)+' GiB']),'GPU 不可用');
    if(!el('libero-train-suite').options.length){choices('libero-train-suite',value.suites.map(s=>[s.id,'LIBERO '+s.title]));trainTasks();
      if(value.gpus.length)el('libero-train-gpu').value=String([...value.gpus].sort((a,b)=>b.free_mb-a.free_mb)[0].index);}
    choices('libero-train-checkpoint',[[value.libero.base_checkpoint,'通用 GR00T N1.6 · 新建训练'],...value.libero.checkpoints.filter(c=>c.resumable).map(c=>[c.path,c.label])]);
    const old=el('libero-eval-checkpoint').value;
    choices('libero-eval-checkpoint',value.libero.checkpoints.map(c=>[c.path,c.label]),'暂无 LIBERO 权重，请先训练');
    if(!old&&el('libero-eval-checkpoint').value)chooseEvaluationCheckpoint();
    work.updateButtons();
  }
  function buttons(state){
    control=state;const blocked=state.busy||state.anyRunning;
    const data=dataState(), available=!!options?.gpus.length;
    el('libero-prepare-data').disabled=blocked||!data.selected.length;
    el('libero-train-start').disabled=blocked||!data.ready||!available;
    el('libero-train-start').title=!data.ready?'请先准备所选任务示教':state.anyRunning?'已有 Benchmark 运行中':'';
    el('libero-train-preview').disabled=state.busy;
    const kind=el('libero-eval-kind').value, item=options?.catalog.find(c=>c.id===kind);
    el('bench-libero-start').disabled=blocked||!available||!item?.available||(kind==='libero_eval'&&!el('libero-eval-checkpoint').value);
    el('bench-libero-start').title=state.anyRunning?'已有 Benchmark 运行中':kind==='libero_eval'&&!el('libero-eval-checkpoint').value?'请先训练 LIBERO 权重':'';
  }
  function trainPayload(kind='libero_train'){
    const payload={kind,suite:el('libero-train-suite').value,tasks:tasks('libero-train-tasks'),gpu:Number(el('libero-train-gpu').value),notes:el('libero-train-notes').value};
    if(kind==='libero_train')Object.assign(payload,{checkpoint:el('libero-train-checkpoint').value,arm:el('libero-train-arm').value,fine_tuning_mode:el('libero-train-mode').value,
      steps:Number(el('libero-train-steps').value),group_size:Number(el('libero-train-group').value),save_interval:Number(el('libero-train-save').value),seed:Number(el('libero-train-seed').value),learning_rate:Number(el('libero-train-lr').value)});
    return payload;
  }
  function evalPayload(){
    const kind=el('libero-eval-kind').value, payload={kind,suite:el('bench-suite').value,tasks:tasks('bench-tasks'),gpu:Number(el('bench-gpu').value),seed:Number(el('bench-seed').value),notes:el('bench-notes').value};
    if(kind==='libero_eval')Object.assign(payload,{checkpoint:el('libero-eval-checkpoint').value,trials:Number(el('libero-eval-trials').value),max_steps:Number(el('bench-steps').value)});
    else payload.steps=Number(el('bench-steps').value);
    return payload;
  }
  function metrics(reports){
    let html='';
    for(const {data:d} of reports){
      if(d.type==='libero_dataset')html+='<h3>示教已准备</h3>'+table(['套件','任务','可用示教','数据大小'],d.datasets.map(t=>[t.suite,'Task'+t.task,t.episodes.length,(t.bytes/1024/1024).toFixed(0)+' MiB']))+'<p class="bench-subtle">可返回训练页启动训练；数据来源与校验信息保存在原始报告。</p>';
      if(d.type==='libero_train'){
        html+='<h3>LIBERO 训练</h3>'+table(['模型','套件 / 任务','已训练步数','最后 Loss'],[[d.arm,d.suite+' / '+d.tasks.join(', '),d.steps,benchmarkNumber(d.training.at(-1)?.loss)]])+'<p class="bench-subtle">Loss 是训练误差；任务表现请通过保存的权重做闭环评估。</p>';
        const points=d.training, max=Math.max(.001,...points.map(p=>p.loss)),minStep=points[0]?.step||0,maxStep=Math.max(minStep+1,points.at(-1)?.step||1);
        html+='<svg class="bench-chart" viewBox="0 0 660 180" role="img" aria-label="LIBERO 训练 Loss"><text x="8" y="20" font-size="11">'+benchmarkNumber(max)+'</text><polyline fill="none" stroke="#285fc6" stroke-width="2" points="'+points.map(p=>(45+595*(p.step-minStep)/(maxStep-minStep)).toFixed(1)+','+(150-130*p.loss/max).toFixed(1)).join(' ')+'"/><text x="540" y="175" font-size="11">'+d.steps+' steps</text></svg>';
        html+='<p><button type="button" data-libero-evaluate="'+esc(d.checkpoint)+'">用此权重评估 →</button></p>';
      }
      if(d.type==='libero_eval'){
        const debug=d.mode==='debug';
        html+='<div class="bench-note">'+(debug?'短回合调试 · 每回合最多 '+d.horizon+' 步，不计为完整时长成功率。':'完整时长评估 · '+d.horizon+' 步上限 / 回合 · 每任务 '+d.trials+' 回合；跨模型对照应使用相同任务、初态和回合数。')+'</div>';
        html+=table(['模型 / 权重步数','完成回合','有效 / 异常','成功回合','成功率'],[[d.arm+' / '+d.checkpoint_step,d.completed_episodes+' / '+d.requested_episodes,d.valid_episodes+' / '+d.invalid_episodes,d.successful_episodes,debug?'调试不统计':d.success_rate==null?'—':(d.success_rate*100).toFixed(1)+'%']]);
        html+='<h3>逐回合结果</h3>'+table(['任务','回合 / 初态','环境步数 / 决策数','结果'],d.episodes.map(e=>['Task'+e.task,e.trial+' / '+e.initial_state_index,e.steps+' / '+e.decisions,!e.valid?'异常：'+e.error:e.success?'成功':debug?'调试结束 · 未成功':'未成功']));
      }
    }
    return html;
  }
  function detail(job,reports){
    document.querySelectorAll('[data-libero-evaluate]').forEach(button=>button.onclick=async()=>{
      await work.loadOptions();el('libero-eval-checkpoint').value=button.dataset.liberoEvaluate;
      el('libero-eval-kind').value='libero_eval';chooseEvaluationCheckpoint();evaluationMode();selectPage('launch');
    });
    const evaluation=reports.find(r=>r.data.type==='libero_eval')?.data;
    if(!evaluation?.episodes.length)return;
    const pane=el('bench-pane-cameras'), episodes=evaluation.episodes;
    pane.innerHTML='<label class="bench-subtle">查看回合 <select id="libero-media-episode">'+episodes.map((e,i)=>'<option value="'+i+'">Task'+e.task+' · 回合 '+e.trial+' · '+(!e.valid?'异常':e.success?'成功':'未成功')+'</option>').join('')+'</select></label><div id="libero-media-content"></div>';
    el('libero-media-episode').value=String(Math.min(mediaSelection[job.id]||0,episodes.length-1));
    const show=()=>{
      const index=Number(el('libero-media-episode').value), row=episodes[index];mediaSelection[job.id]=index;
      const url=(kind,file)=>'/api/benchmarks/'+kind+'?run='+encodeURIComponent(job.id)+'&file='+encodeURIComponent(file);
      const frames=Object.entries(row.frames||{}).flatMap(([when,cameras])=>Object.entries(cameras).map(([camera,file])=>({when,camera,file})));
      el('libero-media-content').innerHTML=(row.video?'<video controls preload="metadata" style="width:100%;max-height:480px;margin:16px 0;background:#17251e" src="'+url('video',row.video)+'"></video>':'')+'<div class="bench-camera-grid">'+frames.map(f=>'<figure><img loading="lazy" alt="'+esc(f.camera)+' '+esc(f.when)+'" src="'+url('image',f.file)+'"><figcaption>'+(f.when==='start'?'初始':'结束')+' · '+(f.camera==='agentview'?'外部相机':'腕部相机')+'</figcaption></figure>').join('')+'</div>';
    };
    const showWithAnnotation=()=>{show();work.liberoAnnotation(job,episodes[Number(el('libero-media-episode').value)],pane);};
    el('libero-media-episode').onchange=showWithAnnotation;showWithAnnotation();
  }
  window.liberoWorkflow={options:updateOptions,buttons,metrics,detail,route:(benchmark)=>{
    el('libero-training-card').hidden=benchmark!=='libero';el('bench-training-unavailable').hidden=benchmark==='libero';
  }};
  el('libero-train-suite').onchange=trainTasks;
  el('libero-train-tasks').onchange=()=>{recommend();work.updateButtons();};
  el('libero-train-checkpoint').onchange=resume;
  el('libero-train-preset').onchange=recommend;
  el('libero-train-steps').oninput=()=>{el('libero-train-preset').value='custom';recommend();};
  el('libero-prepare-data').onclick=()=>work.run(trainPayload('libero_dataset'),'libero-training-message');
  el('libero-training-form').onsubmit=e=>{e.preventDefault();void work.run(trainPayload(),'libero-training-message');};
  el('libero-train-preview').onclick=()=>work.run(trainPayload(),'libero-training-message',true);
  el('bench-libero-form').onsubmit=e=>{e.preventDefault();void work.run(evalPayload(),'bench-libero-message');};
  el('bench-libero-preview').onclick=()=>work.run(evalPayload(),'bench-libero-message',true);
  el('libero-eval-kind').onchange=evaluationMode;
  el('libero-eval-preset').onchange=evaluationPreset;
  el('libero-eval-checkpoint').onchange=chooseEvaluationCheckpoint;
  el('bench-suite').addEventListener('change',()=>{if(el('libero-eval-kind').value==='libero_eval')el('bench-steps').max=options?.libero.horizons[el('bench-suite').value]||520;});
  if(work.options())updateOptions(work.options());
  evaluationMode();
})();
