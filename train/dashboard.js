/* Training page shares the existing workbench, without loading model libraries. */
function recommendTraining({datasets = [], gpus = [], checkpoint, fine_tuning_mode = 'action_full',
  arm = 'memory_v1', validation_enabled = true, validation_percent = 10, seed = 42}) {
  const unavailable = reason => ({supported:false, reason});
  if (!['action_full','full','custom','lora'].includes(fine_tuning_mode)) return unavailable('请选择已支持的微调方式。');
  if (!datasets.length) return unavailable('请选择训练任务。');
  if (!gpus.length) return unavailable('已选 ' + datasets.length + ' 个任务；请选择训练 GPU 后生成推荐。');
  const profile = checkpoint?.training_profile;
  if (!profile || !['base','behavior','trained'].includes(profile.kind))
    return unavailable('自定义权重尚无模型信息；请选已登记的 GR00T 权重，或关闭自动推荐后手动填写。');
  if (datasets.some(data => !data.training_profile?.total_windows))
    return unavailable('所选任务尚缺轨迹长度统计；数据准备完成后会自动生成推荐。');
  const memoryGB = Math.min(...gpus.map(gpu => gpu.total_mb / 1024));
  if (!Number.isFinite(memoryGB)) return unavailable('正在等待训练 GPU 的显存信息。');
  if (memoryGB < 39) return unavailable('当前全参数微调暂无适用于这组显存容量的推荐；可换用至少 40 GiB 的训练卡，或关闭自动推荐手动配置。');
  const windows = Math.max(1, Math.floor(4 / gpus.length)) * gpus.length;
  const percent = validation_enabled ? validation_percent : 0;
  if (!Number.isInteger(percent) || percent < 0 || percent > 50 || (validation_enabled && percent < 1))
    return unavailable('验证比例请填写 1–50 的整数。');
  const rows = datasets.map(data => {
    const stats = data.training_profile;
    const exact = percent === stats.validation_percent && seed === stats.split_seed;
    const trainWindows = percent === 0 ? stats.total_windows : exact ? stats.train_windows : stats.total_windows * (1 - percent/100);
    return {task:data.task, train_windows:Math.ceil(trainWindows),
      train_episodes:Math.round(stats.total_episodes * (1 - percent/100)), estimated:percent !== 0 && !exact};
  });
  if (rows.some(row => windows > row.train_episodes)) return unavailable('训练卡数超过可分配的训练 Episode 数，请减少训练卡数。');
  const passes = {base:10, behavior:6, trained:4}[profile.kind];
  const steps = Math.max(1000, Math.ceil(datasets.length * Math.max(...rows.map(row => row.train_windows)) / windows * passes / 1000) * 1000);
  const existingMemory = arm === 'memory_v1' && profile.memory_enabled;
  const values = {steps, learning_rate:profile.kind === 'trained' ? 5e-6 : 1e-5,
    global_windows_per_step:windows, microbatch:memoryGB >= 79 ? 4 : 2,
    memory_length:existingMemory ? profile.memory_length : 64,
    memory_conditioning:existingMemory ? profile.memory_conditioning : 'residual',
    memory_residual_readout:existingMemory ? profile.memory_residual_readout : 'delta',
    memory_cognitive_residual_projection:existingMemory ? (profile.memory_cognitive_residual_projection ?? false) : false,
    memory_cognitive_residual_scope:existingMemory ? (profile.memory_cognitive_residual_scope ?? 'eos') : 'eos',
    image_augmentation:true, seed, validation_interval:datasets.length > 1 ? 3000 : 1000,
    disable_early_stopping:true};
  if (fine_tuning_mode !== 'action_full') values.microbatch = 1;
  if (fine_tuning_mode === 'full') values.learning_rate = profile.kind === 'trained' ? 1e-6 : 2e-6;
  const notes = [
    ({base:'通用 GR00T 起点',behavior:'已在 B1K 微调的起点',trained:'已训练权重；新建优化器'})[profile.kind] +
      '：以最长任务约 ' + passes + ' 遍作为首轮预算，按任务等频采样，步数向上取整至 1000。',
    gpus.length + ' 张训练卡，全局 ' + windows + ' 窗口（最多 ' + windows*32 + ' 决策 / 步）；Microbatch ' + values.microbatch + ' 按最小单卡显存推荐。增加 GPU 不等同于增加全局 Batch。',
    rows.map(row => 'Task' + row.task + ' 约 ' + (steps/datasets.length*windows/row.train_windows).toFixed(1) + ' 遍').join('；') + '。短尾窗口和调度会使实际遍数略有差异。',
    validation_enabled ? '每 ' + values.validation_interval + ' 步完整验证，每 1000 步保存；关闭自动早停，执行所设预算。' : '已关闭验证，全部示教参与训练。',
    '这是当前训练器的起始建议，需用验证与任务成功率选权重。Baseline / Memory 使用相同训练预算便于对比。'
  ];
  if (rows.some(row => row.estimated)) notes.push('验证比例或种子已变化，训练窗口数按总长度估算；实际划分以检查配置结果为准。');
  if (existingMemory) notes.push('沿用所选记忆权重的容量和接入方式；新建 Residual Memory 请从官方 GR00T 起点开始。');
  if (fine_tuning_mode !== 'action_full') notes.push('当前微调方式可能对 Eagle 反传，Microbatch 从 1 起步；实际显存取决于解冻组件与 LoRA 目标，需检查配置。');
  return {supported:true, values, notes, tasks:rows, target_passes:passes};
}
function smoothTrainingCurve(points, weight) {
  // Bias-corrected EMA; smoothing is display-only and never mutates recorded loss.
  let total = 0, normalization = 0;
  return points.map(point => {
    total = weight * total + (1 - weight) * point[1];
    normalization = weight * normalization + (1 - weight);
    return [point[0], total / normalization, point[2]];
  });
}
if (typeof module !== 'undefined' && module.exports) module.exports = {recommendTraining, smoothTrainingCurve};
(() => {
  if (typeof window === 'undefined') return;
  const pageHTML = window.__grootTrainingPageHTML;
  delete window.__grootTrainingPageHTML;
  const el = id => document.getElementById(id);
  const esc = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = value => value == null ? '—' : Number(value).toPrecision(5);
  const timeSpan = value => {
    if (value == null || !Number.isFinite(value) || value < 0) return '—';
    const seconds = Math.floor(value), hours = Math.floor(seconds/3600), minutes = Math.floor(seconds/60)%60;
    return hours ? hours + ' 时 ' + minutes + ' 分' : minutes ? minutes + ' 分 ' + seconds%60 + ' 秒' : seconds + ' 秒';
  };
  const stateLabels = {training:'训练中',running:'运行中',submitted:'已提交',initializing:'初始化中',resuming:'正在恢复',
    completed:'已完成',complete:'已完成',failed:'异常退出',stopped:'已停止',validating:'验证中',
    queued:'排队中',cancelled:'已取消',waiting_for_four_runs:'等待前四组训练及验证',waiting_for_gpus:'等待GPU释放',
    converting:'转换中',syncing:'传输中',starting:'准备启动',not_prepared:'待准备'};
  const stateName = state => stateLabels[state] || state || '等待记录';
  const stateBadge = state => '<span class="status ' + (state === 'failed' ? 'failed' :
    ['complete','completed','cancelled','stopped'].includes(state) ? 'complete' : 'running') + '">' + esc(stateName(state)) + '</span>';
  let mounted = false, options = null, jobs = [], chosen = null, detail = null, busy = false, refreshing = false, view = 'overview', lastOptionsAt = 0, recommendation = null, jobActionBusy = false;
  let trainingDevice = 'a100';
  const jobDevices=['a100','a6000'], jobSnapshots=new Map();
  const liveStates=new Set(['training','running','submitted','initializing','starting','validating','resuming','queued','waiting_for_gpus','waiting_for_four_runs','converting','syncing']);
  const queueStates=new Set(['queued','waiting_for_gpus','waiting_for_four_runs']);
  const deviceName=d=>d==='a100'?'A100':'RTX 6000 Ada';
  let setupStep = 0;
  const setupLabels = ['实验与模型','数据任务','GPU 与验证','参数与启动'];
  let deleteTarget = null;
  let checkpointDeleteTarget = null, checkpointUndo = null;
  let logSource = 'training';
  const detailTabs = ['curve','controls','info','logs'];
  const deviceLabel = () => trainingDevice === 'a100' ? 'A100' : 'RTX 6000 Ada';
  const versionOpen = new Map(), taskOpen = new Map();
  const jobVersion = job => job?.model_version || job?.version || (job?.name || job?.run || '').match(/^(V_\d+(?:_\d+)*)(?:_|$)/)?.[1] || '未标版本';
  const jobTask = job => job?.task ?? job?.task_id ?? (job?.run || '').match(/(?:^|_)task(\d+)(?:_|$)/)?.[1] ?? '未知';
  const jobTasks = job => (Array.isArray(job?.tasks) ? job.tasks : String(jobTask(job)).split('_')).map(String);
  const taskLabel = job => jobTasks(job).length > 1 ? '联合训练 · ' + jobTasks(job).length + ' 个任务' : 'Task ' + jobTask(job);
  const jobMethod = job => job.method || (job.arm === 'baseline' ? 'baseline' : 'memory');
  const methodLabel = method => ({baseline:'Baseline',memory:'Memory',official:'Official'}[method] || method);
  const methodOrder = method => ({baseline:0,memory:1,official:2}[method] ?? 3);
  function selectView(next) {
    view = next;
    el('train-overview').hidden = next !== 'overview';
    el('train-overview-tab').setAttribute('aria-pressed',String(next==='overview'));
    el('train-config').hidden = next !== 'new';
    el('train-monitor').hidden = next !== 'detail';
    el('train-new').setAttribute('aria-pressed', String(next === 'new'));
    renderJobs();renderOverview();
  }
  function selectDetailTab(tab) {
    for (const name of detailTabs) {
      el('train-pane-' + name).hidden = name !== tab;
      const button = el('train-tab-' + name);
      button.setAttribute('aria-selected', String(name === tab));
      button.tabIndex = name === tab ? 0 : -1;
    }
    if (tab === 'logs') renderRunLog();
  }
  async function api(path, payload) {
    const device=trainingDevice;
    const response = await fetch('/api/training/' + path + (path.includes('?')?'&':'?') + 'device=' + device, payload === undefined ? {cache:'no-store'} :
      {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({...payload,device})});
    const result = await response.json();
    if(device !== trainingDevice){
      const error=Error('设备已切换');error.name='DeviceChanged';throw error;
    }
    if (!response.ok) throw Error(result.error || '训练请求失败');
    return result;
  }
  function message(text, error = false) {
    el('train-message').textContent = text;
    el('train-message').className = 'train-message' + (error ? ' train-error' : '');
  }
  function reportError(error) {
    if(error.name !== 'DeviceChanged') message(error.message,true);
  }
  function selectedGPUs() {
    return [...el('train-gpus').querySelectorAll('input:checked')].map(node => Number(node.value));
  }
  function selectedDatasets() {
    return [...el('train-dataset').selectedOptions].map(option => options?.datasets[Number(option.value)]).filter(Boolean);
  }

  function nativeMode() { return el('train-version').value === 'V_2_1'; }
  let wasNative = false;
  const nativeFields=['scheduler','warmup','accum','wd','clip','normalization','save','validation','phase','attempt','llm','top','vision','dit','projector','vlln'];
  function nativePoint() { return options?.checkpoints?.find(p=>p.path===el('train-checkpoint').value.trim()); }
  function nativeContext() { return nativePoint()?.native_context || options?.native_stage1_context; }
  function nativePreset(mode) {
    if (!nativeMode() || mode==='custom') return;
    const all=mode==='full';
    for(const k of ['llm','vision'])el('train-native-'+k).checked=all;
    for(const k of ['dit','projector','vlln'])el('train-native-'+k).checked=true;
    el('train-native-top').value='0';
  }
  function updateNativeMode() {
    const active=nativeMode(), panel=el('train-native-config');
    panel.hidden=!active;
    for (const k of nativeFields) el('train-native-'+k).disabled=!active;
    const windows=el('train-windows');
    windows.disabled=active; windows.closest('label').hidden=active;
    el('train-arm').querySelector('option[value="memory_v1"]').disabled=active;
    el('train-fine-tuning').querySelector('option[value="lora"]').disabled=active;
    for (const id of ['train-tuning-details','train-advanced','train-memory-details','train-validation-details']) {
      const section=el(id);
      if(active)section.hidden=true;
      else if(id==='train-tuning-details'||id==='train-advanced')section.hidden=false;
      section.querySelectorAll('input,select').forEach(node=>{node.disabled=active;});
    }
    el('train-auto-recommend').disabled=active;
    el('train-restore-recommendation').disabled=active;
    el('train-validation-enabled').disabled=active || !!options?.single_gpu;
    if(active) {
      el('train-validation-enabled').checked=true;
      el('train-arm').value='baseline';
      if(!wasNative) {
        el('train-auto-recommend').checked=false;
        el('train-steps').value='6000';el('train-lr').value='0.0001';el('train-microbatch').value='2';
        el('train-fine-tuning').value='action_full';nativePreset('action_full');
        const context=options?.native_stage1_context;
        if(!nativePoint()?.native_stage1 && context)el('train-checkpoint').value=context.base_checkpoint;
        renderCheckpointChoices();
      }
      el('train-validation-gpu-field').hidden=true;
      el('train-validation-percent-field').hidden=true;
      el('train-validation-gpu').disabled=true;
      el('train-validation-percent').disabled=true;
      el('train-recommendation-status').textContent='原生训练 · 手动配置';
      el('train-recommendation-summary').textContent='保留当前输入；默认值来自已核对的原生配方。';
      el('train-recommendation-details').hidden=true;
    } else if(wasNative) {
      el('train-auto-recommend').checked=true;
      el('train-validation-enabled').checked=!options?.single_gpu;
      renderCheckpointChoices();
    }
    wasNative=active;
  }
  function updateNativePreview() {
    if(!nativeMode())return;
    const p=nativePoint(), context=nativeContext();
    if(p?.native_context) {
      el('train-native-normalization').value=p.native_context.normalization;
      el('train-native-normalization').disabled=true;
    } else el('train-native-normalization').disabled=false;
    const sched=el('train-native-scheduler').value;
    if(sched==='constant')el('train-native-warmup').value='0';
    el('train-native-warmup').disabled=sched==='constant';
    if(el('train-native-llm').checked)el('train-native-top').value='0';
    el('train-native-top').disabled=el('train-native-llm').checked;
    const count=selectedGPUs().length,batch=count*Number(el('train-microbatch').value)*Number(el('train-native-accum').value);
    const steps=Number(el('train-steps').value),lr=Number(el('train-lr').value),w=Math.max(0,Math.min(.5,Number(el('train-native-warmup').value)/100||0));
    const baseStep=p?.native_context?.parent_step;
    el('train-native-context').textContent=(context?'':'缺少原生训练上下文 · ')+'Batch '+batch+' · '+count+'/4 卡'+(baseStep!=null?' · 父权重 '+baseStep+' 步 · 新优化器':' · 独立训练');
    const parts=[['llm','LLM'],['vision','视觉'],['dit','DiT'],['projector','投影'],['vlln','VLLN']].filter(([key])=>el('train-native-'+key).checked).map(([,label])=>label);
    if(Number(el('train-native-top').value)>0&&!el('train-native-llm').checked)parts.unshift('LLM 末 '+el('train-native-top').value+' 层');
    el('train-native-components-summary').textContent=parts.length?'训练 '+parts.join(' / '):'未选训练组件';
    const svg=el('train-native-lr-chart');
    const curve=Array.from({length:101},(_,i)=>{
      const t=i/100,f=sched==='constant'?1:(w&&t<w)?t/w:sched==='constant_with_warmup'?1:sched==='cosine'?(1+Math.cos(Math.PI*(t-w)/(1-w)))/2:1-(t-w)/(1-w);
      return (i?'L':'M')+(50+t*295).toFixed(2)+' '+(95-75*f).toFixed(2);
    }).join(' ');
    svg.innerHTML='<path d="M50 10 V95 H345" fill="none" stroke="currentColor" opacity=".3"/><path d="'+curve+'" fill="none" stroke="#5685a8" stroke-width="2"/>'+
      '<text x="45" y="22" text-anchor="end" font-size="10" fill="currentColor">'+esc(Number.isFinite(lr)?lr.toExponential(1):'—')+'</text><text x="45" y="97" text-anchor="end" font-size="10" fill="currentColor">0</text>'+
      '<text x="50" y="112" font-size="10" fill="currentColor">0</text><text x="345" y="112" font-size="10" text-anchor="end" fill="currentColor">'+esc(steps)+'</text>';
  }
  function nativeParameters(runId) {
    const data=selectedDatasets(), point=nativePoint(), context=nativeContext();
    if(data.length!==1||data[0].task!==40||!data[0].ready)throw Error('原生V2_1仅支持已登记Task40原180/20划分');
    if(!context||!point?.native_stage1)throw Error('请选择工作台登记的通用权重或已发布的原生Stage1父权重');
    if(selectedGPUs().length!==4)throw Error('原生V2_1需要四张A100');
    const preflight=el('train-native-phase').value==='preflight';
    const suffix='__preflight'+String(Number(el('train-native-attempt').value)).padStart(3,'0');
    const tuning={tune_llm:el('train-native-llm').checked,tune_top_llm_layers:Number(el('train-native-top').value),
      tune_visual:el('train-native-vision').checked,tune_projector:el('train-native-projector').checked,
      tune_diffusion_model:el('train-native-dit').checked,tune_vlln:el('train-native-vlln').checked};
    return {version:'V_2_1',recipe:'native_random_v2',stage:'stage1',task:40,arm:'baseline',custom_config:true,
      run_id:runId+(preflight?suffix:''),dataset:context.dataset,split:context.split,official_root:context.official_root,
      checkpoint:point.path,video_backend:context.video_backend,gpus:selectedGPUs(),seed:42,
      steps:Number(el('train-steps').value),learning_rate:Number(el('train-lr').value),microbatch:Number(el('train-microbatch').value),
      accumulation:Number(el('train-native-accum').value),scheduler:el('train-native-scheduler').value,
      warmup_ratio:Number(el('train-native-warmup').value)/100,weight_decay:Number(el('train-native-wd').value),
      max_grad_norm:Number(el('train-native-clip').value),normalization:el('train-native-normalization').value,
      checkpoint_interval:Number(el('train-native-save').value),validation_interval:Number(el('train-native-validation').value),
      native_tuning:tuning,preflight_only:preflight,...(preflight?{}:{accept_preflight:context.proof_directory+'/'+runId+suffix+'.json'}),
      enqueue:el('train-enqueue').checked,share_existing:el('train-share').checked,initiator:'user',notes:el('train-notes').value.trim()};
  }
  function drawDiagnosticCurve(){
    const panel=el('train-diagnostic-curves');
    panel.hidden=!detail?.diagnostic_series;
    if(panel.hidden)return;
    const key=el('train-diagnostic-metric').value;
    const points=(key==='validation'?(detail.validation||[]).map(r=>[r.step,r.validation_loss]):detail.diagnostic_series[key]||[])
      .filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
    const svg=el('train-diagnostic-chart');
    if(!points.length){svg.innerHTML='';el('train-diagnostic-hint').textContent='此指标未记录';return;}
    const maxX=Math.max(1,...points.map(p=>p[0])),maxY=Math.max(1e-12,...points.map(p=>p[1]));
    const x=v=>65+800*v/maxX,y=v=>205-180*v/maxY;
    svg.innerHTML='<path d="M65 20 V205 H865" stroke="currentColor" opacity=".3" fill="none"/>'+
      '<path d="'+points.map((p,i)=>(i?'L':'M')+x(p[0]).toFixed(2)+' '+y(p[1]).toFixed(2)).join(' ')+'" stroke="#5685a8" fill="none" stroke-width="2"/>'+
      '<text x="60" y="28" text-anchor="end" fill="currentColor" font-size="11">'+esc(maxY.toExponential(2))+'</text>'+
      '<text x="60" y="209" text-anchor="end" fill="currentColor" font-size="11">0</text><text x="865" y="225" text-anchor="end" fill="currentColor" font-size="11">'+esc(maxX)+'</text>';
    el('train-diagnostic-hint').textContent='记录数 '+points.length+' · 最近 step '+points.at(-1)[0]+'：'+fmt(points.at(-1)[1])+'；未记录的步不补零。';
  }

  const recommendedFields = {
    steps:'train-steps', learning_rate:'train-lr', global_windows_per_step:'train-windows', microbatch:'train-microbatch',
    memory_length:'train-memory-length', memory_conditioning:'train-memory-conditioning',
    memory_residual_readout:'train-memory-residual-readout', image_augmentation:'train-image-augmentation',
    memory_cognitive_residual_projection:'train-memory-cognitive-projection', memory_cognitive_residual_scope:'train-memory-cognitive-scope',
    seed:'train-seed', validation_interval:'train-validation-interval', disable_early_stopping:'train-early-stopping'
  };
  function updateRecommendation() {
    if(nativeMode()){updateNativePreview();updateBatch();return;}
    const automatic = el('train-auto-recommend').checked, selected = selectedGPUs();
    const result = recommendTraining({datasets:selectedDatasets(), gpus:(options?.gpus || []).filter(gpu => selected.includes(gpu.index)),
      checkpoint:options?.checkpoints.find(point => point.path === el('train-checkpoint').value.trim()),
      fine_tuning_mode:el('train-fine-tuning').value, arm:el('train-arm').value,
      validation_enabled:el('train-validation-enabled').checked, validation_percent:Number(el('train-validation-percent').value),
      seed:Number(el('train-seed').value)});
    const status = el('train-recommendation-status');
    recommendation = result;
    status.textContent = automatic ? result.supported ? '已应用推荐' : '等待推荐' : '手动配置';
    el('train-recommendation-summary').textContent = result.supported ?
      result.values.steps + ' 步 · 学习率 ' + result.values.learning_rate.toExponential() +
      ' · 全局 ' + result.values.global_windows_per_step + ' 窗口 · Microbatch ' + result.values.microbatch +
      (el('train-arm').value === 'memory_v1' ? ' · 记忆 ' + result.values.memory_length + ' / ' + result.values.memory_conditioning : '') : result.reason;
    el('train-recommendation-details').hidden = !result.supported;
    el('train-recommendation-notes').textContent = result.supported ? result.notes.join('\n') : '';
    if (automatic && !result.supported) el('train-steps').value = '';
    if (automatic && result.supported) {
      for (const [key,id] of Object.entries(recommendedFields)) {
        const input = el(id), value = result.values[key];
        if (input.type === 'checkbox') input.checked = key === 'disable_early_stopping' ? !value : value;
        else {
          if (input.tagName === 'SELECT' && ![...input.options].some(option => option.value === String(value)))
            input.add(new Option(String(value), String(value)));
          input.value = value;
        }
      }
    }
    updateBatch();
  }
  function updateBatch() {
    const windows = Number(el('train-windows').value), count = selectedGPUs().length;
    if(nativeMode()){
      el('train-batch').textContent='有效 Batch = '+count+' × '+el('train-microbatch').value+' × '+el('train-native-accum').value+' = '+count*Number(el('train-microbatch').value)*Number(el('train-native-accum').value);
      updateNativePreview();updateSummary();return;
    }
    el('train-batch').textContent = '最多 ' + windows * 32 + ' 决策 / 步 · 种子 ' + el('train-seed').value +
      (count && windows % count ? ' · 全局窗口数须能被训练卡数整除' : '');
    updateSummary();
  }
  function updateSummary() {
    const data = selectedDatasets(), gpus = selectedGPUs();
    const method = el('train-method').value.trim() || (el('train-arm').value === 'baseline' ? 'Baseline' : 'Memory');
    el('train-submit-summary').innerHTML = '<strong>' + esc(el('train-version').value || '未填版本') + ' · ' + esc(method) +
      ' · ' + (el('train-steps').value ? esc(el('train-steps').value) + ' 步' : '待设置训练步数') + '</strong>' +
      esc(data.length ? data.length + ' 个任务 · ' + data.slice(0,3).map(d => 'Task' + d.task).join(' / ') + (data.length>3?' 等':'') : '未选任务') + ' · ' +
      esc(gpus.length ? 'GPU ' + gpus.join(', ') : '未选训练 GPU');
    el('train-submit').textContent = el('train-enqueue').checked ? '加入训练队列' : '启动训练';
    const identity = ['test-id','experiment-id','run-number'].map(id => el('train-'+id).value);
    el('train-step-summary-0').textContent = identity.every(Boolean) ?
      el('train-version').value + ' · test' + identity[0] + ' / exp' + identity[1] + ' / run' + identity[2] : '待填写实验身份';
    el('train-step-summary-1').textContent = data.length ? '已选 ' + data.length + ' 个 · ' + data.filter(d=>d.ready).length + ' 个就绪' : '待选择任务';
    el('train-step-summary-2').textContent = gpus.length ? '训练 GPU ' + gpus.join('、') + (el('train-validation-enabled').checked ? ' · 启用验证' : ' · 关闭验证') : '待选择训练 GPU';
    el('train-step-summary-3').textContent = el('train-steps').value ? el('train-steps').value + ' 步 · ' + (el('train-enqueue').checked ? '排队启动' : '立即启动') : '待设置训练预算';
    const batch = nativeMode() ? gpus.length * Number(el('train-microbatch').value) * Number(el('train-native-accum').value) : Number(el('train-windows').value) * 32;
    const rows = [
      ['实验身份',el('train-step-summary-0').textContent],
      ['方法与微调',method + ' · ' + (el('train-fine-tuning').selectedOptions?.[0]?.textContent || el('train-fine-tuning').value)],
      ['初始权重',el('train-checkpoint').value.split('/').filter(Boolean).at(-1)||'尚未选择',el('train-checkpoint').value],
      ['训练数据',el('train-step-summary-1').textContent],
      ['运行资源',deviceLabel() + ' · ' + (gpus.length ? '训练 GPU '+gpus.join('、') : '未选训练 GPU') +
        (nativeMode()?' · 原生内部验证20条':el('train-validation-enabled').checked ? ' · 验证 GPU '+el('train-validation-gpu').value+' / '+el('train-validation-percent').value+'%' : ' · 不启用验证')],
      [nativeMode()?'有效 Batch':'决策 / 步上限',gpus.length&&batch>0?String(batch):'待配置'],
      ['预算与启动',el('train-step-summary-3').textContent]
    ];
    el('train-review').innerHTML = rows.map(([label,value,full])=>'<div><dt>'+esc(label)+'</dt><dd'+(full?' title="'+esc(full)+'"':'')+'>'+esc(value)+'</dd></div>').join('');
    el('train-memory-details').hidden = nativeMode() || el('train-arm').value !== 'memory_v1';
    el('train-validation-details').hidden = nativeMode() || !el('train-validation-enabled').checked;
    el('train-validation-interval').disabled = nativeMode() || !el('train-validation-enabled').checked;
    el('train-lora-fields').querySelectorAll('input').forEach(input => {
      input.disabled = el('train-fine-tuning').value !== 'lora';
    });
  }
  function showSetupStep(step, focus = false) {
    setupStep = Math.max(0, Math.min(3, step));
    el('train-form').querySelectorAll('[data-train-setup-panel]').forEach(panel => {
      panel.hidden = Number(panel.dataset.trainSetupPanel) !== setupStep;
    });
    el('train-form').querySelectorAll('[data-train-setup-step]').forEach(button => {
      if (Number(button.dataset.trainSetupStep) === setupStep) button.setAttribute('aria-current','step');
      else button.removeAttribute('aria-current');
    });
    el('train-previous').hidden = setupStep === 0;
    el('train-next').hidden = setupStep === 3;
    el('train-next').textContent = '下一步：' + (setupLabels[setupStep+1] || '');
    el('train-check').hidden = el('train-submit').hidden = setupStep !== 3;
    if (focus) el('train-setup-'+setupStep).querySelector('.train-step-intro').focus();
  }
  function validateSetup(step = null) {
    const panels = [...el('train-form').querySelectorAll('[data-train-setup-panel]')];
    for (const panel of panels) {
      const index = Number(panel.dataset.trainSetupPanel);
      if (step != null && index !== step) continue;
      for (const input of panel.querySelectorAll('input,select,textarea')) {
        if (!input.willValidate || input.validity.valid) continue;
        showSetupStep(index);
        for (let parent = input.parentElement; parent && parent !== panel; parent = parent.parentElement)
          if (parent.tagName === 'DETAILS') parent.open = true;
        input.reportValidity();
        return false;
      }
      if (index === 1 && (!selectedDatasets().length || selectedDatasets().some(d=>!d.ready))) {
        showSetupStep(1);
        message(selectedDatasets().length ? '所选任务还有缺失数据，请先准备完成。' : '请至少选择一个训练任务。', true);
        el(selectedDatasets().length ? 'train-prepare-data' : 'train-task-search').focus();
        return false;
      }
      if (index === 2 && !selectedGPUs().length) {
        showSetupStep(2); message('请至少选择一张训练 GPU。', true);
        el('train-gpus').querySelector('input')?.focus();
        return false;
      }
    }
    return true;
  }
  function renderTaskChoices() {
    const list = el('train-task-list'), scroll = list.scrollTop;
    const search = el('train-task-search').value.trim().toLowerCase(), readyOnly = el('train-task-ready').checked;
    const selected = new Set(selectedDatasets().map(d => d.task));
    const visible = (options?.datasets || []).map((data,index) => ({data,index})).filter(({data}) =>
      (!readyOnly || data.ready) && (!search || ('task' + data.task + ' ' + data.task + ' ' + data.task_name).toLowerCase().includes(search.replace(/^task\s+/, 'task'))));
    list.innerHTML = visible.map(({data,index}) => '<label class="train-task-choice"><input type="checkbox" data-task-index="' + index + '"' +
      (selected.has(data.task) ? ' checked' : '') + '><span><strong>Task ' + esc(data.task) + '</strong><span class="train-task-name" title="' +
      esc(data.task_name || '') + '">' + esc((data.task_name || '').replaceAll('_',' ')) + '</span></span><small class="train-data-badge' +
      (data.ready ? ' ready' : '') + '">' + esc(data.ready ? deviceLabel()+' 就绪' : data.preparation?.status ? stateName(data.preparation.status) : data.can_prepare ? '待准备' : '数据未齐') + '</small></label>').join('') || '<div class="empty">没有符合条件的任务</div>';
    list.querySelectorAll('[data-task-index]').forEach(input => input.onchange = () => {
      el('train-dataset').options[Number(input.dataset.taskIndex)].selected = input.checked;
      datasetHint();
    });
    list.scrollTop = scroll;
  }
  function selectTasks(predicate) {
    for (const option of el('train-dataset').options) option.selected = predicate(options.datasets[Number(option.value)]);
    datasetHint();
    invalidatePlan();
  }
  function invalidatePlan() {
    el('train-plan').hidden = true;
    el('train-plan-details').hidden = true;
    el('train-preview-result').hidden = true;
    if (!busy) message('');
  }
  function showDetailLoading() {
    const job = currentJob();
    detail = null;
    logSource = 'training';
    el('train-log').textContent = '';
    el('train-log-feedback').textContent = '';
    el('train-pane-info').querySelectorAll('details[open]').forEach(panel => { panel.open = false; });
    el('train-detail-title').textContent = job?.name || chosen?.run || '训练详情';
    el('train-detail-subtitle').textContent = '';
    el('train-detail-tasks').hidden = true;
    el('train-detail-tasks').open = false;
    el('train-detail-gpu').textContent = '正在读取 '+deviceLabel()+' 记录…';
    el('train-detail-status').textContent = '';
    el('train-detail-loading').hidden = false;
    el('train-monitor-body').hidden = true;
  }
  function syncStatus(state) {
    if (!state) return;
    el('train-autosync').checked = state.enabled;
    el('train-sync-state').textContent = (state.enabled ? '自动同步已开启' : '自动同步已关闭') + ' · ' + state.state +
      (state.last_synced ? '\n最近同步：' + state.last_synced : '') +
      (state.digest ? '\n源码：' + state.digest.slice(0,12) + ' · ' + state.file_count + ' 个文件' : '') +
      (state.error ? '\n错误：' + state.error : '');
    el('train-sync-state').className = 'train-info' + (state.error ? ' train-error' : '');
  }
  function renderCheckpointChoices() {
    const points = (options?.checkpoints || []).filter(p=>!nativeMode()||p.native_stage1), selected = el('train-checkpoint').value;
    const official = point => ['GR00T-N1.6-3B','GR00T-N1.6-BEHAVIOR1k'].includes(point.path.split('/').at(-1));
    const group = (label, values) => values.length ? '<optgroup label="' + label + '">' + values.map(point =>
      '<option value="' + esc(point.path) + '">' + esc(point.label) + '</option>').join('') + '</optgroup>' : '';
    el('train-checkpoint-choice').innerHTML = group('官方初始权重', points.filter(official)) + group('本项目已训练权重', points.filter(p=>!official(p))) +
      (nativeMode()?'':'<option value="custom">自定义设备权重路径…</option>');
    el('train-checkpoint-choice').value = points.some(point=>point.path===selected) ? selected : nativeMode()?(options?.native_stage1_context?.base_checkpoint||''):'custom';
    updateCheckpointChoice();
  }
  function updateCheckpointChoice() {
    const choice = el('train-checkpoint-choice').value, custom = choice === 'custom';
    if (!custom) el('train-checkpoint').value = choice;
    el('train-checkpoint-custom').hidden = !custom;
    const label = choice.endsWith('/GR00T-N1.6-3B') ? '通用预训练起点' :
      choice.endsWith('/GR00T-N1.6-BEHAVIOR1k') ? '已在 BEHAVIOR1k 上微调的起点' : '从所选权重初始化';
    el('train-checkpoint-note').textContent = label + ' · 新建优化器';
    el('train-checkpoint-note').title = el('train-checkpoint').value;
  }
  function datasetHint() {
    const selected=selectedDatasets(),enabled=el('train-validation-enabled').checked;
    const validation=enabled?2*Number(el('train-validation-percent').value):0;
    el('train-split').textContent=selected.length ? selected.length+' 个任务 · 共 '+selected.length*(200-validation)+' 集训练 / '+selected.length*validation+' 集验证'+
      (enabled?' · Episode 隔离':' · 无验证') : '请选择训练任务';
    el('train-split').title=selected.map(data=>data.dataset || '').join('\n');
    el('train-validation-gpu-field').hidden=el('train-validation-percent-field').hidden=nativeMode()||!enabled;
    el('train-validation-gpu').disabled=el('train-validation-percent').disabled=nativeMode()||!enabled;
    if(nativeMode())el('train-split').textContent='Task40 · 180 训练 / 20 验证';
    el('train-prepare-data').disabled=!selected.some(data=>!data.ready && data.can_prepare && !['queued','starting','converting','syncing'].includes(data.preparation?.status));
    el('train-prepare-data').hidden=!selected.some(data=>!data.ready);
    el('train-task-count').textContent = '已选 ' + selected.length + ' 个';
    el('train-task-selection-summary').textContent = '已选 ' + selected.length + ' 个任务';
    el('train-task-selection').innerHTML = selected.map(data => '<button type="button" data-remove-task="' + esc(data.task) + '" aria-label="移除 Task ' + esc(data.task) + '">Task ' + esc(data.task) + ' ×</button>').join('') || '<span class="train-hint">尚未选择任务</span>';
    el('train-task-selection').querySelectorAll('[data-remove-task]').forEach(button => button.onclick = () =>
      selectTasks(data => selected.some(d => d.task === data.task) && String(data.task) !== button.dataset.removeTask));
    const selectedIds = new Set(selected.map(data => data.task));
    el('train-task-list').querySelectorAll('[data-task-index]').forEach(input => { input.checked = selectedIds.has(options.datasets[Number(input.dataset.taskIndex)].task); });
    const queue = options?.dataset_queue;
    el('train-data-status').textContent = selected.length + ' 个任务 · 已就绪 ' + selected.filter(data=>data.ready).length +
      (selected.some(data=>!data.ready)?' · 待准备 ' + selected.filter(data=>!data.ready).length:'') +
      (queue?.status === 'preparing' ? ' · 全部数据准备：' + queue.ready_tasks.length + '/50，当前 Task' + queue.current_task + '，视频 ' + (queue.current_videos || 0) + '/600' : '') +
      selected.filter(data=>data.preparation?.error).map(data=>' · Task'+data.task+' '+data.preparation.error).join('');
    updateRecommendation();
  }
  async function loadOptions() {
    const selected = selectedGPUs(), hadOptions = !!options, previous = new Set(selectedDatasets().map(d=>d.task)), validation = el('train-validation-gpu').value;
    options = await api('options'); lastOptionsAt = Date.now();
    if (options.model_versions) {
      const previousVersion = el('train-version').value;
      el('train-version').innerHTML = Object.entries(options.model_versions)
        .filter(([,entry]) => entry.backend === 'v1' || (entry.backend==='v2_1' && options.native_stage1_context))
        .map(([version,entry]) => '<option value="' + esc(version) + '">' + esc(version + ' · ' + entry.description) + '</option>').join('');
      if ([...el('train-version').options].some(option => option.value === previousVersion)) el('train-version').value = previousVersion;
    }
    el('train-device-label').textContent=deviceLabel();
    updateNativeMode();
    el('train-validation-enabled').disabled=nativeMode()||!!options.single_gpu;
    if(options.single_gpu) el('train-validation-enabled').checked=false;
    el('train-dataset').innerHTML = options.datasets.map((data,i) => '<option value="' + i + '">Task ' + data.task + ' · ' + esc(data.task_name) + ' · ' + (data.ready ? deviceLabel()+' 已就绪' : ['queued','starting','converting','syncing'].includes(data.preparation?.status) ? stateName(data.preparation.status) : data.can_prepare ? '待准备到当前设备' : '本机数据未齐') + '</option>').join('');
    for (const option of el('train-dataset').options) option.selected=hadOptions?previous.has(options.datasets[Number(option.value)].task):Number(option.value)===0;
    el('train-checkpoints').innerHTML = options.checkpoints.map(point => '<option value="' + esc(point.path) + '">' + esc(point.label) + '</option>').join('');
    if (!el('train-checkpoint').value) el('train-checkpoint').value = options.checkpoints[0]?.path || '';
    renderCheckpointChoices();
    const gpuInputs = [...el('train-gpus').querySelectorAll('input')];
    if (gpuInputs.map(input=>Number(input.value)).join(',') !== options.gpus.map(gpu=>gpu.index).join(',')) {
      el('train-gpus').innerHTML = options.gpus.map(gpu => '<label class="train-gpu"><input type="checkbox" value="' + gpu.index + '"' +
        (selected.includes(gpu.index) ? ' checked' : '') + '><span title="' + esc(gpu.name) + '"><strong>GPU ' + gpu.index + '</strong>' +
        '<small data-gpu-reading="' + gpu.index + '"></small></span></label>').join('');
    }
    for (const gpu of options.gpus) el('train-gpus').querySelector('[data-gpu-reading="'+gpu.index+'"]').textContent =
      (gpu.used_mb/1024).toFixed(1) + ' / ' + (gpu.total_mb/1024).toFixed(0) + ' GiB · ' + gpu.utilization + '%';
    if(options.single_gpu && !hadOptions) el('train-gpus').querySelector('input').checked=true;
    el('train-validation-gpu').innerHTML = options.gpus.map(gpu => '<option value="' + gpu.index + '">GPU ' + gpu.index + '</option>').join('');
    if (validation !== '') el('train-validation-gpu').value = validation;
    renderTaskChoices(); datasetHint(); updateBatch(); syncStatus(options.sync);
  }
  function parameters() {
    const version = el('train-version').value.trim();
    const test = Number(el('train-test-id').value), experiment = Number(el('train-experiment-id').value);
    const run = Number(el('train-run-number').value);
    if (![test, experiment, run].every(Number.isSafeInteger) || test < 1 || test > 99 || experiment < 1 || experiment > 99 || run < 1)
      throw Error('请分别填写测试编号、实验编号和独立训练次数；暂停续训请使用原作业');
    const prefix = options?.model_versions?.[version]?.prefix || version.replace(/^V_/, 'v');
    const runId = prefix + '__test' + String(test).padStart(2,'0') + '__exp' + String(experiment).padStart(2,'0') + '__run' + String(run).padStart(3,'0');
    if(nativeMode())return nativeParameters(runId);
    const data=selectedDatasets();
    if (!data.length) throw Error('请至少选择一个任务');
    if (data.some(d=>!d.ready)) throw Error('请先准备全部所选任务数据，完成后刷新再启动训练');
    if (!selectedGPUs().length) throw Error('请至少选择一张训练 GPU');
    if (el('train-auto-recommend').checked && !recommendation?.supported)
      throw Error(recommendation?.reason || '请等待推荐参数生成');
    const enabled=el('train-validation-enabled').checked;
    const tuning = {};
    for (const name of ['llm','vision','vision_projector','action_dit','action_projector','action_vlln','memory']) {
      const value = el('train-freeze-' + name.replaceAll('_','-'))?.value || 'auto';
      if (value !== 'auto') tuning['freeze_' + name] = value === 'true';
    }
    return {version, run_id:runId, initiator:'user', notes:el('train-notes').value.trim(), tasks:data.map(d=>d.task),
      checkpoint:el('train-checkpoint').value.trim(), gpus:selectedGPUs(), validation_enabled:enabled,
      ...(enabled?{validation_gpu:Number(el('train-validation-gpu').value),validation_percent:Number(el('train-validation-percent').value)}:{}),
      enqueue:el('train-enqueue').checked,
      arm:el('train-arm').value, steps:Number(el('train-steps').value), global_windows_per_step:Number(el('train-windows').value),
      microbatch:Number(el('train-microbatch').value), learning_rate:Number(el('train-lr').value), seed:Number(el('train-seed').value),
      memory_length:Number(el('train-memory-length').value), memory_conditioning:el('train-memory-conditioning').value,
      memory_residual_readout:el('train-memory-residual-readout').value,
      memory_cognitive_residual_projection:el('train-memory-cognitive-projection').checked,
      memory_cognitive_residual_scope:el('train-memory-cognitive-scope').value,
      optimizer_eps:Number(el('train-optimizer-eps').value),
      image_augmentation:el('train-image-augmentation').checked,
      fine_tuning_mode:el('train-fine-tuning').value,
      ...tuning, lora_targets:el('train-lora-targets')?.value || 'llm,action_dit',
      lora_rank:Number(el('train-lora-rank')?.value || 16), lora_alpha:Number(el('train-lora-alpha')?.value || 32),
      lora_dropout:Number(el('train-lora-dropout')?.value || 0),
      llm_last_n_layers:Number(el('train-llm-last-n-layers')?.value || 0),
      gradient_checkpointing:el('train-gradient-checkpointing')?.checked || false,
      weight_decay:Number(el('train-weight-decay')?.value ?? .01), max_grad_norm:Number(el('train-max-grad-norm')?.value ?? 1),
      validation_interval:Number(el('train-validation-interval').value),
      disable_early_stopping:!el('train-early-stopping').checked,
      share_existing:el('train-share').checked,
      method:el('train-method').value.trim() || (el('train-arm').value==='baseline'?'baseline':'memory'),
      method_description:el('train-method-description').value.trim()};
  }
  async function submit(start) {
    if (busy) return;
    if (setupStep !== 3) {
      if (validateSetup(setupStep)) showSetupStep(setupStep+1, true);
      return;
    }
    if (!validateSetup()) return;
    busy = true; el('train-submit').disabled = el('train-check').disabled = true;
    try {
      const params = parameters();
      message(start ? '检查 '+deviceLabel()+' 参数、同步源码并提交训练…' : '检查数据划分、GPU、权重和训练参数…');
      const result = await api(start ? 'start' : 'preview', params);
      el('train-plan').hidden = false; el('train-plan').textContent = JSON.stringify(result,null,2);
      el('train-plan-details').hidden = false;
      if (start) {
        chosen = {run:result.run,arm:result.arm};
        message(result.status==='queued'?'已加入训练队列；GPU空闲后启动，源码快照已固定。':'训练已提交，源码快照已固定。');
        showDetailLoading();
        selectView('detail');
        await refresh();
      } else {
        el('train-preview-result').hidden = false;
        el('train-preview-result').innerHTML = '<strong>配置检查通过 · 尚未启动</strong><br>运行名称：' + esc(result.run_name) +
          '<br>全局最多 ' + esc(result.global_decisions) + ' 决策 / 优化步。启动后固定源码快照；完整参数可在下方展开查看。';
        el('train-preview-result').scrollIntoView({block:'nearest',behavior:'smooth'});
        message('配置检查通过，可确认下方结果后启动。');
      }
    } catch (error) { reportError(error); }
    finally { busy = false; el('train-submit').disabled = el('train-check').disabled = false; }
  }
  function renderOverview(){
    const all=jobDevices.flatMap(device=>(jobSnapshots.get(device)?.jobs||[]).map(job=>({device,job})));
    const live=all.filter(({job})=>liveStates.has(job.status));
    const queued=live.filter(({job})=>queueStates.has(job.status));
    const count=(label,n)=>'<div><small>'+label+'</small><strong>'+n+'</strong></div>';
    el('train-active-count').textContent=String(live.length);
    el('train-overview-counts').innerHTML=count('运行中',live.length-queued.length)+count('排队',queued.length)+count('已完成',all.filter(({job})=>['complete','completed'].includes(job.status)).length)+count('异常 / 停止',all.filter(({job})=>['failed','stopped','paused'].includes(job.status)).length);
    el('train-overview-status').textContent=jobDevices.map(d=>deviceName(d)+(jobSnapshots.get(d)?.error?' · 不可查':jobSnapshots.has(d)?' · 已读取':' · 读取中')).join(' / ');
    const list=el('train-overview-list');
    const cards=live.sort((a,b)=>Number(queueStates.has(a.job.status))-Number(queueStates.has(b.job.status)));
    list.innerHTML=cards.length?cards.map(({device,job},i)=>{
      const pct=Math.max(0,Math.min(100,Number(job.step||0)/(Number(job.max_steps)||1)*100));
      return '<button type="button" class="workbench-live-row" data-live-train="'+i+'"><div class="workbench-live-head"><strong>'+esc(jobVersion(job)+' · '+taskLabel(job)+' · '+methodLabel(jobMethod(job)))+'</strong>'+stateBadge(job.status)+'</div><div class="subtle">'+esc(job.run)+' · '+esc(deviceName(device))+' · GPU '+esc(job.gpus??'未知')+' · '+esc(({agent:'Agent',user:'用户'})[job.initiator]||'来源未记录')+'</div><div class="progress-track"><span style="width:'+pct+'%"></span></div><div class="workbench-live-foot"><span>'+esc(job.step??'—')+' / '+esc(job.max_steps??'—')+' 步</span><span>Loss '+fmt(job.last?.loss)+'</span><span>详情 →</span></div></button>';
    }).join(''):'<p class="empty">'+(jobDevices.some(d=>jobSnapshots.get(d)?.error)?'部分设备不可查':'暂无运行任务')+'</p>';
    list.querySelectorAll('[data-live-train]').forEach(button=>button.onclick=async()=>{
      if(busy||jobActionBusy)return;
      const {device,job}=cards[Number(button.dataset.liveTrain)];
      if(device!==trainingDevice){el('train-device').value=device;await el('train-device').onchange();}
      if(device!==trainingDevice)return;
      const found=jobs.find(j=>j.run===job.run&&j.arm===job.arm);
      if(!found){message('记录已变化，请刷新。',true);return;}
      chosen={run:found.run,arm:found.arm};showDetailLoading();selectView('detail');selectDetailTab('curve');
      await loadDetail().catch(reportError);
    });
  }
  function renderJobs() {
    const scroll = el('train-jobs').scrollTop;
    const search = (el('train-job-search').value || '').trim().toLowerCase().replace(/task\s+/g,'task');
    const filter = el('train-job-filter').value || 'all', owner=el('train-job-owner').value||'all', category=el('train-job-category').value||'research', filtering = !!search || filter !== 'all'||owner!=='all'||category==='engineering';
    const stateGroups = {active:['training','running','submitted','initializing','starting','validating','resuming'],
      queued:['queued','waiting_for_gpus','waiting_for_four_runs'],complete:['complete','completed'],failed:['failed','stopped','paused']};
    const visible = jobs.map((job,i)=>({job,i})).filter(({job}) => (category==='all'||(category==='engineering'?job.category==='engineering_preflight':job.category!=='engineering_preflight'||chosen?.run===job.run))&&(owner==='all'||owner===(job.initiator||'unknown'))&&(!stateGroups[filter] || stateGroups[filter].includes(job.status)) &&
      (!search || [job.name,job.run,jobVersion(job),...jobTasks(job).map(task=>'task'+task),taskLabel(job),jobMethod(job),job.training_id,job.notes].join(' ').toLowerCase().includes(search)));
    el('train-count').textContent = visible.length + ' / ' + jobs.length;
    const versions = new Map();
    visible.forEach(({job,i}) => {
      const version = jobVersion(job);
      if (!versions.has(version)) versions.set(version, []);
      versions.get(version).push({job,i});
    });
    const groups = [...versions].sort(([a],[b]) => b.localeCompare(a,'zh-CN',{numeric:true}));
    const currentVersion = view==='detail'&&chosen ? jobVersion(currentJob()) : null;
    el('train-jobs').innerHTML = groups.length ? groups.map(([version,entries]) => {
      const open = filtering || (versionOpen.get(version) ?? (version === currentVersion));
      const tasks = new Map();
      entries.forEach(entry => {
        const task = String(jobTask(entry.job));
        if (!tasks.has(task)) tasks.set(task, []);
        tasks.get(task).push(entry);
      });
      return '<details class="version-group" data-train-version="' + esc(version) + '"' + (open ? ' open' : '') + '>' +
        '<summary>' + esc(version) + '<small>' + entries.length + ' 条</small></summary><div class="version-runs">' +
        [...tasks].sort(([a],[b])=>a.localeCompare(b,'zh-CN',{numeric:true})).map(([task,records]) => {
      const key = version + ':' + task;
      const expanded = filtering || (taskOpen.get(key) ?? (version === currentVersion && task === String(jobTask(currentJob()))));
      records.sort(({job:a},{job:b}) => methodOrder(jobMethod(a))-methodOrder(jobMethod(b)) ||
        jobMethod(a).localeCompare(jobMethod(b)) || (b.train_number || 1)-(a.train_number || 1) || b.run.localeCompare(a.run,undefined,{numeric:true}));
      return '<details class="train-task-group" data-train-task="' + esc(key) + '"' + (expanded ? ' open' : '') + '>' +
        '<summary><span class="train-task-label">' + esc(taskLabel(records[0].job)) + '</span><small>' + records.length + ' 条</small></summary><div class="train-task-runs">' +
        records.map(({job,i}) => {
      const selected = view === 'detail' && chosen?.run === job.run && chosen?.arm === job.arm;
      const progress = Math.max(0, Math.min(100, Number(job.step || 0) / (job.max_steps || 1) * 100));
      return '<button type="button" class="train-job run-card' + (selected ? ' selected' : '') +
        '" aria-pressed="' + selected + '" data-job="' + i + '" title="' + esc(job.name || job.run) + '">' +
        '<div class="run-card-top"><strong>' + esc(methodLabel(jobMethod(job))) + ' · 第 ' + esc(job.train_number || 1) + ' 次训练</strong></div>' +
        '<div class="subtle">' + stateBadge(job.status) + ' · GPU ' + esc(job.gpus) + ' · '+esc(({user:'用户',agent:'Agent'})[job.initiator]||'来源未记录')+'</div>' +
        (job.notes ? '<div class="train-job-note" title="' + esc(job.notes) + '">' + esc(job.notes) + '</div>' : '') +
        '<div class="progress-track"><span style="width:' + progress + '%"></span></div>' +
        '<div class="train-job-stats"><span>' + esc(job.step) + ' / ' + esc(job.max_steps ?? '?') + ' 步</span><span>Loss ' + fmt(job.last?.loss) + '</span></div></button>';
        }).join('') + '</div></details>';
        }).join('') + '</div></details>';
    }).join('') : '<div class="empty">' + (filtering ? '没有符合条件的记录' : '暂无训练记录') + '</div>';
    el('train-jobs').querySelectorAll('[data-train-version]').forEach(group => {
      group.ontoggle = () => { if (group.isConnected && !filtering) versionOpen.set(group.dataset.trainVersion, group.open); };
    });
    el('train-jobs').querySelectorAll('[data-train-task]').forEach(group => {
      group.ontoggle = () => { if (group.isConnected && !filtering) taskOpen.set(group.dataset.trainTask, group.open); };
    });
    el('train-jobs').scrollTop = scroll;
    el('train-jobs').querySelectorAll('[data-job]').forEach(button => button.onclick = async () => {
      const job = jobs[Number(button.dataset.job)]; chosen = {run:job.run,arm:job.arm};
      el('train-stop-message').textContent = '';
      el('train-cancel-message').textContent = '';
      showDetailLoading();
      selectView('detail');
      selectDetailTab('curve');
      await loadDetail().catch(error => message(error.message,true));
    });
  }
  function currentJob() { return jobs.find(job => job.run === chosen?.run && job.arm === chosen?.arm); }
  function updateJobActions() {
    const job = currentJob();
    const resume = job?.resume;
    el('train-resume-open').hidden = !['complete','completed','stopped','paused','failed'].includes(job?.status);
    el('train-resume-open').disabled = jobActionBusy || !resume?.available;
    el('train-resume-open').title = resume?.reason || '恢复最新完整断点，继续当前训练记录';
    el('train-warm-start').hidden=jobVersion(job)!=='V_2_1'||detail?.config?.recipe!=='native_random_v2'||detail?.config?.preflight_only===true;
    el('train-warm-start').disabled=jobActionBusy||!el('train-saved-checkpoint').value||!['complete','completed','stopped','paused','failed'].includes(job?.status);
    el('train-copy-config').hidden=detail?.config?.recipe!=='native_random_v2';
    el('train-formal-from-preflight').hidden=!(detail?.config?.recipe==='native_random_v2'&&detail.config.custom_config&&detail.config.preflight_only&&detail.preflight?.passed&&job?.status==='completed');
    el('train-download-report').disabled=!detail?.analysis_report?.text;
    el('train-download-evidence').disabled=!detail?.analysis_report?.text;
    el('train-download-report').title=detail?.analysis_report?'已归档 step '+detail.analysis_report.step+' · '+detail.analysis_report.status:'尚未生成报告；终态归档服务完成后自动可用';
    const resumeKey = job?.run + '/' + job?.arm + '/' + resume?.step;
    if (el('train-resume-panel').dataset.key !== resumeKey) {
      el('train-resume-panel').dataset.key = resumeKey;
      el('train-resume-panel').hidden = true;
      el('train-resume-steps').value = resume?.default_steps || '';
      el('train-resume-steps').min = resume?.min_steps || 1000;
      el('train-resume-steps').step = resume?.step_interval || 1000;
      el('train-resume-checkpoint-interval').value = resume?.default_checkpoint_interval || resume?.checkpoint_interval || 1000;
      el('train-resume-share').checked = false;
      el('train-resume-message').textContent = '';
    }
    if (!resume?.available) el('train-resume-panel').hidden = true;
    el('train-resume-checkpoint-field').hidden = !resume?.checkpoint_interval_editable;
    el('train-resume-checkpoint-note').hidden = !resume?.checkpoint_interval_editable;
    el('train-resume-checkpoint-interval').disabled = jobActionBusy || !resume?.checkpoint_interval_editable;
    el('train-resume-submit').disabled = jobActionBusy || !resume?.available;
    el('train-resume-summary').textContent = resume?.available ?
      '第 ' + resume.step + ' 步断点 · GPU ' + resume.gpus.join(',') + ' · Seed ' + resume.seed +
      ' · ' + (resume.validation_enabled ? '保留独立验证' : '保留关闭验证') + '\n' + resume.checkpoint : '';
    updateResumeBudget();
    el('train-stop-next').hidden = !job?.can_stop;
    el('train-stop-next').disabled = jobActionBusy;
    el('train-cancel-top').hidden = !job?.can_cancel;
    el('train-cancel-top').disabled = jobActionBusy;
    el('train-delete').disabled = jobActionBusy || !job?.can_delete;
    el('train-checkpoint-delete').disabled = jobActionBusy || !el('train-saved-checkpoint').value;
    el('train-delete').title = job?.can_delete ? '删除训练记录，可选择是否保留输出目录' : job?.delete_block_reason || '请先结束训练和验证';
    el('train-action-hint').textContent = job?.can_stop ?
      (job.stop_after_step ? '将在第 ' + job.stop_after_step + ' 步保存后停止' : '停止时保留权重和日志') :
      job?.can_cancel ? '当前等待 GPU' :
      resume?.available ? '可从 '+resume.step+' 步断点续训' :
      job?.can_delete ? (resume?.reason || '删除记录仅从列表移除，保留权重与日志。') : job?.delete_block_reason || '当前任务暂不支持停止，请查看日志与配置。';
    if (job?.can_stop && Number(el('train-stop-step').value) <= job.step) {
      const interval = job.checkpoint_interval ?? 1000;
      el('train-stop-step').value = Math.min((Math.floor(job.step / interval) + 1) * interval, job.max_steps);
    }
  }
  function updateResumeBudget() {
    const resume = currentJob()?.resume, total = Number(el('train-resume-steps').value);
    el('train-resume-budget').textContent = resume?.available ?
      '总步数包含已完成的 ' + resume.step + ' 步；还将训练 ' + Math.max(0,total-resume.step) +
      ' 步。总步数至少 ' + resume.min_steps + '，按 ' + (resume.step_interval || 1000) + ' 步递增。' : '';
  }
  async function resumeTraining(event) {
    event.preventDefault();
    const job = currentJob();
    if (!job?.resume?.available || jobActionBusy || !el('train-resume-panel').reportValidity()) return;
    const requested = {...chosen,steps:Number(el('train-resume-steps').value),share_existing:el('train-resume-share').checked};
    if (job.resume.checkpoint_interval_editable)
      requested.checkpoint_interval = Number(el('train-resume-checkpoint-interval').value);
    if (!window.confirm('从第 ' + job.resume.step + ' 步续跑至总计 ' + requested.steps + ' 步？\n' +
      job.name + '\n每 ' + (requested.checkpoint_interval || job.checkpoint_interval) + ' 步保存完整检查点。\n恢复优化器和采样状态。')) return;
    jobActionBusy = true; updateJobActions();
    el('train-resume-message').textContent = '正在校验断点和 GPU，完成后启动续跑…';
    try {
      await api('resume',{...requested,preview:true});
      const result = await api('resume',requested);
      message('续跑已提交：从第 ' + result.resume_step + ' 步恢复，总计 ' + result.steps + ' 步。');
      el('train-resume-panel').hidden = true;
      await refresh();
    } catch (error) {
      el('train-resume-message').textContent = error.message;
      message(error.message,true);
    } finally { jobActionBusy = false; updateJobActions(); }
  }
  async function requestTrainingStop(nextCheckpoint) {
    const job = currentJob();
    if (!job?.can_stop || jobActionBusy) return;
    const requested = {...chosen};
    if (nextCheckpoint && !window.confirm('保存权重后停止这次训练？\n' + (job.name || job.run) +
      '\n将在下一个计划保存点停止，已排队验证继续完成。')) return;
    jobActionBusy = true; updateJobActions();
    try {
      const result = await api('stop',{...requested,...(nextCheckpoint ? {next_checkpoint:true} : {step:Number(el('train-stop-step').value)})});
      message('已设定：第 ' + result.stop_after_step + ' 步保存权重后停止训练；已排队验证继续完成。');
      el('train-stop-message').textContent = '停止点：' + result.stop_after_step + ' 步';
      await refresh();
    } catch (error) { reportError(error); }
    finally { jobActionBusy = false; updateJobActions(); }
  }
  function updateDeleteDescription() {
    const files = el('train-delete-files');
    el('train-delete-confirm').textContent = files.checked ? '永久删除记录和目录' : '删除记录';
    el('train-delete-description').textContent = files.checked ?
      '永久删除该训练的输出目录，包括权重和日志，无法恢复。' :
      files.disabled ? '此记录只能从列表移除，输出目录会保留。' : '仅从列表移除记录，保留权重和日志。';
  }
  function openDeleteDialog() {
    const job = currentJob();
    if (!job?.can_delete || jobActionBusy) return;
    deleteTarget = {run:job.run, arm:job.arm, output:job.output, device:trainingDevice};
    el('train-delete-name').textContent = job.name || job.run;
    el('train-delete-device').textContent = deviceLabel() + ' · ' + methodLabel(jobMethod(job));
    el('train-delete-path').textContent = job.output || '未记录输出路径';
    el('train-delete-error').textContent = '';
    el('train-delete-files').checked = false;
    el('train-delete-files').disabled = job.can_delete_files !== true || !job.output;
    updateDeleteDescription();
    el('train-delete-dialog').showModal();
  }
  async function deleteTrainingRecord(event) {
    event.preventDefault();
    if (!deleteTarget || jobActionBusy || deleteTarget.device !== trainingDevice) return;
    const {run, arm, output} = deleteTarget;
    const files = el('train-delete-files');
    const canSelectFiles = !files.disabled;
    const requested = {run, arm};
    if (canSelectFiles && files.checked) Object.assign(requested, {delete_files:true, expected_output:output});
    jobActionBusy = true; updateJobActions();
    files.disabled = el('train-delete-confirm').disabled = el('train-delete-close').disabled = true;
    el('train-delete-error').textContent = '';
    try {
      const result = await api('delete',requested);
      el('train-delete-dialog').close();
      deleteTarget = null;
      jobs = jobs.filter(job => job.run !== run || job.arm !== arm);
      chosen = null;
      detail = null;
      selectView('new');
      el('train-new').focus();
      message(result.files_preserved ? '已删除训练记录，输出目录保留。' :
        '已永久删除训练记录及输出目录，包括权重和日志。');
      if (!result.files_preserved) {
        if (el('train-checkpoint').value.startsWith(output + '/')) el('train-checkpoint').value = '';
        lastOptionsAt = 0;
      }
      await refresh();
    } catch (error) {
      el('train-delete-error').textContent = error.message;
    } finally {
      files.disabled = !canSelectFiles;
      el('train-delete-confirm').disabled = el('train-delete-close').disabled = false;
      jobActionBusy = false;
      updateJobActions();
    }
  }
  function checkpointDeleteMode() {
    const permanent = el('train-checkpoint-delete-permanent').checked;
    el('train-checkpoint-delete-confirm').textContent = permanent ? '永久删除此 checkpoint' : '移入回收目录';
    el('train-checkpoint-delete-mode').textContent = permanent ?
      '仅永久删除上方所选的权重文件夹，无法撤销。' : '默认移入回收目录，可以撤销；回收目录仍占用磁盘空间。';
  }
  async function openCheckpointDelete() {
    const job = currentJob(), path = el('train-saved-checkpoint').value;
    const checkpoint = detail?.checkpoints?.find(p => p.path === path);
    if (!job || !checkpoint || jobActionBusy) return;
    const target = {run:job.run, arm:job.arm, checkpoint:path, device:trainingDevice};
    checkpointDeleteTarget = target;
    el('train-checkpoint-delete-name').textContent = job.run + ' · step ' + checkpoint.step;
    el('train-checkpoint-delete-device').textContent = deviceLabel();
    el('train-checkpoint-delete-path').textContent = path;
    el('train-checkpoint-delete-note').textContent = checkpointNotes.note(trainingDevice,path) || '暂无备注';
    el('train-checkpoint-delete-permanent').checked = false;
    el('train-checkpoint-delete-permanent').disabled = false;
    el('train-checkpoint-delete-confirm').disabled = true;
    el('train-checkpoint-delete-close').disabled = false;
    el('train-checkpoint-delete-warning').textContent = '正在核对 checkpoint 和占用情况…';
    el('train-checkpoint-delete-error').textContent = '';
    checkpointDeleteMode();
    el('train-checkpoint-delete-dialog').showModal();
    try {
      const preview = await api('checkpoint-delete',{...target,preview:true});
      if (checkpointDeleteTarget !== target || !el('train-checkpoint-delete-dialog').open) return;
      target.expected_checkpoint = preview.expected_checkpoint;
      el('train-checkpoint-delete-warning').textContent = preview.warning || '';
      el('train-checkpoint-delete-confirm').disabled = false;
    } catch (error) {
      if (checkpointDeleteTarget !== target) return;
      el('train-checkpoint-delete-warning').textContent = '';
      el('train-checkpoint-delete-error').textContent = error.message;
    }
  }
  async function deleteSelectedCheckpoint(event) {
    event.preventDefault();
    const target = checkpointDeleteTarget;
    if (!target?.expected_checkpoint || target.device !== trainingDevice || jobActionBusy) return;
    const permanent = el('train-checkpoint-delete-permanent').checked;
    jobActionBusy = true; updateJobActions();
    for (const name of ['confirm','close','permanent']) el('train-checkpoint-delete-'+name).disabled = true;
    el('train-checkpoint-delete-error').textContent = '';
    try {
      const result = await api('checkpoint-delete',{...target,permanent,confirm:true});
      el('train-checkpoint-delete-dialog').close();
      checkpointUndo = result.status === 'trashed' ? {...result,device:target.device} : null;
      el('train-checkpoint-undo').hidden = !checkpointUndo;
      el('train-checkpoint-delete-message').textContent = (permanent ? '已永久删除：' : '已移入回收目录（仍占磁盘）：') +
        target.checkpoint + '。训练记录、其他权重、备注和历史结果保留。';
      if (el('train-checkpoint').value === target.checkpoint) el('train-checkpoint').value = '';
      lastOptionsAt = 0;
      window.dispatchEvent(new CustomEvent('checkpoint-deleted',{detail:target}));
      await refresh();
    } catch (error) {
      el('train-checkpoint-delete-error').textContent = error.message;
    } finally {
      jobActionBusy = false; updateJobActions();
      for (const name of ['confirm','close','permanent']) el('train-checkpoint-delete-'+name).disabled = false;
    }
  }
  async function undoCheckpointDelete() {
    const target = checkpointUndo;
    if (!target || jobActionBusy) return;
    jobActionBusy = true; updateJobActions();
    el('train-checkpoint-undo').disabled = true;
    try {
      const response = await fetch('/api/registry/storage',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action:'restore',host:target.device === 'a100' ? 'A100' : 'A6000',path:target.checkpoint,token:target.token})});
      const result = await response.json();
      if (!response.ok) throw Error(result.error || '恢复失败');
      checkpointUndo = null;
      el('train-checkpoint-undo').hidden = true;
      el('train-checkpoint-delete-message').textContent = '已恢复 checkpoint：' + target.checkpoint;
      lastOptionsAt = 0;
      window.dispatchEvent(new CustomEvent('checkpoint-restored',{detail:target}));
      await refresh();
    } catch (error) { el('train-checkpoint-delete-message').textContent = error.message; }
    finally {
      jobActionBusy = false; updateJobActions();
      el('train-checkpoint-undo').disabled = false;
    }
  }
  function drawCurve() {
    if (!detail) return;
    const mode = el('train-curve-mode').value, chart = el('train-chart');
    const points = (mode === 'epochs' ? detail.epochs.filter(e => e.complete).map(e => [e.last,e.loss,e.steps]) :
      mode === 'steps' ? (detail.loss_series || []) : detail.curve).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    const weight = Number(el('train-curve-smoothing').value), showRaw = el('train-curve-raw').checked;
    const sourceLabel = {steps:detail.curve_metadata?.steps_label || '逐步 Loss',
      blocks:detail.curve_metadata?.blocks_label || '50 步加权均值',epochs:'完整 Epoch 加权均值'}[mode];
    el('train-curve-smoothing-value').textContent = weight.toFixed(2);
    el('train-curve-source-label').textContent = sourceLabel;
    el('train-curve-trend-label').textContent = weight ? 'EMA 平滑趋势' : sourceLabel + ' · 未平滑';
    el('train-curve-raw-key').hidden = !showRaw || weight === 0;
    el('train-curve-readout').textContent = '悬停查看具体数值；平滑仅影响显示。';
    chart.onpointermove = chart.onpointerleave = null;
    if (!points.length) { chart.innerHTML = '<text x="60" y="130" fill="#777">暂无对应训练数据</text>'; return; }
    const trend = smoothTrainingCurve(points, weight);
    const maxX = Math.max(1,...points.map(p => p[0]));
    let low = Math.min(...points.map(p => p[1])), high = Math.max(...points.map(p => p[1]));
    const padding = Math.max((high-low)*.12,high*.02,1e-8); low = Math.max(0,low-padding); high += padding;
    const x = value => 85 + value/maxX*780, y = value => 250 - (value-low)/(high-low)*225;
    let svg = '';
    for (let i=0;i<=4;i++) {
      const value = low + (high-low)*i/4, yy=y(value), xx=x(maxX*i/4);
      svg += '<line x1="85" y1="' + yy + '" x2="865" y2="' + yy + '" stroke="#dbe2e8"/>' +
        '<text x="76" y="' + (yy+4) + '" text-anchor="end" font-size="12" fill="#667">' + value.toPrecision(4) + '</text>' +
        '<text x="' + xx + '" y="274" text-anchor="middle" font-size="12" fill="#667">' + Math.round(maxX*i/4) + '</text>';
    }
    const color = chosen.arm === 'baseline' ? '#285fc6' : '#c57626';
    el('train-curve-legend').style.setProperty('--curve-color', color);
    const line = (values, raw) => '<polyline data-loss-series="' + (raw ? 'raw' : 'trend') + '" fill="none" stroke="' + color +
      '" stroke-opacity="' + (raw ? '.22' : '1') + '" stroke-width="' + (raw ? '1' : '2') +
      '" stroke-linejoin="round" stroke-linecap="round" points="' + values.map(p => x(p[0]).toFixed(2)+','+y(p[1]).toFixed(2)).join(' ') + '"/>';
    if (showRaw && weight > 0) svg += line(points, true);
    svg += line(trend, false);
    if (points.length === 1) svg += '<circle cx="' + x(points[0][0]) + '" cy="' + y(points[0][1]) + '" r="3" fill="' + color + '"/>';
    svg += '<g id="train-curve-cursor" visibility="hidden" pointer-events="none"><line y1="25" y2="250" stroke="#8190a2" stroke-dasharray="3 4"/><circle r="4" fill="' + color + '" stroke="white" stroke-width="2"/></g>';
    chart.innerHTML = svg;
    const cursor = el('train-curve-cursor');
    const readout = index => '优化步 ' + points[index][0] + ' · ' + sourceLabel + ' ' + fmt(points[index][1]) +
      (weight ? ' · EMA ' + fmt(trend[index][1]) : '') + (mode === 'steps' ? '' : ' · ' + points[index][2] + ' 步统计');
    el('train-curve-readout').textContent = readout(points.length-1) + ' · 平滑仅影响显示';
    chart.onpointermove = event => {
      const position = chart.createSVGPoint(); position.x = event.clientX; position.y = event.clientY;
      const px = position.matrixTransform(chart.getScreenCTM().inverse()).x;
      const step = (px-85)/780*maxX;
      let left = 0, right = points.length-1;
      while (left < right) { const middle = Math.floor((left+right)/2); if (points[middle][0] < step) left = middle+1; else right = middle; }
      const index = left > 0 && Math.abs(points[left-1][0]-step) < Math.abs(points[left][0]-step) ? left-1 : left;
      const xx = x(points[index][0]);
      cursor.setAttribute('visibility','visible');
      const marker = cursor.querySelector('circle'), guide = cursor.querySelector('line');
      guide.setAttribute('x1',xx); guide.setAttribute('x2',xx);
      marker.setAttribute('cx',xx); marker.setAttribute('cy',y(trend[index][1]));
      el('train-curve-readout').textContent = readout(index);
    };
    chart.onpointerleave = () => { cursor.setAttribute('visibility','hidden'); el('train-curve-readout').textContent = readout(points.length-1) + ' · 平滑仅影响显示'; };
  }
  function parameterList(rows) {
    return '<dl class="train-param-list">' + rows.map(([label,value,path]) => {
      const missing = value == null || value === '';
      const classes = [path ? 'train-parameter-path' : '', missing ? 'train-missing' : ''].filter(Boolean).join(' ');
      return '<div' + (path ? ' class="train-path-row"' : '') + '><dt>' + esc(label) +
        '</dt><dd class="' + classes + '">' + esc(missing ? '未记录' : value) + '</dd></div>';
    }).join('') + '</dl>';
  }
  function parameterGroups(groups) {
    return groups.map(([title,rows]) =>
      '<div class="train-param-group"><h3>' + esc(title) + '</h3>' + parameterList(rows) + '</div>').join('');
  }
  function renderRecordedParameters(job) {
    const p = detail.training_parameters?.values || {};
    const flag = value => value == null ? null : value ? '开启' : '关闭';
    const steps = value => value == null ? null : value + ' 步';
    const tuningModes = {action_full:'动作头 / 记忆全参数微调',full:'全模型微调',lora:'LoRA 微调',custom:'自定义组件冻结'};
    el('train-recorded-params').innerHTML = parameterGroups([
      ['训练设置', [
        ['训练总步数',steps(p.steps)],['学习率',p.learning_rate],['调度器',p.scheduler],['Warmup比例',p.warmup_ratio],['训练 GPU',p.gpus],
        ['梯度累积',p.accumulation??p.gradient_accumulation_steps],['有效Batch',p.global_decisions],['父checkpoint',p.parent_checkpoint?.checkpoint,true],['父累计步数',p.parent_cumulative_step],
        ['全局 Batch',p.global_windows_per_step == null ? null : p.global_windows_per_step + ' 窗口 / 优化步' +
          (p.global_decisions_max == null ? '' : '（最多 '+p.global_decisions_max+' 决策）')],
        ['Microbatch / GPU',p.microbatch == null ? null : p.microbatch + ' 决策 / 次前向'],
        ['微调方式',tuningModes[p.fine_tuning_mode] || p.fine_tuning_mode],['随机种子',p.seed],
        ...(p.stop_after_step ? [['保存后停止点',steps(p.stop_after_step)]] : [])]],
      ['数据与验证', [
        ['训练任务',p.tasks?.map(task => 'Task ' + task).join(' / ')],
        ['训练 Episode',p.train_episodes],['验证 Episode',p.validation_episodes],['独立验证',flag(p.validation_enabled)],
        ...(p.validation_enabled ? [['验证 GPU',p.validation_gpu],['验证间隔',steps(p.validation_interval)]] : []),
        ['保存间隔',steps(p.checkpoint_interval)],['数据集路径',p.dataset,true]]]
    ]);
    const components = {llm:'LLM',vision:'视觉编码器',vision_projector:'视觉投影',action_dit:'动作 DiT',
      action_projector:'状态 / 动作投影',action_vlln:'动作输入 LayerNorm',memory:'记忆模块'};
    el('train-advanced-recorded-params').innerHTML = parameterGroups([
      ['优化器与训练策略', [
        ['Weight decay',p.weight_decay],['AdamW epsilon',p.optimizer_eps],['梯度裁剪',p.max_grad_norm],['归一化',p.normalization],
        ['梯度检查点',flag(p.gradient_checkpointing)],['图像增强',flag(p.image_augmentation)],
        ['自动早停',p.disable_early_stopping == null ? null : flag(!p.disable_early_stopping)],['训练协议',p.training_protocol]]],
      ['组件冻结', [
        ...Object.entries(components).filter(([key]) => key !== 'memory' || chosen.arm === 'memory_v1')
          .map(([key,label]) => {
            const actual=detail.runtime?.components?.[key];
            if(actual)return [label,actual.trainable+' / '+actual.total+' 可训练参数'];
            const map={llm:'tune_llm',vision:'tune_visual',vision_projector:'tune_visual',action_dit:'tune_diffusion_model',action_projector:'tune_projector',action_vlln:'tune_vlln'};
            if(p.native_tuning&&map[key])return [label,key==='llm'&&p.native_tuning.tune_top_llm_layers?'最后 '+p.native_tuning.tune_top_llm_layers+' 层':p.native_tuning[map[key]]?'训练':'冻结'];
            return [label,p['freeze_'+key] == null ? null : p['freeze_'+key] ? '冻结' : '训练'];
          }),
        ['冻结 Eagle',flag(p.frozen_eagle)],
        ['LLM 最后 N 层',p.llm_last_n_layers === 0 ? '全部所选层' : p.llm_last_n_layers]]],
      ...(p.fine_tuning_mode === 'lora' ? [['LoRA', [
        ['目标组件',p.lora_targets],['Rank',p.lora_rank],['Alpha',p.lora_alpha],['Dropout',p.lora_dropout]]]] : []),
      ...(chosen.arm === 'memory_v1' ? [['记忆模块', [
        ['记忆容量',p.memory_length],['记忆接入',p.memory_conditioning],['Residual readout',p.memory_residual_readout],
        ['认知检索归一化',p.memory_cognitive_qk_norm],['认知矩阵投影',flag(p.memory_cognitive_residual_projection)],
        ['认知残差范围',p.memory_cognitive_residual_scope]]]] : [])
    ]);
    const origin = detail.initial_weights || {};
    el('train-initial-weight-label').textContent = origin.label || '未记录';
    el('train-initial-weight-path').textContent = origin.checkpoint || '未记录';
    el('train-initial-weight-mode').textContent = origin.mode === 'resume' ?
      '从第 ' + origin.resume_step + ' 步续训 · ' + (origin.optimizer_restored ? '已恢复优化器状态' : '等待确认优化器恢复') :
      origin.mode === 'fresh' ? '从初始权重启动 · 新建优化器' : '未记录';
    el('train-resume-weight-row').hidden = !origin.resume_checkpoint;
    el('train-resume-weight-path').textContent = origin.resume_checkpoint || '';
    const sources = [...new Set([...(detail.training_parameters?.sources || []),origin.source_file,origin.resume_record].filter(Boolean))];
    el('train-params-source').textContent = sources.length ? '记录来源\n' + sources.join('\n') : '暂无配置来源记录';
    const config = detail.config || {};
    el('train-raw-config').textContent = Object.keys(config).length ? JSON.stringify(config,null,2) : '暂无原始训练配置';

    const q = job?.queue;
    el('train-detail-info').innerHTML = parameterList([
      ['运行名',chosen.run],['训练 ID',job?.training_id || chosen.run],
      ['每步耗时',job?.last?.seconds == null ? null : fmt(job.last.seconds) + ' 秒'],
      ['梯度范数',job?.last?.gradient_norm == null ? null : fmt(job.last.gradient_norm)],
      ['源码哈希',job?.source_digest],['输出目录',job?.output,true],['源码快照',job?.source,true],
      ...(q ? [
        ['队列',q.name + ' · ' + stateName(q.status)],
        ...(q.dependencies?.length ? [['前序作业',q.dependencies.map(d => 'Task ' + d.task + ' · ' + d.arm +
          ' · ' + d.step + ' 步 · ' + (d.complete ? '训练及验证完成' : '等待训练或最终验证')).join('\n')]] : []),
        ...(q.waiting_for_runs?.length ? [['等待前序训练',q.waiting_for_runs.join('\n')]] : []),
        ...(q.unexpected_pids?.length ? [['等待 GPU 进程退出',q.unexpected_pids.join('、')]] : []),
        ...(q.error ? [['队列错误',q.error]] : []),['队列状态文件',q.path,true]] : [])
    ]);
  }
  function runLogText() {
    return (logSource === 'validation' ? detail?.validation_log_tail : detail?.log_tail) || '';
  }
  function renderRunLog(jumpToEnd = false) {
    const validation = logSource === 'validation';
    const log = el('train-log');
    const content = runLogText();
    const scrollTop = log.scrollTop;
    const follow = jumpToEnd || !log.textContent || log.scrollHeight - log.clientHeight - scrollTop <= 24;
    let empty = validation ? '暂无验证日志。' : '暂无训练日志。';
    if (validation && currentJob()?.validation_enabled === false) empty = '本次训练未启用独立验证。';
    const text = content || empty;
    if (log.textContent !== text) log.textContent = text;
    log.scrollTop = follow ? log.scrollHeight : scrollTop;
    log.setAttribute('aria-label', validation ? '验证日志' : '训练日志');
    for (const source of ['training','validation'])
      el('train-log-source-' + source).setAttribute('aria-pressed', String(source === logSource));
    el('train-log-copy').disabled = el('train-log-latest').disabled = !content;
  }
  function selectLogSource(source) {
    logSource = source;
    el('train-log-feedback').textContent = '';
    renderRunLog(true);
  }
  async function loadDetail() {
    if (!chosen) return;
    const requested = chosen;
    const result = await api('detail?run=' + encodeURIComponent(requested.run) + '&arm=' + encodeURIComponent(requested.arm));
    if (chosen !== requested) return;
    detail = result;
    drawDiagnosticCurve();
    const job = currentJob();
    const notes = detail.notes || job?.notes || '';
    el('train-detail-notes').textContent = notes ? '备注：' + notes : '';
    el('train-detail-notes').hidden = !notes;
    renderRecordedParameters(job);
    el('train-detail-title').textContent = job?.name || chosen.run;
    el('train-detail-subtitle').textContent = taskLabel(job) + ' · ' + (job?.method || (chosen.arm === 'baseline' ? 'baseline' : 'memory'));
    const tasks = jobTasks(job);
    el('train-detail-tasks').hidden = tasks.length < 2;
    el('train-detail-task-list').textContent = tasks.map(task=>'Task '+task).join(' / ');
    el('train-detail-gpu').textContent = 'GPU ' + (job?.gpus ?? '—') + ' · 每 10 秒刷新';
    el('train-detail-status').innerHTML = stateBadge(job?.status);
    el('train-runtime-note').hidden = !job;
    el('train-runtime-text').textContent = job?.can_cancel ?
      (job.status === 'waiting_for_gpus' ? '等待所选设备 GPU 释放；可在任务控制中取消排队。' : '已加入训练队列，可在任务控制中查看或取消。') :
      job?.can_stop ? '训练中，可设置保存后停止点；已保存权重可直接用于推理评估。' :
      job?.status === 'failed' ? '训练异常退出，可查看运行日志排查。' : '可选择已保存的权重，继续进行推理评估。';
    el('train-runtime-controls').textContent = job?.can_cancel ? '管理排队 →' : job?.can_stop ? '权重 / 停止设置 →' :
      job?.status === 'failed' ? '查看运行日志 →' : '查看权重 →';
    const metrics = [['训练步数', (job?.step ?? '—') + ' / ' + (job?.max_steps ?? '—'), '优化步'],
      ['最新 Loss', fmt(job?.last?.loss), '训练损失'],
      ['训练吞吐', fmt(job?.last?.decisions_per_second_end_to_end), job?.last?.throughput_unit || '决策 / 秒'],
      ['峰值显存', fmt(job?.last?.peak_allocated_gib), 'GiB · 记录值']];
    el('train-metrics').innerHTML = metrics.map(([label,value,hint],i) => '<div class="metric' + (i === 1 ? ' primary' : '') + '"><label>' + label + '</label><strong>' + esc(value) + '</strong><small>' + hint + '</small></div>').join('');
    const timing = detail.timing || {}, lastSeconds = timing.last_step_seconds ?? job?.last?.seconds;
    el('train-elapsed').textContent = timeSpan(timing.elapsed_seconds);
    el('train-eta').textContent = timing.eta_seconds === 0 ? '训练已结束' : timeSpan(timing.eta_seconds);
    el('train-eta-note').textContent = timing.eta_seconds === 0 ? '训练优化已结束；独立验证状态另计' :
      timing.eta_seconds == null ? '等待运行中的计时记录；不含排队时间' :
      '剩余 ' + timing.remaining_steps + ' 步 × 最近 ' + timing.recent_samples + ' 步均值；' +
        (timing.scope === 'recorded_training_intervals' ? '按日志区间估算，包含区间内保存和验证耗时' : '不含保存和验证');
    el('train-step-time').textContent = timing.recent_mean_step_seconds == null ? '—' : timing.recent_mean_step_seconds.toFixed(2) + ' s / 步';
    el('train-step-time-note').textContent = lastSeconds == null ? '等待计时记录' :
      (timing.scope === 'recorded_training_intervals' ? '最近记录区间平均每步 ' : '最近一步 ') +
      Number(lastSeconds).toFixed(2) + ' s' + (timing.mean_step_seconds == null ? '' : ' · 全程平均 ' + timing.mean_step_seconds.toFixed(2) + ' s');
    const validation = job?.validation || {};
    el('train-validation-info').textContent = job?.validation_enabled===false?'本次未启用验证':
      (detail.validation_scope === 'internal' ? '训练内验证 · ' + detail.validation.length + ' 次记录' : '独立验证 · ' + stateName(validation.status)) +
      (validation.checkpoint_step != null ? ' · step ' + validation.checkpoint_step : '') +
      (validation.completed_episodes != null ? ' · ' + validation.completed_episodes + '/' + validation.total_episodes + ' 集' : '');
    const metadata = detail.curve_metadata;
    const epochOption = el('train-curve-mode').querySelector('option[value="epochs"]');
    epochOption.disabled = metadata?.has_epochs === false;
    epochOption.textContent = epochOption.disabled ? 'Epoch（本训练未记录）' : '完整 Epoch 平均';
    if (epochOption.disabled && el('train-curve-mode').value === 'epochs') el('train-curve-mode').value = 'steps';
    el('train-curve-mode').querySelector('option[value="steps"]').textContent = metadata?.steps_label || '逐步 Loss';
    el('train-epoch-table').innerHTML = (metadata?.has_epochs === false ?
      '<p class="train-hint">本训练未记录 Epoch，Loss 按实际日志记录点显示。</p>' :
      '<table><thead><tr><th>Epoch</th><th>训练步</th><th>平均 Loss</th></tr></thead><tbody>' +
      detail.epochs.map(e => '<tr><td>' + e.epoch + (e.complete ? '' : '（未完整）') + '</td><td>' + e.first + '–' + e.last + '</td><td>' + fmt(e.loss) + '</td></tr>').join('') +
      '</tbody></table>') + '<table><thead><tr><th>验证 checkpoint</th><th>验证 Loss</th>' +
      (detail.validation_scope === 'internal' ? '' : '<th>动作 MAE</th>') + '</tr></thead><tbody>' +
      detail.validation.map(v => '<tr><td>' + v.step + '</td><td>' + fmt(v.validation_loss) + '</td>' +
        (detail.validation_scope === 'internal' ? '' : '<td>' + fmt(v.action_mae?.normal ?? v.action_mae) + '</td>') + '</tr>').join('') + '</tbody></table>';
    const previous = el('train-saved-checkpoint').value;
    el('train-saved-checkpoint').innerHTML = (detail.checkpoints || []).map(p => '<option value="' + esc(p.path) + '">step ' + p.step + esc(checkpointNotes.brief(trainingDevice,p.path)) + '</option>').join('') || '<option value="">暂无已保存权重</option>';
    if ([...el('train-saved-checkpoint').options].some(o => o.value === previous)) el('train-saved-checkpoint').value = previous;
    updateCheckpointPath();
    el('train-use-checkpoint').disabled = !(detail.checkpoints?.length);
    el('train-use-checkpoint').title = '在该训练设备上加载权重，进入推理评估';
    el('train-stop').disabled = !job?.can_stop;
    el('train-stop-panel').hidden = !job?.can_stop;
    el('train-cancel-panel').hidden = !job?.can_cancel;
    const cancelLabel = job?.status === 'waiting_for_gpus' ? '取消 GPU 等待及本组排队' : '取消本组排队';
    el('train-cancel-title').textContent = cancelLabel;
    el('train-cancel').textContent = cancelLabel;
    el('train-cancel').disabled = !job?.can_cancel;
    el('train-cancel-scope').textContent = '将一并撤销以下尚未开始的训练：\n' + (job?.queue?.affected_runs || []).join('\n');
    updateJobActions();
    drawCurve();
    el('train-detail-loading').hidden = true;
    el('train-monitor-body').hidden = false;
    if (!el('train-pane-logs').hidden) renderRunLog();
  }
  function updateCheckpointPath() {
    el('train-checkpoint-path').textContent = el('train-saved-checkpoint').value;
    checkpointNotes.bind('train-saved-checkpoint-note',trainingDevice,el('train-saved-checkpoint').value);
    el('train-checkpoint-delete').disabled = jobActionBusy || !el('train-saved-checkpoint').value;
  }
  window.addEventListener('checkpoint-notes-updated',()=>{
    if (!mounted) return;
    for(const option of el('train-saved-checkpoint').options){
      option.dataset.baseLabel ||= option.textContent.split(' · 备注：')[0];
      option.textContent=option.dataset.baseLabel+checkpointNotes.brief(trainingDevice,option.value);
    }
  });
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      const current=trainingDevice;
      const settled=await Promise.allSettled(jobDevices.map(async device=>{const response=await fetch('/api/training/jobs?device='+device,{cache:'no-store'});const data=await response.json();if(!response.ok||!Array.isArray(data.jobs))throw Error(data.error||'记录不可查');return data;}));
      settled.forEach((r,i)=>jobSnapshots.set(jobDevices[i],r.status==='fulfilled'?r.value:{jobs:[],error:String(r.reason)}));
      renderOverview();
      if(current!==trainingDevice)return;
      const result=jobSnapshots.get(current);if(result.error){jobs=[];renderJobs();throw Error(deviceLabel()+' 不可查');}
      jobs=result.jobs.filter(job=>job.status!=='cancelled');syncStatus(result.sync);
      if (chosen && !currentJob()) {
        chosen = null; detail = null;
        selectView('overview');
      }
      if (view === 'new' && Date.now() - lastOptionsAt > 30000) await loadOptions();
      if (!chosen && jobs.length) chosen = {run:jobs[0].run,arm:jobs[0].arm};
      renderJobs();
      if (view === 'detail') await loadDetail();
    } catch (error) { reportError(error); }
    finally { refreshing = false; }
  }
  async function synchronize(payload) {
    el('train-sync-once').disabled = el('train-autosync').disabled = true;
    try { syncStatus(await api('sync',payload)); }
    catch(error) { reportError(error); }
    finally { el('train-sync-once').disabled = el('train-autosync').disabled = false; }
  }
  function bind() {
    el('train-form').querySelectorAll('[data-train-setup-step]').forEach(button => {
      button.onclick = () => showSetupStep(Number(button.dataset.trainSetupStep), true);
    });
    el('train-previous').onclick = () => showSetupStep(setupStep-1, true);
    el('train-next').onclick = () => { if (validateSetup(setupStep)) { message(''); showSetupStep(setupStep+1, true); } };
    showSetupStep(0);
    el('train-device').onchange = async () => {
      if(busy || jobActionBusy){el('train-device').value=trainingDevice;return;}
      trainingDevice=el('train-device').value;
      options=null;jobs=[];chosen=null;detail=null;lastOptionsAt=0;
      el('train-dataset').innerHTML='';el('train-task-list').innerHTML='';el('train-task-selection').innerHTML='';
      el('train-checkpoint').value='';el('train-gpus').innerHTML='';
      el('train-jobs').innerHTML='<div class="empty">正在读取 '+esc(deviceLabel())+' 记录…</div>';
      el('train-checkpoint-choice').innerHTML='<option value="">正在读取设备权重…</option>';
      el('train-validation-enabled').disabled=trainingDevice!=='a100';
      el('train-validation-enabled').checked=trainingDevice==='a100';
      el('train-device-label').textContent=deviceLabel();
      el('train-device-hint').hidden=trainingDevice==='a100';
      el('train-device-hint').textContent='单卡开发训练；先准备任务数据，训练后单独评估权重。设备离线时不会自动转到其他机器。';
      selectView(view==='new'?'new':'overview');invalidatePlan();
      try{await refresh();await loadOptions();}catch(error){
        if(error.name!=='DeviceChanged') message(deviceLabel()+' 读取失败，设备可能离线：'+error.message,true);
      }
    };
    el('train-prepare-data').onclick = async () => {
      try {
        for (const data of selectedDatasets().filter(d=>!d.ready && d.can_prepare)) data.preparation=await api('dataset',{task:data.task});
        renderTaskChoices();datasetHint();message('所选缺失数据已提交后台准备，完成后自动更新。');
      } catch(error) { reportError(error); }
    };
    el('train-overview-tab').onclick=()=>{selectView('overview');void refresh();};
    el('train-new').onclick = () => {selectView('new');if(!options)void loadOptions().catch(reportError);};
    el('train-records-toggle').onclick = event => {
      const toggle = event.currentTarget;
      const expanded = toggle.getAttribute('aria-expanded') !== 'true';
      toggle.setAttribute('aria-expanded', String(expanded));
      toggle.textContent = expanded ? '收起记录' : '展开记录';
    };
    el('train-job-search').oninput = renderJobs;
    el('train-job-filter').onchange = renderJobs;
    el('train-job-owner').onchange=renderJobs;
    el('train-job-category').onchange=renderJobs;
    el('train-task-search').oninput = renderTaskChoices;
    el('train-task-ready').onchange = renderTaskChoices;
    el('train-checkpoint-choice').onchange = () => { updateCheckpointChoice(); invalidatePlan(); updateRecommendation(); };
    el('train-auto-recommend').onchange = () => { invalidatePlan(); updateRecommendation(); };
    el('train-version').onchange=()=>{updateNativeMode();datasetHint();invalidatePlan();};
    el('train-diagnostic-metric').onchange=drawDiagnosticCurve;
    el('train-download-report').onclick=()=>{
      const report=detail?.analysis_report;
      if(!report?.text)return;
      const url=URL.createObjectURL(new Blob([report.text],{type:'text/markdown;charset=utf-8'}));
      const anchor=document.createElement('a');anchor.href=url;anchor.download=chosen.run+'__training_report.md';anchor.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    };
    el('train-download-evidence').onclick=async()=>{
      if(!chosen||!detail?.analysis_report)return;
      const id=chosen;
      const url='/api/training/evidence?device='+encodeURIComponent(trainingDevice)+'&run='+encodeURIComponent(id.run)+'&arm='+encodeURIComponent(id.arm);
      try{
        const response=await fetch(url,{cache:'no-store'});
        if(!response.ok)throw Error((await response.json()).error||'证据包生成失败');
        const blob=URL.createObjectURL(await response.blob()),a=document.createElement('a');
        a.href=blob;a.download=id.run+'__training_evidence.zip';a.click();setTimeout(()=>URL.revokeObjectURL(blob),1000);
      }catch(error){reportError(error);}
    };
    function fillNativeConfig(p) {
      const fields={'train-steps':p.steps,'train-lr':p.learning_rate,'train-microbatch':p.microbatch,
        'train-native-accum':p.accumulation,'train-native-scheduler':p.scheduler,'train-native-warmup':100*(p.warmup_ratio??.05),
        'train-native-wd':p.weight_decay,'train-native-clip':p.max_grad_norm??1,'train-native-normalization':p.normalization,
        'train-native-save':p.checkpoint_interval,'train-native-validation':p.validation_interval??1000};
      for(const [id,v] of Object.entries(fields))if(v!=null)el(id).value=v;
      const tuning=p.native_tuning||{tune_llm:false,tune_top_llm_layers:0,tune_visual:false,tune_projector:true,tune_diffusion_model:true,tune_vlln:true};
      for(const [k,v] of Object.entries({llm:tuning.tune_llm,vision:tuning.tune_visual,dit:tuning.tune_diffusion_model,projector:tuning.tune_projector,vlln:tuning.tune_vlln}))el('train-native-'+k).checked=!!v;
      el('train-native-top').value=tuning.tune_top_llm_layers||0;
      el('train-fine-tuning').value='custom';
    }
    async function copyNative(formal) {
      const p=detail?.config,job=currentJob();
      if(!p||!job)return;
      try{
        await loadOptions();
        if(!options.checkpoints.some(c=>c.path===p.checkpoint&&c.native_stage1))throw Error('原始初始化权重已不可用');
        el('train-version').value='V_2_1';updateNativeMode();
        el('train-checkpoint').value=p.checkpoint;renderCheckpointChoices();
        selectTasks(d=>d.task===40);fillNativeConfig(p);
        const id=job.run.match(/__test(\d+)__exp(\d+)__run(\d+)(?:__preflight(\d+))?/);
        if(id){el('train-test-id').value=Number(id[1]);el('train-experiment-id').value=Number(id[2]);}
        el('train-run-number').value=formal&&id?Number(id[3]):'';
        el('train-native-attempt').value=formal&&id?Number(id[4]):1;
        el('train-native-phase').value=formal?'formal':'preflight';
        el('train-notes').value=p.notes||'';
        for(const input of el('train-gpus').querySelectorAll('input'))input.checked=(p.gpus||[]).includes(Number(input.value));
        selectView('new');showSetupStep(formal?3:0,true);invalidatePlan();updateBatch();
        message(formal?'已复制预检配方；检查配置后提交正式训练，不能修改已通过的预检条件。':'已复制配方，请填写未使用的新run编号。');
      }catch(error){reportError(error);}
    }
    el('train-copy-config').onclick=()=>void copyNative(false);
    el('train-formal-from-preflight').onclick=()=>void copyNative(true);
    el('train-warm-start').onclick=async()=>{
      const parent=el('train-saved-checkpoint').value,job=currentJob(),p=detail?.config;
      if(!parent||!job)return;
      try {
        await loadOptions();
        if(!options.checkpoints.some(p=>p.path===parent&&p.native_stage1))throw Error('该权重未通过原生Stage1目录核查');
        el('train-version').value='V_2_1';updateNativeMode();
        el('train-checkpoint').value=parent;renderCheckpointChoices();
        selectTasks(d=>d.task===40);
        if(p)fillNativeConfig(p);
        const id=job.run.match(/__test(\d+)__exp(\d+)__run(\d+)/);
        if(id){el('train-test-id').value=Number(id[1]);el('train-experiment-id').value=Number(id[2]);}
        el('train-run-number').value='';
        el('train-native-phase').value='preflight';el('train-native-attempt').value='1';
        el('train-steps').value='6000';el('train-lr').value='0.00001';el('train-native-scheduler').value='constant';el('train-native-warmup').value='0';
        for(const input of el('train-gpus').querySelectorAll('input'))input.checked=(p?.gpus||[]).includes(Number(input.value));
        el('train-notes').value='从 '+job.run+' checkpoint 继续训练（新优化器）';
        selectView('new');showSetupStep(0,true);invalidatePlan();updateBatch();
        message('已继承选定权重；请填写未使用的新run编号，核对资源和参数后预检。');
      }catch(error){reportError(error);}
    };
    el('train-restore-recommendation').onclick = () => {
      el('train-auto-recommend').checked = true; invalidatePlan(); updateRecommendation();
    };
    for (const id of ['train-arm','train-fine-tuning','train-checkpoint']) el(id).onchange = () => {
      if (el('train-lora-fields')) el('train-lora-fields').hidden = el('train-fine-tuning').value !== 'lora';
      if (id === 'train-fine-tuning') {
        if(nativeMode())nativePreset(el(id).value);
        else el('train-tuning-details').open = ['lora','custom'].includes(el(id).value);
      }
      updateRecommendation();
    };
    el('train-select-ready').onclick = () => selectTasks(data=>data.ready);
    el('train-clear-tasks').onclick = () => selectTasks(()=>false);
    el('train-runtime-controls').onclick = () => {
      const tab = currentJob()?.status === 'failed' ? 'logs' : 'controls';
      selectDetailTab(tab);
      if (tab === 'logs') selectLogSource('training');
      el('train-tab-' + tab).focus();
    };
    for (const name of detailTabs) el('train-tab-' + name).onclick = () => selectDetailTab(name);
    el('train-detail-tabs').onkeydown = event => {
      const current = detailTabs.findIndex(name => event.target === el('train-tab-' + name));
      if (current < 0) return;
      const last = detailTabs.length - 1;
      const next = {Home:0,End:last,ArrowLeft:current === 0 ? last : current - 1,
        ArrowRight:current === last ? 0 : current + 1}[event.key];
      if (next == null) return;
      event.preventDefault();
      selectDetailTab(detailTabs[next]);
      el('train-tab-' + detailTabs[next]).focus();
    };
    for (const source of ['training','validation'])
      el('train-log-source-' + source).onclick = () => selectLogSource(source);
    el('train-log-latest').onclick = () => renderRunLog(true);
    el('train-log-copy').onclick = async () => {
      const feedback = el('train-log-feedback');
      try {
        await navigator.clipboard.writeText(runLogText());
        feedback.textContent = '已复制当前日志片段';
      } catch {
        feedback.textContent = '无法访问剪贴板，可选中文字手动复制';
      }
    };
    el('train-form').onsubmit = event => { event.preventDefault(); void submit(true); };
    el('train-check').onclick = () => void submit(false);
    el('train-form').oninput = event => {
      invalidatePlan();
      if(nativeMode()&&['train-native-llm','train-native-top','train-native-vision','train-native-dit','train-native-projector','train-native-vlln'].includes(event.target.id))el('train-fine-tuning').value='custom';
      if (Object.values(recommendedFields).includes(event.target.id)) {
        el('train-auto-recommend').checked = false;
        updateRecommendation();
      }
      updateBatch();
    };
    el('train-form').addEventListener('invalid', event => {
      const section = event.target.closest('details');
      if (section) section.open = true;
    }, true);
    el('train-dataset').onchange = datasetHint;
    el('train-validation-enabled').onchange=datasetHint;
    el('train-validation-percent').oninput=datasetHint;
    el('train-gpus').onchange = updateRecommendation;
    el('train-curve-mode').onchange = drawCurve;
    el('train-curve-smoothing').oninput = drawCurve;
    el('train-curve-raw').onchange = drawCurve;
    el('train-saved-checkpoint').onchange = updateCheckpointPath;
    el('train-refresh').onclick = async () => { try { await loadOptions(); await refresh(); } catch(error) { reportError(error); } };
    el('train-autosync').onchange = () => void synchronize({enabled:el('train-autosync').checked});
    el('train-sync-once').onclick = () => void synchronize({once:true});
    el('train-stop').onclick = () => void requestTrainingStop(false);
    el('train-stop-next').onclick = () => void requestTrainingStop(true);
    el('train-resume-open').onclick = () => {
      if (!currentJob()?.resume?.available) return;
      el('train-resume-panel').hidden = false;
      el('train-resume-steps').focus();
    };
    el('train-resume-close').onclick = () => { el('train-resume-panel').hidden = true; };
    el('train-resume-steps').oninput = updateResumeBudget;
    el('train-resume-panel').onsubmit = event => void resumeTraining(event);
    el('train-delete').onclick = openDeleteDialog;
    el('train-delete-files').onchange = updateDeleteDescription;
    el('train-delete-form').onsubmit = event => void deleteTrainingRecord(event);
    el('train-delete-close').onclick = () => el('train-delete-dialog').close();
    el('train-delete-dialog').oncancel = event => { if (jobActionBusy) event.preventDefault(); };
    el('train-checkpoint-delete').onclick = () => void openCheckpointDelete();
    el('train-checkpoint-delete-permanent').onchange = checkpointDeleteMode;
    el('train-checkpoint-delete-form').onsubmit = event => void deleteSelectedCheckpoint(event);
    el('train-checkpoint-delete-close').onclick = () => el('train-checkpoint-delete-dialog').close();
    el('train-checkpoint-delete-dialog').onclose = () => { checkpointDeleteTarget = null; };
    el('train-checkpoint-delete-dialog').oncancel = event => { if (jobActionBusy) event.preventDefault(); };
    el('train-checkpoint-undo').onclick = () => void undoCheckpointDelete();
    el('train-cancel-top').onclick = () => {
      if (window.confirm('取消本组排队？\n' + (currentJob()?.queue?.affected_runs || []).join('\n')))
        el('train-cancel').click();
    };
    el('train-cancel').onclick = async () => {
      if (!chosen || !currentJob()?.can_cancel) return;
      el('train-cancel').disabled = true;
      try {
        const result = await api('cancel',chosen);
        message('已取消排队并从列表移除：' + result.affected_runs.join('、') + '。');
        await refresh();
      } catch(error) {
        el('train-cancel-message').textContent = error.message;
        await refresh();
      }
    };
    el('train-use-checkpoint').onclick = async () => {
      const path = el('train-saved-checkpoint').value, source = currentJob()?.source;
      if (!path) return;
      selectPage('inference');
      await selectInferenceDevice(trainingDevice);
      el('launch-device').value=trainingDevice==='a6000'?'a6000':'local';
      el('launch-device').dispatchEvent(new Event('change'));
      await loadInferenceOptions();
      el('inference-checkpoint').value = path;
      if (source) el('inference-source').value = source;
      el('inference-checkpoint').dispatchEvent(new Event('input',{bubbles:true}));
      if(currentJob()?.version) el('launch-version').value=currentJob().version;
      el('launch-group').value=currentJob()?.method || (chosen.arm==='baseline'?'baseline':'memory');
      if(currentJob()?.task!=null) el('launch-task').value=String(currentJob().task);
    };
  }
  window.trainingDashboard = {open:async () => {
    try {
      if (!mounted) {
        if (typeof pageHTML !== 'string') throw Error('训练页面版本不匹配，请按 Ctrl+F5 重新加载工作台');
        el('page-training').innerHTML = pageHTML; mounted = true; bind();
        window.benchmarkWorkbench?.trainingReady();selectView('overview');
      }
      await refresh(); await loadOptions();
    } catch(error) {
      if (mounted) reportError(error);
      else el('page-training').textContent = error.message;
    }
  }};
  setInterval(() => { if (mounted && !el('page-training').hidden) void refresh(); },10000);
})();
