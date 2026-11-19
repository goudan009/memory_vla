/* Preserved architecture explanation prototype. All execution is simulated. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const clone = value => JSON.parse(JSON.stringify(value));
  const STORAGE = 'memory-vla.ai-workbench.preview.v2';
  const clock = () => new Date().toLocaleTimeString('zh-CN', {hour:'2-digit', minute:'2-digit', hour12:false});
  const stdName = n => 'STD-' + String(n).padStart(2, '0');
  const kinds = {compare:'验证改进', diagnose:'诊断原因', evaluate:'评估已有权重'};
  const protocolNames = {offline:'动作短窗诊断', b1k:'B1K 闭环评估', libero:'LIBERO · Spatial 闭环评估'};
  const metricNames = {offline:'动作 MAE₈ ↓', b1k:'B1K Q-score ↑', libero:'LIBERO 成功率 ↑'};
  const stageNames = ['定方案', '审代码', '分资源', '跑实验', '看结论'];
  const runStatusNames = {running:'演示运行中', paused:'演示已暂停', failed:'演示失败', completed:'待分析', archived:'已归档'};
  const roles = [
    {id:'planning', name:'研究规划', type:'AI 判断角色', short:'决定验证什么',
      description:'接收你的研究问题，先读已有结论和资源，再提出能回答问题的实验方案。',
      free:'理解问题、提出假设、设计对照、安排优先级；根据负结果调整下一轮方向。',
      fixed:'按统一方案格式写清变化因素、共享条件、预算、需要的材料和判定依据。',
      input:'你的问题 + 共享研究记录', output:'待审阅的实验方案与分支', tools:'研究记录查询、资源目录查询、方案创建与修订', boundary:'提出方案不等于启动实验；缺少已有材料时先列为待补。'},
    {id:'code', name:'代码工程', type:'AI 判断角色', short:'实现想法并审查',
      description:'实现方法与修复问题，保留清晰版本；由独立审查上下文检查作者的修改。',
      free:'选择实现方式、定位数值或接口问题、解释变更；审查角色提出修改意见。',
      fixed:'创建实验工作区、记录变更、运行已登记检查、保存源码快照并关联实验。',
      input:'已确认方案 + 当前代码版本', output:'变更集、检查报告、审查意见、源码快照', tools:'代码工作区、变更审阅、既有检查、版本与快照服务', boundary:'作者与审查者分开；改动评分定义或实验条件，需要修订方案。'},
    {id:'resources', name:'资源管理', type:'固定服务为主', short:'找到并分配资源',
      description:'统一管理机器、环境、数据与权重位置，按能力和占用处理资源请求。',
      free:'发生资源冲突时，由研究角色决定是否换设备、缩小预算或继续等待。',
      fixed:'查询已有目录、核对所需材料、检查占用、排队、分配和释放资源。',
      input:'已确认任务的能力与预算需求', output:'资源分配记录或明确等待原因', tools:'现有设备目录、Manager options、实验台账、共享队列', boundary:'不根据模型猜测设备可用；实时占用和实际路径以工作台返回为准。'},
    {id:'execution', name:'实验执行', type:'固定服务为主', short:'按方案可靠运行',
      description:'把已确认的方案变成可追踪的检查、训练、推理和评估任务。',
      free:'无法按方案继续时，携带失败证据交回研究或代码角色处理。',
      fixed:'依赖衔接、状态追踪、暂停与恢复、有限重试、日志与产物采集、运行身份记录。',
      input:'源码快照 + 方案 + 标准版本 + 资源分配', output:'真实作业 ID、状态、日志、权重与指标产物', tools:'TrainingManager、InferenceManager、LaunchManager、BenchmarkManager', boundary:'正式作业走已有入口；失败和暂停有明确状态，不靠聊天记忆持续运行。'},
    {id:'evaluation', name:'评估分析', type:'AI 判断角色', short:'用证据回答问题',
      description:'固定程序负责计算指标，分析角色判断可比性、解释证据并给出下一步建议。',
      free:'区分支持、不支持和证据不足；识别混杂因素，设计需要补充的实验。',
      fixed:'调用绑定的评估协议，保留每项指标的来源，关联相同实例和对照条件。',
      input:'实验指标、协议、对照条件与产物', output:'结论、证据、局限与后续任务建议', tools:'既有评估结果、指标比较、证据引用、研究记录归档', boundary:'分析文本不替代指标计算；不同 benchmark 和诊断指标分别展示。'}
  ];

  function sampleRows(t) {
    const names = arms(t).map(a => a[0]);
    if(t.protocol === 'offline') return names.map((name, i) => [name, (t.kind === 'diagnose' ? ['0.0528','0.0581','0.0527'] : ['0.0581','0.0528'])[i] || '0.0528', '6 条示例 Episode']);
    if(t.protocol === 'b1k') return names.map((name, i) => [name, i ? '0.1410' : '0.1240', '0 / 10']);
    return names.map((name, i) => [name, i ? '20%' : '10%', i ? '2 / 10' : '1 / 10']);
  }
  function arms(t) {
    if(t.kind === 'evaluate') return [['选定的已有权重', '复用已登记代码和模型，不新增训练。']];
    if(t.kind === 'diagnose') return [['正确历史', '保持原始历史信息。'], ['空历史', '去掉历史，检查有无历史的影响。'], ['同任务错误历史', '替换历史，检查正确内容是否带来收益。']];
    return [['当前版本 · 对照组', '保留当前方法，作为本轮基线。'], ['候选方法 · 实验组', '只改变本次研究方案指定的因素。']];
  }
  function summaryFor(t) {
    if(t.kind === 'diagnose') return '示例中，正确历史与同任务错误历史的误差接近，尚不能证明模型利用正确历史获得收益。下一轮应检查历史依赖样本与独立 Episode 的配对结果。';
    if(t.kind === 'evaluate') return '示例展示了单个已有权重在指定协议下的表现。它可以作为后续比较的起点，不能单独回答新方法是否更好。';
    return '示例中候选方法的指标优于当前版本，但这些合成数字只用于展示。真实判断仍需核对配对条件、独立样本和不确定性，不能仅凭训练 loss 下结论。';
  }
  function initialState() {
    const standards = {revision:1, defaultSteps:100, maxSteps:3000, retryLimit:1, checkOnChange:true, queueOnAvailable:true, evaluateOnCheckpoint:true};
    const base = () => ({revision:1, protocol:'offline', tasks:[0,34,40], steps:100, device:'auto', checkpoint:'base', mode:'controlled', allowCode:true, phase:0, paused:false, review:'pending', codeRound:1, retries:0, parentId:null, standard:clone(standards), runs:[], events:[]});
    const a = {...base(), id:'R-001', kind:'compare', question:'这次 Memory 修改，是否真的比当前版本更好？', criterion:'相同数据、权重起点和训练预算下比较动作预测误差；保留对照组，不把 loss 下降直接当成方法有效。'};
    a.events.push({time:'10:00', actor:'你', text:'交办研究问题', detail:'等待确认示例方案；未读取真实项目资源。'});
    const b = {...base(), id:'R-002', kind:'diagnose', question:'模型真的利用了正确历史吗？', criterion:'固定当前观测，比较正确、空和同任务错误历史下的动作误差。', phase:5, allowCode:false};
    b.runs.push({id:'SIM-0001', revision:1, status:'archived', attempt:1, snapshot:clone({...b, runs:undefined, events:undefined}), source:'preview://R-002/r1/source', rows:sampleRows(b), finding:summaryFor(b), created:'09:20', finished:'09:25'});
    b.events.push({time:'09:25', actor:'评估分析', text:'示例结论已归档', detail:'合成数据；用于查看结果和共享研究记录。'});
    const c = {...base(), id:'R-003', kind:'evaluate', question:'已有权重在固定 B1K 协议下表现如何？', criterion:'复用已有权重及其源码，记录 Q-score 和完整任务成功回合。', protocol:'b1k', tasks:[0], checkpoint:'existing', allowCode:false, steps:0};
    c.events.push({time:'09:40', actor:'你', text:'准备已有权重评估方案', detail:'真实权重和实例待接入后选择。'});
    return {schema:2, selected:a.id, standards, tasks:[a,b,c], nextTask:4, nextRun:2};
  }

  let state = initialState();
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE));
    if(saved?.schema === 2 && saved.tasks?.length && saved.standards && saved.tasks.every(t => Array.isArray(t.runs) && Number.isInteger(t.phase))) state = saved;
  } catch (_) { /* Browser-local previews also work without persistent storage. */ }
  let view = 'research', tab = 'plan', query = '', knowledgeQuery = '', roleId = 'planning', editId = null, parentId = null, formStandard, toastTimer;
  const current = () => state.tasks.find(t => t.id === state.selected) || state.tasks[0];
  const activeRun = t => t.runs.slice().reverse().find(r => r.revision === t.revision);
  const log = (t, actor, text, detail) => t.events.push({time:clock(), actor, text, detail});
  const benchmark = t => t.protocol === 'libero' ? 'LIBERO · Spatial' : 'BEHAVIOR-1K';
  const checkpoint = t => ({base:'通用预训练权重', behavior:'B1K 专用权重', existing:'已登记训练权重'}[t.checkpoint]);
  function taskStatus(t) {
    const run = activeRun(t);
    if(t.phase === 5) return ['complete', '结论已归档'];
    if(t.paused) return ['paused', '演示已暂停'];
    if(run?.status === 'failed') return ['failed', '需要处理'];
    return [['waiting','待确认方案'], ['waiting','待代码审查'], ['waiting','待分配资源'], ['running','实验演示中'], ['waiting','待分析结果']][t.phase];
  }
  const badge = (type, text) => '<span class="badge '+type+'">'+esc(text)+'</span>';
  function save() {
    try { localStorage.setItem(STORAGE, JSON.stringify(state)); $('save-state').textContent = '演示已保存到此浏览器 · '+clock(); }
    catch (_) { $('save-state').textContent = '浏览器存储不可用，当前页面仍可体验'; }
  }
  function toast(message) {
    clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false;
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3200);
  }
  const facts = rows => '<dl class="fact-grid">'+rows.map(([a,b]) => '<div><dt>'+esc(a)+'</dt><dd>'+esc(b)+'</dd></div>').join('')+'</dl>';

  function renderList() {
    $('task-count').textContent = state.tasks.length;
    const tasks = state.tasks.filter(t => (t.question+' '+t.id).toLowerCase().includes(query));
    $('task-list').innerHTML = tasks.map(t => '<button class="task-card" type="button" data-task="'+esc(t.id)+'" aria-pressed="'+(t.id === current().id)+'"><span class="card-top"><span>'+esc(t.id)+'</span>'+badge(...taskStatus(t))+'</span><h3>'+esc(t.question)+'</h3><span class="card-foot">'+esc(kinds[t.kind])+' · 方案 r'+t.revision+'</span></button>').join('') || '<p class="empty-state">没有匹配的研究问题。</p>';
    const selected = $('task-list').querySelector('[aria-pressed=true]');
    if(selected && matchMedia('(max-width:820px)').matches) $('task-list').scrollLeft = selected.offsetLeft - $('task-list').offsetLeft;
  }
  function actionInfo(t) {
    const run = activeRun(t);
    if(t.phase === 5) return {owner:'评估分析 → 研究规划', title:'这轮已经回答到哪里？', text:'查看示例结论与证据，再决定是否交办下一轮。', button:'交办后续实验', action:'followup'};
    if(t.paused) return {owner:'实验执行服务 · 已暂停', title:'当前演示已暂停', text:'继续当前方案，或调整方案创建新修订；已有运行记录会保留。', button:'继续演示运行', action:'resume'};
    if(run?.status === 'failed') return {owner:'实验执行 → 代码工程', title:'检查失败，流程已停止', text:'示例：输出出现非有限数值。失败证据已保留；本方案已重试 '+t.retries+' / '+t.standard.retryLimit+' 次。', button:t.retries < t.standard.retryLimit ? '模拟修复后重试' : '调整方案后再运行', action:t.retries < t.standard.retryLimit ? 'retry' : 'edit', issue:true};
    if(t.phase === 0) return {owner:'研究规划 → 你', title:'请确认这次怎么验证', text:'先看下面的实验分支、控制条件和预算。方案确认后才交给代码与执行环节。', button:'确认方案 · 演示', action:'approve-plan'};
    if(t.phase === 1 && t.review === 'changes') return {owner:'独立代码审查 → 代码工程', title:'审查要求补充修改', text:'示例审查意见：补充空历史检查，明确候选方法只改变本次研究因素。', button:'查看修订后的示例', action:'revise-code', issue:true};
    if(t.phase === 1) return {owner:'代码工程 → 独立代码审查', title:'检查实现是否符合已确认方案', text:'在“代码与执行”中查看示例变更和检查项；可以通过，也可以要求修改。', button:'通过代码审查 · 演示', action:'approve-code'};
    if(t.phase === 2) return {owner:'资源管理 → 实验执行', title:'为已确认的实验匹配资源', text:'复用工作台的设备、数据和模型目录。当前仅演示分配，不读取真实 GPU 占用。', button:t.mode === 'staged' ? '确认资源分配 · 演示' : '模拟分配资源', action:'allocate'};
    if(t.phase === 3) return {owner:'实验执行服务', title:t.kind === 'evaluate' ? '按既定协议评估已有权重' : '按固定流程运行配对实验', text:'这里用按钮模拟后台完成。也可以体验暂停或检查失败，不会启动真实作业。', button:'模拟完成实验', action:'complete-run'};
    return {owner:'固定评估程序 → 评估分析', title:'用结果回答最初的问题', text:'查看示例指标、结论和运行身份。确认后归档到所有角色共享的研究记录。', button:'归档示例结论', action:'archive'};
  }
  function renderTask() {
    const t = current(), info = actionInfo(t);
    $('task-heading').innerHTML = '<div class="task-title-block"><div class="task-eyebrow"><span>'+esc(t.id)+' · '+esc(benchmark(t))+' · 方案 r'+t.revision+'</span>'+badge(...taskStatus(t))+'</div><h2>'+esc(t.question)+'</h2><p class="criterion">希望得到的证据：'+esc(t.criterion)+'</p><div class="task-tools"><button type="button" class="text-button" data-action="edit">调整方案</button><button type="button" class="text-button" data-action="export">导出方案</button><span>演示任务 · '+esc(stdName(t.standard.revision))+'</span></div></div>';
    $('stage-track').innerHTML = stageNames.map((name,i) => '<li class="'+(i === 1 && !t.allowCode ? 'skipped' : i < t.phase ? 'done' : i === t.phase ? 'current' : '')+'"'+(i === t.phase?' aria-current="step"':'')+'><b>'+(i === 1 && !t.allowCode ? '—' : i < t.phase ? '✓' : i+1)+'</b><span>'+name+'</span></li>').join('');
    $('next-action').className = 'next-action'+(info.issue?' issue':'');
    $('next-action').innerHTML = '<div><div class="owner">当前负责：'+esc(info.owner)+'</div><h3>'+esc(info.title)+'</h3><p>'+esc(info.text)+'</p></div><button id="advance-task" class="primary" type="button" data-action="'+info.action+'">'+esc(info.button)+'</button>';
    renderPlan(t); renderWork(t); renderResults(t); showTab(tab);
  }
  function renderPlan(t) {
    const variants = arms(t);
    const related = state.tasks.filter(other => other.id !== t.id && other.protocol === t.protocol && other.phase === 5).slice(0,2);
    const parent = state.tasks.find(other => other.id === t.parentId);
    $('panel-plan').innerHTML = '<section class="block"><div class="section-heading"><h3>本轮实验方案</h3><span class="tag">'+esc(kinds[t.kind])+'</span></div><p>只改变研究问题指定的因素，其余条件保持一致。以下为按模板组织的示例方案，尚未调用 AI。</p><div class="branch-grid" style="margin-top:14px">'+variants.map(([name,description],i) => '<div class="branch-card"><span class="branch-label">分支 '+String.fromCharCode(65+i)+'</span><strong>'+esc(name)+'</strong><p>'+esc(description)+'</p></div>').join('')+'</div></section><section class="block"><h3>对照条件与预算</h3>'+facts([
      ['判断依据',t.criterion], ['验证协议',protocolNames[t.protocol]], ['数据与评估实例','各分支使用相同划分、实例和 seed 42；接入后绑定真实 ID'], ['权重起点',checkpoint(t)+' · 实际版本待核对'], ['任务集合','Task '+t.tasks.join(' / ')], ['训练预算',t.kind === 'evaluate' ? '复用已有权重，不新增训练' : variants.length+' 组 × '+t.steps.toLocaleString()+' 步 = '+(variants.length*t.steps).toLocaleString()+' 步'], ['工作范围',t.kind === 'evaluate' ? '仅评估，不修改代码或训练' : (t.allowCode?'实现与审查代码、':'复用已有代码、')+'短训练与固定评估'], ['人工介入',t.mode === 'controlled'?'确认方案和代码后按预算执行':'每次交接由我确认']
    ])+'</section><section class="block"><h3>先复用工作台已有能力</h3><div class="reuse-list"><div class="reuse-row"><strong>代码与检查</strong><span>当前登记源码、版本与开发快测</span><small>真实材料待核对</small></div><div class="reuse-row"><strong>资源与模型</strong><span>设备目录、数据集、已登记权重</span><small>资源服务待接入</small></div><div class="reuse-row"><strong>训练与评估</strong><span>现有 Manager、参数检查与任务入口</span><a href="'+workbenchUrl('/')+'">查看工作台 ↗</a></div></div></section>'+
      (related.length || parent ? '<section class="block"><h3>'+(parent?'这轮从哪里来':'同协议已有示例')+'</h3>'+[...(parent?[parent]:[]),...related.filter(r=>r.id!==parent?.id)].map(r=>'<p><button type="button" class="text-button" data-task="'+esc(r.id)+'">'+esc(r.id)+' · '+esc(r.question)+' →</button></p>').join('')+'<p class="section-note">正式接入时，规划角色会检索已有研究记录，再决定是否补实验。</p></section>' : '')+
      '<details class="block"><summary>方案版本与共享标准</summary>'+facts([['方案修订','r'+t.revision],['绑定标准',stdName(t.standard.revision)],['每分支预算上限',t.standard.maxSteps.toLocaleString()+' 步'],['失败重试上限',t.standard.retryLimit+' 次']])+'<p class="section-note">调整本方案会创建新修订。修改共享默认标准不会回写当前方案和已有运行。</p></details>';
  }
  function renderWork(t) {
    const run = activeRun(t), reviewed = t.phase > 1;
    const review = t.allowCode ? '<section class="block review-card"><div class="section-heading"><h3>代码实现与独立审查</h3>'+badge(reviewed?'complete':'waiting', reviewed?'示例审查通过':t.review==='changes'?'需要修改':'等待示例审查')+'</div><p>本任务的实现必须对应已确认的问题：“'+esc(t.question)+'”。这里展示审查卡片样式，不对应真实项目修改。</p><div class="review-meta"><span>实验代码副本 · 示例</span><span>审查轮次 '+t.codeRound+'</span><span>代码作者与审查者分开</span></div><pre aria-label="示例配置差异"><span class="diff-minus">− method_variant: current</span>\n<span class="diff-plus">+ method_variant: candidate</span>\n  data_split: same_as_control\n  evaluation_protocol: '+esc(t.protocol)+'</pre><ul class="plain-list"><li>变更只影响本轮方案指定的因素。</li><li>沿用已有网络回归和短窗检查入口。</li><li>运行前保存源码与配置快照。</li></ul>'+(t.review==='changes'?'<div class="feedback">示例意见：补充空历史检查，并明确候选方法与对照组的差异。点击上方“查看修订后的示例”继续。</div>':t.phase===1?'<div class="inline-actions"><button type="button" data-action="request-changes">要求补充修改 · 演示</button><span class="muted">通过审查使用上方主按钮</span></div>':'')+'</section>' : '<section class="block review-card"><h3>复用已有代码与权重</h3><p style="margin-top:9px">本任务跳过代码实现和审查。执行前仍需核对所选权重、对应源码与推理服务身份；当前尚未绑定真实材料。</p></section>';
    const executionNames = ['检查源码、配置与权重身份', ...(t.kind==='evaluate'?[]:['运行既有检查与短训练']), '按固定协议计算指标', '收集产物并交给评估分析'];
    const executionStatus = run?.status === 'failed' ? ['failed','失败，等待处理'] : run?.status === 'paused' ? ['paused','已暂停'] : ['running','演示执行中'];
    $('panel-work').innerHTML = review+'<section class="block"><div class="section-heading"><h3>资源分配</h3><span class="muted small">示意，未查询真实占用</span></div><div class="resource-grid"><div class="resource-item"><span>训练 / 推理候选</span><strong>'+esc(t.device==='local'?'本机 GPU':t.device==='a100'?'A100':'由能力目录匹配')+'</strong><p>'+esc(t.kind==='evaluate'?'仅需要推理，不创建训练任务':'与手动训练共用资源目录和队列')+'</p></div><div class="resource-item"><span>验证方式</span><strong>'+esc(t.protocol==='offline'?'离线诊断环境':'对应 benchmark 仿真')+'</strong><p>'+esc(t.protocol==='offline'?'不申请闭环仿真任务':'仿真与推理按各自能力匹配')+'</p></div><div class="resource-item"><span>当前分配</span><strong>'+(run?'已创建模拟分配':'等待方案与代码确认')+'</strong><p>数据、环境与权重位置待真实接入</p></div></div></section><section class="block"><div class="section-heading"><h3>固定执行流程</h3>'+badge(run?executionStatus[0]:'waiting',run ? run.status==='archived'||run.status==='completed'?'示例执行完成':executionStatus[1]:'尚未分配')+'</div><ol class="execution-steps">'+executionNames.map((name,i)=>'<li><span class="step-dot"></span><span>'+name+'</span>'+badge(run?.status==='archived'||run?.status==='completed'?'complete':'',!run?'待执行':run.status==='archived'||run.status==='completed'?'演示完成':i===0?executionStatus[1]:'等待前序')+'</li>').join('')+'</ol>'+(t.phase===3 && run?.status==='running'?'<div class="inline-actions"><button type="button" data-action="pause">暂停演示运行</button><button type="button" data-action="fail">模拟检查失败</button><span class="muted">不会操作真实进程</span></div>':'')+'</section>'+
      '<details class="block"><summary>查看运行身份与历史尝试（'+t.runs.length+' 次）</summary>'+(t.runs.length ? t.runs.slice().reverse().map(r=>'<div class="execution-card" style="margin-top:12px"><div class="section-heading"><h3>'+esc(r.id)+'</h3>'+badge(r.status==='failed'?'failed':r.status==='archived'?'complete':'',runStatusNames[r.status])+'</div>'+facts([['方案修订','r'+r.revision],['标准快照',stdName(r.snapshot.standard.revision)],['训练预算',r.snapshot.kind==='evaluate'?'不新增训练':r.snapshot.steps+' 步 / 组'],['源码身份',r.source],['协议',protocolNames[r.snapshot.protocol]],['尝试序号',String(r.attempt)]])+'</div>').join('') : '<p class="section-note">模拟分配资源后创建运行记录；记录将保留当时的方案与标准。</p>')+'</details><details class="block"><summary>查看交接记录（'+t.events.length+' 条）</summary><div class="history-list">'+t.events.slice().reverse().map(e=>'<div class="history-row"><time>'+esc(e.time)+'</time><div><strong>'+esc(e.actor)+' · '+esc(e.text)+'</strong><p>'+esc(e.detail)+'</p></div></div>').join('')+'</div></details>';
  }
  function renderResults(t) {
    const run = activeRun(t);
    if(!run?.rows) {
      $('panel-results').innerHTML = '<div class="empty-state"><h3>这轮还没有结果</h3><p style="margin-top:10px">按上方提示确认方案并推进演示，完成后在这里看指标、结论和证据。</p><button type="button" class="text-button" data-task="R-002" style="margin-top:17px">先看一个已完成的示例 →</button></div>';
      return;
    }
    $('panel-results').innerHTML = '<p class="result-intro">以下为合成数据，只用于评审结果展示，不代表真实模型效果。</p><section class="block"><div class="section-heading"><h3>实验结果对比</h3><span class="tag">'+esc(protocolNames[t.protocol])+'</span></div><div class="table-scroll"><table><thead><tr><th>分支 / 条件</th><th>'+esc(metricNames[t.protocol])+'</th><th>'+esc(t.protocol==='offline'?'独立样本范围':'完整成功回合')+'</th></tr></thead><tbody>'+run.rows.map(row=>'<tr>'+row.map(value=>'<td>'+esc(value)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div></section><section class="block conclusion"><h3>这能回答最初的问题吗？</h3><p>'+esc(run.finding || summaryFor(t))+'</p></section><section class="block"><h3>证据应当能追溯到哪里</h3>'+facts([['实验身份',run.id+' · 方案 r'+run.revision],['源码与配置',run.source],['固定协议',protocolNames[run.snapshot.protocol]+' · '+stdName(run.snapshot.standard.revision)],['数据与权重',checkpoint(t)+'；真实 ID 待接入后绑定']])+'<p class="section-note">真实接入后，这里链接训练记录、指标文件、checkpoint、日志及适用的评估视频。本原型没有创建这些实际产物。</p></section><div class="followup-strip"><p>把这轮的问题与证据带入下一次实验，保留任务关系。</p><button type="button" data-action="followup">交办后续实验 →</button></div>';
  }

  function renderTeam() {
    $('role-picker').innerHTML = roles.map((r,i)=>'<button type="button" class="role-card" data-role="'+r.id+'" aria-pressed="'+(r.id===roleId)+'"><span class="role-number">0'+(i+1)+'</span><strong>'+esc(r.name)+'</strong><small>'+esc(r.short)+'</small></button>').join('');
    const r = roles.find(item=>item.id===roleId);
    $('role-detail').innerHTML = '<div class="section-heading"><h2>'+esc(r.name)+'</h2><span class="tag">'+esc(r.type)+'</span></div><p>'+esc(r.description)+'</p><div class="role-boundaries"><div><h3>需要自由判断</h3><p>'+esc(r.free)+'</p></div><div><h3>建立标准后固定执行</h3><p>'+esc(r.fixed)+'</p></div></div>'+facts([['接收什么',r.input],['交付什么',r.output],['使用哪些能力',r.tools],['职责边界',r.boundary]])+'<p class="section-note">以上为计划中的职责与工具边界，当前尚未连接真实 Agent 或执行服务。</p>';
  }
  function renderStandards() {
    const s=state.standards, form=$('standards-form');
    $('standard-version').textContent=stdName(s.revision)+' · 本地示例';
    for(const name of ['defaultSteps','maxSteps','retryLimit']) form.elements[name].value=s[name];
    for(const name of ['checkOnChange','queueOnAvailable','evaluateOnCheckpoint']) form.elements[name].checked=s[name];
  }
  function renderKnowledge() {
    const records = state.tasks.flatMap(t=>t.runs.filter(r=>['archived','failed'].includes(r.status)).map(run=>({t,run})));
    const filtered=records.filter(({t,run})=>(t.question+' '+(run.finding||'检查失败')+' '+run.id).toLowerCase().includes(knowledgeQuery));
    $('knowledge-list').innerHTML=filtered.map(({t,run})=>'<article class="finding-card"><div class="section-heading"><h2>'+esc(run.snapshot.question)+'</h2>'+badge(run.status==='failed'?'failed':'complete',run.status==='failed'?'失败记录':'示例结论')+'</div><p>'+esc(run.finding||'模拟检查发现非有限数值，本次尝试停止。保留失败证据，供代码工程定位；原型未运行实际检查。')+'</p><div class="finding-footer"><span>'+esc(run.id)+' · 方案 r'+run.revision+'</span><span>'+esc(protocolNames[run.snapshot.protocol])+'</span><span>'+esc(stdName(run.snapshot.standard.revision))+'</span><button type="button" class="text-button" data-record-task="'+esc(t.id)+'">查看所属研究 →</button></div></article>').join('') || '<div class="surface empty-state">没有匹配记录。实验结论归档后会出现在这里，失败尝试也会保留。</div>';
  }
  function showView(next) {
    view=next;
    document.querySelectorAll('[data-view]').forEach(b=>{if(b.dataset.view===next)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
    for(const name of ['research','team','standards','knowledge']) $('view-'+name).hidden=name!==next;
    if(next==='team')renderTeam();
    if(next==='standards')renderStandards();
    if(next==='knowledge')renderKnowledge();
  }
  function showTab(next, focus=false) {
    tab=next;
    document.querySelectorAll('[data-tab]').forEach(b=>{const on=b.dataset.tab===next;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;if(on&&focus)b.focus();});
    for(const name of ['plan','work','results']) $('panel-'+name).hidden=name!==next;
  }
  function render() { renderList(); renderTask(); if(view==='knowledge')renderKnowledge(); }
  function workbenchUrl(path) { return location.port==='8772' ? location.protocol+'//'+location.hostname+':8765'+path : path; }

  function startRun(t) {
    const {runs,events,...plan}=t;
    const run={id:'SIM-'+String(state.nextRun++).padStart(4,'0'), revision:t.revision, status:'running', attempt:t.retries+1, snapshot:clone(plan), source:'preview://'+t.id+'/r'+t.revision+'/code-'+t.codeRound, rows:null, finding:null, created:clock()};
    t.runs.push(run); t.phase=3; t.paused=false;
    log(t,'资源管理 → 实验执行','创建模拟分配与运行记录',run.id+' · 冻结方案 r'+t.revision+' / '+stdName(t.standard.revision));
  }
  function act(action) {
    const t=current(), run=activeRun(t);
    if(action==='edit'){openForm(t);return;}
    if(action==='followup'){openForm(t,true);return;}
    if(action==='export'){exportTask(t);return;}
    if(action==='approve-plan' && t.phase===0){t.phase=t.allowCode?1:2;tab='work';log(t,'你 → '+(t.allowCode?'代码工程':'资源管理'),'确认方案 r'+t.revision,'保留对照条件与预算；仅推进演示。');}
    else if(action==='approve-code' && t.phase===1 && t.review!=='changes'){t.review='approved';t.phase=2;log(t,'独立代码审查 → 资源管理','示例代码审查通过','后续运行关联代码版本，不修改真实文件。');}
    else if(action==='request-changes' && t.phase===1){t.review='changes';tab='work';log(t,'独立代码审查 → 代码工程','要求补充修改','示例意见：补充空历史检查，明确变化因素。');}
    else if(action==='revise-code' && t.phase===1 && t.review==='changes'){t.review='pending';t.codeRound++;tab='work';log(t,'代码工程 → 独立代码审查','提交修订示例','审查轮次 '+t.codeRound+'；等待独立确认。');}
    else if(action==='allocate' && t.phase===2){startRun(t);tab='work';}
    else if(action==='pause' && t.phase===3 && run?.status==='running'){t.paused=true;run.status='paused';log(t,'你 → 实验执行','暂停演示运行',run.id+' 状态保留。');}
    else if(action==='resume' && t.paused && run?.status==='paused'){t.paused=false;run.status='running';log(t,'你 → 实验执行','继续演示运行',run.id+' 沿用原快照。');}
    else if(action==='fail' && t.phase===3 && run?.status==='running'){run.status='failed';run.finished=clock();log(t,'实验执行 → 代码工程','模拟检查失败，停止推进','示例：输出存在非有限数值；没有实际运行检查。');}
    else if(action==='retry' && t.phase===3 && run?.status==='failed' && t.retries<t.standard.retryLimit){t.retries++;startRun(t);tab='work';log(t,'代码工程 → 实验执行','模拟修复后重试','新的尝试保留旧失败记录；不增加方案预算。');}
    else if(action==='complete-run' && t.phase===3 && run?.status==='running'){run.status='completed';run.rows=sampleRows(t);run.finished=clock();t.phase=4;tab='results';log(t,'固定评估 → 评估分析','生成示例指标','全部是合成数据，不属于真实实验结果。');}
    else if(action==='archive' && t.phase===4 && run?.status==='completed'){run.finding=summaryFor(t);run.status='archived';t.phase=5;tab='results';log(t,'评估分析 → 共享研究记录','归档示例结论',run.id+' 的问题、证据与局限一并保留。');}
    else return;
    save();render();
    toast(action==='archive'?'示例结论已保存到研究记录':'已更新演示状态，未调用真实执行接口');
  }

  const form=$('task-form');
  function configureKind(keepSettings=false) {
    const kind=form.elements.kind.value, evaluating=kind==='evaluate';
    $('steps-field').hidden=evaluating; form.elements.steps.disabled=evaluating;
    form.elements.allowCode.disabled=evaluating;
    if(evaluating){form.elements.allowCode.checked=false;form.elements.steps.value=0;form.elements.checkpoint.value='existing';if(!keepSettings)form.elements.protocol.value='b1k';}
    else if(!keepSettings){form.elements.steps.value=formStandard.defaultSteps;form.elements.allowCode.checked=true;form.elements.checkpoint.value='base';if(kind==='diagnose')form.elements.protocol.value='offline';}
    form.elements.protocol.disabled=kind==='diagnose';
    if(kind==='diagnose')form.elements.protocol.value='offline';
  }
  function openForm(task=null, followup=false) {
    form.reset(); editId=task&&!followup?task.id:null;parentId=followup?task.id:null;
    formStandard=clone(editId?task.standard:state.standards);
    $('form-error').hidden=true;$('advanced-settings').open=false;
    $('dialog-title').textContent=editId?'调整方案 · 新建修订':followup?'交办后续研究任务':'交办一个研究问题';
    form.querySelector('[type=submit]').textContent=editId?'保存新修订，重新确认':'准备示例方案';
    form.elements.steps.max=formStandard.maxSteps;
    if(task){for(const name of ['question','criterion','kind','protocol','steps','device','checkpoint','mode'])form.elements[name].value=task[name];form.elements.tasks.value=task.tasks.join(', ');form.elements.allowCode.checked=task.allowCode;if(followup)form.elements.question.value='后续研究：'+task.question;}
    else form.elements.steps.value=formStandard.defaultSteps;
    configureKind(true);
    $('form-standard').textContent=stdName(formStandard.revision)+(editId?' · 保留当前方案的标准':' · 使用当前共享标准');
    $('task-dialog').showModal();
  }
  form.addEventListener('change',e=>{
    if(e.target.name==='kind')configureKind();
    if(e.target.name==='protocol')form.elements.tasks.value=e.target.value==='libero'?'0':'0, 34, 40';
  });
  form.addEventListener('invalid',()=>{$('advanced-settings').open=true;},true);
  form.addEventListener('submit',e=>{
    e.preventDefault();
    const data=new FormData(form), question=data.get('question').trim(), criterion=data.get('criterion').trim();
    const kind=data.get('kind'), protocol=kind==='diagnose'?'offline':data.get('protocol'), tasks=[...new Set(data.get('tasks').split(',').map(v=>Number(v.trim())))];
    const maximum=protocol==='libero'?9:49, steps=kind==='evaluate'?0:Number(data.get('steps'));
    const error=!question||!criterion?'请说明研究问题和希望看到的证据。':tasks.some(id=>!Number.isInteger(id)||id<0||id>maximum)?'任务编号须在 0–'+maximum+' 之间。':kind!=='evaluate'&&(!Number.isInteger(steps)||steps<1||steps>formStandard.maxSteps)?'每分支步数须在 1–'+formStandard.maxSteps+' 之间。':'';
    if(error){$('form-error').textContent=error;$('form-error').hidden=false;$('advanced-settings').open=true;return;}
    const fields={question,criterion,kind,protocol,tasks,steps,device:data.get('device'),checkpoint:data.get('checkpoint'),mode:data.get('mode'),allowCode:kind!=='evaluate'&&data.has('allowCode')};
    let t=editId?state.tasks.find(item=>item.id===editId):null;
    if(t){const old=activeRun(t);if(old?.status==='running'){old.status='paused';log(t,'你 → 实验执行','调整方案前暂停模拟运行',old.id+' 保留原方案和标准快照。');}Object.assign(t,fields);t.revision++;}
    else{t={...fields,id:'R-'+String(state.nextTask++).padStart(3,'0'),revision:1,parentId,standard:clone(formStandard),runs:[],events:[]};state.tasks.unshift(t);}
    Object.assign(t,{phase:0,paused:false,review:'pending',codeRound:1,retries:0});
    log(t,'你 → 研究规划',editId?'创建方案修订 r'+t.revision:'交办研究问题',parentId?'继承自 '+parentId+'；等待确认新方案。':'按模板组织示例方案，等待确认。');
    state.selected=t.id;query='';$('task-search').value='';tab='plan';showView('research');save();render();$('task-dialog').close();toast('方案已准备，请查看分支、条件与预算');
  });
  $('standards-form').addEventListener('submit',e=>{
    e.preventDefault();const data=new FormData(e.currentTarget), defaults=Number(data.get('defaultSteps')), maximum=Number(data.get('maxSteps'));
    if(defaults>maximum){$('standard-error').textContent='默认步数不能超过每分支预算上限。';$('standard-error').hidden=false;return;}
    $('standard-error').hidden=true;
    state.standards={revision:state.standards.revision+1,defaultSteps:defaults,maxSteps:maximum,retryLimit:Number(data.get('retryLimit')),checkOnChange:data.has('checkOnChange'),queueOnAvailable:data.has('queueOnAvailable'),evaluateOnCheckpoint:data.has('evaluateOnCheckpoint')};
    save();renderStandards();$('standard-saved').textContent='已保存 '+stdName(state.standards.revision)+'，仅影响新方案';toast('新标准已保存，已有方案和运行快照保持原版本');
  });
  function exportTask(t) {
    const blob=new Blob([JSON.stringify({schema:'ai-research-workbench-preview-v2',demo:true,executable:false,task:t},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=t.id+'-r'+t.revision+'-preview.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已导出演示方案与记录');
  }
  function selectTask(id, result=false) {
    const t=state.tasks.find(item=>item.id===id);if(!t)return;
    state.selected=id;tab=result||t.phase>=4?'results':t.phase===0?'plan':'work';showView('research');save();render();
  }
  document.addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b)return;
    if(b.dataset.view)showView(b.dataset.view);
    if(b.dataset.tab)showTab(b.dataset.tab);
    if(b.dataset.task)selectTask(b.dataset.task);
    if(b.dataset.recordTask)selectTask(b.dataset.recordTask,true);
    if(b.dataset.role){roleId=b.dataset.role;renderTeam();$('role-picker').querySelector('[data-role="'+roleId+'"]').focus();}
    if(b.dataset.action)act(b.dataset.action);
    if(b.dataset.close)$(b.dataset.close).close();
  });
  $('new-task-button').onclick=()=>openForm();
  $('task-search').oninput=e=>{query=e.target.value.toLowerCase();renderList();};
  $('knowledge-search').oninput=e=>{knowledgeQuery=e.target.value.toLowerCase();renderKnowledge();};
  $('reset-button').onclick=()=>$('reset-dialog').showModal();
  $('confirm-reset').onclick=()=>{state=initialState();query='';knowledgeQuery='';tab='plan';$('task-search').value='';$('knowledge-search').value='';showView('research');save();render();$('reset-dialog').close();toast('已恢复初始研究示例');};
  document.querySelector('.detail-tabs').addEventListener('keydown',e=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
    const names=['plan','work','results'],i=names.indexOf(tab),next=e.key==='Home'?0:e.key==='End'?2:(i+(e.key==='ArrowRight'?1:2))%3;
    e.preventDefault();showTab(names[next],true);
  });
  document.querySelectorAll('[data-workbench-link]').forEach(a=>{a.href=workbenchUrl(a.getAttribute('href'));});
  render();
})();
