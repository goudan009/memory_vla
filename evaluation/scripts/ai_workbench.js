/* Human research control console. All agents, resources and runs are simulated. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const clone = v => JSON.parse(JSON.stringify(v));
  const KEY = 'memory-vla.research-console.preview.v3';
  const time = () => new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false});
  const tabs = ['plan','code','execution','results','records'];
  const stages = [
    {name:'研究方案',owner:'研究规划 Agent',input:'研究目标、已有结论与资源目录',output:'实验分支、控制条件与预算',tab:'plan'},
    {name:'代码审查',owner:'代码工程 / 独立审查',input:'已确认方案与当前源码',output:'变更集、检查报告与源码快照',tab:'code'},
    {name:'资源准备',owner:'资源管理服务',input:'资源需求与已审查代码',output:'设备、数据、环境与权重分配',tab:'execution'},
    {name:'实验运行',owner:'实验执行服务',input:'方案快照、代码与资源分配',output:'训练记录、checkpoint 与日志',tab:'execution'},
    {name:'固定评估',owner:'评估程序',input:'权重、固定协议与评估实例',output:'指标文件与配对结果',tab:'results'},
    {name:'结论归档',owner:'评估分析 Agent',input:'可追溯的指标与实验产物',output:'研究结论、局限与下一轮建议',tab:'results'}
  ];
  const protocols = {offline:'离线动作诊断',b1k:'B1K 闭环评估',libero:'LIBERO · Spatial'};
  const metrics = {offline:'动作 MAE₈ ↓',b1k:'B1K Q-score ↑',libero:'LIBERO 成功率 ↑'};
  const materials = [
    {id:'code',name:'当前项目源码',detail:'V1 / 当前登记版本 · 示例',type:'代码'},
    {id:'candidate',name:'候选实现版本',detail:'实验工作区 / 待审查变更 · 示例',type:'代码'},
    {id:'dataset',name:'Task0 / 34 / 40 数据',detail:'已有训练与验证划分 · 示例',type:'数据'},
    {id:'base',name:'GR00T 预训练权重',detail:'已有模型目录 / 待绑定身份 · 示例',type:'权重'},
    {id:'checkpoint',name:'已登记训练权重',detail:'实验台账 / 待选择 checkpoint · 示例',type:'权重'},
    {id:'findings',name:'历史利用诊断记录',detail:'已有研究结论与失败经验 · 示例',type:'记录'}
  ];
  const material = id => materials.find(m=>m.id===id);
  const kindNames = {compare:'方法改进',diagnose:'原因诊断',evaluate:'已有权重评估'};
  const fixed = {revision:1,maxSteps:3000,retries:1,seed:42};
  function branches(t) {
    return t.kind==='evaluate' ? ['已有权重'] : t.kind==='diagnose' ? ['正确历史','空历史','同任务错误历史'] : ['当前版本 · 对照','候选版本 · 实验'];
  }
  const skipped = (t,i) => t.kind==='evaluate' && (i===1 || i===3);
  function freshTask(id, fields={}) {
    return {id,title:'验证 Memory 修改的动作预测收益',criterion:'固定数据划分、初始化、随机种子和训练预算，比较独立 Episode 上的配对动作误差。',kind:'compare',protocol:'offline',tasks:[0,34,40],steps:100,revision:1,phase:0,review:'pending',codeRound:1,mode:'controlled',device:'auto',materials:['code','candidate','dataset','base'],standard:clone(fixed),runs:[],events:[],notes:[],retries:0,paused:false,issue:null,parent:null,...fields};
  }
  const log = (t,actor,action,detail) => t.events.push({time:time(),actor,action,detail});
  function snapshot(t) {
    const {runs,events,notes,...rest}=t;return clone(rest);
  }
  function makeRun(t,id) {
    return {id,revision:t.revision,snapshot:snapshot(t),status:'running',stage:t.kind==='evaluate'?'evaluation':'training',source:'preview://'+t.id+'/r'+t.revision+'/source-'+t.codeRound,rows:null,finding:null,attempt:t.retries+1,started:time(),allocation:{training:'A100 · 示例分配',evaluation:t.protocol==='offline'?'离线诊断环境':'对应仿真环境'},failure:null};
  }
  function rowsFor(t) {
    return branches(t).map((name,i)=>[name,t.protocol==='offline'?(t.kind==='diagnose'?['0.0528','0.0581','0.0527'][i]:['0.0581','0.0528'][i]):t.protocol==='b1k'?(i?'0.1410':'0.1240'):(i?'20%':'10%'),t.protocol==='offline'?'6 条示例 Episode':t.protocol==='b1k'?'0 / 10':(i?'2 / 10':'1 / 10')]);
  }
  const findingFor = t => t.kind==='diagnose'?'正确历史与同任务错误历史的示例误差接近，证据不足以支持“正确历史带来可靠收益”。下一步应验证独立的历史依赖样本。':t.kind==='evaluate'?'已形成所选权重的固定协议评估记录，可作为后续比较的起点。单组评估不支持新方法优于旧方法的结论。':'候选版本的示例指标优于对照版本，但需要进一步核对配对样本和不确定性。当前数字均为合成数据，不能认定真实方法有效。';
  function initialState() {
    const a=freshTask('R-001');
    a.events=[{time:'09:32',actor:'研究规划 Agent',action:'提交方案 r1',detail:'完成分支、控制条件、预算与已有入口的示例匹配。'}];
    const b=freshTask('R-002',{title:'当前训练权重的 B1K 闭环复测',criterion:'复用已有权重和固定任务实例，分别记录 Q-score 与完整成功回合。',kind:'evaluate',protocol:'b1k',tasks:[0],steps:0,phase:4,materials:['code','checkpoint'],review:'skipped'});
    b.runs=[makeRun(b,'SIM-0001')];
    b.events=[{time:'09:05',actor:'你',action:'确认评估方案',detail:'沿用已有权重与对应源码，跳过训练。'},{time:'09:07',actor:'资源管理服务',action:'分配模拟推理与仿真环境',detail:'未查询真实设备占用。'},{time:'09:10',actor:'评估程序',action:'执行固定协议评估',detail:'等待示例指标返回；后台尚未接入。'}];
    const c=freshTask('R-003',{title:'正确历史是否优于错误历史',criterion:'固定当前观测，比较正确、空和同任务错误历史的动作误差。',kind:'diagnose',phase:6,review:'approved',materials:['code','dataset','base','findings']});
    const r=makeRun(c,'SIM-0002');Object.assign(r,{status:'archived',stage:'complete',rows:rowsFor(c),finding:findingFor(c)});c.runs=[r];
    c.events=[{time:'08:40',actor:'评估分析 Agent',action:'提交证据与结论',detail:'保留示例负结果与研究局限。'},{time:'08:45',actor:'你',action:'确认归档',detail:'研究记录关联源码、配置、指标和结论。'}];
    return {schema:3,selected:a.id,tasks:[a,b,c],nextTask:4,nextRun:3};
  }
  let state=initialState();
  try { const saved=JSON.parse(localStorage.getItem(KEY));if(saved?.schema===3 && saved.tasks?.length && saved.tasks.every(t=>Array.isArray(t.runs)&&Array.isArray(t.events)&&Array.isArray(t.notes))) state=saved; } catch (_) { /* Local previews can run without storage. */ }
  let page='overview',tab='plan',filter='all',query='',newMaterials=['code','dataset','base'],materialTarget='new',toastTimer;
  const current=()=>state.tasks.find(t=>t.id===state.selected)||state.tasks[0];
  const runFor=t=>t.runs.slice().reverse().find(r=>r.revision===t.revision);
  function status(t) {
    if(t.phase===6)return ['complete','已归档'];
    if(t.issue)return ['attention',t.issue==='resource'?'资源等待':'执行异常'];
    if(t.paused)return ['paused','已暂停'];
    return [['attention','待方案审核'],['attention',t.review==='changes'?'代码待修订':'待代码审查'],['attention','待资源确认'],['running','训练运行中'],['running','固定评估中'],['attention','待结论确认']][t.phase];
  }
  const pill=(type,label)=>'<span class="status-pill '+type+'"><i class="status-dot '+type+'"></i>'+esc(label)+'</span>';
  const needsAttention=t=>status(t)[0]==='attention';
  const stepProgress=t=>Math.round(stages.filter((s,i)=>!skipped(t,i)&&i<t.phase).length/stages.filter((s,i)=>!skipped(t,i)).length*100);
  function save(){try{localStorage.setItem(KEY,JSON.stringify(state));$('save-state').textContent='演示状态已保存 · '+time();}catch(_){$('save-state').textContent='本次操作仅保留在当前页面';}}
  function toast(s){clearTimeout(toastTimer);$('toast').textContent=s;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,3000);}
  const facts=items=>'<dl class="detail-facts">'+items.map(([a,b])=>'<div><dt>'+esc(a)+'</dt><dd>'+esc(b)+'</dd></div>').join('')+'</dl>';
  const block=(title,body,meta='')=>'<section class="surface block"><div class="section-heading"><h2>'+esc(title)+'</h2>'+(meta?'<span>'+esc(meta)+'</span>':'')+'</div>'+body+'</section>';
  const budget=t=>t.kind==='evaluate'?'不新增训练':branches(t).length+' 组 × '+t.steps.toLocaleString()+' 步';
  function showPage(next){page=next;$('project-workspace').hidden=next!=='overview';$('empty-workspace').hidden=next!=='new';$('task-workspace').hidden=next!=='task';$('breadcrumb-current').textContent={overview:'研究总览',new:'新建研究任务',task:current().id}[next];if(next==='overview')$('project-overview').setAttribute('aria-current','page');else $('project-overview').removeAttribute('aria-current');}
  function renderList(){
    $('task-count').textContent=state.tasks.length;$('attention-count').textContent=state.tasks.filter(needsAttention).length;
    const tasks=state.tasks.filter(t=>(filter==='all'||needsAttention(t))&&(t.title+' '+t.id).toLowerCase().includes(query));
    $('task-list').innerHTML=tasks.map(t=>'<button class="task-item" type="button" data-task="'+esc(t.id)+'" aria-pressed="'+(page==='task'&&t.id===current().id)+'"><h3>'+esc(t.title)+'</h3><div class="task-item-meta"><i class="status-dot '+status(t)[0]+'"></i><span>'+esc(status(t)[1])+'</span><time>'+esc(t.id)+'</time></div></button>').join('')||'<div class="empty-list">无匹配任务</div>';
    document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.filter===filter)));
  }
  function controls(t){
    if(t.phase===6)return {title:'研究已归档',description:'保留完整证据链；后续探索创建关联的新任务。',action:'followup',label:'新建后续研究',type:'complete'};
    if(t.issue==='resource')return {title:'资源等待 · 执行未开始',description:'示例：所需设备被占用。可继续排队或修订资源方案，尚未提交训练。',action:'queue',label:'保留方案并继续排队',type:'attention'};
    if(t.issue==='failure')return {title:'执行异常 · 前置检查失败',description:'模拟环境检查发现非有限输出，尚未消耗训练步数。已重试 '+t.retries+' / '+t.standard.retries+' 次；失败证据已保留。',action:t.retries<t.standard.retries?'retry':'edit',label:t.retries<t.standard.retries?'模拟修复后重试':'修订方案',type:'attention'};
    if(t.paused)return {title:'执行已暂停',description:'当前运行保留原方案与源码快照。恢复沿用该快照，修改参数需要新修订。',action:'resume',label:'恢复执行',type:'paused'};
    if(t.phase===0)return {title:'方案审核',description:'确认研究假设、对照条件、预算和验收标准后，才能交给代码与执行环节。',action:'approve-plan',label:'批准方案 r'+t.revision,type:'attention'};
    if(t.phase===1&&t.review==='changes')return {title:'代码审查未通过',description:'示例意见：补充空历史检查，确认候选版本没有改变数据与评估条件。',action:'revise-code',label:'模拟提交代码修订',type:'attention'};
    if(t.phase===1)return {title:'独立代码审查',description:'核对变更范围、既有检查结果和源码身份。审查通过后进入资源准备。',action:'approve-code',label:'通过代码审查',type:'attention'};
    if(t.phase===2)return {title:'资源与启动确认',description:'核对代码、环境、数据和权重的绑定关系。确认后创建固定快照并提交运行。',action:'allocate',label:'确认资源并提交',type:'attention'};
    if(t.phase===3)return {title:'实验执行中',description:'执行服务按已批准方案运行。查看子任务、依赖和日志；参数修改不会回写运行快照。',action:'pause',label:'暂停运行',type:'running'};
    if(t.phase===4)return {title:'固定评估中',description:'按绑定的协议计算指标。结果就绪后交给评估分析角色，不能跳过评估直接验收。',action:'pause',label:'暂停评估',type:'running'};
    return {title:'结论验收',description:'审核证据是否回答研究问题、是否存在混杂因素，以及下一轮是否需要补实验。',action:'archive',label:'接受结论并归档',type:'attention'};
  }
  function renderOverview(){
    const running=state.tasks.filter(t=>status(t)[0]==='running'),attention=state.tasks.filter(needsAttention),complete=state.tasks.filter(t=>t.phase===6);
    $('project-metrics').innerHTML=[['研究任务',state.tasks.length,'统一管理研究目标与实验分支'],['执行中',running.length,'包含训练与固定评估'],['待审 / 异常',attention.length,'需要人工决定或处理'],['已归档',complete.length,'已形成可追溯的研究记录']].map(([label,value,note])=>'<div><span>'+label+'</span><strong>'+value+'</strong><small>'+note+'</small></div>').join('');
    $('project-task-table').innerHTML='<table class="data-table overview-table"><thead><tr><th>研究任务</th><th>当前阶段 / 负责人</th><th>整体进度</th><th>状态</th><th></th></tr></thead><tbody>'+state.tasks.map(t=>'<tr><td><button type="button" class="table-task" data-task="'+esc(t.id)+'">'+esc(t.title)+'</button><small>'+esc(t.id)+' · r'+t.revision+' · '+esc(kindNames[t.kind])+'</small></td><td>'+esc(t.phase===6?'归档完成':stages[t.phase].name)+'<small>'+esc(t.phase===6?'共享研究记录':stages[t.phase].owner)+'</small></td><td><div class="mini-progress"><span style="width:'+stepProgress(t)+'%"></span></div><small>'+stepProgress(t)+'% · '+(t.phase===6?'全部阶段完成':'产物与阶段进度')+'</small></td><td>'+pill(...status(t))+'</td><td><button class="text-button" type="button" data-task="'+esc(t.id)+'">打开 →</button></td></tr>').join('')+'</tbody></table>';
    $('control-count').textContent=attention.length+' 项';
    $('control-list').innerHTML=attention.map(t=>'<button type="button" class="control-item" data-task="'+esc(t.id)+'"><span class="control-item-icon">'+(t.issue?'!':'◇')+'</span><span><strong>'+esc(controls(t).title)+'</strong><small>'+esc(t.title)+'</small></span><span>→</span></button>').join('')||'<div class="empty-list">暂无待审或异常事项</div>';
    const events=state.tasks.flatMap(t=>t.events.slice(-2).map((e,i)=>({...e,task:t.id,title:t.title,index:i}))).sort((a,b)=>b.time.localeCompare(a.time)||b.index-a.index).slice(0,6);
    $('project-activity').innerHTML='<div class="activity-table">'+events.map(e=>'<div><time>'+esc(e.time)+'</time><span>'+esc(e.actor)+'</span><strong>'+esc(e.action)+'</strong><button type="button" data-task="'+esc(e.task)+'">'+esc(e.task)+' ↗</button></div>').join('')+'</div>';
  }
  function renderHeader(t){
    const run=runFor(t);
    $('task-header').innerHTML='<div><div class="task-heading-kicker"><span>'+esc(t.id)+'</span><span>'+esc(kindNames[t.kind])+'</span><span>方案 r'+t.revision+'</span></div><h1>'+esc(t.title)+'</h1><div class="task-header-sub"><span>'+esc(protocols[t.protocol])+'</span><span>STD-01</span><span>'+esc(budget(t))+'</span>'+(t.parent?'<span>派生自 '+esc(t.parent)+'</span>':'')+'</div></div><div class="task-header-actions">'+pill(...status(t))+'<button type="button" data-action="edit">修订方案</button><button type="button" data-action="export">导出记录</button></div>';
    $('task-milestones').innerHTML=stages.map((s,i)=>{const skip=skipped(t,i),done=i<t.phase;return '<button type="button" class="milestone '+(skip?'skipped':done?'done':i===t.phase?'active':'')+'" data-stage="'+i+'"'+(i===t.phase?' aria-current="step"':'')+'><span class="milestone-index">'+(skip?'—':done?'✓':String(i+1).padStart(2,'0'))+'</span><span><strong>'+s.name+'</strong><small>'+esc(skip?'本任务跳过':s.owner)+'</small></span><i></i></button>';}).join('');
    const c=controls(t);
    $('control-gate').className='control-gate '+c.type;
    $('control-gate').innerHTML='<div class="gate-label">'+(c.type==='attention'?'阶段控制点':'执行状态')+'</div><div class="gate-info"><h2>'+esc(c.title)+'</h2><p>'+esc(c.description)+'</p></div><div class="gate-actions">'+(t.phase===1&&t.review!=='changes'?'<button type="button" data-action="request-changes">退回修改</button>':'')+(t.phase===5?'<button type="button" data-action="followup">要求补充实验</button>':'')+'<button id="primary-action" type="button" class="primary" data-action="'+c.action+'">'+esc(c.label)+'</button></div>';
    $('task-context').innerHTML='<section class="context-section"><h2>当前执行责任</h2><div class="context-owner"><span class="owner-avatar">'+(t.phase===6?'✓':'AI')+'</span><div><strong>'+esc(t.phase===6?'归档完成':stages[t.phase].owner)+'</strong><small>'+esc(t.phase===6?'可创建关联后续任务':t.paused?'等待恢复':t.issue?'异常处理中':'负责当前阶段交付')+'</small></div></div></section><section class="context-section"><h2>运行约束</h2>'+facts([['方案修订','r'+t.revision],['预算',budget(t)],['随机种子','42'],['重试上限','1 次 · 前置检查'],['决策规则','方案、代码、启动、结论审核'],['当前运行',run?.id||'未创建']])+'</section><section class="context-section"><h2>关联材料</h2>'+t.materials.map(id=>{const m=material(id);return m?'<div class="context-material"><span class="file-icon">'+esc(m.type[0])+'</span><div><strong>'+esc(m.name)+'</strong><small>模拟台账资源</small></div></div>':'';}).join('')+'<button class="text-button context-add" type="button" data-action="materials">管理关联材料</button></section><section class="context-section"><h2>产物追踪</h2>'+[['plan','实验方案 r'+t.revision],...(t.phase>=2&&t.kind!=='evaluate'?[['code','代码审查报告']]:[]),...(run?[['identity','运行与源码快照']]:[]),...(run?.rows?[['metrics','评估指标文件']]:[]),...(t.phase>=5?[['report','研究结论报告']]:[])].map(([id,name])=>'<button type="button" class="context-artifact" data-artifact="'+id+'"><span>▤</span>'+esc(name)+'<span>↗</span></button>').join('')+'</section><section class="context-section"><p class="context-preview">当前只模拟人机交互，未连接真实 Agent、资源状态或执行服务。</p></section>';
  }
  function renderPlan(t){
    const names=branches(t);
    $('panel-plan').innerHTML=block('研究定义',facts([['研究问题',t.title],['判定依据',t.criterion],['已有证据','读取关联研究记录；真实内容尚未接入'],['研究类型',kindNames[t.kind]]]),'研究规划 Agent · r'+t.revision)+block('实验设计','<div class="branch-design">'+names.map((name,i)=>'<div><span>分支 '+String.fromCharCode(65+i)+'</span><h3>'+esc(name)+'</h3><p>'+esc(t.kind==='evaluate'?'复用已有权重和对应源码':t.kind==='diagnose'?'只改变历史条件，保持当前观测一致':'只改变本次研究因素，其余条件配对')+'</p></div>').join('')+'</div>'+facts([['共享条件','相同数据划分、初始化、随机种子与评估实例'],['任务集合','Task '+t.tasks.join(' / ')],['每组预算',t.kind==='evaluate'?'不新增训练':t.steps+' 个优化步'],['评估协议',protocols[t.protocol]+' · '+metrics[t.protocol]],['不纳入范围','不改变任务评分定义、动作接口与评估时长']]))+block('阶段输入与验收标准','<div class="table-scroll"><table class="data-table"><thead><tr><th>阶段</th><th>输入</th><th>交付 / 通过条件</th></tr></thead><tbody>'+stages.map((s,i)=>'<tr class="'+(skipped(t,i)?'row-skipped':'')+'"><td>'+s.name+(skipped(t,i)?'<small>本任务跳过</small>':'')+'</td><td>'+s.input+'</td><td>'+s.output+'</td></tr>').join('')+'</tbody></table></div>','固定交接 · 方法内容可探索');
  }
  function renderCode(t){
    if(t.kind==='evaluate'){$('panel-code').innerHTML=block('复用已有源码','<div class="stage-empty"><h3>本任务不修改代码</h3><p>运行前仍检查所选权重与源码、推理服务的身份一致性。代码实现与训练阶段跳过。</p></div>');return;}
    const approved=t.phase>=2,changes=t.review==='changes';
    $('panel-code').innerHTML=block('AI 实施与审查任务','<div class="work-items">'+[
      ['代码工程 Agent','实现本轮研究因素',t.phase===0?'等待方案批准':changes?'修订中':approved?'示例已提交':'示例已提交','变更集 / 工作区'],
      ['固定检查程序','运行既有网络与短窗检查',t.phase===0?'未执行':changes?'需补充检查':'示例检查通过','检查输出'],
      ['独立审查角色','核对范围、接口与对照条件',approved?'审查通过':changes?'已退回':'等待审核','审查意见'],
      ['版本服务','固化源码与配置身份',approved?'待运行绑定':'等待审查通过','不可变快照']
    ].map(([owner,action,s,output])=>'<div><span class="work-avatar">'+(owner.includes('Agent')?'AI':'✓')+'</span><div><strong>'+esc(action)+'</strong><small>'+esc(owner)+' → '+esc(output)+'</small></div><span class="work-status">'+esc(s)+'</span></div>').join('')+'</div>','审查轮次 '+t.codeRound)+block('变更与检查证据','<div class="change-summary"><div><span class="file-icon">Δ</span><div><h3>候选实现变更集</h3><p>隔离实验副本 · 示例内容，不对应真实代码修改</p></div><button class="text-button" type="button" data-artifact="code">查看审查报告 ↗</button></div><pre class="code-preview">− method_variant: current\n+ method_variant: candidate\n  data_split: paired\n  evaluation_protocol: '+esc(t.protocol)+'</pre></div><ul class="check-list"><li>保持动作接口和评估定义一致</li><li>保持分支的数据划分、初始化与预算一致</li><li>关联已有检查输出与代码版本</li></ul>'+(changes?'<div class="inline-alert">审查意见：补充空历史检查，明确候选改动没有影响对照条件。</div>':''));
  }
  function jobs(t,r){
    if(!r)return [];
    const s=r.status,done=['review','archived'].includes(s),evalStage=r.stage==='evaluation';
    const waiting=s==='paused'?'已暂停':s==='failed'?'前置失败':'等待依赖';
    return [['身份 / 环境检查','无前序依赖',s==='failed'?'失败':'示例通过','资源管理服务', '检查报告'],...branches(r.snapshot).flatMap((name,i)=>[
      ...(t.kind==='evaluate'?[]:[[name+' · 训练','检查通过',done||evalStage?'示例完成':s==='running'?(i===0?'执行中':'等待分支资源'):waiting,'实验执行服务','checkpoint / 训练日志']]),
      [name+' · 评估',t.kind==='evaluate'?'身份检查通过':'对应 checkpoint 就绪',done?'示例完成':evalStage&&s==='running'?(i===0?'执行中':'等待分支资源'):waiting,'固定评估程序','配对指标 / 评估日志']
    ]),['分析与汇总','全部分支评估完成',done?'示例完成':'等待指标','评估分析 Agent','研究结论与证据']];
  }
  function renderExecution(t){
    const run=runFor(t),jobRows=jobs(t,run);
    $('panel-execution').innerHTML=block('资源需求与分配','<div class="resource-cards">'+[
      ['训练 / 推理设备',t.kind==='evaluate'?'推理能力':'训练与推理能力',run?(t.device==='local'?'本机 GPU · 示例':run.allocation.training):'待分配'],
      ['验证环境',protocols[t.protocol],run?run.allocation.evaluation:'待协议与资源确认'],
      ['数据与权重','从已登记材料中绑定',run?'已创建模拟绑定':'等待关联材料核对']
    ].map(([label,requirement,assigned])=>'<div><span>'+esc(label)+'</span><h3>'+esc(assigned)+'</h3><p>'+esc(requirement)+'</p></div>').join('')+'</div><p class="section-note">示例分配，未读取实时设备占用。正式资源以共享目录和队列返回为准。</p>')+block('执行依赖与子任务',jobRows.length?'<div class="table-scroll"><table class="data-table"><thead><tr><th>子任务</th><th>依赖</th><th>状态</th><th>执行主体 / 输出</th></tr></thead><tbody>'+jobRows.map(([name,dep,s,owner,output])=>'<tr><td>'+esc(name)+'</td><td>'+esc(dep)+'</td><td><span class="job-status '+(s==='执行中'?'active':s==='失败'?'error':'')+'">'+esc(s)+'</span></td><td>'+esc(owner)+'<small>'+esc(output)+'</small></td></tr>').join('')+'</tbody></table></div>':'<div class="stage-empty"><h3>尚未提交运行</h3><p>方案与代码确认后，资源服务创建分配，执行服务据此创建子任务。</p></div>',run?.id||'未创建运行')+block('运行快照与历史尝试',t.runs.length?'<div class="run-history">'+t.runs.slice().reverse().map(r=>'<div class="run-history-item"><div class="section-heading"><h3>'+esc(r.id)+'</h3>'+pill(r.status==='failed'?'attention':r.status==='archived'?'complete':r.status==='paused'?'paused':'running',({running:'运行中',paused:'已暂停',failed:'前置检查失败',review:'待结论确认',archived:'已归档'})[r.status])+'</div>'+facts([['方案 / 标准','r'+r.revision+' / STD-'+String(r.snapshot.standard.revision).padStart(2,'0')],['训练预算',budget(r.snapshot)],['协议',protocols[r.snapshot.protocol]],['源码快照',r.source],['尝试序号',String(r.attempt)],['训练已消耗',r.status==='failed'?'0 步 · 模拟前置失败':r.rows?budget(r.snapshot):'待后端上报']])+'</div>').join('')+'</div>':'<p class="section-note">提交后固定保存当时的方案、源码、资源分配和协议；之后的修订不覆盖历史。</p>');
  }
  function metricTable(t,rows){return '<div class="table-scroll"><table class="data-table result-data"><thead><tr><th>分支 / 条件</th><th>'+esc(metrics[t.protocol])+'</th><th>'+esc(t.protocol==='offline'?'独立样本范围':'完整成功回合')+'</th></tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
  function renderResults(t){
    const r=runFor(t);
    $('panel-results').innerHTML=block('固定评估配置',facts([['指标口径',metrics[t.protocol]],['输入条件','对应分支权重、相同实例与 seed'],['可比性要求','控制变量与方案 r'+t.revision+' 保持一致'],['当前状态',t.phase<4?'等待前序实验完成':t.phase===4?'固定程序计算指标中':'示例指标已就绪']]))+(!r?.rows?block('指标与结论','<div class="stage-empty"><h3>指标尚未就绪</h3><p>等待固定评估完成后生成比较结果，再由评估分析角色形成结论。</p></div>'):block('分支指标比较',metricTable(r.snapshot,r.rows)+'<p class="section-note">全部为合成数字，用于评审展示。按本任务绑定的评估协议记录。</p>','模拟数据 · 不代表模型效果')+block('研究结论与验收','<div class="finding"><span class="finding-label">'+(t.phase===6?'已验收归档':'待人工验收')+'</span><h3>'+esc(t.kind==='diagnose'?'正确历史收益尚无充分证据':t.kind==='evaluate'?'已形成固定协议评估基线':'候选方法存在示例改善，需补充证据')+'</h3><p>'+esc(r.finding||findingFor(t))+'</p></div>'+facts([['回答的研究问题',r.snapshot.title],['证据来源',r.id+' / 方案 r'+r.revision],['局限','当前数据为合成示例；真实分析需配对独立样本与不确定性'],['后续方向',t.kind==='evaluate'?'建立可比的候选模型实验':'补充独立样本与机制对照']])+'<div class="evidence-actions"><button type="button" data-artifact="report">查看研究报告</button><button type="button" data-artifact="metrics">查看指标文件</button><button type="button" data-artifact="identity">核对运行身份</button></div>'));
  }
  function renderRecords(t){
    $('decision-records').innerHTML=block('决策与阶段交接','<div class="record-list">'+t.events.slice().reverse().map(e=>'<div><time>'+esc(e.time)+'</time><div><strong>'+esc(e.action)+'</strong><p>'+esc(e.actor)+' · '+esc(e.detail)+'</p></div></div>').join('')+'</div>')+(t.notes.length?block('研究决策备注','<div class="record-list">'+t.notes.slice().reverse().map(n=>'<div><time>'+esc(n.time)+'</time><div><strong>人工记录</strong><p>'+esc(n.text)+'</p></div></div>').join('')+'</div>'):'');
  }
  function renderTask(){const t=current();renderHeader(t);renderPlan(t);renderCode(t);renderExecution(t);renderResults(t);renderRecords(t);showTab(tab);}
  function render(){renderList();renderOverview();renderTask();showPage(page);}
  function showTab(next,focus=false){tab=next;document.querySelectorAll('[data-tab]').forEach(b=>{const on=b.dataset.tab===next;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1;if(on&&focus)b.focus();});tabs.forEach(n=>$('panel-'+n).hidden=n!==next);}
  function selectTask(id){if(!state.tasks.some(t=>t.id===id))return;state.selected=id;page='task';tab=current().phase===6?'results':stages[current().phase].tab;save();render();const selected=$('task-list').querySelector('[aria-pressed=true]');if(selected&&matchMedia('(max-width:760px)').matches)$('task-list').scrollLeft=selected.offsetLeft-$('task-list').offsetLeft;}
  function createRun(t){const r=makeRun(t,'SIM-'+String(state.nextRun++).padStart(4,'0'));t.runs.push(r);t.phase=t.kind==='evaluate'?4:3;t.issue=null;t.paused=false;log(t,'资源管理 → 实验执行','固定快照并提交模拟运行',r.id+' · 方案 r'+t.revision+' · '+budget(t));}
  function act(action){
    const t=current(),r=runFor(t);
    if(action==='edit'){openPlan();return;}
    if(action==='export'){download('research-record',t);return;}
    if(action==='materials'){openMaterials('task');return;}
    if(action==='followup'){openNew(t);return;}
    if(action==='approve-plan'&&t.phase===0){t.phase=t.kind==='evaluate'?2:1;tab=stages[t.phase].tab;log(t,'你 → '+stages[t.phase].owner,'批准方案 r'+t.revision,'预算、对照条件与协议进入受控执行。');}
    else if(action==='request-changes'&&t.phase===1){t.review='changes';tab='code';log(t,'独立审查 → 代码工程','退回代码修改','要求补充空历史检查，确认不改变对照条件。');}
    else if(action==='revise-code'&&t.phase===1&&t.review==='changes'){t.review='pending';t.codeRound++;log(t,'代码工程 Agent','模拟提交修订代码','审查轮次 '+t.codeRound+'；等待独立审核。');}
    else if(action==='approve-code'&&t.phase===1&&t.review!=='changes'){t.review='approved';t.phase=2;tab='execution';log(t,'独立审查 → 资源管理','通过代码审查','保留变更与检查报告；等待启动确认。');}
    else if(action==='allocate'&&t.phase===2&&!t.issue){createRun(t);tab='execution';}
    else if(action==='pause'&&r?.status==='running'){t.paused=true;r.status='paused';log(t,'你 → 执行服务','暂停运行',r.id+' 沿用当前快照。');}
    else if(action==='resume'&&t.paused&&r?.status==='paused'){t.paused=false;r.status='running';log(t,'你 → 执行服务','恢复运行',r.id+' 按原快照继续。');}
    else if(action==='queue'&&t.issue==='resource'){t.issue=null;log(t,'你 → 资源管理','继续等待资源','原方案不变；未创建运行。');}
    else if(action==='retry'&&t.issue==='failure'&&t.retries<t.standard.retries){t.retries++;createRun(t);log(t,'代码工程 → 执行服务','模拟修复并重试','失败发生于前置检查，训练消耗为 0；保留失败尝试。');}
    else if(action==='archive'&&t.phase===5&&r?.rows){t.phase=6;r.status='archived';r.finding=findingFor(t);tab='results';log(t,'你 → 共享研究记录','接受结论并归档','保留实验产物、证据与研究局限。');}
    else return;
    save();render();toast('演示状态已更新，未调用真实执行接口');
  }
  function openDemo(){
    const t=current(),r=runFor(t);
    const actions=[];
    if(page==='task'&&t.phase===2&&!t.issue)actions.push(['resource','模拟资源冲突','资源进入等待状态，实验不启动。']);
    if(page==='task'&&r?.status==='running'){
      actions.push(['advance',t.phase===3?'模拟训练产物就绪':'模拟评估与分析就绪',t.phase===3?'训练结束后进入固定评估。':'形成指标与结论，等待人工验收。']);
      if(t.phase===3||t.kind==='evaluate')actions.push(['fail','模拟前置检查失败','保留失败证据；训练消耗记为 0，仅用于原型演示。']);
    }
    $('demo-actions').innerHTML=actions.length?actions.map(([id,title,description])=>'<button type="button" data-demo="'+id+'">'+title+'<small>'+description+'</small></button>').join(''):'<p class="section-note">选择正在执行或等待资源的任务，模拟后台事件。其他阶段请使用页面上的方案与审查操作。</p>';
    $('demo-dialog').showModal();
  }
  function simulate(event){
    const t=current(),r=runFor(t);
    if(event==='resource'&&t.phase===2){t.issue='resource';log(t,'资源管理服务','模拟资源占用冲突','所需设备暂不可用；保持排队，未提交任务。');}
    else if(event==='fail'&&r?.status==='running'&&(t.phase===3||t.kind==='evaluate')){t.issue='failure';r.status='failed';r.failure='前置检查发现非有限输出';log(t,'执行服务','模拟前置检查失败',r.id+' 已停止，尚未消耗训练步数。');}
    else if(event==='advance'&&r?.status==='running'&&t.phase===3){t.phase=4;r.stage='evaluation';tab='execution';log(t,'执行服务 → 固定评估','模拟训练产物就绪','对应分支 checkpoint 交给已绑定的评估协议。');}
    else if(event==='advance'&&r?.status==='running'&&t.phase===4){t.phase=5;r.status='review';r.stage='analysis';r.rows=rowsFor(t);r.finding=findingFor(t);tab='results';log(t,'评估分析 Agent','模拟提交指标与研究结论','合成数据；等待人工验收。');}
    else return;
    $('demo-dialog').close();save();render();toast('已模拟后台事件');
  }
  function openNew(parent=null){
    $('new-form').reset();newMaterials=parent?clone(parent.materials):['code','dataset','base'];
    $('new-form').dataset.parent=parent?.id||'';
    if(parent){$('new-question').value='后续研究：'+parent.title;$('new-criterion').value=parent.criterion;$('new-kind').value=parent.kind;$('new-protocol').value=parent.protocol;}
    page='new';renderNewContext();renderList();showPage('new');$('new-question').focus();
  }
  function renderNewContext(){$('new-context').innerHTML=newMaterials.map(id=>'<span class="attachment-chip">'+esc(material(id)?.type)+' · '+esc(material(id)?.name)+'</span>').join('')||'<span class="section-note">尚未关联材料，可在方案准备前选择。</span>';}
  function openMaterials(target){materialTarget=target;const ids=target==='new'?newMaterials:current().materials;$('materials-list').innerHTML=materials.map(m=>'<label class="material-option"><input type="checkbox" name="material" value="'+m.id+'"'+(ids.includes(m.id)?' checked':'')+'><span><strong>'+esc(m.name)+'</strong><small>'+esc(m.detail)+'</small></span><span class="material-type">'+m.type+'</span></label>').join('');$('materials-dialog').showModal();}
  function updatePlanScope(){const evaluate=$('plan-form').elements.scope.value==='evaluate';$('budget-field').hidden=evaluate;$('plan-form').elements.steps.disabled=evaluate;}
  function openPlan(){const t=current(),f=$('plan-form');$('plan-error').hidden=true;f.elements.question.value=t.title;f.elements.criterion.value=t.criterion;f.elements.protocol.value=t.protocol;f.elements.tasks.value=t.tasks.join(', ');f.elements.steps.value=t.steps||100;f.elements.scope.value=t.kind==='evaluate'?'evaluate':'research';$('plan-edit-note').textContent='保存为 r'+(t.revision+1)+' 并重新审核；当前运行若未结束会先转为暂停。';updatePlanScope();$('plan-dialog').showModal();}
  function revise(t,fields,detail){
    const r=runFor(t);if(r?.status==='running'){r.status='paused';log(t,'你 → 执行服务','修订前暂停模拟运行',r.id+' 保留原源码、配置与协议。');}
    Object.assign(t,fields,{revision:t.revision+1,phase:0,review:'pending',codeRound:1,retries:0,paused:false,issue:null});
    log(t,'你 → 研究规划','提交方案修订 r'+t.revision,detail);tab='plan';save();render();
  }
  function artifact(id){
    const t=current(),r=runFor(t);const title={plan:'研究方案 r'+t.revision,code:'代码变更与审查报告',identity:'运行身份与源码快照',metrics:'评估指标文件',report:'研究结论报告'}[id];
    $('artifact-title').textContent=title;
    let content=id==='plan'?facts([['研究问题',t.title],['判定依据',t.criterion],['方案修订','r'+t.revision],['预算',budget(t)],['协议',protocols[t.protocol]],['任务集合',t.tasks.join(', ')]]):id==='code'?'<p class="artifact-note">示例变更与独立审查结果，不对应真实代码修改。</p>'+facts([['审查轮次',String(t.codeRound)],['审查结论',t.review==='approved'?'示例通过':t.review==='changes'?'需修订':'待审查'],['变更范围','仅实现本轮研究因素'],['检查入口','复用已有网络 / 短窗检查']]):id==='identity'?facts([['运行 ID',r?.id||'未创建'],['源码身份',r?.source||'未创建'],['方案修订',r?'r'+r.revision:'未创建'],['标准快照',r?'STD-01':'未创建'],['预算',r?budget(r.snapshot):'未创建'],['实际作业 ID','后端待接入']]):id==='metrics'?(r?.rows?metricTable(r.snapshot,r.rows)+'<p class="artifact-note">以上指标为合成示例。</p>':'<p class="artifact-note">指标尚未生成。</p>'):'<p class="artifact-note">'+esc(r?.finding||'研究结论尚未生成。')+'</p>'+(r?.rows?metricTable(r.snapshot,r.rows):'');
    $('artifact-body').innerHTML=content+'<button type="button" class="download-button" data-download="'+id+'">导出示例 JSON</button>';
    $('artifact-dialog').showModal();
  }
  function download(type,t){const r=runFor(t),payload={schema:'human-research-console-v3',demo:true,executable:false,type,taskId:t.id,planRevision:t.revision,data:type==='metrics'?{protocol:t.protocol,rows:r?.rows||null}:type==='identity'?r:type==='report'?{finding:r?.finding,run:r?.id}:t};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=t.id+'-'+type+'-preview.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已导出带演示标记的记录');}
  $('new-form').addEventListener('submit',e=>{
    e.preventDefault();const title=$('new-question').value.trim(),criterion=$('new-criterion').value.trim();if(!title||!criterion)return;
    const kind=$('new-kind').value,protocol=kind==='diagnose'?'offline':$('new-protocol').value;
    const t=freshTask('R-'+String(state.nextTask++).padStart(3,'0'),{title,criterion,kind,protocol,tasks:protocol==='libero'?[0]:[0,34,40],steps:kind==='evaluate'?0:100,materials:clone(newMaterials),parent:$('new-form').dataset.parent||null});
    log(t,'研究规划 Agent','建立方案草稿 r1','按模板组织方案，真实 AI 和资源查询待接入。');state.tasks.unshift(t);query='';filter='all';$('task-search').value='';selectTask(t.id);
  });
  $('materials-form').addEventListener('submit',e=>{e.preventDefault();const ids=new FormData(e.currentTarget).getAll('material');if(materialTarget==='new'){newMaterials=ids;renderNewContext();}else revise(current(),{materials:ids},'修改关联材料，重新审核输入与实验设计。');$('materials-dialog').close();});
  $('plan-form').elements.scope.addEventListener('change',updatePlanScope);
  $('plan-form').addEventListener('submit',e=>{
    e.preventDefault();const d=new FormData(e.currentTarget),t=current(),protocol=d.get('protocol'),ids=[...new Set(d.get('tasks').split(',').map(v=>Number(v.trim())))],maximum=protocol==='libero'?9:49;
    const title=d.get('question').trim(),criterion=d.get('criterion').trim();
    if(!title||!criterion||ids.some(i=>!Number.isInteger(i)||i<0||i>maximum)){$('plan-error').textContent='请填写研究问题、判定依据，并使用 0–'+maximum+' 的任务编号。';$('plan-error').hidden=false;return;}
    const evaluating=d.get('scope')==='evaluate',steps=evaluating?0:Number(d.get('steps'));
    if(steps>t.standard.maxSteps){$('plan-error').textContent='每组预算不能超过当前标准的 '+t.standard.maxSteps+' 步。';$('plan-error').hidden=false;return;}
    revise(t,{title,criterion,protocol,tasks:ids,steps,kind:evaluating?'evaluate':t.kind==='evaluate'?'compare':t.kind},'调整研究定义、预算或协议；旧运行保留原快照。');$('plan-dialog').close();toast('新修订等待重新审核');
  });
  $('message-form').addEventListener('submit',e=>{e.preventDefault();const value=$('message').value.trim();if(!value)return;current().notes.push({time:time(),text:value});$('message').value='';save();renderRecords(current());toast('决策记录已保存，运行参数保持原方案');});
  document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.task)selectTask(b.dataset.task);if(b.dataset.tab)showTab(b.dataset.tab);if(b.hasAttribute('data-stage'))showTab(stages[Number(b.dataset.stage)].tab);if(b.dataset.action)act(b.dataset.action);if(b.dataset.artifact)artifact(b.dataset.artifact);if(b.dataset.download)download(b.dataset.download,current());if(b.dataset.demo)simulate(b.dataset.demo);if(b.dataset.close)$(b.dataset.close).close();if(b.dataset.filter){filter=b.dataset.filter;renderList();}});
  $('task-search').oninput=e=>{query=e.target.value.toLowerCase();renderList();};
  $('new-task').onclick=$('overview-new-task').onclick=()=>openNew();$('cancel-new-task').onclick=$('project-overview').onclick=()=>{page='overview';render();};
  $('new-materials').onclick=()=>openMaterials('new');$('demo-button').onclick=openDemo;
  $('reset-demo').onclick=()=>{state=initialState();page='overview';tab='plan';filter='all';query='';$('task-search').value='';$('demo-dialog').close();save();render();toast('已恢复初始示例任务');};
  document.querySelector('.task-tabs').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;const i=tabs.indexOf(tab),next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:tabs.length-1))%tabs.length;e.preventDefault();showTab(tabs[next],true);});
  document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='n'&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!e.target.closest('input,textarea,select,dialog')){e.preventDefault();openNew();}});
  document.querySelectorAll('[data-workbench-link]').forEach(a=>{if(location.port==='8772')a.href=location.protocol+'//'+location.hostname+':8765'+a.getAttribute('href');});
  render();
})();
