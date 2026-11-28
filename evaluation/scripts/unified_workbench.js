(() => {
'use strict';
const $=id=>document.getElementById(id);
const labels={experiment:'实验管理',training:'模型训练',launch:'推理评估',results:'结果分析',registry:'资源管理'};
const benchmarks={all:'全部',b1k:'B1K-2025',b1k2026:'B1K-2026',libero:'LIBERO',mikasa:'MIKASA',simpler:'SimplerEnv'};
const esc=v=>String(v??'—').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const version=v=>{if(!v)return '未标版本';const s=String(v),m=s.match(/^V[_ ]?(\d+(?:[._]\d+)*)$/i);return m?'V'+m[1].replaceAll('_','.'):s;};
const stateLabel=s=>({complete:'已完成',completed:'已完成',training:'训练中',running:'运行中',failed:'异常',queued:'排队',stopped:'已停止',paused:'已暂停',submitted:'已提交',interrupted:'中断'})[s]||s||'未记录';
const jsonKey=(...parts)=>JSON.stringify(parts);
const params=new URLSearchParams(location.search);
const page='experiment';let benchmark=Object.hasOwn(benchmarks,params.get('benchmark'))?params.get('benchmark'):'all';
const priorPage=params.get('page');if(priorPage&&priorPage!=='experiment'&&Object.hasOwn(labels,priorPage)){location.replace('/?'+new URLSearchParams({page:priorPage,benchmark:benchmark==='all'?'b1k':benchmark}));return;}
let selected=params.get('node'),expanded=new Set(),nodes=new Map(),roots=[],refreshing=false,first=true,includeEngineering=false;
let cache={train:[],results:[],native:[],libero:[],experiments:[],launch:[]};
const detailCache=new Map(),detailRequests=new Map(),checkpointByTraining=new Map();
let graphFocus=null,graphObserver=null,wireFrame=0;
const nodeTypes={version:'代码版本',train:'训练',checkpoint:'权重',eval:'评估',group:'分组',document:'文档'};
const pathValue=v=>typeof v==='string'?v:typeof v?.path==='string'?v.path:typeof v?.checkpoint==='string'?v.checkpoint:typeof v?.directory==='string'?v.directory:null;
function status(msg,error=false){$('message').className='notice'+(error?' error':'');$('message').textContent=msg||'';}
async function api(path){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),18000);try{const r=await fetch(path,{cache:'no-store',signal:controller.signal});const d=await r.json();if(!r.ok||d.error)throw Error(d.error||('HTTP '+r.status));return d;}finally{clearTimeout(timer);}}
function add(n,parent){if(nodes.has(n.id))return nodes.get(n.id);n.children=[];n.parent=parent?.id||null;nodes.set(n.id,n);if(parent)parent.children.push(n.id);else roots.push(n.id);return n;}
function versionNode(v){return add({id:jsonKey('version',v),kind:'version',title:v,version:v});}
function weightNode(parent,path,meta){const recorded=nodes.get(jsonKey('checkpoint',parent.id,path));if(recorded){recorded.raw={...meta,...recorded.raw};if(recorded.raw.step!=null)recorded.title='Step '+recorded.raw.step;return recorded;}return add({id:jsonKey('checkpoint',parent.id,path),kind:'checkpoint',title:meta.step!=null?'Step '+meta.step:meta.title||String(path).split('/').filter(Boolean).pop()||'权重',path,version:parent.version,benchmark:meta.benchmark||parent.benchmark,device:parent.device,train:parent.kind==='train'?parent.id:null,raw:meta,official:meta.official===true},parent);}
function identity(r){return r.policy_identity||r.configuration?.policy_identity||{};}
function rebuild(){
  const previous=selected;nodes=new Map();roots=[];
  const trainIds=new Map();
  for(const row of cache.train){
    const j=row.job,v=version(j.model_version||j.version),parent=versionNode(v),id=jsonKey('train',row.device,j.run,j.arm);
    const n=add({id,kind:'train',title:(j.method||j.arm||'训练')+' · '+(j.train_number?'Train '+String(j.train_number).padStart(2,'0'):j.run)+(j.task!=null?' · Task '+j.task:''),version:v,benchmark:'b1k',device:row.device,run:j.run,arm:j.arm,raw:j,engineering:j.category==='engineering_preflight'},parent);
    for(const key of new Set([j.training_id,j.run].filter(Boolean))){if(!trainIds.has(key))trainIds.set(key,[]);trainIds.get(key).push(n);}
    for(const c of checkpointByTraining.get(id)||[]){if(typeof c.path==='string'&&c.path)weightNode(n,c.path,c);}
  }
  for(const r of cache.results){
    const raw=identity(r),p={...raw,checkpoint:pathValue(raw.checkpoint)},v=version(r.version||p.version),parent=versionNode(v);
    let candidates=(trainIds.get(p.training_id)||[]).filter(n=>n.version===v);
    if(candidates.length>1&&p.checkpoint){const byPath=candidates.filter(n=>(checkpointByTraining.get(n.id)||[]).some(c=>c.path===p.checkpoint));if(byPath.length===1)candidates=byPath;}
    let owner;
    if(candidates.length===1&&p.checkpoint)owner=weightNode(candidates[0],p.checkpoint,{step:p.checkpoint_step,benchmark:'b1k',recordedReference:true});
    else if(p.checkpoint&&p.method==='official')owner=weightNode(parent,p.checkpoint,{title:'官方权重',official:true,benchmark:'b1k'});
    else owner=add({id:jsonKey('unlinked',v,'b1k'),kind:'group',title:'未关联评估',version:v,benchmark:'b1k'},parent);
    add({id:jsonKey('eval','b1k',r._folder),kind:'eval',title:'Task '+(r.task_id??r.configuration?.task_id??'—')+' · '+(r.method||p.method||String(r._folder||'').split('/').pop()),version:v,benchmark:'b1k',folder:r._folder,raw:r,engineering:r.evaluation_mode==='fast-debug'},owner);
  }
  for(const r of cache.launch.filter(j=>j.kind!=='b1k2026_websocket'&&!cache.results.some(x=>x._folder===j.folder))){
    const raw=identity(r),p={...raw,checkpoint:pathValue(raw.checkpoint)},v=version(r.version||p.version),parent=versionNode(v),candidates=(trainIds.get(p.training_id)||[]).filter(n=>n.version===v);
    const owner=candidates.length===1&&p.checkpoint?weightNode(candidates[0],p.checkpoint,{step:p.checkpoint_step,benchmark:'b1k',recordedReference:true}):add({id:jsonKey('unlinked',v,'b1k'),kind:'group',title:'未关联评估',version:v,benchmark:'b1k'},parent);
    add({id:jsonKey('job','b1k',r.id,r.device),kind:'eval',title:'Task '+(r.task_id??'—')+' · '+(r.group||'评估作业'),version:v,benchmark:'b1k',run:r.id,raw:r,pending:true},owner);
  }
  for(const r of cache.native){
    const raw=identity(r),p={...raw,checkpoint:pathValue(raw.checkpoint)},v=version(r.version),parent=versionNode(v);
    let owner;
    if((r.official===true||p.method==='official')&&p.checkpoint)owner=weightNode(parent,p.checkpoint,{title:'官方 Radio 权重',official:true,benchmark:'b1k2026',sha:r.checkpoint_sha256});
    else owner=add({id:jsonKey('unlinked',v,'b1k2026'),kind:'group',title:'未关联评估',version:v,benchmark:'b1k2026'},parent);
    add({id:jsonKey('eval','b1k2026',r.id),kind:'eval',title:(r.task_id==='turning_on_radio'?'Radio':r.task_id||'原生评估')+' · '+({official_default:'默认指令',task_description:'任务原文'}[r.language?.mode]||'指令未记录')+(r.attempt?' · '+r.attempt:''),version:v,benchmark:'b1k2026',run:r.id,raw:r},owner);
  }
  for(const r of cache.libero.filter(j=>j.kind?.startsWith('libero'))){
    const v=version(r.config?.version||r.version),parent=versionNode(v);
    const group=add({id:jsonKey('unlinked',v,'libero'),kind:'group',title:'LIBERO · 未关联记录',version:v,benchmark:'libero'},parent);
    add({id:jsonKey('eval','libero',r.id),kind:'eval',title:r.title||r.kind,version:v,benchmark:'libero',run:r.id,raw:r,engineering:r.kind==='libero'||r.config?.max_steps>0},group);
  }
  if(cache.experiments.length){const group=add({id:'documents',kind:'group',title:'科研实验文档',version:'文档'});for(const entry of cache.experiments){const id=typeof entry==='string'?entry:entry?.id;if(typeof id==='string')add({id:jsonKey('document',id),kind:'document',title:id.includes('/')?id.replace('/',' / '):id,experiment:id},group);}}
  roots.sort((a,b)=>nodes.get(b).title.localeCompare(nodes.get(a).title,undefined,{numeric:true}));
  if(previous&&nodes.has(previous)&&scope(previous))selected=previous;
  else {selected=roots.find(id=>nodes.get(id).title==='V2.1'&&matchesBranch(id))||roots.find(id=>nodes.get(id).kind==='version'&&matchesBranch(id))||roots.find(matchesBranch)||null;}
  if(first&&selected){
    const root=nodes.get(selected);
    if(!params.has('node')&&root.kind==='version'){
      const child=root.children.map(id=>nodes.get(id)).find(n=>scope(n.id)&&n.kind==='train')||root.children.map(id=>nodes.get(id)).find(n=>scope(n.id)&&n.kind==='checkpoint');
      if(child)selected=child.id;
    }
    ancestors(nodes.get(selected)).forEach(n=>expanded.add(n.id));graphFocus=selected;first=false;
  }
}
function visibleSelf(n){return (includeEngineering||!n.engineering)&&(!n.benchmark||benchmark==='all'||n.benchmark===benchmark);}
function scope(id){const n=nodes.get(id);if(!n||!visibleSelf(n))return false;if(n.benchmark)return true;if(n.kind==='document')return true;return n.children.some(scope)||n.kind==='group'&&n.id==='documents';}
function matchesBranch(id){return scope(id);}
function nav(){
 $('nav').innerHTML=Object.entries(labels).map(([k,v])=>'<button data-page="'+k+'" class="'+(page===k?'active':'')+'" aria-current="'+(page===k?'page':'false')+'">'+v+'</button>').join('');
 const select=$('benchmarks');if(!select.options.length)select.innerHTML=Object.entries(benchmarks).map(([k,v])=>'<option value="'+k+'">'+v+'</option>').join('');select.value=benchmark;
}
function treeHTML(ids){
 return '<ul>'+ids.filter(matchesBranch).map(id=>{
  const n=nodes.get(id),children=n.children.filter(matchesBranch),open=expanded.has(id),expandable=children.length>0||n.kind==='train';
  const type=n.kind==='version'||n.kind==='group'?'':nodeTypes[n.kind];
  return '<li><div class="tree-row kind-'+n.kind+'"><button class="toggle" data-expand="'+esc(id)+'" aria-label="'+(open?'收起':'展开')+'" aria-expanded="'+open+'" '+(!expandable?'disabled':'')+'>'+(expandable?(open?'▾':'▸'):'')+'</button><button class="pick '+(selected===id?'active':'')+'" data-node="'+esc(id)+'" data-kind="'+n.kind+'" aria-current="'+(selected===id?'true':'false')+'" title="'+esc(nodeTypes[n.kind]+' · '+n.title)+'">'+(type?'<small class="tree-type">'+esc(type)+'</small>':'')+'<span class="tree-name">'+esc(n.title)+'</span>'+(children.length?'<small class="count">'+children.length+'</small>':'')+'</button></div>'+(open&&children.length?treeHTML(children):'')+'</li>';
 }).join('')+'</ul>';
}
function drawTree(){const el=$('tree'),scroll=el.scrollTop;el.innerHTML=treeHTML(roots)||'<div class="empty">暂无记录</div>';el.scrollTop=scroll;}
function descendants(n,kind){return n?(n.kind===kind?[n]:[]).concat(n.children.flatMap(id=>descendants(nodes.get(id),kind))).filter(x=>visibleSelf(x)):[];}
function ancestors(n){const rows=[];for(let x=n;x;x=nodes.get(x.parent))rows.unshift(x);return rows;}
function getTrain(n){return n?.kind==='train'?n:ancestors(n).find(x=>x.kind==='train');}
function getWeight(n){return n?.kind==='checkpoint'?n:ancestors(n).find(x=>x.kind==='checkpoint');}
function currentBenchmark(n){return n?.benchmark||(benchmark==='all'?'b1k':benchmark);}
function section(title,rows){rows=rows.filter(([,v])=>v!==undefined&&v!==null&&v!=='');if(!rows.length)return '';return '<div class="group-title">'+esc(title)+'</div><dl class="kv">'+rows.map(([k,v])=>'<dt>'+esc(k)+'</dt><dd>'+esc(Array.isArray(v)?v.join(', '):v)+'</dd>').join('')+'</dl>';}
function pct(n){return finite(n)?(100*n).toFixed(1)+'%':'—';}
function summary(n){const r=n.raw||{},p=identity(r);if(n.benchmark==='b1k2026'){const m=(r.native_metrics||[]).filter(x=>typeof x.success==='boolean');return {rate:m.length?m.filter(x=>x.success).length/m.length:null,episodes:m.length,state:r.state};}return {rate:r.success_rate,episodes:r.completed_episodes??r.recorded_episodes,state:r.status||r.state,checkpoint:p.checkpoint};}
function lineage(n){
 if(!n)return '<div class="empty">从左侧选择实验记录</div>';
 const focused=nodes.get(graphFocus);
 const chain=focused&&scope(focused.id)&&ancestors(focused).some(x=>x.id===n.id)?ancestors(focused):ancestors(n),levels=[];
 const tip=chain.at(-1);
 for(let i=0;i<chain.length;i++){
   const focus=chain[i],parent=chain[i-1];
   const siblings=parent?parent.children.map(id=>nodes.get(id)).filter(x=>scope(x.id)):[focus];
   let group=siblings.slice(0,3);
   if(!group.some(x=>x.id===focus.id))group=group.slice(0,2).concat(focus);
   levels.push({group,focus:focus.id,more:Math.max(0,siblings.length-group.length)});
 }
 const children=tip.children.map(id=>nodes.get(id)).filter(x=>scope(x.id));
 if(children.length)levels.push({group:children.slice(0,3),focus:null,more:Math.max(0,children.length-3)});
 const subtitle=x=>{const r=x.raw||{},p=r.parameter_summary||{};if(x.kind==='train')return [r.status?stateLabel(r.status):null,p.learning_rate!=null?'LR '+p.learning_rate:null].filter(Boolean).join(' · ');if(x.kind==='eval'){const s=summary(x);return [s.state?stateLabel(s.state):null,s.rate!=null?'SR '+pct(s.rate):null].filter(Boolean).join(' · ');}if(x.kind==='checkpoint')return [x.official?'官方来源':null,x.children.length?x.children.length+' 次评估':null].filter(Boolean).join(' · ');if(x.kind==='version')return x.children.length?x.children.length+' 项直接记录':'';return '';};
 return '<div class="lineage graph" id="lineage-graph"><svg class="graph-wires" aria-hidden="true"></svg>'+levels.map((level,i)=>{
  const kinds=[...new Set(level.group.map(x=>x.kind))],label=kinds.length===1?nodeTypes[kinds[0]]:'关联记录';
  return '<div class="graph-row"><div class="graph-label">'+esc(label)+(level.more?'<small>+'+level.more+'</small>':'')+'</div><div class="lineage-level" style="--columns:'+level.group.length+'">'+level.group.map(x=>'<button class="lineage-node kind-'+x.kind+' '+(x.id===selected?'active':x.id===level.focus?'branch':'')+'" data-node="'+esc(x.id)+'" data-kind="'+x.kind+'" aria-pressed="'+(x.id===selected)+'" title="'+esc(nodeTypes[x.kind]+' · '+x.title)+'"><span class="node-type">'+esc(nodeTypes[x.kind])+'</span><b>'+esc(x.title)+'</b>'+(subtitle(x)?'<small>'+esc(subtitle(x))+'</small>':'')+'</button>').join('')+'</div></div>';
 }).join('')+'</div>';
}
function drawWires(){
 const graph=$('lineage-graph');if(!graph)return;
 const svg=graph.querySelector('.graph-wires'),bounds=graph.getBoundingClientRect();
 if(!svg||!bounds.width||!bounds.height)return;
 svg.setAttribute('viewBox','0 0 '+bounds.width+' '+bounds.height);
 const rendered=new Map([...graph.querySelectorAll('.lineage-node')].map(el=>[el.dataset.node,el]));
 const branch=new Set(ancestors(nodes.get(selected)).map(n=>n.id));
 const ns='http://www.w3.org/2000/svg',fragment=document.createDocumentFragment();
 for(const [id,child] of rendered){
   const parentId=nodes.get(id)?.parent,parent=rendered.get(parentId);
   if(!parent)continue;
   const a=parent.getBoundingClientRect(),b=child.getBoundingClientRect();
   const x1=a.left+a.width/2-bounds.left,y1=a.bottom-bounds.top,x2=b.left+b.width/2-bounds.left,y2=b.top-bounds.top;
   if(y2<=y1)continue;
   const mid=(y1+y2)/2,edge=document.createElementNS(ns,'path');
   edge.setAttribute('d','M '+x1+' '+y1+' V '+mid+' H '+x2+' V '+y2);
   edge.setAttribute('class','graph-edge'+(branch.has(id)&&branch.has(parentId)?' selected-edge':''));
   edge.dataset.parent=parentId;edge.dataset.child=id;fragment.append(edge);
 }
 svg.replaceChildren(fragment);
}
function scheduleWires(){
 if(wireFrame)cancelAnimationFrame(wireFrame);
 wireFrame=requestAnimationFrame(()=>{wireFrame=0;drawWires();});
}
function observeGraph(){
 graphObserver?.disconnect();
 const graph=$('lineage-graph');if(!graph)return;
 if(typeof ResizeObserver==='function'){graphObserver=new ResizeObserver(scheduleWires);graphObserver.observe(graph);graph.querySelectorAll('.lineage-node').forEach(el=>graphObserver.observe(el));}
 scheduleWires();
}
window.addEventListener('resize',scheduleWires);
function saveSelection(){const u=new URL(location.href);u.searchParams.set('benchmark',benchmark);if(selected)u.searchParams.set('node',selected);else u.searchParams.delete('node');history.replaceState(null,'',u);}
function navigate(p,target,n){const b=target?.benchmark||(benchmark!=='all'?benchmark:currentBenchmark(n||nodes.get(selected)));const q=new URLSearchParams({page:p,benchmark:b});if(target)q.set('lineage_target',JSON.stringify(target));if(selected)q.set('lineage_node',selected);location.assign('/?'+q);}
function details(n){
 $('inspector-kind').textContent=n?nodeTypes[n.kind]:'';
 if(!n){$('inspector').innerHTML='<div class="empty">选择节点后查看详情</div>';return;}
 const r=n.raw||{},p=r.parameter_summary||{},s=n.kind==='eval'?summary(n):null,tr=getTrain(n),w=getWeight(n),owner=nodes.get(n.parent);
 let rows=[],actions=[];
 if(n.kind==='version'){
   rows=[['代码版本',n.version],['训练记录',descendants(n,'train').length],['已索引权重',descendants(n,'checkpoint').length],['评估记录',descendants(n,'eval').length]];
 }else if(n.kind==='train'){
   rows=[['代码版本',n.version],['任务',r.task],['方法',r.method||n.arm],['状态',stateLabel(r.status)],['设备',n.device],['微调',p.fine_tuning_mode],['学习率',p.learning_rate],['Batch',p.global_decisions??p.global_decisions_max],['步数',r.last?.step??r.step],['Loss',r.last?.loss],['训练 ID',r.training_id||n.run],['备注',r.notes]];
   actions.push(['training','查看训练详情 →',true]);
 }else if(n.kind==='checkpoint'){
   rows=[['代码版本',n.version],['来源',n.official?'官方权重':tr?'训练产物':'未记录'],['所属训练',tr?.title],['保存步数',r.step],['设备',n.device],['关联评估',descendants(n,'eval').length],['权重路径',n.path],['SHA256',r.sha]];
   if(tr&&n.benchmark==='b1k')actions.push(['evaluate','使用此权重评估 →',true],['training','查看所属训练 →',false]);
   else actions.push(['weights','查看模型资源 →',true]);
 }else if(n.kind==='eval'){
   rows=[['代码版本',n.version],['任务',r.task_id??r.task??r.config?.suite],['状态',stateLabel(s.state)],['回合',s.episodes],['成功率',pct(s.rate)],['权重',w?.title],['协议',typeof r.protocol==='string'?r.protocol:r.evaluation_mode],['指令',r.language?.mode],['RTC',r.rtc==null?null:r.rtc?'开启':'关闭'],['备注',r.notes]];
   actions.push([n.pending?'launch':'result',n.pending?'查看运行 →':'查看评估结果 →',true]);
 }else if(n.kind==='document')actions.push(['document','打开实验文档 →',true]);
 else rows.push(['关联记录',n.children.filter(scope).length]);
 const children=n.children.map(id=>nodes.get(id)).filter(x=>scope(x.id));
 const links=children.length?'<div class="group-title">直接关联 · '+children.length+'</div><div class="related-list">'+children.slice(0,6).map(x=>'<button data-node="'+esc(x.id)+'" title="'+esc(x.title)+'"><small>'+esc(nodeTypes[x.kind])+'</small><span>'+esc(x.title)+'</span><span aria-hidden="true">›</span></button>').join('')+(children.length>6?'<small class="muted">其余记录可在左侧目录查看</small>':'')+'</div>':'';
 $('inspector').innerHTML='<div class="identity"><h3>'+esc(n.title)+'</h3>'+(n.benchmark?'<small>'+esc(benchmarks[n.benchmark])+'</small>':'')+'</div><div class="node-actions">'+actions.map(([a,t,primary])=>'<button data-action="'+a+'" class="'+(primary?'primary':'')+'">'+t+'</button>').join('')+'</div>'+section('记录信息',rows)+links;
}
function render(){nav();drawTree();const n=nodes.get(selected);$('crumb').textContent=n?ancestors(n).map(x=>x.title).join(' / '):'';$('content').innerHTML=lineage(n);observeGraph();details(n);}
function choose(id){const n=nodes.get(id);if(!n||!scope(id))return;selected=id;ancestors(n).forEach(x=>expanded.add(x.id));const old=nodes.get(graphFocus);if(!old||!ancestors(old).some(x=>x.id===id))graphFocus=id;saveSelection();status('');render();if(n.kind==='train')void loadCheckpoints(n);}
async function loadCheckpoints(n){if(detailRequests.has(n.id))return detailRequests.get(n.id);const promise=(async()=>{try{const d=await api('/api/training/detail?'+new URLSearchParams({device:n.device,run:n.run,arm:n.arm}));detailCache.set(n.id,d);checkpointByTraining.set(n.id,Array.isArray(d.checkpoints)?d.checkpoints.filter(c=>c&&typeof c.path==='string'):[]);rebuild();render();}catch(e){status('权重读取失败：'+e.message,true);}})();detailRequests.set(n.id,promise);try{await promise;}finally{detailRequests.delete(n.id);}}
async function refresh(){if(refreshing)return;refreshing=true;$('reload').disabled=true;const errors=[];
 const sources=[['a100','/api/training/jobs?device=a100'],['a6000','/api/training/jobs?device=a6000'],['results','/api/results'],['native','/api/launch/jobs?benchmark=b1k2026'],['launch','/api/launch/jobs'],['libero','/api/benchmarks/jobs'],['experiments','/api/agent-experiments']];
 try{await Promise.all(sources.map(async ([key,url])=>{try{const d=await api(url);if(key==='a100'||key==='a6000'){if(!Array.isArray(d.jobs))throw Error('缺少 jobs');cache.train=cache.train.filter(x=>x.device!==key).concat(d.jobs.map(job=>({device:key,job})));}else{const rows=key==='results'?d.runs:key==='experiments'?d.experiments:d.jobs;if(!Array.isArray(rows))throw Error('记录格式错误');cache[key]=rows;}}catch(e){errors.push(key+'：'+e.message);}}));
 const requested=first?params.get('node'):null;rebuild();render();
 if(requested&&!nodes.has(requested)){try{const key=JSON.parse(requested);if(key[0]==='checkpoint'&&nodes.get(key[1])?.kind==='train'){const owner=nodes.get(key[1]);await loadCheckpoints(owner);if(nodes.has(requested)&&scope(requested))choose(requested);}}catch{}}
 const active=nodes.get(selected);if(active?.kind==='train'&&!detailCache.has(active.id))void loadCheckpoints(active);
 $('source-state').textContent=errors.length?'部分来源不可查 · '+errors.length+' 项':'已同步 · '+new Date().toLocaleTimeString();$('source-state').title=errors.join('；');
 }finally{$('reload').disabled=false;refreshing=false;}}
async function devices(){const el=$('device-content');el.textContent='读取设备…';try{const d=await api('/api/services');el.innerHTML=[['remote','A100'],['local','4090'],['a6000','RTX 6000 Ada']].map(([k,label])=>{const h=d[k]||{},g=Array.isArray(h.gpus)?h.gpus:[];return '<section class="gpu-group"><strong>'+label+'</strong><div class="gpu-grid">'+(h.ok!==false&&g.length?g.map(x=>{const total=Number(x.total_mb),used=Number(x.used_mb),has=Number.isFinite(total)&&total>0&&Number.isFinite(used);return '<div class="gpu-cell"><b>GPU '+esc(x.index)+'</b><div>'+esc(x.name)+'</div><div>'+ (has?(used/1024).toFixed(1)+' / '+(total/1024).toFixed(1)+' GiB':'显存未记录')+'</div>'+(has?'<progress max="'+total+'" value="'+Math.max(0,Math.min(used,total))+'"></progress>':'')+'<div>利用率 '+(finite(x.utilization)?x.utilization+'%':'—')+'</div></div>';}).join(''):'<span class="muted">'+esc(h.error||'设备状态不可查')+'</span>')+'</div></section>';}).join('');}catch(e){el.textContent='设备读取失败：'+e.message;}}
$('open-device-dialog').onclick=()=>{$('device-dialog').showModal();void devices();};$('close-device-dialog').onclick=()=>$('device-dialog').close();
$('nav').onclick=e=>{const b=e.target.closest('[data-page]');if(b&&b.dataset.page!=='experiment')navigate(b.dataset.page);};
$('benchmarks').onchange=e=>{benchmark=e.target.value;if(selected&&!scope(selected)){selected=null;graphFocus=null;rebuild();}if(selected)ancestors(nodes.get(selected)).forEach(x=>expanded.add(x.id));saveSelection();render();};
$('tree').onclick=e=>{const exp=e.target.closest('[data-expand]');if(exp){const id=exp.dataset.expand,n=nodes.get(id);if(expanded.has(id))expanded.delete(id);else{expanded.add(id);if(n?.kind==='train')void loadCheckpoints(n);}drawTree();return;}const b=e.target.closest('[data-node]');if(b)choose(b.dataset.node);};
$('content').onclick=e=>{const b=e.target.closest('[data-node]');if(b)choose(b.dataset.node);};
$('engineering').onchange=e=>{includeEngineering=e.target.checked;if(selected&&!scope(selected)){selected=null;graphFocus=null;rebuild();}render();};
$('collapse').onclick=()=>{expanded.clear();drawTree();};$('reload').onclick=()=>void refresh();
$('inspector').onclick=e=>{const pick=e.target.closest('[data-node]');if(pick){choose(pick.dataset.node);return;}const a=e.target.closest('[data-action]')?.dataset.action,n=nodes.get(selected);if(!a||!n)return;const tr=getTrain(n),w=getWeight(n);
 if(a==='document'){location.assign('/ai?'+new URLSearchParams({experiment:n.experiment,benchmark:currentBenchmark(n)}));return;}
 if(a==='training'&&tr)navigate('training',{type:'train',benchmark:tr.benchmark,device:tr.device,run:tr.run,arm:tr.arm},tr);
 if(a==='evaluate'&&tr&&w)navigate('training',{type:'checkpoint',benchmark:'b1k',device:tr.device,run:tr.run,arm:tr.arm,path:w.path},tr);
 if(a==='result')navigate('results',{type:'eval',benchmark:n.benchmark,run:n.run,folder:n.folder},n);
 if(a==='launch')navigate('launch',null,n);
 if(a==='weights')navigate('registry',null,n);
 if(a==='new-training')navigate('training',{type:'new',page:'training',benchmark:currentBenchmark(n)},n);
};
render();void refresh();setInterval(refresh,45000);
})();
