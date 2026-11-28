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
let page=Object.hasOwn(labels,params.get('page'))?params.get('page'):'experiment',benchmark=Object.hasOwn(benchmarks,params.get('benchmark'))?params.get('benchmark'):'all';
let selected=null,tab=page==='registry'?'datasets':page==='launch'?'operations':page==='experiment'?'lineage':'overview',expanded=new Set(),nodes=new Map(),roots=[],sourceErrors=[],refreshing=false,first=true,frameEpoch=0,frameURL='',frameRequest='',compare=new Set(),search='',includeEngineering=false;
let cache={train:[],results:[],native:[],libero:[],experiments:[],launch:[]};
const detailCache=new Map(),detailRequests=new Map(),checkpointByTraining=new Map();
let resourceInfo=null;
const resourceViews={datasets:'数据集',weights:'模型权重',downloads:'下载任务',disks:'磁盘空间'};
const pathValue=v=>typeof v==='string'?v:typeof v?.path==='string'?v.path:typeof v?.checkpoint==='string'?v.checkpoint:typeof v?.directory==='string'?v.directory:null;
function status(msg,error=false){$('message').className='notice'+(error?' error':'');$('message').textContent=msg||'';}
async function api(path){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),18000);try{const r=await fetch(path,{cache:'no-store',signal:controller.signal});const d=await r.json();if(!r.ok||d.error)throw Error(d.error||('HTTP '+r.status));return d;}finally{clearTimeout(timer);}}
function add(n,parent){if(nodes.has(n.id))return nodes.get(n.id);n.children=[];n.parent=parent?.id||null;nodes.set(n.id,n);if(parent)parent.children.push(n.id);else roots.push(n.id);return n;}
function versionNode(v){return add({id:jsonKey('version',v),kind:'version',title:v,version:v});}
function weightNode(parent,path,meta){return add({id:jsonKey('checkpoint',parent.id,path),kind:'checkpoint',title:meta.step!=null?'Step '+meta.step:meta.title||String(path).split('/').filter(Boolean).pop()||'权重',path,version:parent.version,benchmark:meta.benchmark||parent.benchmark,device:parent.device,train:parent.kind==='train'?parent.id:null,raw:meta,official:meta.official===true},parent);}
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
  if(cache.experiments.length){const group=add({id:'documents',kind:'group',title:'科研实验文档',version:'文档'});for(const id of cache.experiments)add({id:jsonKey('document',id),kind:'document',title:id.includes('/')?id.replace('/',' / '):'test_01 / '+id,experiment:id},group);}
  roots.sort((a,b)=>nodes.get(b).title.localeCompare(nodes.get(a).title,undefined,{numeric:true}));
  if(previous&&nodes.has(previous)&&scope(previous))selected=previous;
  else {selected=roots.find(id=>nodes.get(id).title==='V2.1'&&matchesBranch(id))||roots.find(id=>nodes.get(id).kind==='version'&&matchesBranch(id))||roots.find(matchesBranch)||null;}
  if(first&&selected){expanded.add(selected);first=false;}
}
function visibleSelf(n){return (includeEngineering||!n.engineering)&&(!n.benchmark||benchmark==='all'||n.benchmark===benchmark);}
function scope(id){const n=nodes.get(id);if(!n||!visibleSelf(n))return false;if(n.benchmark)return true;if(n.kind==='document')return true;return n.children.some(scope)||n.kind==='group'&&n.id==='documents';}
function matchesBranch(id){const n=nodes.get(id);if(!scope(id))return false;const text=[n.title,n.version,n.run,n.path,n.raw?.task_id,n.raw?.task,n.raw?.notes,n.raw?.training_id].join(' ').toLowerCase();return !search||text.includes(search)||n.children.some(matchesBranch);}
function nav(){ $('nav').innerHTML=Object.entries(labels).map(([k,v])=>'<button data-page="'+k+'" class="'+(page===k?'active':'')+'">'+v+'</button>').join(''); $('benchmarks').innerHTML=Object.entries(benchmarks).map(([k,v])=>'<button data-benchmark="'+k+'" class="'+(benchmark===k?'active':'')+'">'+v+'</button>').join('');}
function treeHTML(ids){return '<ul>'+ids.filter(matchesBranch).map(id=>{const n=nodes.get(id),children=n.children.filter(matchesBranch),open=expanded.has(id)||!!search,folder=['version','train','checkpoint','group'].includes(n.kind);return '<li><div class="tree-row"><button class="toggle" data-expand="'+esc(id)+'" aria-label="'+(open?'收起':'展开')+'" '+(!children.length&&n.kind!=='train'?'disabled':'')+'>'+(children.length||n.kind==='train'?(open?'▾':'▸'):'·')+'</button><button class="pick '+(selected===id?'active':'')+'" data-node="'+esc(id)+'" title="'+esc(n.title)+'"><i class="'+(folder?'folder':'file')+'"></i><span>'+esc(n.title)+'</span>'+(children.length?'<small class="count">'+children.length+'</small>':'')+'</button></div>'+(open&&children.length?treeHTML(children):'')+'</li>';}).join('')+'</ul>';}
function drawTree(){const el=$('tree'),scroll=el.scrollTop;el.innerHTML=treeHTML(roots)||'<div class="empty">暂无记录</div>';el.scrollTop=scroll;}
function descendants(n,kind){return n?(n.kind===kind?[n]:[]).concat(n.children.flatMap(id=>descendants(nodes.get(id),kind))).filter(x=>visibleSelf(x)):[];}
function ancestors(n){const rows=[];for(let x=n;x;x=nodes.get(x.parent))rows.unshift(x);return rows;}
function getTrain(n){return n?.kind==='train'?n:ancestors(n).find(x=>x.kind==='train');}
function getWeight(n){return n?.kind==='checkpoint'?n:ancestors(n).find(x=>x.kind==='checkpoint');}
function currentBenchmark(n){return n?.benchmark||(benchmark==='all'?'b1k':benchmark);}
function closeFrame(){frameEpoch++;frameURL='';frameRequest='';$('panel').onload=null;$('panel').hidden=true;$('panel').removeAttribute('src');}
function openResource(){
 const view=resourceViews[tab]?tab:'datasets';tab=view;
 const url='/registry?embedded=1&section='+encodeURIComponent(view);
 $('content').innerHTML='';$('panel').hidden=false;
 if(frameURL===url)return;
 frameEpoch++;frameRequest='';frameURL=url;resourceInfo=null;status('');resourceInspector();
 $('panel').onload=null;$('panel').src=url;
}
function resourceInspector(){
 $('inspector-title').textContent='资源详情';$('kind').textContent='';
 const item=resourceInfo;
 const view=resourceViews[tab]||'数据集';
 $('inspector').innerHTML=item?'<div class="identity"><h3>'+esc(item.name||'未命名资源')+'</h3><small>'+esc(item.kind==='weights'?'模型权重':'数据集')+'</small></div>'+section('资源信息',[['存放设备',item.host],['容量',item.size],['更新时间',item.updated],['文件数',item.count],['完整路径',item.path],['状态',item.removed?'已移除':'已登记']]):'<div class="identity"><h3>'+esc(view)+'</h3><small>'+({'datasets':'选择数据集，查看真实目录与元数据。','weights':'选择权重，查看文件及配置。','downloads':'下载操作需通过预检。','disks':'实时容量与设备状态。'}[tab]||'')+'</small></div>';
 $('selection').innerHTML='';
}
function openFrame(p,n,target){
  const b=currentBenchmark(n),query=new URLSearchParams({embedded:'1',benchmark:b});
  if(p!=='results'&&p!=='experiment')query.set('page',p);
  if(n?.kind==='document')query.set('experiment',n.experiment);
  const url=(p==='experiment'?'/ai':'/')+'?'+query,requestKey=JSON.stringify(target||{});
  $('content').innerHTML='';$('panel').hidden=false;
  if(frameURL===url&&frameRequest===requestKey)return;
  const token=++frameEpoch;frameURL=url;frameRequest=requestKey;status('');
  const select=async()=>{if(token!==frameEpoch||!target)return;try{const bridge=$('panel').contentWindow.unifiedWorkbenchPanel;if(!bridge)throw Error('详情面板版本不匹配');await bridge.select(target);if(token===frameEpoch)status('');}catch(e){if(token===frameEpoch)status('无法打开记录：'+e.message,true);}};
  if($('panel').getAttribute('src')===url&&$('panel').contentWindow?.unifiedWorkbenchPanel){void select();return;}
  $('panel').onload=()=>void select();$('panel').src=url;
}
function section(title,rows){rows=rows.filter(([,v])=>v!==undefined&&v!==null&&v!=='');if(!rows.length)return '';return '<div class="group-title">'+esc(title)+'</div><dl class="kv">'+rows.map(([k,v])=>'<dt>'+esc(k)+'</dt><dd>'+esc(Array.isArray(v)?v.join(', '):v)+'</dd>').join('')+'</dl>';}
function pct(n){return finite(n)?(100*n).toFixed(1)+'%':'—';}
function summary(n){const r=n.raw||{},p=identity(r);if(n.benchmark==='b1k2026'){const m=(r.native_metrics||[]).filter(x=>typeof x.success==='boolean');return {rate:m.length?m.filter(x=>x.success).length/m.length:null,episodes:m.length,state:r.state};}return {rate:r.success_rate,episodes:r.completed_episodes??r.recorded_episodes,state:r.status||r.state,checkpoint:p.checkpoint};}
function inspect(n){if(page==='registry'){resourceInspector();return;}$('inspector-title').textContent='节点详情';$('kind').textContent=n?.kind?({version:'版本',train:'训练',checkpoint:'权重',eval:'评估',document:'文档',group:'分组'})[n.kind]:'';if(!n){$('inspector').innerHTML='<div class="empty">选择目录节点</div>';$('selection').innerHTML='';return;}
 const tr=getTrain(n),ck=getWeight(n),r=n.raw||{},p=tr?.raw?.parameter_summary||{},i=identity(r);$('inspector').innerHTML='<div class="identity"><h3>'+esc(n.title)+'</h3><small>'+esc(n.benchmark?benchmarks[n.benchmark]:n.kind==='version'?'网络版本':n.version||'科研文档')+'</small></div>'+section('A · 网络',[['版本',n.version],['方法',tr?.raw?.method||r.method||i.method],['源码',tr?.raw?.source_digest||r.source_revision]])+section('T · 训练',[['训练 ID',tr?.raw?.training_id||tr?.run],['微调',p.fine_tuning_mode],['学习率',p.learning_rate],['Batch',p.global_decisions??p.global_decisions_max],['调度器',p.scheduler],['初始权重',p.checkpoint],['设备',tr?.device]])+section('I · 推理',[['权重',ck?.path||pathValue(i.checkpoint)],['记忆模式',i.history_mode||r.configuration?.history_mode],['RTC',r.rtc==null?null:r.rtc?'开启':'关闭'],['指令',r.language?.mode]])+section('E · 评估',[['Benchmark',n.benchmark?benchmarks[n.benchmark]:null],['任务',r.task_id??r.config?.suite],['状态',r.state||r.status?stateLabel(r.state||r.status):null],['结果',n.kind==='eval'?pct(summary(n).rate):null],['备注',r.notes||r.config?.notes]]);
 $('selection').innerHTML=compare.size===0?'':'<h2>对比篮 · '+compare.size+'</h2>'+[...compare].filter(id=>nodes.has(id)).map(id=>'<div class="tag">'+esc(nodes.get(id).title)+' <button data-remove="'+esc(id)+'" aria-label="移除">×</button></div>').join('')+'<button data-action="compare" '+(compare.size<2?'disabled':'')+'>对比所选</button><button data-action="clear-compare">清空</button>';
}
function metric(label,value){return '<div class="metric"><small>'+esc(label)+'</small><strong>'+esc(value)+'</strong></div>';}
function table(headers,rows){return '<div class="table-scroll"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function nodeButton(n){return '<button class="record-link" data-node="'+esc(n.id)+'">'+esc(n.title)+'</button>';}
function pickButton(n){return '<button data-add="'+esc(n.id)+'">'+(compare.has(n.id)?'已加入':'对比 ＋')+'</button>';}
function overview(n){const ts=descendants(n,'train'),es=descendants(n,'eval'),ws=descendants(n,'checkpoint');let h='<div class="metrics">'+metric('训练',ts.length)+metric('已索引权重',ws.length)+metric('评估 / 记录',es.length)+'</div>';if(ts.length)h+='<div class="panel-card"><h3>训练总览</h3>'+table(['记录','状态','LR','Batch','Loss','操作'],ts.map(t=>{const j=t.raw,p=j.parameter_summary||{};return [nodeButton(t),esc(stateLabel(j.status)),esc(p.learning_rate),esc(p.global_decisions??p.global_decisions_max),esc(j.last?.loss),pickButton(t)];}))+'</div>';if(es.length)h+='<div class="panel-card"><h3>评估总览</h3>'+table(['记录','状态','成功率','回合','操作'],es.map(e=>{const s=summary(e);return [nodeButton(e),esc(stateLabel(s.state)),esc(pct(s.rate)),esc(s.episodes),pickButton(e)];}))+'</div>';if(n?.kind==='document')h+='<button data-action="document">查看实验方案与进度</button>';return h;}
function lineage(n){
 if(!n)return '<div class="empty">选择实验节点</div>';
 const chain=ancestors(n),levels=[];
 // Show siblings at each layer but descend only along the selected branch.
 // This avoids visual edges between unrelated training runs.
 for(let i=0;i<chain.length;i++){
   const focus=chain[i],parent=chain[i-1];
   const siblings=parent?parent.children.map(id=>nodes.get(id)).filter(x=>scope(x.id)):[focus];
   let group=siblings.slice(0,3);
   if(!group.some(x=>x.id===focus.id))group=group.slice(0,2).concat(focus);
   levels.push({group,focus:focus.id,kind:focus.kind,more:Math.max(0,siblings.length-group.length)});
 }
 const children=n.children.map(id=>nodes.get(id)).filter(x=>scope(x.id));
 if(children.length){const group=children.slice(0,3);levels.push({group,focus:null,kind:group.every(x=>x.kind===group[0].kind)?group[0].kind:'group',more:Math.max(0,children.length-group.length)});}
 const names={version:'网络版本',train:'训练记录',checkpoint:'模型权重',eval:'关联评估',group:'关联记录',document:'实验文档'};
 const subtitle=x=>{const r=x.raw||{},p=r.parameter_summary||{};if(x.kind==='train')return [stateLabel(r.status),p.learning_rate!=null?'LR '+p.learning_rate:null,p.global_decisions!=null?'Batch '+p.global_decisions:p.global_decisions_max!=null?'Batch '+p.global_decisions_max:null].filter(Boolean).join(' · ');if(x.kind==='eval'){const s=summary(x);return [stateLabel(s.state),s.rate!=null?'SR '+pct(s.rate):null].filter(Boolean).join(' · ');}if(x.kind==='checkpoint')return [x.official?'官方权重':null,x.children.length?x.children.length+' 次评估':null].filter(Boolean).join(' · ');return x.children.length?x.children.length+' 项记录':'';};
 const wire=(previous,level)=>{
   const a=previous.group.findIndex(x=>x.id===previous.focus);if(a<0)return '';
   const sx=300*(a+.5)/previous.group.length;
   const xs=level.group.map((_,i)=>300*(i+.5)/level.group.length);
   return '<svg class="lineage-wire" viewBox="0 0 300 28" preserveAspectRatio="none" aria-hidden="true"><path d="M'+sx+' 0 V14 M'+Math.min(sx,...xs)+' 14 H'+Math.max(sx,...xs)+' '+xs.map(x=>'M'+x+' 14 V28').join(' ')+'" stroke="#9eb0c8" stroke-width="1.1" fill="none" vector-effect="non-scaling-stroke"/></svg>';
 };
 return '<div class="lineage panel-card"><div class="lineage-title">实验谱系</div>'+levels.map((level,i)=>(i?wire(levels[i-1],level):'')+'<div class="layer-label">'+esc(names[level.kind]||'关联记录')+(level.more?' · 另 '+level.more+' 项见目录':'')+'</div><div class="lineage-level '+(level.group.length===1?'single':'')+'">'+level.group.map(x=>'<button class="lineage-node '+(x.id===selected?'active':x.id===level.focus?'branch':'')+'" data-node="'+esc(x.id)+'" title="'+esc(x.title)+'"><b>'+esc(x.title)+'</b>'+(subtitle(x)?'<small>'+esc(subtitle(x))+'</small>':'')+'</button>').join('')+'</div>').join('')+'</div>';
}
function lossOverlay(items){
 const runs=items.filter(n=>n.kind==='train');
 if(runs.length<2)return '';
 const curves=runs.map(n=>({n,points:(detailCache.get(n.id)?.loss_series||[]).filter(p=>Array.isArray(p)&&finite(p[0])&&finite(p[1]))}));
 const pending=runs.filter(n=>!detailCache.has(n.id));
 if(pending.length)return '<div class="panel-card"><h3>Loss 曲线对照</h3><p class="notice">尚未加载 '+pending.length+' 条曲线。</p><button data-action="load-curves">读取真实曲线</button></div>';
 const valid=curves.filter(c=>c.points.length);
 if(!valid.length)return '<div class="panel-card"><h3>Loss 曲线对照</h3><p class="notice">没有可绘制的原始 Loss 数据。</p></div>';
 let xmax=1,ymin=Infinity,ymax=-Infinity;
 for(const c of valid)for(const [x,y] of c.points){xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
 if(ymax===ymin){ymax+=.001;ymin-=.001;}
 const colors=['#285fc6','#75869f','#966cb4','#11879b','#b07f38','#567a5b'];
 const xy=p=>[52+548*p[0]/xmax,206-184*(p[1]-ymin)/(ymax-ymin)];
 const lines=valid.map((c,i)=>{const stride=Math.max(1,Math.ceil(c.points.length/3000)),points=c.points.filter((_,k)=>k%stride===0||k===c.points.length-1);return '<polyline fill="none" stroke="'+colors[i%colors.length]+'" stroke-width="1.6" points="'+points.map(p=>xy(p).map(v=>v.toFixed(2)).join(',')).join(' ')+'"/>';}).join('');
 const grid=Array.from({length:5},(_,i)=>{const y=22+46*i,value=ymax-(ymax-ymin)*i/4;return '<path d="M52 '+y+' H600" stroke="#e0e7f1"/><text x="46" y="'+(y+4)+'" text-anchor="end" font-size="10" fill="#687c98">'+esc(value.toPrecision(3))+'</text>';}).join('');
 return '<div class="panel-card"><h3>Loss 曲线对照</h3><svg class="chart" viewBox="0 0 620 234" role="img" aria-label="原始训练 Loss 对照">'+grid+lines+'<text x="52" y="225" font-size="10" fill="#687c98">0</text><text x="600" y="225" text-anchor="end" font-size="10" fill="#687c98">'+esc(xmax)+' steps</text></svg>'+valid.map((c,i)=>'<span class="tag" style="border-left:3px solid '+colors[i%colors.length]+'">'+esc(c.n.title)+'</span>').join('')+'<p class="notice">使用记录中的 optimizer step 与 Loss；不同损失定义、数据或归一化条件不应直接比较。</p></div>';
}
function comparison(){const items=[...compare].map(id=>nodes.get(id)).filter(Boolean);if(items.length<2)return '<div class="empty">至少选择两条记录</div>';const rows=[['节点类型',n=>n.kind],['代码版本',n=>n.version],['Benchmark',n=>benchmarks[n.benchmark]],['方法',n=>n.raw?.method||identity(n.raw||{}).method],['微调',n=>getTrain(n)?.raw?.parameter_summary?.fine_tuning_mode],['学习率',n=>getTrain(n)?.raw?.parameter_summary?.learning_rate],['Batch',n=>getTrain(n)?.raw?.parameter_summary?.global_decisions??getTrain(n)?.raw?.parameter_summary?.global_decisions_max],['Scheduler',n=>getTrain(n)?.raw?.parameter_summary?.scheduler],['权重',n=>getWeight(n)?.path||pathValue(identity(n.raw||{}).checkpoint)],['任务',n=>n.raw?.task_id??n.raw?.task],['指令模式',n=>n.raw?.language?.mode],['评估协议',n=>n.raw?.evaluation_mode||n.raw?.protocol],['成功率',n=>n.kind==='eval'?pct(summary(n).rate):null],['最后 Loss',n=>n.raw?.last?.loss]];return lossOverlay(items)+'<div class="panel-card"><h3>配置对照</h3><p class="notice">不同协议、数据集或 Loss 定义的数值不代表可直接比较。</p>'+table(['配置',...items.map(n=>n.title)],rows.map(([key,get])=>{const values=items.map(get),diff=new Set(values.filter(v=>v!=null).map(String)).size>1;return [esc(key),...values.map(v=>'<span class="'+(diff?'diff':'')+'">'+esc(v)+'</span>')];}))+'</div>';}
function tabs(n){if(page==='registry'){$('tabs').innerHTML=Object.entries(resourceViews).map(([k,v])=>'<button data-tab="'+k+'" class="'+(tab===k?'active':'')+'">'+v+'</button>').join('');return;}const list=[['overview','总览'],['lineage','谱系']];if(page==='experiment'&&n?.kind==='document')list.push(['document','方案与进度']);if(page==='training'&&getTrain(n))list.push(['detail','训练详情']);if(page==='results'&&n?.kind==='eval')list.push(['detail','回合与录像']);if(page==='launch')list.push(['operations','运行 / 新建']);if(compare.size>=2)list.push(['compare','配置对比']);$('tabs').innerHTML=list.map(([k,v])=>'<button data-tab="'+k+'" class="'+(tab===k?'active':'')+'">'+v+'</button>').join('');}
function renderCenter(){const n=nodes.get(selected);document.body.dataset.page=page;if(page==='registry'){$('crumb').textContent='资源管理';$('title').textContent='资源管理';$('actions').innerHTML='';if(!resourceViews[tab])tab='datasets';tabs(n);openResource();resourceInspector();return;}$('crumb').textContent=[labels[page],...ancestors(n).map(x=>x.title)].join(' / ');$('title').textContent=n?n.title:labels[page];let acts=[];const tr=getTrain(n),ck=getWeight(n);if(n&&['train','eval','checkpoint'].includes(n.kind))acts.push('<button data-add="'+esc(n.id)+'">'+(compare.has(n.id)?'已加入对比':'加入对比')+'</button>');if(tr&&page!=='training')acts.push('<button data-action="training">训练详情</button>');if(n?.kind==='eval'&&page!=='results')acts.push('<button data-action="result">评估详情</button>');if(ck&&ck.train&&ck.benchmark==='b1k')acts.push('<button class="primary" data-action="evaluate">使用权重评估</button>');else if(tr)acts.push('<button class="primary" data-action="evaluate-latest">选择权重评估</button>');if(page==='training')acts.push('<button '+(tr?'':'class="primary"')+' data-action="new-training">＋ 新建训练</button>');if(page==='launch'&&!tr)acts.push('<button class="primary" data-action="new-launch">＋ 新建评估</button>');$('actions').innerHTML=acts.join('');tabs(n);inspect(n);
 if(tab==='detail'&&page==='training'&&tr){openFrame('training',tr,{type:'train',device:tr.device,run:tr.run,arm:tr.arm});return;}
 if(tab==='detail'&&page==='results'&&n?.kind==='eval'){if(n.pending){openFrame('launch',n);status('尚未归档结果，请在评估作业中查看运行进度。');}else openFrame('results',n,{type:'eval',run:n.run,folder:n.folder});return;}
 if(tab==='document'&&n?.kind==='document'){openFrame('experiment',n,{type:'experiment'});return;}
 if(tab==='operations'&&['launch','registry'].includes(page)){openFrame(page,n);return;}
 closeFrame();$('content').innerHTML=tab==='compare'?comparison():tab==='lineage'?lineage(n):overview(n);
}
function render(){nav();drawTree();renderCenter();}
function route(p){if(page!==p){closeFrame();resourceInfo=null;status('');}page=p;tab=p==='training'&&getTrain(nodes.get(selected))?'detail':p==='results'&&nodes.get(selected)?.kind==='eval'?'detail':p==='registry'?'datasets':p==='launch'?'operations':p==='experiment'?'lineage':'overview';const u=new URL(location.href);u.searchParams.set('page',page);u.searchParams.set('benchmark',benchmark);history.replaceState(null,'',u);render();}
function choose(id){const n=nodes.get(id);if(!n)return;selected=id;ancestors(n).forEach(x=>expanded.add(x.id));tab=n.kind==='document'&&page==='experiment'?'document':page==='training'&&getTrain(n)?'detail':page==='results'&&n.kind==='eval'?'detail':page==='experiment'?'lineage':page==='registry'?(resourceViews[tab]?tab:'datasets'):page==='launch'?'operations':'overview';render();if(n.kind==='train')void loadCheckpoints(n);}
async function loadCheckpoints(n){if(detailRequests.has(n.id))return detailRequests.get(n.id);const promise=(async()=>{try{const d=await api('/api/training/detail?'+new URLSearchParams({device:n.device,run:n.run,arm:n.arm}));detailCache.set(n.id,d);checkpointByTraining.set(n.id,Array.isArray(d.checkpoints)?d.checkpoints.filter(c=>c&&typeof c.path==='string'):[]);rebuild();drawTree();if(selected===n.id&&tab!=='detail'&&tab!=='compare')renderCenter();}catch(e){status('权重读取失败：'+e.message,true);}})();detailRequests.set(n.id,promise);try{await promise;}finally{detailRequests.delete(n.id);}}
async function refresh(){if(refreshing)return;refreshing=true;$('reload').disabled=true;const errors=[];
 const sources=[['a100','/api/training/jobs?device=a100'],['a6000','/api/training/jobs?device=a6000'],['results','/api/results'],['native','/api/launch/jobs?benchmark=b1k2026'],['launch','/api/launch/jobs'],['libero','/api/benchmarks/jobs'],['experiments','/api/agent-experiments']];
 await Promise.all(sources.map(async ([key,url])=>{try{const d=await api(url);if(key==='a100'||key==='a6000'){if(!Array.isArray(d.jobs))throw Error('缺少 jobs');cache.train=cache.train.filter(x=>x.device!==key).concat(d.jobs.map(job=>({device:key,job})));}else{const rows=key==='results'?d.runs:key==='experiments'?d.experiments:d.jobs;if(!Array.isArray(rows))throw Error('记录格式错误');cache[key]=rows;}}catch(e){errors.push(key+'：'+e.message);}}));
 sourceErrors=errors;rebuild();if(selected&&!scope(selected))selected=roots.find(matchesBranch)||null;drawTree();if($('panel').hidden)renderCenter();else inspect(nodes.get(selected));$('source-state').textContent=errors.length?'部分设备 / 接口不可查 · '+errors.length+' 项':'已同步 · '+new Date().toLocaleTimeString();$('source-state').title=errors.join('；');$('reload').disabled=false;refreshing=false;
}
$('nav').onclick=e=>{const b=e.target.closest('[data-page]');if(b)route(b.dataset.page);};
$('benchmarks').onclick=e=>{const b=e.target.closest('[data-benchmark]');if(!b)return;benchmark=b.dataset.benchmark;if(selected&&!scope(selected))selected=roots.find(matchesBranch)||null;if(page==='registry'){nav();drawTree();return;}closeFrame();route(page);};
$('tree').onclick=e=>{const exp=e.target.closest('[data-expand]');if(exp){const id=exp.dataset.expand,n=nodes.get(id);if(expanded.has(id))expanded.delete(id);else{expanded.add(id);if(n?.kind==='train')void loadCheckpoints(n);}drawTree();return;}const b=e.target.closest('[data-node]');if(b)choose(b.dataset.node);};
$('search').oninput=e=>{search=e.target.value.trim().toLowerCase();drawTree();};$('engineering').onchange=e=>{includeEngineering=e.target.checked;if(selected&&!scope(selected)){selected=roots.find(matchesBranch)||null;closeFrame();tab='overview';}render();};$('clear').onclick=()=>{$('search').value='';search='';drawTree();};$('collapse').onclick=()=>{expanded.clear();drawTree();};$('reload').onclick=()=>void refresh();
$('tabs').onclick=e=>{const b=e.target.closest('[data-tab]');if(b){tab=b.dataset.tab;renderCenter();}};
document.body.addEventListener('click',async e=>{if(e.target.closest('#tree'))return;const b=e.target.closest('[data-action],[data-node],[data-add],[data-remove]');if(!b)return;if(b.dataset.node){choose(b.dataset.node);return;}if(b.dataset.add){compare.add(b.dataset.add);inspect(nodes.get(selected));tabs(nodes.get(selected));return;}if(b.dataset.remove){compare.delete(b.dataset.remove);renderCenter();return;}const n=nodes.get(selected),tr=getTrain(n),ck=getWeight(n);
 switch(b.dataset.action){case 'compare':tab='compare';renderCenter();break;case 'load-curves':{const trains=[...compare].map(id=>nodes.get(id)).filter(x=>x?.kind==='train');status('读取原始 Loss 曲线…');await Promise.all(trains.map(loadCheckpoints));if(tab==='compare')renderCenter();status('');break;}case 'clear-compare':compare.clear();tab='overview';renderCenter();break;case 'training':route('training');break;case 'result':route('results');break;case 'document':tab='document';renderCenter();break;case 'new-training':page='training';tab='detail';nav();openFrame('training',n,{type:'new',page:'training'});break;case 'new-launch':page='launch';tab='operations';nav();openFrame('launch',n,{type:'new',page:'launch'});break;case 'evaluate':if(ck&&tr){page='launch';tab='operations';nav();openFrame('training',tr,{type:'checkpoint',benchmark:'b1k',device:tr.device,run:tr.run,arm:tr.arm,path:ck.path});}break;case 'evaluate-latest':if(tr){await loadCheckpoints(tr);const cp=(checkpointByTraining.get(tr.id)||[]).slice().sort((a,b)=>(b.step||0)-(a.step||0))[0];if(!cp){status('暂无已发布权重；请查看训练详情。',true);return;}const id=jsonKey('checkpoint',tr.id,cp.path);choose(id);page='launch';tab='operations';nav();openFrame('training',tr,{type:'checkpoint',benchmark:'b1k',device:tr.device,run:tr.run,arm:tr.arm,path:cp.path});}break;}
});
window.addEventListener('message',e=>{if(e.origin!==location.origin||e.source!==$('panel').contentWindow||$('panel').hidden)return;if(e.data?.type==='memoryvla-resource'){if(page==='registry'&&e.data.section===tab){resourceInfo=e.data.item||null;resourceInspector();}return;}if(e.data?.type!=='memoryvla-embedded-page'||page==='registry')return;const p=e.data.page;if(p==='launch'&&page!=='launch'){page='launch';tab='operations';nav();tabs(nodes.get(selected));}});
render();void refresh();setInterval(refresh,45000);
})();