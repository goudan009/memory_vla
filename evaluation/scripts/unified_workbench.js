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
let graphFocus=null,form=null;
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
  if(first&&selected){
    const root=nodes.get(selected);
    if(page==='experiment'&&root.kind==='version'){
      const child=root.children.map(id=>nodes.get(id)).find(n=>scope(n.id)&&n.kind==='train')||root.children.map(id=>nodes.get(id)).find(n=>scope(n.id)&&n.kind==='checkpoint');
      if(child)selected=child.id;
    }
    ancestors(nodes.get(selected)).forEach(n=>expanded.add(n.id));graphFocus=selected;first=false;if(page!=='registry'&&page!=='launch')tab=defaultTab(page,nodes.get(selected));
  }
}
function visibleSelf(n){return (includeEngineering||!n.engineering)&&(!n.benchmark||benchmark==='all'||n.benchmark===benchmark);}
function scope(id){const n=nodes.get(id);if(!n||!visibleSelf(n))return false;if(n.benchmark)return true;if(n.kind==='document')return true;return n.children.some(scope)||n.kind==='group'&&n.id==='documents';}
function matchesBranch(id){const n=nodes.get(id);if(!scope(id))return false;const text=[n.title,n.version,n.run,n.path,n.raw?.task_id,n.raw?.task,n.raw?.notes,n.raw?.training_id].join(' ').toLowerCase();return !search||text.includes(search)||n.children.some(matchesBranch);}
function nav(){
 $('nav').innerHTML=Object.entries(labels).map(([k,v])=>'<button data-page="'+k+'" class="'+(page===k?'active':'')+'" aria-current="'+(page===k?'page':'false')+'">'+v+'</button>').join('');
 const select=$('benchmarks');if(!select.options.length)select.innerHTML=Object.entries(benchmarks).map(([k,v])=>'<option value="'+k+'">'+v+'</option>').join('');select.value=benchmark;
}
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
 frameEpoch++;frameRequest='';frameURL=url;status('');
 $('panel').onload=null;$('panel').src=url;
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
function compareItems(){
 return '<div class="compare-items">'+[...compare].filter(id=>nodes.has(id)).map(id=>'<span class="tag">'+esc(nodes.get(id).title)+' <button data-remove="'+esc(id)+'" aria-label="移除">×</button></span>').join('')+(compare.size?'<button data-action="clear-compare">清空</button>':'')+'</div>';
}
function metric(label,value){return '<div class="metric"><small>'+esc(label)+'</small><strong>'+esc(value)+'</strong></div>';}
function table(headers,rows){return '<div class="table-scroll"><table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function nodeButton(n){return '<button class="record-link" data-node="'+esc(n.id)+'">'+esc(n.title)+'</button>';}
function pickButton(n){return '<button data-add="'+esc(n.id)+'">'+(compare.has(n.id)?'已加入':'对比 ＋')+'</button>';}
function overview(n){const ts=descendants(n,'train'),es=descendants(n,'eval'),ws=descendants(n,'checkpoint');let h='<div class="metrics">'+metric('训练',ts.length)+metric('已索引权重',ws.length)+metric('评估 / 记录',es.length)+'</div>';if(ts.length&&page!=='results')h+='<div class="panel-card"><h3>训练总览</h3>'+table(['记录','状态','LR','Batch','Loss','操作'],ts.map(t=>{const j=t.raw,p=j.parameter_summary||{};return [nodeButton(t),esc(stateLabel(j.status)),esc(p.learning_rate),esc(p.global_decisions??p.global_decisions_max),esc(j.last?.loss),pickButton(t)];}))+'</div>';if(es.length&&page!=='training')h+='<div class="panel-card"><h3>评估总览</h3>'+table(['记录','状态','成功率','回合','操作'],es.map(e=>{const s=summary(e);return [nodeButton(e),esc(stateLabel(s.state)),esc(pct(s.rate)),esc(s.episodes),pickButton(e)];}))+'</div>';if(n?.kind==='document')h+='<button data-action="document">查看实验方案与进度</button>';return h;}
function lineage(n){
 if(!n)return '<div class="empty">选择实验节点</div>';
 const focused=nodes.get(graphFocus);const chain=focused&&ancestors(focused).some(x=>x.id===n.id)?ancestors(focused):ancestors(n),levels=[];const tip=chain.at(-1);
 // Show siblings at each layer but descend only along the selected branch.
 // This avoids visual edges between unrelated training runs.
 for(let i=0;i<chain.length;i++){
   const focus=chain[i],parent=chain[i-1];
   const siblings=parent?parent.children.map(id=>nodes.get(id)).filter(x=>scope(x.id)):[focus];
   let group=siblings.slice(0,3);
   if(!group.some(x=>x.id===focus.id))group=group.slice(0,2).concat(focus);
   levels.push({group,focus:focus.id,kind:focus.kind,more:Math.max(0,siblings.length-group.length)});
 }
 const children=tip.children.map(id=>nodes.get(id)).filter(x=>scope(x.id));
 if(children.length){const group=children.slice(0,3);levels.push({group,focus:null,kind:group.every(x=>x.kind===group[0].kind)?group[0].kind:'group',more:Math.max(0,children.length-group.length)});}
 const names={version:'网络版本',train:'训练记录',checkpoint:'模型权重',eval:'关联评估',group:'关联记录',document:'实验文档'};
 const subtitle=x=>{const r=x.raw||{},p=r.parameter_summary||{};if(x.kind==='train')return [stateLabel(r.status),p.learning_rate!=null?'LR '+p.learning_rate:null,p.global_decisions!=null?'Batch '+p.global_decisions:p.global_decisions_max!=null?'Batch '+p.global_decisions_max:null].filter(Boolean).join(' · ');if(x.kind==='eval'){const s=summary(x);return [stateLabel(s.state),s.rate!=null?'SR '+pct(s.rate):null].filter(Boolean).join(' · ');}if(x.kind==='checkpoint')return [x.official?'官方权重':null,x.children.length?x.children.length+' 次评估':null].filter(Boolean).join(' · ');return x.children.length?x.children.length+' 项记录':'';};
 const wire=(previous,level)=>{
   const a=previous.group.findIndex(x=>x.id===previous.focus);if(a<0)return '';
   const sx=300*(a+.5)/previous.group.length;
   const xs=level.group.map((_,i)=>300*(i+.5)/level.group.length);
   return '<svg class="lineage-wire" viewBox="0 0 300 28" preserveAspectRatio="none" aria-hidden="true"><path d="M'+sx+' 0 V14 M'+Math.min(sx,...xs)+' 14 H'+Math.max(sx,...xs)+' '+xs.map(x=>'M'+x+' 14 V28').join(' ')+'" stroke="#9eb0c8" stroke-width="1.1" fill="none" vector-effect="non-scaling-stroke"/></svg>';
 };
 return '<div class="lineage panel-card"><div class="lineage-context"><h3>实验谱系</h3><small>点击节点选择 · 上方按钮打开详情</small></div>'+levels.map((level,i)=>(i?wire(levels[i-1],level):'')+'<div class="layer-label">'+esc(names[level.kind]||'关联记录')+(level.more?' · 另 '+level.more+' 项见目录':'')+'</div><div class="lineage-level '+(level.group.length===1?'single':'')+'">'+level.group.map(x=>'<button class="lineage-node '+(x.id===selected?'active':x.id===level.focus?'branch':'')+'" data-node="'+esc(x.id)+'" title="'+esc(x.title)+'"><b>'+esc(x.title)+'</b>'+(subtitle(x)?'<small>'+esc(subtitle(x))+'</small>':'')+'</button>').join('')+'</div>').join('')+'</div>';
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
function defaultTab(p,n){
 if(p==='experiment')return n?.kind==='document'?'document':'lineage';
 if(p==='training')return getTrain(n)?'detail':'overview';
 if(p==='results')return n?.kind==='eval'&&!n.pending?'detail':'overview';
 if(p==='launch')return 'operations';
 if(p==='registry')return 'datasets';
 return 'overview';
}
function tabList(n){
 if(page==='registry')return Object.entries(resourceViews);
 const list=[];
 if(page==='experiment'){if(n?.kind==='document')list.push(['document','实验文档']);else list.push(['lineage','实验谱系']);list.push(['overview','实验记录']);}
 if(page==='training'){list.push(['overview','训练记录']);if(getTrain(n))list.push(['detail','训练详情']);if(form?.type==='new-training')list.push(['new-training','新建训练']);}
 if(page==='launch'){list.push(['operations','运行总览']);if(form)list.push(['new-launch','新建评估']);}
 if(page==='results'){list.push(['overview','评估记录']);if(n?.kind==='eval'&&!n.pending)list.push(['detail','回合与录像']);}
 if(compare.size>0&&page!=='registry')list.push(['compare','对比 ('+compare.size+')']);
 return list;
}
function tabs(n){const list=tabList(n);if(!list.some(([k])=>k===tab))tab=defaultTab(page,n);$('tabs').innerHTML=list.map(([k,v])=>'<button data-tab="'+k+'" class="'+(tab===k?'active':'')+'" aria-selected="'+(tab===k)+'">'+v+'</button>').join('');}
function renderHeader(n){
 document.body.dataset.page=page;
 $('crumb').textContent=page==='registry'?'':[labels[page],...ancestors(n).map(x=>x.title)].join(' / ');
 const tr=getTrain(n),ck=getWeight(n),acts=[];
 if(page==='registry'){$('title').textContent='资源管理';$('actions').innerHTML='';tabs(n);return;}
 if(form&&['new-training','new-launch'].includes(tab))$('title').textContent=form.type==='new-training'?'新建训练':form.target?'新建评估 · '+form.target.title:'新建评估';
 else $('title').textContent=n?n.title:labels[page];
 if(!form||!['new-training','new-launch'].includes(tab)){
  if(n&&['train','eval','checkpoint'].includes(n.kind)&&page!=='launch')acts.push('<button data-add="'+esc(n.id)+'" aria-pressed="'+compare.has(n.id)+'">'+(compare.has(n.id)?'已选对比 ✓':'加入对比')+'</button>');
  if(page==='experiment'&&tr)acts.push('<button data-action="training">查看训练 →</button>');
  if(page==='experiment'&&n?.kind==='eval'&&!n.pending)acts.push('<button data-action="result">查看评估 →</button>');
  if((page==='experiment'||page==='results')&&n?.pending)acts.push('<button data-action="launch">查看运行 →</button>');
  if(page==='experiment'||page==='training'){
    if(ck&&ck.train&&ck.benchmark==='b1k')acts.push('<button class="primary" data-action="evaluate">使用权重评估 →</button>');
    else if(tr)acts.push('<button class="primary" data-action="evaluate-latest">最新权重评估 →</button>');
  }
  if(page==='training')acts.push('<button '+(tr?'':'class="primary"')+' data-action="new-training">＋ 新建训练</button>');
  if(page==='launch')acts.push('<button class="primary" data-action="new-launch">＋ 新建评估</button>');
 }
 $('actions').innerHTML=acts.join('');tabs(n);
}
function renderCenter(){
 const n=nodes.get(selected);renderHeader(n);
 if(page==='registry'){openResource();return;}
 const tr=getTrain(n);
 if(tab==='new-training'&&form?.type==='new-training'){openFrame('training',form.node,{type:'new',page:'training'});return;}
 if(tab==='new-launch'&&form?.type==='new-launch'){
   if(form.inFrame&&!$('panel').hidden)return;
   if(form.target){const t=form.target,owner=form.owner;openFrame('training',owner,{type:'checkpoint',benchmark:'b1k',device:owner.device,run:owner.run,arm:owner.arm,path:t.path});}
   else openFrame('launch',form.node,{type:'new',page:'launch'});
   return;
 }
 if(tab==='detail'&&page==='training'&&tr){openFrame('training',tr,{type:'train',device:tr.device,run:tr.run,arm:tr.arm});return;}
 if(tab==='detail'&&page==='results'&&n?.kind==='eval'&&!n.pending){openFrame('results',n,{type:'eval',run:n.run,folder:n.folder});return;}
 if(tab==='document'&&n?.kind==='document'){openFrame('experiment',n,{type:'experiment'});return;}
 if(tab==='operations'&&page==='launch'){openFrame('launch',n);return;}
 closeFrame();$('content').innerHTML=tab==='compare'?compareItems()+comparison():tab==='lineage'?lineage(n):overview(n);
}
function render(){nav();drawTree();renderCenter();}
function saveRoute(){const u=new URL(location.href);u.searchParams.set('page',page);u.searchParams.set('benchmark',benchmark);history.replaceState(null,'',u);}
function leaveForm(){
 if(!form)return true;
 if(!window.confirm('当前配置面板将关闭，尚未提交的配置不会保存。继续切换？'))return false;
 form=null;closeFrame();return true;
}
function route(p){
 if(p===page)return;
 if(!leaveForm())return;
 closeFrame();status('');page=p;tab=defaultTab(page,nodes.get(selected));saveRoute();render();
}
function choose(id,fromGraph=false){
 const n=nodes.get(id);if(!n||!scope(id))return;
 if(id===selected&&!fromGraph)return;
 if(page!=='registry'&&!leaveForm())return;
 selected=id;ancestors(n).forEach(x=>expanded.add(x.id));
 const oldFocus=nodes.get(graphFocus);
 if(!oldFocus||!ancestors(oldFocus).some(x=>x.id===id))graphFocus=id;
 if(page==='experiment'){
   if(n.kind==='document')tab='document';
   else if(fromGraph||tab==='document'||tab==='compare')tab='lineage';
 }else if(page!=='registry')tab=defaultTab(page,n);
 status('');render();if(n.kind==='train')void loadCheckpoints(n);
}
function startForm(type,n,target=null,owner=null){
 if(!leaveForm())return;
 closeFrame();status('');page=type==='new-training'?'training':'launch';tab=type;
 form={type,node:n,target,owner};saveRoute();render();
}
async function loadCheckpoints(n){if(detailRequests.has(n.id))return detailRequests.get(n.id);const promise=(async()=>{try{const d=await api('/api/training/detail?'+new URLSearchParams({device:n.device,run:n.run,arm:n.arm}));detailCache.set(n.id,d);checkpointByTraining.set(n.id,Array.isArray(d.checkpoints)?d.checkpoints.filter(c=>c&&typeof c.path==='string'):[]);rebuild();drawTree();if(selected===n.id&&tab!=='detail'&&tab!=='compare')renderCenter();}catch(e){status('权重读取失败：'+e.message,true);}})();detailRequests.set(n.id,promise);try{await promise;}finally{detailRequests.delete(n.id);}}
async function refresh(){if(refreshing)return;refreshing=true;$('reload').disabled=true;const errors=[];
 const sources=[['a100','/api/training/jobs?device=a100'],['a6000','/api/training/jobs?device=a6000'],['results','/api/results'],['native','/api/launch/jobs?benchmark=b1k2026'],['launch','/api/launch/jobs'],['libero','/api/benchmarks/jobs'],['experiments','/api/agent-experiments']];
 await Promise.all(sources.map(async ([key,url])=>{try{const d=await api(url);if(key==='a100'||key==='a6000'){if(!Array.isArray(d.jobs))throw Error('缺少 jobs');cache.train=cache.train.filter(x=>x.device!==key).concat(d.jobs.map(job=>({device:key,job})));}else{const rows=key==='results'?d.runs:key==='experiments'?d.experiments:d.jobs;if(!Array.isArray(rows))throw Error('记录格式错误');cache[key]=rows;}}catch(e){errors.push(key+'：'+e.message);}}));
 sourceErrors=errors;rebuild();if(selected&&!scope(selected))selected=roots.find(matchesBranch)||null;const active=nodes.get(selected);if(active?.kind==='train'&&!detailCache.has(active.id))void loadCheckpoints(active);drawTree();if($('panel').hidden)renderCenter();else renderHeader(nodes.get(selected));$('source-state').textContent=errors.length?'部分设备 / 接口不可查 · '+errors.length+' 项':'已同步 · '+new Date().toLocaleTimeString();$('source-state').title=errors.join('；');$('reload').disabled=false;refreshing=false;
}
$('nav').onclick=e=>{const b=e.target.closest('[data-page]');if(b)route(b.dataset.page);};
$('benchmarks').onchange=e=>{
 const value=e.target.value;if(value===benchmark)return;
 if(!leaveForm()){e.target.value=benchmark;return;}
 benchmark=value;if(selected&&!scope(selected))selected=roots.find(matchesBranch)||null;
 if(selected)ancestors(nodes.get(selected)).forEach(x=>expanded.add(x.id));
 if(page!=='registry'){closeFrame();status('');tab=defaultTab(page,nodes.get(selected));}
 saveRoute();render();
};
$('tree').onclick=e=>{const exp=e.target.closest('[data-expand]');if(exp){const id=exp.dataset.expand,n=nodes.get(id);if(expanded.has(id))expanded.delete(id);else{expanded.add(id);if(n?.kind==='train')void loadCheckpoints(n);}drawTree();return;}const b=e.target.closest('[data-node]');if(b)choose(b.dataset.node);};
$('search').oninput=e=>{search=e.target.value.trim().toLowerCase();drawTree();};
$('engineering').onchange=e=>{const checked=e.target.checked;includeEngineering=checked;if(selected&&!scope(selected)){if(!leaveForm()){includeEngineering=!checked;e.target.checked=!checked;return;}selected=roots.find(matchesBranch)||null;closeFrame();tab=defaultTab(page,nodes.get(selected));}render();};
$('clear').onclick=()=>{$('search').value='';search='';drawTree();};
$('collapse').onclick=()=>{expanded.clear();drawTree();};
$('reload').onclick=()=>void refresh();
$('tabs').onclick=e=>{const b=e.target.closest('[data-tab]');if(!b||b.dataset.tab===tab)return;if(!leaveForm())return;tab=b.dataset.tab;status('');renderCenter();};
document.body.addEventListener('click',async e=>{
 if(e.target.closest('#tree'))return;
 const b=e.target.closest('[data-action],[data-node],[data-add],[data-remove]');if(!b)return;
 if(b.dataset.node){choose(b.dataset.node,!!b.closest('.lineage'));return;}
 if(b.dataset.add){const id=b.dataset.add;if(compare.has(id))compare.delete(id);else compare.add(id);renderCenter();return;}
 if(b.dataset.remove){compare.delete(b.dataset.remove);if(!compare.size)tab=defaultTab(page,nodes.get(selected));renderCenter();return;}
 const n=nodes.get(selected),tr=getTrain(n),ck=getWeight(n);
 switch(b.dataset.action){
 case 'load-curves':{const trains=[...compare].map(id=>nodes.get(id)).filter(x=>x?.kind==='train');status('读取原始 Loss 曲线…');await Promise.all(trains.map(loadCheckpoints));if(tab==='compare')renderCenter();status('');break;}
 case 'clear-compare':compare.clear();tab=defaultTab(page,n);renderCenter();break;
 case 'training':route('training');break;
 case 'result':route('results');break;
 case 'launch':route('launch');break;
 case 'document':tab='document';renderCenter();break;
 case 'new-training':startForm('new-training',n);break;
 case 'new-launch':startForm('new-launch',n);break;
 case 'evaluate':if(ck&&tr)startForm('new-launch',n,ck,tr);break;
 case 'evaluate-latest':if(tr){
   const original=selected;await loadCheckpoints(tr);if(selected!==original)return;
   const cp=(checkpointByTraining.get(tr.id)||[]).slice().sort((a,b)=>(b.step||0)-(a.step||0))[0];
   if(!cp){status('暂无已发布权重；请查看训练详情。',true);return;}
   const target=nodes.get(jsonKey('checkpoint',tr.id,cp.path));if(!target){status('权重索引尚未就绪',true);return;}
   selected=target.id;ancestors(target).forEach(x=>expanded.add(x.id));graphFocus=target.id;
   startForm('new-launch',target,target,tr);
 }break;
 }
});
window.addEventListener('message',e=>{
 if(e.origin!==location.origin||e.source!==$('panel').contentWindow||$('panel').hidden||page==='registry')return;
 // Only a completed, explicit checkpoint hand-off can change the outer page.
 if(e.data?.type==='memoryvla-checkpoint-form'&&['training','launch'].includes(page)){
   const d=e.data,owner=nodes.get(jsonKey('train',d.device,d.run,d.arm));
   if(!owner||typeof d.path!=='string'||!d.path)return;
   const target=nodes.get(jsonKey('checkpoint',owner.id,d.path));
   if(target){selected=target.id;ancestors(target).forEach(x=>expanded.add(x.id));graphFocus=target.id;}
   else {selected=owner.id;ancestors(owner).forEach(x=>expanded.add(x.id));}
   page='launch';tab='new-launch';form={type:'new-launch',node:owner,target:target||{path:d.path,title:d.path.split('/').pop()},owner,inFrame:true};
   saveRoute();nav();drawTree();renderHeader(nodes.get(selected));
 }
});
render();void refresh();setInterval(refresh,45000);
})();