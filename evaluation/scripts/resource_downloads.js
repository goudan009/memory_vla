(() => {
  const $=id=>document.getElementById(id);
  const panel=$('resource-download-panel'), choice=$('resource-download-choice'), start=$('resource-download-start');
  const refresh=$('resource-download-refresh'), preview=$('resource-download-preview');
  const status=$('resource-download-message'), detail=$('resource-download-detail'), progress=$('resource-download-progress');
  const summary=$('resource-download-summary'), raw=$('resource-download-raw'), register=$('resource-download-register');
  let recipes=[], previewed=null, busy=false, lastStatus=null;
  const gib=value=>Number.isFinite(value)?(value/1024**3).toFixed(2)+' GiB':'—';
  const shortPath=value=>String(value||'').split('/').filter(Boolean).at(-1)||'—';
  function summaryCard(label,value){const card=document.createElement('div'),key=document.createElement('small'),val=document.createElement('strong');key.textContent=label;val.textContent=value;card.append(key,val);return card;}
  async function api(action,id,write=false,expected){
    const payload={action,...(id?{recipe_id:id}:{}),...(expected?{expected_job_id:expected}:{})};
    const response=await fetch('/api/resources/downloads'+(write?'':'?'+new URLSearchParams(payload)),
      write?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{cache:'no-store'});
    const data=await response.json();if(!response.ok||data.error)throw Error(data.error||'下载接口异常');return data;
  }
  function controls(){
    choice.disabled=busy||!recipes.length;refresh.disabled=busy;
    preview.disabled=busy||!choice.value||['running','starting','complete','existing'].includes(lastStatus);
    start.disabled=busy||!previewed||previewed.id!==choice.value;
  }
  function show(data){
    lastStatus=data.status||null;
    detail.textContent=JSON.stringify(data,null,2);
    const recipe=recipes.find(r=>r.id===choice.value)||{};
    if(summary){const cards=[summaryCard('设备',recipe.host||'—'),summaryCard('目标',shortPath(data.target||recipe.target)),summaryCard('总量',gib(data.total_bytes))];if(Number.isFinite(data.free_bytes))cards.push(summaryCard('可用空间',gib(data.free_bytes)));if(Number.isFinite(data.verified_bytes))cards.push(summaryCard('已校验',gib(data.verified_bytes)));summary.replaceChildren(...cards);summary.hidden=false;}
    if(raw)raw.hidden=false;
    if(register)register.hidden=!['complete','existing'].includes(data.status)||!recipe.target;
    progress.hidden=!(data.total_bytes>0 && data.status);
    progress.value=Math.min(1,(data.verified_bytes||0)/(data.total_bytes||1));
    status.textContent=({not_started:'未启动',starting:'已提交，等待后台',running:'下载及校验中',complete:'下载完成 · 请登记目录',existing:'目标已存在 · 请核对后登记',failed:'下载失败 · '+String(data.error||'请查看详情').slice(0,140),interrupted:'下载中断 · 可重新预检'})[data.status]||'预检通过 · 确认目标后下载';
  }
  function reset(){previewed=null;lastStatus=null;detail.textContent='';progress.hidden=true;if(summary)summary.hidden=true;if(raw){raw.hidden=true;raw.open=false;}if(register)register.hidden=true;controls();}
  async function load(){
    if(busy)return;busy=true;reset();status.textContent='正在查询下载配方…';
    try{
      const old=choice.value;
      recipes=(await api('options')).recipes;
      choice.replaceChildren(...recipes.map(r=>new Option(r.id+' · '+r.host+' · '+r.repo,r.id)));
      if(recipes.some(r=>r.id===old))choice.value=old;
      if(!choice.value){status.textContent='暂无下载配方，请在设备配置中登记。';return;}
      show(await api('status',choice.value));
    }catch(e){status.textContent=e.message;}finally{busy=false;controls();}
  }
  choice.onchange=()=>void load();
  refresh.onclick=()=>void load();
  preview.onclick=async()=>{
    if(busy||!choice.value||preview.disabled)return;busy=true;reset();const id=choice.value;
    status.textContent='正在预检…';
    try{const data=await api('preview',id,true);show(data);previewed={id,job_id:data.job_id};}
    catch(e){status.textContent=e.message;}finally{busy=false;controls();}
  };
  start.onclick=async()=>{
    if(busy||previewed?.id!==choice.value)return;
    if(!confirm('确认下载？已有目标不会覆盖。'))return;
    busy=true;const id=choice.value,expected=previewed.job_id;previewed=null;controls();
    try{show(await api('start',id,true,expected));}catch(e){status.textContent=e.message+'；先查询状态再决定是否重试。';lastStatus='starting';}finally{busy=false;controls();}
  };
  if(register)register.onclick=()=>{const recipe=recipes.find(r=>r.id===choice.value);if(recipe&&['complete','existing'].includes(lastStatus))window.registryStorage?.registerDownload(recipe);};
  panel.addEventListener('toggle',()=>{if(panel.open&&!panel.hidden)void load();});
  setInterval(async()=>{if(!panel.open||panel.hidden||!panel.getClientRects().length||document.hidden||previewed||busy||!choice.value||!['starting','running'].includes(lastStatus))return;busy=true;controls();try{show(await api('status',choice.value));}catch(e){status.textContent='状态查询失败 · '+e.message;}finally{busy=false;controls();}},15000);
  controls();
})();
