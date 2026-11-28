(() => {
  const $=id=>document.getElementById(id);
  const panel=$('resource-download-panel'), choice=$('resource-download-choice'), start=$('resource-download-start');
  const refresh=$('resource-download-refresh'), preview=$('resource-download-preview');
  const status=$('resource-download-message'), detail=$('resource-download-detail'), progress=$('resource-download-progress');
  let recipes=[], previewed=null, busy=false, lastStatus=null;
  async function api(action,id,write=false){
    const payload={action,...(id?{recipe_id:id}:{})};
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
    progress.hidden=!(data.total_bytes>0 && data.status);
    progress.value=Math.min(1,(data.verified_bytes||0)/(data.total_bytes||1));
    status.textContent=({not_started:'未启动',starting:'已提交，等待后台',running:'下载及校验中',complete:'已校验完成，可通过添加已有目录登记',existing:'目标已存在，请核对并登记',failed:'下载失败，检查错误后可重试同配方',interrupted:'下载中断，检查后可重试'})[data.status]||'预检通过，请核对设备、路径、容量和版本';
  }
  function reset(){previewed=null;lastStatus=null;detail.textContent='';progress.hidden=true;controls();}
  async function load(){
    if(busy)return;busy=true;reset();status.textContent='正在查询下载配方…';
    try{
      const old=choice.value;
      recipes=(await api('options')).recipes;
      choice.replaceChildren(...recipes.map(r=>new Option(r.id+' · '+r.host+' · '+r.repo,r.id)));
      if(recipes.some(r=>r.id===old))choice.value=old;
      if(!choice.value){status.textContent='尚无可执行配方。请管理员按 configs/workbench_downloads.example.json 配置实际设备与 Python。';return;}
      show(await api('status',choice.value));
    }catch(e){status.textContent=e.message;}finally{busy=false;controls();}
  }
  choice.onchange=()=>void load();
  refresh.onclick=()=>void load();
  preview.onclick=async()=>{
    if(busy||!choice.value||preview.disabled)return;busy=true;reset();const id=choice.value;
    status.textContent='正在检查环境、磁盘与目标…';
    try{const data=await api('preview',id,true);show(data);previewed={id,job_id:data.job_id};}
    catch(e){status.textContent=e.message;}finally{busy=false;controls();}
  };
  start.onclick=async()=>{
    if(busy||previewed?.id!==choice.value)return;
    if(!confirm('确认启动该下载？不会覆盖已有目标。'))return;
    busy=true;const id=choice.value;previewed=null;controls();
    try{show(await api('start',id,true));}catch(e){status.textContent=e.message+'；先查询状态再决定是否重试。';lastStatus='starting';}finally{busy=false;controls();}
  };
  panel.addEventListener('toggle',()=>{if(panel.open)void load();});
  setInterval(()=>{if(panel.open&&panel.getClientRects().length&&!document.hidden&&!previewed&&!busy&&choice.value)void load();},15000);
  controls();
})();
