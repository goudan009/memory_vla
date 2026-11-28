(() => {
  const $=id=>document.getElementById(id);
  const panel=$('resource-download-panel'), choice=$('resource-download-choice'), start=$('resource-download-start');
  const status=$('resource-download-message'), detail=$('resource-download-detail'), progress=$('resource-download-progress');
  let recipes=[], previewed=null, busy=false;
  async function api(action,id,write=false){
    const payload={action,...(id?{recipe_id:id}:{})};
    const response=await fetch('/api/resources/downloads'+(write?'':'?'+new URLSearchParams(payload)),
      write?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{cache:'no-store'});
    const data=await response.json();if(!response.ok||data.error)throw Error(data.error||'下载接口异常');return data;
  }
  function show(data){
    detail.textContent=JSON.stringify(data,null,2);
    progress.hidden=!(data.total_bytes>0);
    progress.value=Math.min(1,(data.verified_bytes||0)/(data.total_bytes||1));
    status.textContent=({not_started:'未启动',starting:'已提交，等待后台',running:'下载及校验中',complete:'已校验完成，可通过添加已有目录登记',failed:'下载失败，检查错误后可重试同配方',interrupted:'下载中断，检查后可重试'})[data.status]||'预检通过，请核对设备、路径、容量和版本';
  }
  function reset(){previewed=null;start.disabled=true;detail.textContent='';progress.hidden=true;}
  async function load(){
    if(busy)return;busy=true;reset();
    try{
      const old=choice.value;
      recipes=(await api('options')).recipes;
      choice.replaceChildren(...recipes.map(r=>new Option(r.id+' · '+r.host+' · '+r.repo,r.id)));
      if(recipes.some(r=>r.id===old))choice.value=old;
      if(!choice.value){status.textContent='尚无可执行配方。请管理员按 configs/workbench_downloads.example.json 配置实际设备与 Python。';return;}
      show(await api('status',choice.value));
    }catch(e){status.textContent=e.message;}finally{busy=false;}
  }
  choice.onchange=()=>{reset();void load();};
  $('resource-download-refresh').onclick=()=>void load();
  $('resource-download-preview').onclick=async()=>{
    if(busy||!choice.value)return;busy=true;reset();const id=choice.value;
    try{const data=await api('preview',id,true);if(choice.value===id){show(data);previewed=id;start.disabled=false;}}
    catch(e){status.textContent=e.message;}finally{busy=false;}
  };
  start.onclick=async()=>{
    if(busy||previewed!==choice.value)return;
    if(!confirm('确认启动该下载？不会覆盖已有目标。'))return;
    busy=true;start.disabled=true;const id=choice.value;previewed=null;
    try{show(await api('start',id,true));}catch(e){status.textContent=e.message+'；先查询状态再决定是否重试。';}finally{busy=false;}
  };
  panel.addEventListener('toggle',()=>{if(panel.open)void load();});
  setInterval(()=>{if(panel.open&&!document.hidden&&!previewed&&!busy&&choice.value)void load();},15000);
})();
