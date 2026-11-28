/* Live resource contents use the same workbench and configured SSH devices. */
(() => {
  const $ = id => document.getElementById(id);
  const pathOf = value => String(value||'').replace(/^[^/]+:\//,'/');
  const bytes = value => {
    if(value==null)return '—';
    const units=['B','KiB','GiB','TiB'];
    if(value<1024)return value+' B';
    if(value<1024**2)return (value/1024).toFixed(1)+' KiB';
    if(value<1024**3)return (value/1024**2).toFixed(1)+' MiB';
    const index=value>=1024**4?3:2;
    return (value/1024**(index+1)).toFixed(1)+' '+units[index];
  };
  const modified = value => value ? new Date(value*1000).toLocaleString('zh-CN') : '—';
  let item=null, current=null, browsing=0, diskSelection=null, diskVersion=0, dialogAction=null, query='', timer;
  let lastTrash;
  try { lastTrash=JSON.parse(sessionStorage.getItem('registry-last-trash')||'null'); } catch { lastTrash=null; }
  const api=async (host,payload,write=false)=>{
    const response=await fetch('/api/registry/storage'+(write?'':'?'+new URLSearchParams({host,...payload})),
      write?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({host,...payload})}:{cache:'no-store'});
    const data=await response.json();
    if(!response.ok||data.error)throw Error(data.error||'资源操作失败');
    return data;
  };
  function feedback(text){
    const target=$('res-message');
    target.textContent=text;
    if(lastTrash){
      const button=document.createElement('button');button.type='button';button.textContent='恢复刚删除的项目';
      button.style.margin='8px';button.onclick=restore;
      target.append(button);
    }
  }
  async function restore(){
    const request=lastTrash;
    if(!request)return;
    try{
      await api(request.host,{action:'restore',path:request.path,token:request.token},true);
      lastTrash=null;sessionStorage.removeItem('registry-last-trash');feedback('已恢复：'+request.path);
      if(item?.host===request.host)await open(current?.type==='directory'?current.path:pathOf(item.path));
    }catch(error){feedback('恢复失败：'+error.message);}
  }
  function diskCard(host,disk,stale=false){
    return `<div class="storage-disk${stale?' stale':''}"><strong>${esc(host)} · ${esc(disk.mount)}</strong>
      <progress max="100" value="${disk.percent}"></progress><div>剩余 ${bytes(disk.free)} · 已用 ${bytes(disk.used)}</div>
      <div class="hint">总容量 ${bytes(disk.total)} · 已用 ${disk.percent}%${stale?' · 上次扫描':' · 实时查询'}</div></div>`;
  }
  async function showDisks(host='all',force=false){
    if(diskSelection===host&&!force)return;
    diskSelection=host;const version=++diskVersion;
    const hosts=host==='all'?window.registryStorage.hosts.map(row=>row.host):[host];
    const summaries=hosts.map(label=>label+' 查询中…');
    const updateSummary=()=>{$('storage-capacity-summary').textContent=summaries.join('；');if($('disk-visible-status'))$('disk-visible-status').textContent=summaries.some(s=>s.includes('查询中'))?'查询中…':'查询结束';};
    updateSummary();
    const slots=hosts.map(label=>{
      const slot=document.createElement('div');slot.className='storage-disk';slot.textContent=label+' · 正在查询磁盘…';return slot;
    });
    $('storage-disks').replaceChildren(...slots);
    await Promise.allSettled(hosts.map(async(label,index)=>{
      const slot=slots[index];
      try{
        const result=await api(label,{action:'disks'});
        if(version!==diskVersion)return;
        summaries[index]=label+' · '+(result.disks.slice(0,2).map(disk=>disk.mount+' 剩余 '+bytes(disk.free)).join(' / ')||'暂无可访问的数据盘')+(result.disks.length>2?' · 另 '+(result.disks.length-2)+' 个挂载点':'');
        updateSummary();
        slot.className='storage-host-disks';slot.innerHTML=result.disks.map(disk=>diskCard(label,disk)).join('')||`<div class="storage-disk">${esc(label)} · 暂无可访问的数据盘</div>`;
      }catch(error){
        if(version!==diskVersion)return;
        summaries[index]=label+' 暂时不可达 · 展开查看详情';updateSummary();
        const cached=D?.hosts?.find(row=>row.host===label);
        slot.className='storage-host-disks';slot.innerHTML=`<div class="storage-disk stale"><strong>${esc(label)} · 暂时不可达</strong><div class="hint">${esc(error.message)}</div></div>`+
          (cached?.disk||[]).map(disk=>diskCard(label,{mount:disk.mount,total:disk.total_gb*1024**3,used:disk.used_gb*1024**3,free:disk.avail_gb*1024**3,percent:disk.pct},true)).join('');
      }
    }));
  }
  function breadcrumbs(path){
    const base=pathOf(item.path).replace(/\/$/,'');
    if(path!==base&&!path.startsWith(base+'/'))return `<span class="storage-path">${esc(path)}</span>`;
    const parts=path.slice(base.length).split('/').filter(Boolean);
    return `<button type="button" data-open-path="${esc(base)}">${esc(item.name||base.split('/').at(-1))}</button>`+
      parts.map((part,index)=>`<span aria-hidden="true">/</span><button type="button" data-open-path="${esc(base+'/'+parts.slice(0,index+1).join('/'))}">${esc(part)}</button>`).join('');
  }
  function draw(data){
    const panel=$('resource-browser');
    if(!panel)return;
    const directory=data.type==='directory';
    panel.innerHTML=`<strong>${directory?'目录内容':'文件详情'}</strong><div class="storage-breadcrumbs" aria-label="当前位置">${breadcrumbs(data.path)}</div>
      <div class="storage-path">${esc(data.path)}${data.is_symlink?`<br>软链接 → ${esc(data.link_target)}`:''}</div>
      <div class="storage-actions">${data.parent?'<button type="button" data-storage-parent>↑ 上一级</button>':''}<button type="button" data-storage-refresh>刷新</button>
      ${directory?'<button type="button" data-storage-mkdir>＋ 新建文件夹</button>':''}
      ${directory||data.can_delete?`<details class="storage-menu"><summary>更多操作</summary>${directory?'<button type="button" data-storage-size>统计目录大小</button>':''}${data.can_delete?`<button type="button" class="danger" data-storage-trash="${esc(data.path)}">删除${directory?'当前目录':'当前文件'}</button>`:''}</details>`:''}</div>
      <div id="storage-browser-message" class="hint storage-feedback" role="status"></div>
      ${data.disk?`<div class="hint">所在盘 ${esc(data.disk.mount)} · 剩余 ${bytes(data.disk.free)} / 总容量 ${bytes(data.disk.total)}</div>`:''}
      ${directory?`<input type="search" id="storage-search" class="storage-search" aria-label="筛选当前目录" placeholder="筛选当前目录的文件名…" value="${esc(query)}">
        <div class="storage-entries">${data.entries.map(row=>`<div class="storage-entry"><button type="button" data-open-path="${esc(row.path)}">${row.type==='directory'?'▸ 📁':'▤'} ${esc(row.name)}<small>${row.is_symlink?'软链接 · ':''}${modified(row.modified)}</small></button><span class="storage-size">${row.type==='directory'?'文件夹':bytes(row.size)}</span><details class="storage-menu"><summary aria-label="${esc(row.name)} 的操作">⋯</summary>${row.can_delete===false?'<span class="hint">受保护</span>':`<button type="button" class="storage-delete" data-storage-trash="${esc(row.path)}" aria-label="删除 ${esc(row.name)}">删除</button>`}</details></div>`).join('')||'<div class="empty">当前目录没有匹配的文件</div>'}</div>
        <div class="storage-actions"><span class="hint">共 ${data.total} 项${data.total?` · 显示 ${data.offset+1}–${data.offset+data.entries.length}`:''}</span>${data.offset?'<button type="button" data-storage-prev>上一页</button>':''}${data.has_more?'<button type="button" data-storage-next>下一页</button>':''}</div>`:
        `<dl class="detail-meta"><div><dt>文件大小</dt><dd>${bytes(data.size)}</dd></div><div><dt>修改时间</dt><dd>${modified(data.modified)}</dd></div></dl>
        ${data.preview==null?'<p class="hint">二进制文件，不作为文本加载。可查看文件大小、修改时间与同目录配置文件。</p>':`<pre class="storage-preview" tabindex="0">${esc(data.preview)}</pre>${data.truncated?'<p class="hint">仅预览前 64 KiB 内容。</p>':''}`}`}`;
    $('storage-search')?.addEventListener('input',event=>{
      query=event.target.value;clearTimeout(timer);
      timer=setTimeout(async()=>{const wanted=query;await open(data.path,0,false);if($('storage-search')&&query===wanted){$('storage-search').focus();}},300);
    });
  }
  async function open(path,offset=0,resetQuery=true){
    const version=++browsing, selected=item;
    if(resetQuery){query='';clearTimeout(timer);}
    const panel=$('resource-browser');
    if(!panel)return;
    panel.setAttribute('aria-busy','true');
    try{
      const data=await api(selected.host||'4090',{action:'browse',path:pathOf(path),offset,query});
      if(version!==browsing||item!==selected||!panel.isConnected)return;
      current=data;draw(data);
    }catch(error){
      if(version===browsing&&panel.isConnected){panel.textContent='读取目录失败：'+error.message;}
    }finally{panel.removeAttribute('aria-busy');}
  }
  function browse(selected){
    clearTimeout(timer);
    item=selected;current=null;query='';
    $('resource-browser').textContent='正在读取实际目录…';
    void open(pathOf(selected.path));
  }
  function dialog(action,path){
    dialogAction={action,path,host:item?.host||'4090'};
    const register=action==='register',deleting=action==='trash';
    $('storage-dialog-title').textContent=register?'添加已有数据 / 权重目录':deleting?'删除资源':'新建文件夹';
    $('storage-dialog-submit').textContent=deleting?'移入回收站':register?'添加到台账':'创建';
    $('storage-host-field').hidden=$('storage-kind-field').hidden=$('storage-path-field').hidden=!register;
    $('storage-name-field').hidden=deleting;
    $('storage-name').value='';$('storage-path').value='';
    $('storage-path').required=register;$('storage-name').required=action==='mkdir';
    $('storage-host').value=$('res-host').value==='all'?(item?.host||'4090'):$('res-host').value;
    $('storage-name-field').firstChild.textContent=register?'显示名称（可选）':'文件夹名称';
    $('storage-dialog-path').textContent=register?'登记此设备上已有的文件夹；不会复制或上传数据。':
      path+(deleting?'\n移入同级回收站后可恢复；回收站中的文件仍占用磁盘空间。':'');
    $('storage-dialog-error').textContent='';openLedgerDialog($('storage-dialog'));
  }
  $('storage-form').addEventListener('submit',async event=>{
    event.preventDefault();const button=$('storage-dialog-submit');if(button.disabled)return;button.disabled=true;
    const operation={...dialogAction};
    if(operation.action==='register')Object.assign(operation,{host:$('storage-host').value,path:$('storage-path').value.trim(),kind:$('storage-kind').value,name:$('storage-name').value.trim()});
    if(operation.action==='mkdir')operation.name=$('storage-name').value.trim();
    try{
      const result=await api(operation.host,operation,true);
      $('storage-dialog').close();
      if(operation.action==='register'){
        await load(false);selectedResource=resourceKey(result.item);$('res-kind').value='all';$('res-host').value=result.item.host;$('res-q').value='';$('res-state').value='active';renderRes();feedback('已添加目录：'+result.item.path);
      }else if(operation.action==='trash'){
        lastTrash={host:operation.host,path:result.path,token:result.token};sessionStorage.setItem('registry-last-trash',JSON.stringify(lastTrash));
        feedback('已移入回收站：'+result.path);
        await open(current?.path===operation.path?(current.parent||pathOf(item.path)):current.path);
      }else{await open(current.path);feedback('已创建文件夹：'+operation.name);}
    }catch(error){$('storage-dialog-error').textContent=error.message;}
    finally{button.disabled=false;}
  });
  $('storage-dialog-cancel').onclick=()=>$('storage-dialog').close();
  $('storage-register').onclick=()=>dialog('register');
  $('storage-refresh-disks').onclick=()=>void showDisks($('res-host').value,true);
  $('resource-detail').addEventListener('click',async event=>{
    const button=event.target.closest('button');if(!button||!item)return;
    if(button.hasAttribute('data-open-path'))void open(button.dataset.openPath);
    if(button.hasAttribute('data-storage-parent'))void open(current.parent);
    if(button.hasAttribute('data-storage-refresh'))void open(current.path,current.offset||0,false);
    if(button.hasAttribute('data-storage-next'))void open(current.path,current.offset+100,false);
    if(button.hasAttribute('data-storage-prev'))void open(current.path,Math.max(0,current.offset-100),false);
    if(button.hasAttribute('data-storage-mkdir'))dialog('mkdir',current.path);
    if(button.hasAttribute('data-storage-trash'))dialog('trash',button.dataset.storageTrash);
    if(button.hasAttribute('data-storage-size')){
      const target=$('storage-browser-message');button.disabled=true;target.textContent='正在统计目录占用…';
      try{const result=await api(item.host,{action:'size',path:current.path});if(target.isConnected)target.textContent='目录占用：'+bytes(result.size);}
      catch(error){if(target.isConnected)target.textContent='统计失败：'+error.message;}
      finally{button.disabled=false;}
    }
  });
  function registerDownload(recipe){
    show('res');dialog('register');
    $('storage-host').value=recipe.host;
    $('storage-kind').value=recipe.kind;
    $('storage-path').value=recipe.target;
    $('storage-name').value=recipe.target.split('/').filter(Boolean).at(-1)||'';
  }
  window.registryStorage={registerDownload,hosts:[{host:'4090',label:'4090 · 本机'},{host:'A100',label:'A100'},{host:'A6000',label:'RTX 6000 Ada'}],browse,showDisks};
  api('4090',{action:'hosts'}).then(result=>{
    window.registryStorage.hosts=result.hosts;$('storage-host').innerHTML=result.hosts.map(row=>`<option value="${esc(row.host)}">${esc(row.label)}</option>`).join('');
    if(D){fillHostSelects();if(VIEW==='disks')showDisks($('res-host').value,true);}
  }).catch(error=>{feedback('读取资源设备失败：'+error.message);});
  if(lastTrash)feedback('上次删除的项目可恢复：'+lastTrash.path);
  if(D&&VIEW==='res')renderRes();
})();
