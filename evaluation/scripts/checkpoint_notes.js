/* One checkpoint note shared by training, inference and evaluation views. */
(() => {
  let records = {}, devices = {}, loaded = false, loading = null;
  const drafts = new Map(), editors = new Map(), saving = new Set();
  const pathOf = value => {
    const path = String(value || '').trim();
    return (path.startsWith('/') ? '/' : '') + path.split('/').filter(part => part && part !== '.').join('/');
  };
  const deviceOf = value => devices[value] ? value : Object.keys(devices).find(key => devices[key].host === String(value || '').split('@').pop());
  const keyOf = (device, path) => JSON.stringify([deviceOf(device), pathOf(path)]);
  const note = (device, path) => records[keyOf(device, path)]?.notes || '';
  const brief = (device, path) => {
    const text = note(device, path).replace(/\s+/g, ' ');
    return text ? ' · 备注：' + (text.length > 42 ? text.slice(0, 42) + '…' : text) : '';
  };
  async function load() {
    if (loaded) return;
    if (!loading) loading = (async () => {
      const response = await fetch('/api/checkpoint-notes', {cache:'no-store'});
      const data = await response.json();
      if (!response.ok) throw Error(data.error || '读取备注失败');
      if(!data.notes||typeof data.notes!=='object'||!data.devices||typeof data.devices!=='object')throw Error('备注接口返回格式错误');
      records = data.notes; devices = data.devices; loaded = true;
      window.dispatchEvent(new Event('checkpoint-notes-updated'));
    })().finally(() => { loading = null; });
    return loading;
  }
  function paint(editor) {
    const device = deviceOf(editor.device), path = pathOf(editor.path);
    const valid = loaded && device && path.startsWith('/') && path !== '/' && !path.split('/').includes('..');
    const key = valid ? keyOf(device, path) : null;
    editor.key = key;
    const original = key ? note(device, path) : '';
    const value = key ? (drafts.get(key) ?? original) : '';
    if (editor.input.value !== value) editor.input.value = value;
    editor.input.disabled = !valid;
    editor.save.disabled = !valid || saving.has(key) || value === original;
    editor.target.textContent = valid ? (devices[device].label + ' · ' + path.split('/').pop()) : '先选择已保存的 checkpoint；备注会关联到这份权重。';
    editor.target.title = valid ? path : '';
    if(editor.summary){const preview=original.replace(/\s+/g,' ');editor.summary.textContent='Checkpoint 备注'+(value!==original?' · 未保存':preview?' · '+(preview.length>48?preview.slice(0,48)+'…':preview):'');editor.summary.title=original;}
    editor.status.textContent = editor.error || (!loaded ? '正在读取备注…' : saving.has(key) ? '正在保存…' :
      value !== original ? '尚未保存' : original ? '已保存 · 训练与评估共用' : '暂无备注');
  }
  function paintAll() { for (const editor of editors.values()) paint(editor); }
  function create(id) {
    const root = document.getElementById(id);
    if (!root) return null;
    root.classList.add('checkpoint-note');
    root.innerHTML = '<label for="' + id + '-text">Checkpoint 备注</label><small class="checkpoint-note-target"></small>' +
      '<textarea id="' + id + '-text" rows="3" maxlength="2000" placeholder="例如：3000 步；验证效果较好；待评估 Task40。清空后保存可删除备注。"></textarea>' +
      '<div class="checkpoint-note-actions"><small role="status" aria-live="polite"></small><button type="button">保存备注</button></div>';
    let summary=null;
    if(id!=='result-checkpoint-note'){
      const fold=document.createElement('details');fold.className='checkpoint-note-fold';
      summary=document.createElement('summary');summary.textContent='Checkpoint 备注';
      const body=document.createElement('div');body.className='checkpoint-note-body';body.append(...root.childNodes);
      fold.append(summary,body);root.append(fold);
    }
    const editor = {root, summary, input:root.querySelector('textarea'), target:root.querySelector('.checkpoint-note-target'),
      status:root.querySelector('[role=status]'), save:root.querySelector('button')};
    editor.input.oninput = event => {
      event.stopPropagation();
      if (editor.key) drafts.set(editor.key, editor.input.value);
      editor.error = ''; paintAll();
    };
    editor.save.onclick = async () => {
      const key = editor.key;
      if (!key || saving.has(key)) return;
      const [device, checkpoint] = JSON.parse(key), text = editor.input.value;
      saving.add(key); editor.error = ''; paintAll();
      try {
        const response = await fetch('/api/checkpoint-notes', {method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({device, checkpoint, notes:text})});
        const data = await response.json();
        if (!response.ok) throw Error(data.error || '保存失败');
        if (data.notes) records[key] = data; else delete records[key];
        if (drafts.get(key) === text) drafts.delete(key);
        window.dispatchEvent(new Event('checkpoint-notes-updated'));
      } catch (error) { editor.error = '保存失败：' + error.message + '；输入已保留，可重试。'; }
      finally { saving.delete(key); paintAll(); }
    };
    editors.set(id, editor);
    return editor;
  }
  function bind(id, device, path) {
    const editor = editors.get(id) || create(id);
    if (!editor) return;
    if (editor.device !== device || editor.path !== path) editor.error = '';
    editor.device = device; editor.path = path;
    paint(editor);
    if (!loaded) void load().then(paintAll).catch(error => {
      editor.error = '读取失败：' + error.message + '；请刷新页面重试。'; paint(editor);
    });
  }
  window.checkpointNotes = {bind, note, brief, ready:load};
})();
