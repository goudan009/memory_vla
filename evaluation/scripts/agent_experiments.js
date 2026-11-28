(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const recordId = record => record?.id ?? record?.name;
  const testDirectory = id => id.includes('/') ? id.split('/')[0] : 'test_01';
  const panes = ['task', 'progress', 'results'];
  const states = {
    '已完成': 'done', '已通过': 'done', '进行中': 'active', '待开始': 'waiting', '待验收': 'waiting',
    '待权重重载': 'waiting', '排队中': 'waiting', '待资源': 'waiting',
    '未通过': 'blocked', '受阻': 'blocked', '状态未标记': 'unknown'
  };
  const jobStates = {
    running: '进行中', training: '进行中', starting: '进行中', loading: '进行中',
    queued: '排队中', waiting_for_gpu: '待资源', waiting_for_gpus: '待资源',
    complete: '已完成', completed: '已完成', failed: '未通过'
  };
  const phases = {
    closed_loop: '闭环评估', initializing: '初始化', complete: '回合已结束', full_task_eval: '整任务评估',
    task40_collect: '轨迹采集', task40_forward: '冻结前向', task40_restore: '恢复评估',
    renderer_probe: '固定动作恢复', fixed_action_replay: '重放归档动作', restore_sim: '恢复仿真',
    frozen_forward: '冻结前向', replaying_probe: '探针回放'
  };
  let experiment = new URLSearchParams(location.search).get('experiment');
  let catalogSignature, allNames=[];
  const trainingCache=new Map();
  let displayed, displayedLive;
  let workflowExperiment, workflowVersion, workflowNode, workflowSignature, workflowObserver;
  let requestId = 0;

  function text(tag, value, classes) {
    const element = document.createElement(tag);
    element.textContent = value;
    if (classes) element.className = classes;
    return element;
  }

  function openPane(name, focus = false) {
    for (const pane of panes) {
      const active = pane === name;
      const button = byId('agent-tab-' + pane);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      byId('agent-' + pane).hidden = !active;
      if (active && focus) button.focus();
    }
  }

  function selectExperiment(name) {
    experiment = name;
    const url = new URL(location.href);
    url.searchParams.set('experiment', name);
    history.replaceState(null, '', url);
    void refresh();
  }

  function renderList(names) {
    const list = byId('agent-experiment-list');
    allNames=names;
    const query=byId('agent-experiment-search')?.value.trim().toLowerCase()||'';
    names=names.filter(id=>(testDirectory(id)+'/'+id).toLowerCase().includes(query));
    const signature = JSON.stringify(names);
    if (catalogSignature !== signature) {
      const groups = new Map();
      for (const id of names) {
        const directory = testDirectory(id);
        if (!groups.has(directory)) groups.set(directory, []);
        groups.get(directory).push(id);
      }
      const activeDirectory = experiment ? testDirectory(experiment) : groups.keys().next().value;
      const openFolders = new Map(Array.from(list.children, folder => [folder.dataset.directory, folder.open]));
      list.replaceChildren(...Array.from(groups, ([directory, ids]) => {
        const folder = text('details', '', 'version-group');
        folder.dataset.directory = directory;
        folder.open = openFolders.get(directory) ?? (directory === activeDirectory);
        const summary = text('summary', 'Test ' + directory.slice('test_'.length) + ' · ' + directory);
        summary.append(text('small', ids.length + ' 个实验'));
        const contents = text('div', '', 'agent-version-experiments');
        for (const id of ids) {
          const button = text('button', id.split('/').at(-1));
          button.type = 'button';
          button.dataset.experiment = id;
          button.onclick = () => selectExperiment(id);
          contents.append(button);
        }
        folder.append(summary, contents);
        return folder;
      }));
      catalogSignature = signature;
    }
    byId('agent-experiment-count').textContent = names.length + ' 个实验';
    for (const button of list.querySelectorAll('button[data-experiment]')) {
      button.setAttribute('aria-pressed', String(button.dataset.experiment === experiment));
    }
  }

  function sourceLabel(record) {
    const updated = new Date(record.updated_at);
    return text('p', record.file + ' · 文件更新 ' + updated.toLocaleString('zh-CN', {hour12: false}), 'subtle');
  }

  function fullDocument(record, label) {
    const details = document.createElement('details');
    details.append(text('summary', label), text('pre', record.markdown));
    if (record.truncated) details.append(text('p', '预览已截断，完整内容见实验目录。', 'subtle'));
    return details;
  }

  function excerpt(value) {
    return value
      .replace(/^[-*]\s+/, '')
      .replace(/^>\s*/, '')
      .replace(/\*\*|\x60/g, '')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .trim();
  }

  function renderTask(record) {
    const pane = byId('agent-task');
    if (!record) {
      pane.replaceChildren(text('p', '任务安排未记录', 'subtle'));
      return;
    }
    const lines = record.markdown.split('\n').map(line => line.trim());
    const heading = lines.find(line => /^#\s/.test(line));
    const summary = lines.find(line => /^>\s*摘要[：:]/.test(line));
    const elements = [];
    if (heading) elements.push(text('h3', excerpt(heading.replace(/^#\s+/, ''))));
    if (summary) elements.push(text('p', excerpt(summary).replace(/^摘要[：:]\s*/, '')));
    else {
      const intro = lines.slice(heading ? lines.indexOf(heading) + 1 : 0);
      const paragraph = intro.slice(0, 16).find(line => line && !/^[#-*|>]/.test(line));
      if (paragraph) {
        const copy = excerpt(paragraph);
        elements.push(text('p', copy.length > 220 ? copy.slice(0, 220) + '…' : copy));
      }
    }
    pane.replaceChildren(...elements, sourceLabel(record), fullDocument(record, '查看完整任务安排'));
  }

  function progressStage(stage, status, notes, action) {
    const [label, ...extra] = status.split(' · ');
    const item = (kind, badge, detail = notes) => ({stage, kind, status: badge, detail: detail.join(' · ')});
    if (Object.hasOwn(states, label)) return item(states[label], status);
    if (Object.hasOwn(jobStates, label)) {
      const state = jobStates[label];
      return item(states[state], [state, ...extra.map(part => phases[part] || part)].join(' · '));
    }
    // Some automatic records put a measurement in the status column, without a lifecycle state.
    if (/^\d+\s*\/\s*\d+$/.test(status)) {
      if (stage === '正式训练') {
        const code = action.match(/^训练 ([a-z_]+)(?:[；;\s]|$)/)?.[1];
        if (Object.hasOwn(jobStates, code)) return item(states[jobStates[code]], jobStates[code], [status + ' 步', ...notes]);
      }
      return item('recorded', status);
    }
    if (/^\d+ 个作业已完成$/.test(status)) return item('recorded', status);
    if (stage === '实例留出 Probe') {
      if (status === '暂无') return item('waiting', '暂无产出');
      if (/^\d+(?:,\s*\d+)*$/.test(status)) return item('recorded', '已产出', ['权重步数 ' + status, ...notes]);
    }
    if (stage === '持久巡检' && /^\d+ 秒$/.test(status)) return item('recorded', '每 ' + status);
    return item('unknown', '状态未标记', [status, ...notes]);
  }

  function progressStages(lines, action) {
    const cells = line => line.split('|').slice(1, -1).map(excerpt);
    const header = lines.findIndex(line => {
      const [stage, state] = cells(line);
      return stage === '阶段' && state === '状态';
    });
    if (header >= 0) {
      const stages = [];
      const jobColumn = cells(lines[header]).indexOf('训练作业');
      for (const line of lines.slice(header + 2)) {
        if (!line.startsWith('|')) break;
        const entries = cells(line);
        const [stage, status] = entries;
        if (!stage || !status) continue;
        const notes = entries.slice(2).filter((_, i) => i + 2 !== jobColumn);
        const training = entries[jobColumn]?.match(/^(a100|a6000):([\w.-]+)$/);
        stages.push({...progressStage(stage, status, notes, action), training: training?.slice(1)});
      }
      return stages;
    }
    const current = lines.findIndex(line => /^##\s+(当前进展|当前动作|Current Status)\s*$/i.test(line));
    if (current < 0) return [];
    const section = lines.slice(current + 1);
    const next = section.findIndex(line => /^##\s/.test(line));
    return (next < 0 ? section : section.slice(0, next)).filter(line => /^[-*]\s/.test(line))
      .map(line => ({stage: '进展记录', kind: 'unknown', status: '状态未标记', detail: excerpt(line)}));
  }

  function quotedField(lines, key) {
    const pattern = new RegExp('^>\\s*' + key + '[：:]\\s*');
    const line = lines.find(value => pattern.test(value));
    return line ? excerpt(line.replace(pattern, '')) : '';
  }

  function focusCard(label, body, classes) {
    const card = text('div', '', classes);
    card.append(text('strong', label), text('p', body));
    return card;
  }

  function workflowSteps(plan, records, jobs, gates, history) {
    const steps = new Map();
    const nodes = new Map(plan.nodes.map(node => [node.id, node]));
    function resolve(id) {
      if (steps.has(id)) return steps.get(id);
      const node = nodes.get(id);
      const dependencies = node.after.map(resolve);
      const record = records.get(node.record || node.title);
      const job = record?.training && jobs.get(record.training.join(':'));
      const unmet = dependencies.filter(dep => dep.kind !== 'done').map(dep => dep.title);
      let kind = record?.kind;
      let status = record?.status;
      let jobProgress;
      const evidence = [];
      if (record?.detail) evidence.push(record.detail);
      if (job) {
        if (Number.isFinite(job.step) && job.max_steps) jobProgress = `${job.step}/${job.max_steps} 步`;
        evidence.push(['工作台', job.name || job.run, job.status, jobProgress].filter(Boolean).join(' · '));
        if (kind !== 'done' && job.status === 'failed') [kind, status] = ['blocked', '作业失败'];
        if (kind !== 'done' && job.status === 'cancelled') [kind, status] = ['blocked', '作业已取消'];
        if (kind === 'waiting' && ['training', 'running', 'starting', 'loading'].includes(job.status)) [kind, status] = ['active', '进行中'];
        if (kind === 'active' && job.training_complete) [kind, status] = ['waiting', '待验收'];
      }
      if (!kind || (kind === 'waiting' && status === '待开始')) {
        [kind, status] = unmet.length ? ['waiting', '等待前置'] : ['ready', '可开始'];
      }
      if (node.review_experiment && !history) {
        const gate = gates[node.review_experiment];
        if (gate !== 'true') [kind, status] = ['blocked', '等待前置实验验收'];
        else if (kind !== 'done') [kind, status] = ['waiting', '待验收'];
      }
      if (history) [kind, status] = ['historical', ''];
      const level = Math.max(-1, ...dependencies.map(dep => dep.level)) + 1;
      const step = {...node, level, kind, status, unmet, evidence, jobProgress};
      steps.set(id, step);
      return step;
    }
    return plan.nodes.map(node => resolve(node.id));
  }

  function drawWorkflowGraph(canvas, cards, steps, mainline) {
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * scale);
    canvas.height = Math.round(bounds.height * scale);
    const context = canvas.getContext('2d');
    context.scale(scale, scale);
    const byId = new Map(steps.map(step => [step.id, step]));
    for (const step of steps) for (const dependency of step.after) {
      const source = cards.get(dependency).getBoundingClientRect();
      const target = cards.get(step.id).getBoundingClientRect();
      const x1 = source.left + source.width / 2 - bounds.left;
      const x2 = target.left + target.width / 2 - bounds.left;
      const y1 = source.bottom - bounds.top;
      const y2 = target.top - bounds.top;
      const primary = mainline[mainline.indexOf(step.id) - 1] === dependency;
      context.strokeStyle = primary ? '#529180' : '#9caea7';
      context.lineWidth = primary ? 2.1 : 1.2;
      context.beginPath();
      context.moveTo(x1, y1);
      if (step.level - byId.get(dependency).level > 1) {
        const gutter = x1 <= bounds.width / 2 ? 8 : bounds.width - 8;
        context.lineTo(x1, y1 + 8);
        context.lineTo(gutter, y1 + 8);
        context.lineTo(gutter, y2 - 10);
        context.lineTo(x2, y2 - 10);
      } else {
        context.bezierCurveTo(x1, (y1 + y2) / 2, x2, (y1 + y2) / 2, x2, y2 - 3);
      }
      context.lineTo(x2, y2);
      context.moveTo(x2 - 3, y2 - 5);
      context.lineTo(x2, y2);
      context.lineTo(x2 + 3, y2 - 5);
      context.stroke();
    }
  }

  function renderWorkflow(record, jobs = new Map(), force = false) {
    const pane = byId('agent-workflow');
    if (workflowExperiment !== recordId(record)) {
      workflowExperiment = recordId(record);
      workflowVersion = workflowNode = null;
    }
    const {versions = [], errors = []} = record?.workflow || {};
    const latest = versions.at(-1);
    const plan = versions.find(version => version.version === workflowVersion) || latest;
    const history = plan !== latest;
    const lines = (record?.progress?.markdown || '').split('\n').map(line => line.trim());
    const rows = history ? [] : progressStages(lines, quotedField(lines, '当前动作'));
    const steps = plan ? workflowSteps(plan, new Map(rows.map(row => [row.stage, row])), jobs, record.gates || {}, history) : [];
    const signature = JSON.stringify([recordId(record), versions, errors, plan?.version, steps]);
    if (!force && signature === workflowSignature) return;
    workflowSignature = signature;
    workflowObserver?.disconnect();
    if (!plan) {
      const message = record?.workflow ? [text('p', '尚未记录版本化计划。', 'subtle')] : [];
      pane.replaceChildren(...message, ...errors.map(error => text('p', error, 'subtle')));
      return;
    }
    const active = steps.filter(step => step.kind === 'active' || step.kind === 'blocked' || step.status === '待验收');
    const ready = steps.filter(step => step.kind === 'ready');
    const main = plan.mainline.map(id => steps.find(step => step.id === id)).find(step => step.kind !== 'done');
    const current = history ? [] : active.length ? active : ready.length ? ready : main ? [main] : [];
    const currentIds = new Set(current.map(step => step.id));
    const heading = text('div', '', 'workflow-heading');
    const selector = document.createElement('select');
    for (const version of versions.slice().reverse()) {
      const option = text('option', `${version.version}${version === latest ? ' · 当前版' : ''}`);
      option.value = version.version;
      selector.append(option);
    }
    selector.value = plan.version;
    selector.onchange = () => {
      workflowVersion = selector.value === latest.version ? null : selector.value;
      workflowNode = null;
      renderWorkflow(record, jobs, true);
      pane.querySelector('select').focus();
    };
    const label = text('label', '计划版本');
    label.append(selector);
    heading.append(text('h3', '实验流程'), label);
    const summary = text('div', '', 'workflow-summary' + (main?.kind === 'blocked' ? ' blocked' : ''));
    if (history) summary.append(text('strong', '历史计划 · ' + plan.version));
    else {
      summary.append(text('strong', main ? `主线 · ${main.title} · ${main.status}` : '主线已完成'));
      if (main?.jobProgress) summary.append(text('small', '工作台训练 · ' + main.jobProgress));
      summary.append(text('small', `${steps.filter(step => step.kind === 'done').length}/${steps.length} 个步骤已验收`));
      const parallel = current.filter(step => step !== main);
      if (parallel.length) summary.append(text('small', parallel.map(step => `${step.title} ${step.status}`).join(' · ')));
    }
    const graph = text('div', '', 'workflow-graph');
    graph.setAttribute('aria-label', '计划依赖流程图');
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    graph.append(canvas);
    const detail = text('div', '', 'workflow-node-detail');
    const cards = new Map();
    function showNode(node) {
      workflowNode = node.id;
      for (const [id, card] of cards) card.setAttribute('aria-pressed', String(id === node.id));
      detail.replaceChildren(text('strong', node.title), text('p', '验收条件 · ' + node.acceptance));
      if (node.after.length) detail.append(text('p', '依赖 · ' + node.after.map(id => steps.find(step => step.id === id).title).join('、'), 'workflow-meta'));
      if (node.review_experiment) {
        const link = text('a', '查看实验 ' + Number(node.review_experiment.replace('experiment_', '')));
        link.href = '/ai?' + new URLSearchParams({experiment: node.review_experiment});
        detail.append(link);
      }
      if (!history && node.unmet.length && node.status === '等待前置') detail.append(text('p', '等待 · ' + node.unmet.join('、')));
      if (!history) for (const evidence of node.evidence) detail.append(text('p', evidence));
    }
    const mainIds = new Set(plan.mainline);
    for (let level = 0; level <= Math.max(...steps.map(step => step.level)); level++) {
      const lane = text('div', '', 'workflow-level');
      const members = steps.filter(step => step.level === level).sort((a, b) => Number(mainIds.has(b.id)) - Number(mainIds.has(a.id)));
      for (const node of members) {
        const card = text('button', '', 'workflow-node workflow-' + node.kind);
        card.type = 'button';
        card.dataset.node = node.id;
        card.dataset.main = String(mainIds.has(node.id));
        card.dataset.current = String(currentIds.has(node.id));
        card.append(text('strong', node.title));
        if (node.status) card.append(text('span', node.status, 'agent-badge agent-' + node.kind));
        card.onclick = () => {
          showNode(node);
          const bounds = detail.getBoundingClientRect();
          if (bounds.top > innerHeight - 60) detail.scrollIntoView({block: 'nearest'});
        };
        cards.set(node.id, card);
        lane.append(card);
      }
      graph.append(lane);
    }
    const prior = versions[versions.indexOf(plan) - 1];
    const notes = text('details', '', 'workflow-version-notes');
    notes.open = pane.querySelector('.workflow-version-notes')?.open || false;
    notes.append(text('summary', '版本说明'), text('p', `${plan.version} · ${plan.created_at} · ${plan.author}`), text('p', plan.reason));
    if (prior) {
      const oldNodes = new Map(prior.nodes.map(node => [node.id, node]));
      const edits = plan.nodes.filter(node => JSON.stringify(oldNodes.get(node.id)) !== JSON.stringify(node)).map(node => node.title);
      const removed = prior.nodes.filter(node => !cards.has(node.id)).map(node => node.title);
      if (edits.length) notes.append(text('p', '新增或调整 · ' + edits.join('、')));
      if (removed.length) notes.append(text('p', '移除 · ' + removed.join('、')));
    }
    pane.replaceChildren(heading, text('p', plan.goal, 'workflow-goal'), summary, graph, detail, notes,
      ...errors.map(error => text('p', error, 'subtle')));
    showNode(steps.find(step => step.id === workflowNode) || current[0] || steps[0]);
    workflowObserver = new ResizeObserver(() => drawWorkflowGraph(canvas, cards, steps, plan.mainline));
    workflowObserver.observe(graph);
  }

  async function workflowTraining(record) {
    if (!record) return new Map();
    const lines = (record.progress?.markdown || '').split('\n').map(line => line.trim());
    const tracked = progressStages(lines, quotedField(lines, '当前动作')).flatMap(row => row.training ? [row.training] : []);
    const match=(record.path||'').match(/test_(\d+)\/experiment_(\d+)$/);
    const pattern=match?'__test'+match[1]+'__exp'+match[2]+'__':null;
    const devices = new Set([...(pattern?['a100','a6000']:[]),...tracked.map(([device]) => device)]);
    const entries = await Promise.allSettled([...devices].map(async device => {
      let cached=trainingCache.get(device);
      if(!cached || Date.now()-cached.at>15000){
        const response=await fetch('/api/training/jobs?'+new URLSearchParams({device}),{cache:'no-store'});
        if(!response.ok)throw Error('设备状态不可查');
        const {jobs=[]}=await response.json();
        cached={at:Date.now(),jobs};trainingCache.set(device,cached);
      }
      return cached.jobs.filter(job => (pattern && String(job.run).includes(pattern)) || tracked.some(([host, run]) => host === device && run === job.run))
        .map(job => [device + ':' + job.run, job]);
    }));
    const result=new Map(entries.flatMap(item => item.status === 'fulfilled' ? item.value : []));
    result.partial=entries.some(item=>item.status!=='fulfilled');
    return result;
  }

  function renderLinkedTraining(jobs){
    const panel=byId('agent-live-training');
    if(!panel)return;
    panel.replaceChildren(text('h3','关联训练 · 工作台实际记录'));
    if(!jobs.size)panel.append(text('p','暂无可关联训练记录；历史无规范身份的记录仅按明确绑定显示。','subtle'));
    for(const [key,job] of jobs){
      const who={agent:'Agent',user:'人工'}[job.initiator]||'来源未记录';
      const details=[job.version||job.model_version,who,job.status,job.step==null?null:job.step+'/'+(job.max_steps??'—')+' 步',job.notes,key].filter(Boolean).join(' · ');
      panel.append(focusCard(job.run,details,'agent-now'));
    }
    if(jobs.partial)panel.append(text('p','部分设备不可查；未将未知状态当作空闲或完成。','subtle'));
  }

  function renderLive(live) {
    const signature = JSON.stringify(live || null);
    if (signature === displayedLive) return;
    displayedLive = signature;
    const panel = byId('agent-live');
    panel.replaceChildren();
    if (!live) return;
    if (live.jobs.length) {
      const focus = text('div', '', 'agent-focus');
      for (const job of live.jobs) {
        const device = {local: '4090', a6000: 'RTX 6000 Ada'}[job.device || 'local'] || job.device;
        const machine = job.gpus?.length ? device + ' GPU ' + job.gpus.join(', ') : device;
        const progress = job.progress || {};
        const action = job.state === 'running' ? '工作台运行中' : job.state === 'queued' ? '工作台排队中' : '工作台启动中';
        const details = [phases[progress.stage] || phases[job.phase] || phases[job.kind], progress.trial,
          progress.decisions == null ? null : progress.decisions + ' 次决策',
          progress.steps == null ? null : progress.steps + ' 仿真步', machine, '作业 ' + job.id];
        focus.append(focusCard(action, details.filter(Boolean).join(' · '), 'agent-now'));
      }
      panel.append(focus);
    }
    if (live.partial) panel.append(text('p', '部分设备的实时作业状态暂不可查', 'subtle'));
  }

  function renderProgress(record, hasPlan) {
    const pane = byId('agent-progress-record');
    if (!record) {
      pane.replaceChildren(text('p', '运行进度未记录', 'subtle'));
      return;
    }
    const lines = record.markdown.split('\n').map(line => line.trim());
    const action = quotedField(lines, '当前动作');
    const stages = progressStages(lines, action);
    const focus = text('div', '', 'agent-focus');
    const next = quotedField(lines, '下一步');
    const origin = quotedField(lines, '记录依据');
    const completed = quotedField(lines, '完成量');
    if (completed) {
      const completion = text('div', '', 'agent-completion');
      completion.append(text('span', completed));
      const counts = completed.match(/(?:^|[，,]\s*)(\d+)\s*\/\s*(\d+)/)
        || completed.match(/(?:^| · )训练\s+(\d+)\s*\/\s*(\d+)\s*步/);
      if (counts && Number(counts[2]) > 0 && Number(counts[1]) <= Number(counts[2])) {
        const bar = document.createElement('progress');
        bar.value = Number(counts[1]);
        bar.max = Number(counts[2]);
        bar.setAttribute('aria-label', completed);
        completion.append(bar);
      }
      focus.append(completion);
    }
    if (action) focus.append(focusCard('最近记录', action, 'agent-now'));
    if (next) focus.append(focusCard('下一步', next, 'agent-next'));
    const totals = {done: 0, active: 0, waiting: 0, blocked: 0, recorded: 0, unknown: 0};
    for (const item of stages) totals[item.kind]++;
    const counts = text('div', '', 'agent-counts');
    for (const [kind, label] of Object.entries({
      done: '已完成', active: '进行中', waiting: '待处理', blocked: '未通过或受阻',
      recorded: '仅记录进展', unknown: '状态未标记'
    })) {
      if ((kind === 'recorded' || kind === 'unknown') && !totals[kind]) continue;
      const card = text('div', '', 'agent-count agent-' + kind);
      card.append(text('strong', totals[kind]), text('span', label));
      counts.append(card);
    }
    const progress = document.createElement('div');
    for (const {stage, kind, status, detail} of stages) {
      const row = text('div', '', 'agent-stage');
      const title = document.createElement('div');
      title.append(text('strong', stage), text('span', status, 'agent-badge agent-' + kind));
      row.append(title, text('span', detail));
      progress.append(row);
    }
    const metadata = lines.filter(line => /^>\s*(Agent Session|机器|代码版本|配置)[：:]/i.test(line))
      .map(line => text('p', excerpt(line), 'subtle'));
    const full = fullDocument(record, '查看进度原文');
    full.open = stages.length === 0;
    const summary = stages.length ? [text('h3', '阶段进度'), counts, progress] : [];
    const archive = document.createElement('details');
    archive.append(text('summary', '查看阶段明细'), ...summary);
    pane.replaceChildren(...(focus.childElementCount ? [focus] : []),
      ...(hasPlan && summary.length ? [archive] : summary),
      ...(origin ? [text('p', '记录依据 · ' + origin, 'subtle')] : []),
      sourceLabel(record), ...metadata, full);
  }

  function resultTable(lines) {
    const header = lines.findIndex((line, index) => line.startsWith('|') && /^\|\s*:?-{2,}/.test(lines[index + 1] || ''));
    if (header < 0) return null;
    const table = document.createElement('table');
    const head = table.createTHead().insertRow();
    const body = table.createTBody();
    const cells = line => line.split('|').slice(1, -1).map(excerpt);
    head.append(...cells(lines[header]).map(value => text('th', value)));
    for (const line of lines.slice(header + 2)) {
      if (!line.startsWith('|') || body.rows.length >= 6) break;
      body.insertRow().append(...cells(line).map(value => text('td', value)));
    }
    const wrapper = text('div', '', 'agent-result-table');
    wrapper.append(table);
    return wrapper;
  }

  function renderResults(summary, report) {
    const pane = byId('agent-results');
    if (!summary && !report) {
      pane.replaceChildren(text('p', '实验结果未记录', 'subtle'));
      return;
    }
    const record = summary || report;
    const lines = record.markdown.split('\n').map(line => line.trim());
    const heading = lines.find(line => /^#\s/.test(line));
    const intro = quotedField(lines, '结论') || excerpt(lines.find(line => line && !/^[#|>]/.test(line)) || '');
    const limitation = quotedField(lines, '待验证');
    const table = resultTable(lines);
    const content = [text('h3', heading ? excerpt(heading.replace(/^#\s+/, '')) : '结果摘要')];
    if (intro) content.push(text('p', intro.length > 300 ? intro.slice(0, 300) + '…' : intro, 'agent-result-intro'));
    if (table) content.push(table);
    if (limitation) content.push(focusCard('待验证', limitation, 'agent-next'));
    content.push(sourceLabel(record));
    if (report) {
      const details = fullDocument(report, '查看完整实验报告');
      if (!report.truncated) {
        const download = text('a', '下载报告 Markdown', 'agent-report-download');
        download.href = '#';
        download.download = report.file;
        download.onclick = event => {
          event.preventDefault();
          const url = URL.createObjectURL(new Blob([report.markdown], {type: 'text/markdown;charset=utf-8'}));
          const link = document.createElement('a');
          link.href = url;
          link.download = report.file;
          link.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        };
        details.append(download);
      }
      content.push(details);
    } else if (summary) content.push(fullDocument(summary, '查看结果原文'));
    pane.replaceChildren(...content);
  }

  function display(record) {
    const signature = JSON.stringify(record);
    if (displayed === signature) return;
    displayed = signature;
    byId('agent-experiment-path').textContent = record?.path || 'Output';
    byId('agent-experiment-title').textContent = record?.name || '未发现实验目录';
    renderTask(record?.task);
    renderProgress(record?.progress, record?.workflow?.versions.length > 0);
    renderResults(record?.result, record?.report);
  }

  async function refresh() {
    const id = ++requestId;
    const status = byId('agent-error');
    const query = experiment ? '?' + new URLSearchParams({experiment}) : '';
    try {
      const response = await fetch('/api/agent-experiments' + query, {cache: 'no-store'});
      const data = await response.json();
      if (id !== requestId) return;
      if (!response.ok) throw Error(data.error || '读取实验记录失败');
      experiment = recordId(data.detail);
      renderList(data.experiments);
      display(data.detail);
      renderLive(data.live);
      if (workflowExperiment !== recordId(data.detail) || !workflowSignature) renderWorkflow(data.detail);
      const jobs = await workflowTraining(data.detail);
      if (id !== requestId) return;
      renderWorkflow(data.detail, jobs);
      renderLinkedTraining(jobs);
      status.hidden = true;
      byId('caption').textContent = '实验管理 · ' + data.experiments.length + ' 个实验';
    } catch (error) {
      if (id !== requestId) return;
      status.textContent = '读取实验记录失败 · ' + error.message;
      status.hidden = false;
      byId('caption').textContent = '实验管理';
    }
  }

  for (const [index, name] of panes.entries()) {
    const button = byId('agent-tab-' + name);
    button.onclick = () => openPane(name);
    button.onkeydown = event => {
      const keys = {ArrowLeft: (index + panes.length - 1) % panes.length,
        ArrowRight: (index + 1) % panes.length, Home: 0, End: panes.length - 1};
      if (!Object.hasOwn(keys, event.key)) return;
      event.preventDefault();
      openPane(panes[keys[event.key]], true);
    };
  }
  byId('agent-experiment-search')?.addEventListener('input',()=>renderList(allNames));
  window.agentExperimentBoard = {refresh};
})();
