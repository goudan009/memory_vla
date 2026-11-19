/* Task40 stages share the workbench's existing Episode and video selection. */
(() => {
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const stateNames = {unknown:'未知',not_reached:'未到达',reached:'已到达',attempted:'已尝试',completed:'已完成',failed:'失败'};
  const errors = {none:'无',unknown:'无法判断',navigation_error:'导航错误',task_order_error:'任务顺序错误',state_order_error:'当前状态与后续动作冲突',history_state_error:'历史状态利用错误（需额外证据）',execution_failure:'操作执行失败',grasp_failure:'抓取失败',contact_failure:'接触失败',repetitive_stuck:'停滞或重复操作'};
  const behaviorNames = {unknown:'无法判断',in_kitchen:'到达厨房',navigate_kitchen:'前往厨房',choose_microwave:'选择微波炉',pull_door:'尝试拉门',leave_for_bag:'离开并前往袋子',grasp_bag:'尝试抓袋',place_bag:'尝试放入',close_door:'尝试关门',press_start:'尝试启动'};
  let context = null, loaded = '', data = null, generation = 0, reviewEvents = [];
  const fraction = metric => metric ? `${metric.numerator}/${metric.denominator}${metric.unknown_n ? ` · 未知 ${metric.unknown_n}`:''}` : '未知';
  const truth = value => value == null ? '未知' : value ? '是' : '否';
  const options = mapping => Object.entries(mapping).map(([key,value]) => `<option value="${key}">${escape(value)}</option>`).join('');
  const json = async (url, payload) => {
    const response = await fetch(url, payload ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)} : {cache:'no-store'});
    const value = await response.json();
    if (!response.ok || value.error) throw new Error(value.error || `HTTP ${response.status}`);
    return value;
  };
  function seek(step, episode) {
    if (step == null) return;
    const player = $('video-player');
    if (!player?.getAttribute('src')) return;
    const seconds = videoSeconds(step, episode);
    player.currentTime = Number.isFinite(player.duration) ? Math.min(seconds, player.duration) : seconds;
    // Reuse the three-camera replay controls and their existing seek listeners.
    if (!$('preview-toggle').checked) $('preview-toggle').click();
  }
  function videoSeconds(step, episode) {
    const frames = episode.time_mapping.frames || [];
    if (frames.length) {
      const closest = frames.reduce((a,b) => Math.abs(a.action_steps-step) <= Math.abs(b.action_steps-step) ? a : b);
      return closest.frame_index/closest.video_fps;
    }
    return step*episode.time_mapping.video_seconds_per_action_step;
  }
  function currentEpisode() {
    return data?.episodes.find(ep => ep.identity.episode_index === context?.episode?.episode_index);
  }
  function showStage(id) {
    const episode = currentEpisode(), stage = episode?.stages.find(item => item.id === id);
    if (!stage) return;
    const step = stage.first_attempt_step ?? stage.first_reached_step ?? stage.first_completed_step;
    seek(step, episode);
    const failures = episode.failures.filter(f => f.stage_id === id);
    $('task40-evidence').innerHTML = `<b>${escape(id)} ${escape(stage.name)} · ${stateNames[stage.status]}</b>
      <div>到达：${truth(stage.reached)} · 尝试：${truth(stage.attempted)} · 观察到完成：${truth(stage.completed)} · 观察到失败：${truth(stage.failed)}</div>
      <div>首次到达 / 尝试 / 完成动作步：${escape(stage.first_reached_step)} / ${escape(stage.first_attempt_step)} / ${escape(stage.first_completed_step)}</div>
      <div>${step == null ? '缺少可定位的阶段证据。' : episode.time_mapping.frames?.length ? '已定位到最接近该动作步的实录视频帧；帧间动作不可见。' : '已定位到近似视频时间，三路相机沿用播放器同步回放。'}</div>
      ${failures.map(f => `<div>${f.role === 'first_observed_failure' ? '首次观察到的失败' : '后续错误'}：${escape(f.explanation)}；前序事件 ${escape(f.preceding_failure_id)}；因果关系尚未确认</div>`).join('')}
      <details><summary>物理与视频事件原始证据</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:300px;overflow:auto">${escape(JSON.stringify({stage_evidence:stage.evidence,failures},null,2))}</pre></details>`;
  }
  function renderEvents() {
    const list = $('task40-review-events');
    if (!list) return;
    list.innerHTML = reviewEvents.length ? reviewEvents.map((ev,i) => `<div style="margin:6px 0">${escape(ev.stage_id)} @${escape(ev.action_start)}–${escape(ev.action_end)} ${escape(behaviorNames[ev.observed_behavior])} · ${escape(errors[ev.error_type])}
      <button type="button" data-edit="${i}">编辑</button> <button type="button" data-remove="${i}">移除标注</button></div>`).join('') : '暂无视频事件；物理记录保持原样。';
    for (const button of list.querySelectorAll('[data-remove]')) button.onclick = () => {reviewEvents.splice(Number(button.dataset.remove),1); renderEvents();};
    for (const button of list.querySelectorAll('[data-edit]')) button.onclick = () => {
      const ev = reviewEvents[Number(button.dataset.edit)];
      for (const [key,value] of Object.entries(ev)) if ($('task40-edit-'+key)) $('task40-edit-'+key).value = Array.isArray(value) ? value.join(',') : value;
      $('task40-event-index').value = button.dataset.edit;
    };
  }
  function draw() {
    const root = $('task40-stage-body'), episode = currentEpisode();
    if (!root || !episode) {if(root) root.textContent = '请选择已结束的 Task40 回合。';return;}
    const s = data.summary;
    reviewEvents = structuredClone(episode.events);
    root.innerHTML = `<div class="subtle">阶段配置为待示教与人工校准的草案。动作步是事实轴，旧视频定位为近似；未知不按失败计算。</div>
      <div style="margin:10px 0">有效回合 ${s.valid_n}/${s.episodes} · 官方完整成功 ${fraction(s.official_success_rate)} · 官方平均 Q ${escape(s.official_mean_q)}（N=${s.q_valid_n}）<br>
      错误转换 ${fraction(s.invalid_transition_rate)} · 有证据选择的正确率 ${fraction(s.decision_accuracy)}</div>
      <details><summary>整轮阶段指标与首次失败分布</summary><div style="overflow:auto"><table><thead><tr><th>阶段</th><th>确认到达 / 有效集数（下界）</th><th>执行成功 / 尝试且结果已知</th><th>确认完成集数</th></tr></thead><tbody>
      ${Object.entries(s.stages).map(([id,m]) => `<tr><td>${escape(id)} ${escape(m.name)}</td><td>${escape(fraction(m.reach_rate))}</td><td>${escape(fraction(m.execution_success_rate))}</td><td>${m.completed}</td></tr>`).join('')}</tbody></table></div>
      <div>首次观察到的失败（不代表已证实根因）：${escape(JSON.stringify(s.first_observed_failure_distribution))}</div></details>
      <div style="margin:10px 0;display:flex;gap:6px;flex-wrap:wrap" id="task40-timeline">${episode.stages.map(stage => `<button type="button" class="episode-button" data-stage="${stage.id}" title="点击查看证据并定位录像">${escape(stage.id)} ${escape(stage.name)} · ${stateNames[stage.status]}</button>`).join('')}</div>
      <div>本集物理样本 ${episode.trace_samples} · Custom Stage Completion：确认 ${episode.custom_stage_completion.completed}/${episode.custom_stage_completion.total}；未知 ${episode.custom_stage_completion.unknown_stages}</div>
      <div><button type="button" id="task40-vl">分析本集候选短片段</button> <span id="task40-vl-message" role="status">候选事件 ${episode.events.length} 条；用于指标 ${episode.events_used_for_metrics} 条${episode.event_review_required ? '，待人工复核':''}</span></div>
      <details><summary>VL / 复核事件及视频证据（${episode.events.length}条）</summary>
        ${episode.events.map(ev=>`<article style="margin:8px 0;padding:8px;border:1px solid var(--line,#ddd)"><button type="button" data-event-step="${ev.action_start}">${escape(ev.stage_id)} @${ev.time_start == null ? '≈'+videoSeconds(ev.action_start,episode).toFixed(2) : Number(ev.time_start).toFixed(2)}秒</button>
        ${escape(behaviorNames[ev.observed_behavior])} · ${escape(ev.outcome)} · ${escape(errors[ev.error_type])} · 自报置信度 ${escape(ev.confidence)}<br>
        ${escape(ev.explanation)}<br>${ev.semantic_validation ? `阶段证据检查：${ev.semantic_validation.accepted ? '字段齐全，图像事实仍待复核' : '缺少目标位置或释放证据，结果保留不确定'}；模型原判 ${escape(ev.model_outcome)}<br>` : ''}${ev.stage_evidence ? `模型图像观察：${escape(JSON.stringify(ev.stage_evidence))}<br>` : ''}证据视频帧：${escape((ev.evidence_frames||[]).join(', '))}${episode.event_review_required ? ' · 尚未确认，不计入指标':''}</article>`).join('') || '无候选事件不代表未尝试。'}
      </details>
      <details><summary>Memory 运行观测（${episode.memory_diagnostics?.sample_count || 0}次采样）</summary>
        <div>${escape(episode.memory_diagnostics?.interpretation || '已有录像未采集 Memory 内部状态。')} 决策序号不能直接当作视频时间。</div>
        <div>证据文件：${escape(episode.memory_diagnostics?.evidence_file)} · ${escape(episode.memory_diagnostics?.warning || '')}</div>
        <pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto">${escape(JSON.stringify(episode.memory_diagnostics?.samples || [],null,2))}</pre>
      </details>
      <div id="task40-evidence" style="margin:10px 0;padding:8px;border:1px solid var(--line,#ddd)">点击阶段查看时间、判定和证据。</div>
      <details><summary>物理状态时间线：历史完成与当前状态分开</summary>
      <div>最后采样状态：门开 ${truth(episode.current_state.door_open)} / 持握 ${truth(episode.current_state.bag_held)} / 袋在炉内 ${truth(episode.current_state.bag_inside)} / 已启动 ${truth(episode.current_state.microwave_on)}</div>
      <div style="max-height:240px;overflow:auto"><table><thead><tr><th>动作步 / 视频秒</th><th>门开</th><th>持握</th><th>在炉内</th><th>启动</th></tr></thead><tbody>${episode.timeline.map(point => `<tr><td><button type="button" data-step="${point.action_steps}">${point.action_steps} / ${point.video_time_approximate ? '≈' : ''}${point.video_seconds.toFixed(2)}s</button></td><td>${truth(point.facts.door_open)}</td><td>${truth(point.facts.bag_held)}</td><td>${truth(point.facts.bag_inside)}</td><td>${truth(point.facts.microwave_on)}</td></tr>`).join('')}</tbody></table></div></details>
      <details id="task40-review"><summary>人工复核与修正（独立标注）${episode.human_review ? ' · '+escape(episode.human_review.decision):''}</summary>
        <label>复核结论 <select id="task40-review-decision"><option value="confirmed">确认现有判断</option><option value="modified">修改视频判断</option><option value="unknown">证据不足，标记不确定</option></select></label>
        <div id="task40-review-events" style="margin:8px 0"></div>
        <details><summary>添加或编辑视频事件</summary><input id="task40-event-index" type="hidden" value="">
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0">
          <label>阶段 <select id="task40-edit-stage_id">${episode.stages.map(st => `<option>${st.id}</option>`).join('')}</select></label>
          <label>行为 <select id="task40-edit-observed_behavior">${options(behaviorNames)}</select></label>
          <label>结果 <select id="task40-edit-outcome"><option value="attempted">尝试</option><option value="completed">完成</option><option value="failed">失败</option><option value="unknown">不确定</option><option value="not_reached">未到达（需整集证据）</option></select></label>
          <label>错误 <select id="task40-edit-error_type">${options(errors)}</select></label>
          <label>开始动作步 <input id="task40-edit-action_start" type="number" min="0" value="0" style="width:90px"></label>
          <label>结束动作步 <input id="task40-edit-action_end" type="number" min="0" value="80" style="width:90px"></label>
          <label>置信度 <input id="task40-edit-confidence" type="number" min="0" max="1" step="0.05" value="0.8" style="width:70px"></label>
          <label>证据帧编号 <input id="task40-edit-evidence_frames" placeholder="如 40,41,42" style="width:130px"></label>
        </div><label>看到的证据 <textarea id="task40-edit-explanation" style="width:100%;min-height:60px"></textarea></label>
        <button type="button" id="task40-add-event">加入复核事件</button></details>
        <label>复核备注 <textarea id="task40-review-notes" style="width:100%;min-height:55px">${escape(episode.human_review?.notes || '')}</textarea></label>
        <button type="button" id="task40-save-review">保存独立标注</button> <span id="task40-review-message" role="status"></span>
      </details>
      <details><summary>同任务模型 / Checkpoint 比较</summary><div>仅按相同实例、种子、协议配对；显示样本数与无法配对原因。</div>
      <select id="task40-compare-run" aria-label="选择对比运行">${context.runs.filter(r => Number(r.task_id ?? r.configuration?.task_id) === 40 && r._folder !== context.run._folder).map(r => `<option value="${escape(r._folder)}">${escape(r._folder)} · ${escape(r.configuration?.policy_identity?.checkpoint_step || '')}</option>`).join('')}</select>
      <button type="button" id="task40-compare">比较</button><div id="task40-comparison"></div></details>
      <div style="margin-top:10px">导出当前整轮：${['json','csv','md'].map(format => `<a style="margin-right:12px" href="/api/vlm/stages/export?run=${encodeURIComponent(context.run._folder)}&format=${format}">${format.toUpperCase()}</a>`).join('')}<button type="button" id="task40-refresh">重新读取记录</button> <button type="button" id="task40-archive">保存整轮阶段报告</button> <span id="task40-archive-message" role="status"></span></div>`;
    if (data.recommendations) root.insertAdjacentHTML('beforeend',`<details><summary>下一轮验证建议（按原始证据）</summary><p>${escape(data.recommendations.interpretation)}</p>${data.recommendations.actions.map(a=>`<div style="margin:8px 0">${escape(a.stage_id || '')} ${escape(a.recommendation)} · 证据事件 ${a.observed_events ?? '—'}</div>`).join('')}<p>${escape(data.recommendations.next_experiment)}</p></details>`);
    $('task40-review-decision').value = episode.human_review?.decision || 'confirmed';
    for (const button of root.querySelectorAll('[data-stage]')) button.onclick = () => showStage(button.dataset.stage);
    for (const button of root.querySelectorAll('[data-step]')) button.onclick = () => seek(Number(button.dataset.step),episode);
    for (const button of root.querySelectorAll('[data-event-step]')) button.onclick = () => seek(Number(button.dataset.eventStep),episode);
    $('task40-add-event').onclick = () => {
      const event = {};
      for (const key of ['stage_id','observed_behavior','outcome','error_type','explanation']) event[key] = $('task40-edit-'+key).value;
      for (const key of ['action_start','action_end','confidence']) event[key] = Number($('task40-edit-'+key).value);
      event.evidence_frames = $('task40-edit-evidence_frames').value.split(',').filter(x => x.trim()).map(Number);
      const index = $('task40-event-index').value;
      if (index === '') reviewEvents.push(event); else reviewEvents[Number(index)] = event;
      $('task40-event-index').value = ''; renderEvents();
    };
    $('task40-save-review').onclick = async () => {
      const message = $('task40-review-message'), identity = context.run._folder;
      try {
        await json('/api/vlm/stages/review',{run:identity,episode_index:episode.identity.episode_index,
          decision:$('task40-review-decision').value,notes:$('task40-review-notes').value,events:reviewEvents});
        message.textContent = '已保存，原始物理记录与模型输出保持原样。';
      } catch(error) {message.textContent = '保存失败：'+error.message;}
    };
    $('task40-compare').onclick = async () => {
      const target = $('task40-comparison');
      try {
        const result = await json('/api/vlm/stages/compare?'+new URLSearchParams({run:context.run._folder,other:$('task40-compare-run').value}));
        target.innerHTML = `<div>有效配对 ${result.matched_n} · 左侧未配对 ${result.unmatched_left} / 右侧 ${result.unmatched_right} · 协议不一致或缺失 ${result.rejected.length}</div>
          <div>两轮完整集数的官方成功：${escape(fraction(result.left_summary.official_success_rate))} / ${escape(fraction(result.right_summary.official_success_rate))}（不等同配对子集）</div>
          <div>配对子集官方成功：${escape(fraction(result.paired_success.left))} / ${escape(fraction(result.paired_success.right))}</div>
          ${result.paired_summaries ? `<div>配对子集错误转换：${escape(fraction(result.paired_summaries.left.invalid_transition_rate))} / ${escape(fraction(result.paired_summaries.right.invalid_transition_rate))}<br>
          配对子集有证据决策正确：${escape(fraction(result.paired_summaries.left.decision_accuracy))} / ${escape(fraction(result.paired_summaries.right.decision_accuracy))}<br>
          首次观察到失败（左 / 右）：${escape(JSON.stringify(result.paired_summaries.left.first_observed_failure_distribution))} / ${escape(JSON.stringify(result.paired_summaries.right.first_observed_failure_distribution))}</div>` : ''}
          <div style="overflow:auto"><table><thead><tr><th>配对阶段</th><th>左 / 右确认到达下界</th><th>左 / 右执行成功</th></tr></thead><tbody>${Object.entries(result.paired_stage_metrics).map(([id,m])=>`<tr><td>${escape(id)} ${escape(m.name)}</td><td>${escape(fraction(m.left_reach))} / ${escape(fraction(m.right_reach))}</td><td>${escape(fraction(m.left_execution))} / ${escape(fraction(m.right_execution))}</td></tr>`).join('')}</tbody></table></div>
          <details><summary>配对阶段与原始指标</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:350px;overflow:auto">${escape(JSON.stringify(result,null,2))}</pre></details>`;
      } catch(error) {target.textContent = error.message;}
    };
    $('task40-refresh').onclick = () => {loaded=''; void render(context.run,context.episode,context.runs);};
    $('task40-vl').onclick = async () => {
      const button = $('task40-vl'), message = $('task40-vl-message');
      button.disabled=true;message.textContent='正在分析候选短片段…';
      try {
        const result = await json('/api/vlm/stages/diagnose',{run:context.run._folder,episode_index:episode.identity.episode_index});
        message.textContent=`已分析 ${result.analyzed_windows}/${result.candidate_windows} 个窗口，格式失败 ${result.failed_windows || 0}，产生 ${result.events.length} 条候选事件；点击重新读取并复核。`;
      } catch(error) {message.textContent=error.message;}
      button.disabled=false;
    };
    $('task40-archive').onclick = async () => {
      const message = $('task40-archive-message');
      try {
        const result = await json('/api/vlm/stages/batch',{run:context.run._folder});
        message.textContent = `已保存 ${result.summary.episodes} 集的 JSON / CSV / Markdown 报告。`;
      } catch(error) {message.textContent = error.message;}
    };
    renderEvents();
  }
  async function render(run, episode, runs) {
    const root = $('task40-stage-body');
    if (!root) return;
    context = {run,episode,runs};
    $('task40-stages').hidden = Number(run?.task_id ?? run?.configuration?.task_id) !== 40;
    if ($('task40-stages').hidden) return;
    const key = `${run._folder}:${episode?.episode_index}:${run.completed_episodes}`;
    if (key === loaded) return;
    loaded = key;
    const requestGeneration = ++generation;
    root.textContent = '正在读取已有物理记录…';
    try {
      const received = await json('/api/vlm/stages?run='+encodeURIComponent(run._folder));
      if (requestGeneration === generation) {data=received; draw();}
    } catch(error) {if (requestGeneration === generation) root.textContent = '阶段分析失败：'+error.message;}
  }
  window.task40Diagnostics = {render};
})();
