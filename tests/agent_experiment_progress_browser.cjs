const {chromium} = require('../outputs/browser-tools/node_modules/playwright');
const assert = require('node:assert/strict');

const base = process.argv[2] || 'http://127.0.0.1:8765';
const progress = `# Experiment 11 · 运行进度

> 当前动作：训练 training；2 个门前/完整任务作业运行。
> 完成量：2026-11-22 17:57:03 CST · 训练 6437/10000 步 · 21 个已核验回合文件。

| 阶段 | 状态 | 记录 |
| --- | --- | --- |
| V1 续训审计 | 已通过 | CONTINUE_4000 |
| 正式训练 | 6437/10000 | train04 |
| 门前恢复 · 官方 | 8/30 | 逐回合与物理真值 |
| 门前恢复 · V1 8000 | 待权重重载 | 逐回合与物理真值 |
| 门前恢复 · step-004000 | running · initializing | 作业 ID |
| 固定输入 A/B/C | 1 个作业已完成 | 另一个作业 running |
| 权重专属探针导出 | 0 个作业已完成 | 作业 running |
| 实例留出 Probe | 4000, 6000 | 机制指标 |
| 持久巡检 | 45 秒 | 巡检日志 |
| 汇总 | 进行中 · 4/10 回合 | 最新记录 |
| 自定义 | 自定义状态 | 仍保留原文 |
`;

(async () => {
  const browser = await chromium.launch({args: ['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport: {width: 390, height: 844}});
    page.setDefaultTimeout(3000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/agent-experiments*', route => route.fulfill({json: {
      experiments: ['experiment_11'],
      detail: {name: 'experiment_11', path: 'Output/test_01/experiment_11',
        progress: {file: 'progress.md', updated_at: '2026-11-22T09:57:03Z', markdown: progress}},
      live: {jobs: [], partial: false}
    }}));
    await page.goto(base + '/ai?experiment=experiment_11');
    const rows = page.locator('#agent-progress .agent-stage');
    await rows.first().waitFor();
    async function check(title, badge, kind, note) {
      const row = rows.filter({has: page.getByText(title, {exact: true})});
      const marker = row.locator('.agent-badge');
      assert.equal(await marker.textContent(), badge);
      assert.ok((await marker.getAttribute('class')).split(' ').includes('agent-' + kind));
      if (note) assert.ok((await row.innerText()).includes(note));
    }
    await check('V1 续训审计', '已通过', 'done');
    await check('正式训练', '进行中', 'active', '6437/10000');
    await check('门前恢复 · 官方', '8/30', 'recorded');
    await check('门前恢复 · V1 8000', '待权重重载', 'waiting');
    await check('门前恢复 · step-004000', '进行中 · 初始化', 'active');
    await check('固定输入 A/B/C', '1 个作业已完成', 'recorded');
    await check('权重专属探针导出', '0 个作业已完成', 'recorded');
    await check('实例留出 Probe', '已产出', 'recorded', '4000, 6000');
    await check('持久巡检', '每 45 秒', 'recorded');
    await check('汇总', '进行中 · 4/10 回合', 'active');
    await check('自定义', '状态未标记', 'unknown', '自定义状态');
    const counts = await page.locator('.agent-count').allTextContents();
    assert.ok(counts.includes('3进行中') && counts.includes('5仅记录进展'));
    const bar = page.locator('#agent-progress progress');
    assert.deepEqual(await bar.evaluate(el => [el.value, el.max]), [6437, 10000]);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    assert.deepEqual(errors, []);
    console.log('Experiment 11 statuses, counters and training bar verified');
    await page.unroute('**/api/agent-experiments*');
    const prepare = {id: 'prepare', title: '协议', after: [], acceptance: '锁定种子'};
    const preflight = {id: 'preflight', title: '真实预检', after: ['prepare'], acceptance: '核对严格重载'};
    const evaluation = {id: 'eval', title: '闭环', after: ['preflight'], acceptance: '归档回合'};
    const probe = {id: 'probe', title: '探针', after: ['preflight'], acceptance: '归档机制证据'};
    const report = {id: 'report', title: '报告', after: ['eval', 'probe'], acceptance: '汇总对照'};
    const plan = {author: 'Agent', goal: '同条件比较', created_at: '2026-11-23', mainline: ['prepare', 'preflight', 'eval', 'report']};
    const versions = [
      {...plan, version: 'v001', reason: '首版', nodes: [prepare, preflight, evaluation, {...report, after: ['eval']}]},
      {...plan, version: 'v002', reason: '新增并行探针', nodes: [prepare, preflight, evaluation, probe, report]}
    ];
    await page.route('**/api/agent-experiments*', route => route.fulfill({json: {
      experiments: ['experiment_12'],
      detail: {name: 'experiment_12', workflow: {versions}, progress: {
        file: 'progress.md', updated_at: '2026-11-23T05:00:00Z',
        markdown: '| 阶段 | 状态 | 记录 | 训练作业 |\n| --- | --- | --- | --- |\n' +
          '| 协议 | 已完成 | 种子归档 | |\n| 真实预检 | 进行中 | 日志 | a100:trial-run |'
      }}, live: {jobs: []}
    }}));
    await page.route('**/api/training/jobs?device=a100', route => route.fulfill({json: {
      jobs: [{run: 'trial-run', name: 'trial-run', status: 'failed', step: 2, max_steps: 2}]
    }}));
    await page.goto(base + '/ai?experiment=experiment_12');
    await page.getByText('主线 · 真实预检 · 作业失败', {exact: true}).waitFor();
    await page.locator('[data-node=eval]').getByText('等待前置', {exact: true}).waitFor();
    assert.equal(await page.locator('.workflow-level').filter({has: page.locator('[data-node=eval]')}).locator('button').count(), 2);
    await page.locator('[data-node=report]').click();
    await page.getByText('依赖 · 闭环、探针', {exact: true}).waitFor();
    await page.getByLabel('计划版本').selectOption('v001');
    await page.getByText('历史计划 · v001', {exact: true}).waitFor();
    assert.equal(await page.locator('.workflow-node .agent-badge').count(), 0);
    assert.equal(await page.locator('[data-node=probe]').count(), 0);
    await page.getByLabel('计划版本').selectOption('v002');
    await page.getByText('主线 · 真实预检 · 作业失败', {exact: true}).waitFor();
    await page.getByText('版本说明', {exact: true}).click();
    await page.getByText('新增并行探针', {exact: true}).waitFor();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    assert.deepEqual(errors, []);
    console.log('Workflow dependencies, history, and live failure override verified');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
