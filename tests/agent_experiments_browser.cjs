/* Read-only acceptance check against the ten existing experiment folders. */
const {chromium} = require('../outputs/browser-tools/node_modules/playwright');
const assert = require('node:assert/strict');

const base = process.argv[2] || 'http://127.0.0.1:8765';
const expected = Array.from({length: 10}, (_, index) => 'experiment_' + String(index + 1).padStart(2, '0'));
const sources = {
  experiment_07: 'Task40_Experiment07_Master_Taskbook.md',
  experiment_09: 'TASK.md',
  experiment_10: 'task.md'
};

(async () => {
  const browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 900}, acceptDownloads: true});
    page.setDefaultTimeout(8000);
    const errors = [], writes = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.method() !== 'GET') writes.push(request.method() + ' ' + request.url());
    });

    const services = page.waitForResponse(response => response.url().endsWith('/api/services') && response.ok());
    await page.goto(base + '/ai?experiment=experiment_10');
    await services;
    await page.locator('#agent-experiment-title').getByText('experiment_10').waitFor();
    await page.locator('#service-overview summary').waitFor();
    await page.getByText('设备资源', {exact: true}).waitFor();
    assert.deepEqual((await page.locator('#agent-experiment-list button').allTextContents()).slice(0, expected.length), expected);
    assert.equal(await page.getByRole('tab', {name: '运行进度'}).getAttribute('aria-selected'), 'true');
    await page.locator('#agent-progress').getByText('闭环评估', {exact: true}).waitFor();
    await page.locator('#agent-progress .agent-badge').getByText('已完成', {exact: true}).first().waitFor();
    await page.locator('#agent-progress .agent-badge').getByText('未通过', {exact: true}).first().waitFor();
    assert.ok(await page.locator('#agent-progress progress').isVisible());

    for (const name of expected) {
      await page.getByRole('button', {name, exact: true}).click();
      await page.locator('#agent-experiment-title').getByText(name).waitFor();
      assert.equal(await page.locator('#agent-experiment-path').textContent(), 'Output/test_01/' + name);
      await page.getByRole('tab', {name: '任务安排'}).click();
      await page.locator('#agent-task').getByText(sources[name] || '任务安排未记录').first().waitFor();
      await page.getByRole('tab', {name: '运行进度'}).click();
      await page.locator('#agent-progress .agent-stage').first().waitFor();
      const progressFile = ['experiment_07', 'experiment_09'].includes(name) ? 'PROGRESS.md' : 'progress.md';
      await page.locator('#agent-progress').getByText(progressFile + ' · 文件更新', {exact: false}).waitFor();
      await page.getByRole('tab', {name: '实验结果'}).click();
      await page.locator('#agent-results .agent-result-intro').waitFor();
      await page.locator('#agent-results table tbody tr').first().waitFor();
    }
    await page.getByText('查看完整实验报告', {exact: true}).click();
    const download = page.waitForEvent('download');
    await page.getByText('下载报告 Markdown', {exact: true}).click();
    assert.equal((await download).suggestedFilename(), 'report.md');
    await page.getByRole('tab', {name: '运行进度'}).click();
    await page.locator('#agent-progress .agent-stage').first().waitFor();
    await page.setViewportSize({width: 390, height: 844});
    await page.reload();
    await page.locator('#agent-experiment-title').getByText('experiment_10').waitFor();
    await page.getByRole('tab', {name: '运行进度'}).click();
    const completion = await page.locator('#agent-progress progress').boundingBox();
    assert.ok(completion && completion.y + completion.height <= 844, 'mobile completion stays on the first screen');
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await page.getByRole('tab', {name: '实验结果'}).click();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    assert.deepEqual(errors, []);
    assert.deepEqual(writes, []);
    console.log('10 experiments checked; both monitors and mobile layout verified');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
