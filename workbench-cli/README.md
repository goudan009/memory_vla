# MemoryVLA 工作台 CLI

需要 Python 3.10 或更高版本。在项目根目录安装。

```sh
python3 -m pip install --user ./workbench-cli
```

CLI 默认连接当前机器的 `http://127.0.0.1:8765`。在其他机器使用时，通过 `--url` 或环境变量 `MEMORY_WORKBENCH_URL` 指定可访问的工作台地址。

```sh
memory-workbench status
memory-workbench status --json
memory-workbench train options --device a100
memory-workbench infer jobs --device a100
memory-workbench eval jobs
memory-workbench results
```

`train`、`infer`、`eval` 共用 `preview` 和 `start`。配置文件使用工作台 API 的 JSON 字段，可通过 `--device` 指定设备。`start` 先做预检，通过后提交任务并输出作业记录。

```sh
memory-workbench eval preview -f eval.json
memory-workbench eval start -f eval.json
memory-workbench eval jobs
```

在部署了 MemoryVLA 项目的主机上，可以从仓库根目录执行 `memory-workbench serve` 启动服务，也可用 `--repo` 指定仓库。当前已有工作台服务时，直接使用查询和提交命令。任务配置、资源规则和 API 字段见 [工作台执行手册](../docs/workbench/agent.md)。
