# 统一资源管理（人 / Agent 共用）

资源管理是唯一日常资源入口。已有训练、推理、评估不受页面更名影响。历史任务对比暂留，避免丢失旧记录。

## 执行顺序

1. 查询 `memory-workbench resource hosts`、`resource disks --host A100` 和已有目录；优先复用已有权重/数据，核对版本、设备、完整性和实际路径。
2. 下载只使用管理员配置的配方：`resource download-options` → `resource download-status ID` → `resource download-preview ID` → `resource download-start ID`。未配置或预检失败即报告阻塞，不擅自 curl/wget/git clone、创建环境、换源、改变版本或下载目录。
3. 后台作业在目标设备指定 Python 中执行；固定 HF revision 和文件清单，逐文件大小与 SHA-256/Git blob SHA-1 校验，先暂存，再完成发布。锁防止同目标并行下载。状态持久化，服务重启后可查询；提交超时先查 status。失败重试同配方，保留已校验文件。完成后使用 register 登记。
4. 目录登记、创建、回收、恢复都使用统一 resource API/CLI。删除前核对依赖、数据价值及用户授权；不直接 rm，训练 checkpoint 使用训练详情受保护接口。回收不释放磁盘。未接入的永久清理、外部传输和非 HF 来源应报告阻塞，不自行绕过。
5. 下载不占用 GPU 作为训练任务，不为下载修改模型配置。正在运行的旧下载不要接管或重启；核对原任务后登记完整资源。

## 部署

复制 `configs/workbench_downloads.example.json` 为本地 `configs/workbench_downloads.json`（Git 忽略）。管理员填写实际 host、已存在的资源根目录、专用目标目录、实际下载 Python（可导入 huggingface_hub）、仓库内固定清单及 resource 名称。示例目标与旧 N1.7 目标不同，避免冲突；不要把示例 Python 当实际环境。未配置时 UI/CLI 显示无配方，不执行任何下载。远程 SSH 复用现有设备配置。每个配方只允许一个明确目标；目标存在拒绝覆盖。当前仅支持带完整文件摘要的 Hugging Face 配方，其他下载来源需要显式接入。目标根目录必须为真实绝对路径，禁止软链接。检查磁盘保守预留完整清单+1 GiB；空间不足时先人工检查，禁止自动删除。

API: GET /api/resources/downloads?action=options 或 action=status&recipe_id=ID；POST 同路径 JSON action=preview|start、recipe_id=ID。既有 /api/registry/storage 支持 hosts/disks/browse/stat/size/register/mkdir/trash/restore。CLI `memory-workbench resource --help`。下载任务状态以真实校验字节为准，不把下载开始当作完成。

## 实验管理与 Agent 正式训练

Agent 新训练经 HTTP/CLI 提交时必须显式填写 `initiator=agent`、简短 `notes`、规范 `run_id` 和 `agent_plan: {"version":"v001","node":"train"}`（节点值必须取当前真实计划）。工作台校验最新有效计划、节点及显式前置实验验收，记录实际提交回执于 outputs/workbench_agent_bindings/。预检不写提交记录。提交失败/超时先核对 jobs，pending 回执需人工核对后处理，不自行改 run_id 重试。该校验不等于对全部科学参数的用户批准，也不替代任务书。旧作业及严格续训不回填、不改身份。实验管理会按规范 run ID 自动关联训练记录，人工/Agent 来源按实际字段显示。

## 页面整合

训练记录默认隐藏已明确标记的工程预检（仍可切换查看，未分类历史不猜测）。推理评估仅在初始选择时优先匹配既有地址的就绪服务，否则只有唯一就绪服务时自动匹配；多服务需人工确认。结果分析页提供历史任务对比和单次结果 JSON 导出（含结果记录中的配置与回合数据、视频索引，不含视频文件）；历史对比兼容入口保留。通用资源回收在同设备存在活动/排队作业或状态不可查时保守拒绝，避免破坏当前实验；训练 checkpoint 仍走原受保护入口。此机制尚不是逐文件依赖图。

下载 start 请求必须带上 preview 返回的 `job_id` 作为 `expected_job_id`，GUI/CLI 自动传递，防止管理员在预检后改变配方。已存在的目标显示 existing，不当作下载完整。磁盘预检不重复计算已暂存且大小一致的完整文件，重启仍重新校验摘要。通用资源回收还核对服务监控及 4090 Benchmark 作业；状态不明时保留资源。

## 工作台交互

资源管理默认显示数据与权重；下载任务、磁盘空间分别为独立视图。下载状态摘要仅显示关键字段，原始 JSON 在详情中，已完成或已有目标可打开预填登记窗口，仍需确认后登记。训练页把低频配置与次要操作折叠；原始参数和证据仍可查看。任务对比打开时仅显示比较视图，关闭返回单次结果。旧历史对比入口保留兼容。上述 UI 改动不改变资源、训练或评估的后端参数与作业身份。
