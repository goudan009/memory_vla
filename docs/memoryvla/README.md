# MemoryVLA 源码讲解与前向分析材料

直接交给动画制作模型的主文档：

[MemoryVLA_源码解析与动画分镜.md](MemoryVLA_源码解析与动画分镜.md)

它包含可直接复制的制作要求、12 分钟 / 16 镜头中文旁白、逐模块 input/output、动画布局、记忆状态变化和源码索引。主文档可独立使用；需要真实数据时，再附上 [forward_trace.json](forward_trace.json)。

| 文件 | 用途 |
| --- | --- |
| `MemoryVLA_源码解析与动画分镜.md` | 完整分析与视频脚本 |
| `forward_trace.json` | 本次 GPU 执行的 492 条事件及动作数组 |
| `probe_forward.py` | 运行原始网络并记录输入输出 |
| `forward_run.log` | 成功运行的终端输出 |

## 执行范围

参考源码位于 `src/references_code/memoryvla`，基于 OpenVLA。探针不修改该目录；它直接载入 `vla/memory_vla.py`，跳过 `vla/__init__.py` 中与前向无关的 TensorFlow 数据管道导入。

DINOv2、SigLIP、LLaMA-2 7B、记忆模块和 DiT-L 都保留原尺寸。网络使用随机初始化权重，图像/动作输入及动作缩放统计是合成数据；没有加载训练 checkpoint，也没有 backward、参数更新、仿真或真实机器人控制。

构造阶段用临时 mock 让视觉主干不下载预训练权重，用显式 LLaMA-2 7B 配置替代联网取配置，并注入本地原始分词器。进入 forward / predict_action 后使用的是参考类及第三方依赖的实际计算。模型 seed 初始设为 20261113；原始 Prismatic 构造器内部的再次设 seed 行为也被保留。

本次 PyTorch 是 2.7.1+cu128，与上游 README 声明的 2.2.0 不同；Transformers 和 timm 分别使用上游指定的 4.40.1 和 0.9.10。BF16 + SDPA 的执行结果用于验证结构，不能当作原始训练环境的性能复现。

## 本机复现

在 `/nvme3/projects/memory_vla` 运行。现有分析依赖位于 `/tmp/memoryvla-video-deps.R6cH0E`，原始分词器位于该目录下 `tokenizer.model`。这些临时依赖只补充本机已有的 GR00T Python 环境，没有覆盖其已安装包。

```bash
env PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/tmp/memoryvla-video-deps.R6cH0E \
    src/v1_groot_memoryvla/.venv/bin/python -u \
    docs/memoryvla/probe_forward.py \
    --tokenizer /tmp/memoryvla-video-deps.R6cH0E/tokenizer.model \
    --device cuda:0 \
    --output docs/memoryvla/forward_trace_rerun.json
```

探针本身需要 GPU 可见；若在隔离执行环境里 CUDA 不可见，应通过该环境的 GPU 授权运行机制执行。原始记录另存为 `forward_trace.json`，上述命令用不同文件名保存复跑结果。

若临时目录已经清理，先创建新的分析目录，并记下输出的实际路径：

```bash
mktemp -d /tmp/memoryvla-video-deps.XXXXXX
```

将下列 `<分析目录>` 换成上一步返回的路径，再安装本次用到的小型补充依赖：

```bash
uv pip install \
  --python src/v1_groot_memoryvla/.venv/bin/python \
  --target <分析目录> --cache-dir <分析目录>/cache --no-deps \
  timm==0.9.10 transformers==4.40.1 tokenizers==0.19.1 \
  sentencepiece==0.1.99 draccus==0.8.0
```

这一安装命令依赖本机已有的 torch、torchvision、numpy、accelerate 等基础包；它不是一份从空白系统安装训练环境的说明。

分词器来源是 `openvla/openvla-7b` 的 `tokenizer.model`，约 488 KiB，不需要下载模型权重。可以使用 Hugging Face CLI：

```bash
src/v1_groot_memoryvla/.venv/bin/python -m huggingface_hub.commands.huggingface_cli download \
  openvla/openvla-7b tokenizer.model --local-dir <分析目录>/tokenizer
```

本次直连 Hugging Face 超时，实际从 `https://hf-mirror.com/openvla/openvla-7b/resolve/main/tokenizer.model` 获取了同名文件。JSON 的 `tokenizer_sha256` 固定了本次具体内容。重新下载后，可用 `sha256sum` 对照，再将运行命令中的 `PYTHONPATH` 和 `--tokenizer` 改为新路径。

## 已核对的证据

- `forward()` 返回有限标量损失及 VLM 输出，训练 batch=2 经 R=4 重复后动作 batch=8。
- 三次部署返回两份 `[16,7]` 数组，每次调用 DiT 10 次、每个记忆库更新一次。
- 首帧和重置帧没有检索旧历史；第二次请求执行两路各两层检索。
- `cur_timestep` 为 `1→2→1`，对应 bank 时间戳 `[0]→[0,1]→[0]`。
- 满库时认知检索的历史输入为 `[1,16,4096]`，感知为 `[1,4096,256]`。
- 18 次原尺寸记忆写入后，双库仍各有 16 条；另用容量 3 的手算样例核对相邻平均合并。
- 原始记录中 838 个带 `finite` 的数值记录均通过；99 个参考 Python 源文件的 SHA-256 与运行后的源码一致。

记录包含计时，但 hook 的数值统计会同步 GPU，且没有做标准化的预热/重复采样。视频不应把这些计时转述为部署延迟、帧率或模型性能成绩。

排查时使用了 `systematic-debugging` 的调用链检查：初次探针在扩散 `q_sample` 产生 FP32 后与 BF16 动作头发生 dtype 不匹配，原因是缺少官方训练循环的外层 autocast。探针补齐该上下文后，训练和部署路径全部通过；参考实现保持原样。
