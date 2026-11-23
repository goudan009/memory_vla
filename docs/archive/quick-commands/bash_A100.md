# # A100

模型训练

```bash
  cd /nvme3/projects/memory_vla

  # 公共路径，先设置一次
  train_common=(
    --dataset /nvme3/data_wyh/datasets/b1k2025_groot_task21_v1_200
    --checkpoint /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k
    --split /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/20261109_2100_task21_ddp_async_s42/config/data_split.json
  )
  train_output_root=/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training

  # M1 + baseline：各两张卡，同时训练，验证与训练并行
  ./train/run_groot_train.sh "${train_common[@]}" \
    --output "$train_output_root/NEW_PAIRED_RUN" \
    --gpus 2,3 --paired --baseline-gpus 0,1 \
    --async-validation-gpu 1 --baseline-validation-gpu 0 \
    --steps 10000 --global-windows-per-step 2 --microbatch 4 \
    --learning-rate 1e-5 --seed 42 --dry-run

  # baseline：仅训练无记忆对照，GPU0/1双卡
  # 当前单组模式的验证与训练串行
  ./train/run_groot_train.sh "${train_common[@]}" \
    --output "$train_output_root/NEW_BASELINE_RUN" \
    --arm baseline --gpus 0,1 \
    --steps 10000 --global-windows-per-step 2 --microbatch 4 \
    --learning-rate 1e-5 --seed 42 --dry-run

  # M1：仅训练记忆模型，GPU2/3双卡
  # 当前单组模式的验证与训练串行
  ./train/run_groot_train.sh "${train_common[@]}" \
    --output "$train_output_root/NEW_MEMORY_RUN" \
    --arm memory_v1 --gpus 2,3 \
    --steps 10000 --global-windows-per-step 2 --microbatch 4 \
    --learning-rate 1e-5 --seed 42 --dry-run

  以上均为预览命令。实际训练时换成自己的新输出目录，去掉 --dry-run；三个示例按需选
  择。

  # 参数说明（按需添加或修改）

  --dataset 数据集目录           # 已转换的Task21示教数据
  --checkpoint 初始权重目录      # 模型起点；不是完整断点恢复
  --split 划分JSON路径           # 固定180训练/20独立验证Episode
  --output 新运行目录            # 保存日志、配置、曲线和权重

  --arm memory_v1                # memory_v1 / baseline；默认memory_v1
  --gpus 2,3                     # 当前组GPU；单卡可以填 --gpus 2
  --paired                       # 同时训练M1和baseline
  --baseline-gpus 0,1            # paired时必填，两组GPU不能重叠
  --async-validation-gpu 1       # paired时启用异步验证，须属于baseline训练卡
  --baseline-validation-gpu 0    # 对照验证GPU；不填则两组验证共享上一张卡

  --steps 10000                  # 每组优化步上限；正式模式为1000的正整数倍
  --global-windows-per-step 2    # 每步总窗口数；须能被各组GPU数量整除
  --microbatch 4                 # 每rank一次动作头计算的决策数，默认4
  --learning-rate 1e-5           # 固定学习率，默认1e-5
  --seed 42                      # 随机种子，公平对照保持一致

  --resume-step 1000             # 从原output最新完整断点恢复，要求paired+异步验证
  --train-only                   # 单组工程短测，不验证、不保存checkpoint
  --source-dir 源码根目录        # 可选，覆盖默认源码位置
  --python-path Python完整路径   # 可选，覆盖默认Python环境

  --dry-run                      # 只打印启动计划，不启动训练
  --help                         # 查看帮助
```

进度查看

```bash
cd /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training
cd /训练次数
cd logs
tail -n 3 -F xxx.log
```

启动评估-开启推理服务

```bash
cd /nvme3/projects/memory_vla

bash evaluation/scripts/run_groot_server.sh \
    --checkpoint /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/20261110_1745_task0_baseline_ddp_s42/baseline/checkpoints/step-003000 \
    --gpu-index 1 \
    --port 5561 \
    --foreground

bash evaluation/scripts/run_groot_server.sh \
    --checkpoint /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/20261109_2100_task21_ddp_async_s42/baseline/checkpoints/step-010000 \
    --gpu-index 1 \
    --port 5560 \
    --foreground

bash evaluation/scripts/run_groot_server.sh \
    --checkpoint /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k \
    --gpu-index 0 \
    --port 5556 \
    --foreground


  # 忘记 PID，可以按端口查：5561改造的|5560baseline
  ss -ltnp | grep ':5561'
 # checkpoint存放地
  /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/
  training/XXXXXXXXXXXXXXXXXXX/
  ├── baseline/checkpoints/       无记忆对照
  └── memory_v1/checkpoints/      M1
```
