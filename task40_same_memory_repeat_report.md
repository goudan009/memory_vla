# Task40 同检查点、同正常记忆三次恢复

**结论：三次均完成 10 次决策 × 8 步。观测、动作和运动轨迹有明显漂移，但在本锚点的 80 步内，未导致可观测的任务阶段分歧。** 三次都靠近微波炉并伸右臂，但门均未打开。没有跨阶段正例，因此不能据此认为漂移在门把手接触或阶段切换边界也无害。

## 设置与控制变量

- 只读复用上一轮实例 4、控制步 72／下一决策 9 的门前 `sim_checkpoint.pt`。新仿真进程初始化该实例后连续加载三次；第一轮前是环境初始化，后两轮前是上一轮的 80 步后状态。这是顺序恢复，不是三个独立冷启动。
- 三轮都走同一个加载与刷新路径，恢复环境和策略 RNG。每次在推理服务清空记忆，按原始 9 次请求重建正常记忆，再与保存的在线库核对：两库各 9 条，索引 0–8，张量逐元素相同（最大差 `0`），容量和写入时钟相同。三轮使用同一权重，不测试空记忆或错误记忆。
- 每次加载后测得机器人、微波炉和袋子的平移差为 `0`，机器人和微波炉关节差为 `0`，四元数分量最大绝对差 `1.19e-7`，控制器 dump 全等。对齐本地控制步 73–152 和决策 9–18，不用不回拨的全局时钟对齐。

## 首帧与十次闭环差异

RGB 按 `uint8` 通道值计算 MAE（范围 0–255）；动作比较实际执行的 8×23 控制块，表中是混合控制量的最大分量差，不是物理距离。位移为 80 步内同一控制步的世界坐标距离最大值。

| 恢复轮次 | 首帧 RGB MAE：左腕／右腕／头部 | 首动作最大差 | 十块动作最大差 | 右末端／基座 XY 最大位移差 | 阶段不一致步数 |
|---|---:|---:|---:|---:|---:|
| 1 vs 2 | 98.75／53.35／84.73 | 0.18984 | 0.20409 | 7.82／5.12 cm | 0/80 |
| 1 vs 3 | 101.50／79.20／82.98 | 0.18984 | 0.20795 | 4.40／3.98 cm | 0/80 |
| 2 vs 3 | 9.13／67.10／10.15 | 0.04102 | 0.20393 | 7.89／6.30 cm | 0/80 |

- 首请求的 37 个本体字段中，只有 `state.joint_qeffort` 有差；三组配对的最大分量差分别为 `1177.46 / 1109.01 / 299.55`（请求记录值，不据此推断模型使用了这个字段）。实际首动作最大分量差来自底盘；10 个决策块中，右臂控制的最大分量差达 `0.16235`，右夹爪达 `0.00781`。
- `first_policy_rgb.png` 是从首请求 NPZ 直接解码的拼图（行是轮次，列是左腕、右腕、头部）。第一轮头部看到的是厨房另一侧，第二、三轮看到的是微波炉。这不是相同视野内的小像素噪声，表明首轮 RGB 与已恢复的门前位姿没有同步；与读到加载前相机帧相符，但尚未确定是哪个渲染／传感器更新环节。第二次决策时，三组头部 RGB MAE 降至 `15.89 / 17.44 / 4.48`，此后仍存在轨迹和视图差异。
- 视频、逐步真值和门关节读数一致：三轮均持续闭门、两臂均未抓到袋子、未放入、未启动，两项原生目标持续为假，评估器累计进度均保持 `25%`（不代表完成开门）。右末端到微波炉原点的最小距离为 `28.35 / 31.09 / 30.18 cm`。没有看到开门或跳到后续成功阶段；策略没有输出内部阶段标签，不能用真值相同证明模型内部决策完全一样。

**对消融的判断：** 在这一个窗口，漂移改变了执行动作和接近轨迹，未改变阶段结果。首帧视角错位与约 8 cm 的末端分歧不能当作可忽略的对照噪声。正式消融前，先验证恢复后的三路图像确实对应锚点，再用含明确开门转移的短片段复测阶段边沿是否稳定；不要把本次 3/3 未开门当成确定性证据。

## 复现与证据

实测目录 `Output/task40_same_memory.3N9GkN/`：`pairwise_audit.json` 含三对、每对十次的全部观测和动作差值；`restore_compare_*.json` 与 `memory_compare_*.json` 为各次锚点对照；`restore_*/truth.jsonl` 与 `restore_*/short.mp4` 分别为 80 条逐步真值和 80 帧三路视频。`sim.log` 记录 `TASK40_REPEAT_DONE 3`，`policy.log` 记录三次 `TASK40_MEMORY_MATCH 9`。记录模块测试 4 项通过。

```bash
source_run="$PWD/Output/task40_replay_validation.GXcT5J"
run=$(mktemp -d "$PWD/Output/task40_same_memory.XXXXXX")
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /nvme3/projects/GR00T/official-baseline-env/bin/python -u evaluation/scripts/task40_replay_policy.py \
  --checkpoint "$PWD/Output/task40_memory_overnight/replay/original_step4000" \
  --source-run "$source_run" --output "$run" --port 5571 > "$run/policy.log" 2>&1 &
# 等 policy.log 出现 TASK40_POLICY_READY 后再启动仿真。
env -u OMNIGIBSON_DATA_PATH OMNIGIBSON_GPU_ID=0 bash scripts/b1k_python.sh \
  evaluation/scripts/task40_replay_probe.py --source-run "$source_run" --output "$run" --port 5571 \
  --behavior /nvme3/projects/GR00T/BEHAVIOR-1K-task-progress \
  --groot /nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1 > "$run/sim.log" 2>&1
python3 evaluation/scripts/task40_same_memory_audit.py "$run" > "$run/pairwise_audit.json"
```
