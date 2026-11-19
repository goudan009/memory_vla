> 当前目录与正式协议以 [evaluation/evaluation.md](../evaluation/evaluation.md) 和[使用说明](../evaluation/README.md)为准。默认结果在 V_1/baseline，实验用 `--version V_1 --group experment_N`，也支持自定义名称；网页读取 JSON 结果，不再生成或追加 results.md。

 在项目根目录运行：

  cd /nvme3/projects/memory_vla

# 默认：单卡单环境，10个完整 Episode

  ./evaluation/scripts/run_groot_eval.sh

# 单卡双环境：每个环境5个，共10个

  ./evaluation/scripts/run_groot_eval.sh --mode single2

# 指定使用 GPU0

  ./evaluation/scripts/run_groot_eval.sh --mode single2 --gpu-index 0

# 双卡四环境：一次命令，同时启动两张卡

  ./evaluation/scripts/run_groot_eval.sh --mode dual2

  双卡四环境内部会启动两个独立 GPU 队列，每卡2个环境，无需手动执行两次。默认总共10个实例：GPU0负责0～4，GPU1负责5～9，每卡两个
  环境分别完成2、3个。两卡日志和实例分配隔离，结果合并到同一个运行目录。

  新目录自动命名为 evaluation/V_1/baseline/groot_task21_1/、groot_task21_2/，默认只保留：
  results.json（先看成功率、进度、状态；配置和性能也在里面）、episodes.jsonl（每回合立即保存）、gpuN.log（每卡一份日志）。
  运行时辅助文件结束后清理；需要详细诊断才加 --diagnostics。旧运行不自动搬动或删除。

# 后续 Memory 模型结果名称（按 grootmeory 拼写；模型由对应 Policy Server 提供）

  ./evaluation/scripts/run_groot_eval.sh --model-name grootmeory --mode single2

  输出例如 evaluation/V_1/baseline/grootmeory_task21_1/；实验可加 --group experment_1。--model-name 只控制名称，不切换服务端模型。

  种子支持以下方式，适用于上述所有模式：

# 每个环境、每个 Episode 都固定使用42

  ./evaluation/scripts/run_groot_eval.sh --seed-mode fixed --seed 42

# 每个 Episode 使用不同的伪随机种子；相同配置可重复生成同一组种子

  ./evaluation/scripts/run_groot_eval.sh --seed-mode random --seed 42

# 每次启动生成新的随机基准种子，再生成各 Episode 的种子

  ./evaluation/scripts/run_groot_eval.sh --seed-mode random --seed random

  不指定时保留原有递增种子规则，从0开始。实际使用的种子会记录到结果中；Baseline 与 Memory 对比建议显式指定相同的种子配置。

  只有快速测试才设置 runner 步数上限：

# 双卡四环境，每环境最多1200动作步，约150轮外层循环

  ./evaluation/scripts/run_groot_eval.sh \
    --mode dual2 \
    --preset fast-debug \
    --debug-max-steps 1200

  正式模式使用 B1K 原生 Horizon，不接受 --debug-max-steps。快速测试截断结果不会计入正式成功率。
