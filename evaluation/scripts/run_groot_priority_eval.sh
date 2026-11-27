#!/usr/bin/env bash
# Bash 严格模式：-e 命令失败即退出；-u 使用未定义变量报错；pipefail 管道中任一命令失败则整体失败。
set -euo pipefail

# 解析当前启动脚本的绝对路径。注意这里的 $0 是脚本路径，不是项目目录。
LAUNCHER_PATH=$(realpath "$0")
# 项目根目录：优先使用外部 PROJECT_ROOT；否则从 evaluation/scripts 向上三级。
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$LAUNCHER_PATH")/../.." && pwd)}
# GR00T N1.6 源码目录。后续在其中定位 gr00t/eval/rollout_policy.py。
GROOT_REPO=${GROOT_REPO:-/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1}
# 仿真 Python 包装启动脚本：真正运行时调用它，而不是直接调用下面的 B1K_PYTHON。
B1K_PYTHON_LAUNCHER=${B1K_PYTHON_LAUNCHER:-$PROJECT_ROOT/scripts/b1k_python.sh}
# 保存仿真 Python 环境位置，用于路径检查和元数据记录；本文件没有直接用它启动子进程。
B1K_PYTHON=${B1K_PYTHON:-/nvme3/envs/b1k2025-sim/bin/python}
# OmniGibson 源码路径，需要加入子进程 Python 搜索路径。
OMNIGIBSON_ROOT=${OMNIGIBSON_ROOT:-/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/OmniGibson}
# BDDL（BEHAVIOR Domain Definition Language）任务定义相关源码。
BDDL_ROOT=${BDDL_ROOT:-/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress/bddl}
# BEHAVIOR 场景/物体等资源所在目录。
# 评测固定使用项目资产；其他项目遗留的 OMNIGIBSON_DATA_PATH 不应改变评测输入。
# 需要外部资产根目录时，显式设置 BEHAVIOR_DATA。
BEHAVIOR_DATA=${BEHAVIOR_DATA:-$PROJECT_ROOT/assets/simulation}
# 本轮评测的标识：默认按当前时间生成，作为日志和任务输出目录的后缀。
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

# 导出 Bash 变量，使后面接管执行的 Python 可以通过 os.environ 读取。
export LAUNCHER_PATH PROJECT_ROOT GROOT_REPO B1K_PYTHON_LAUNCHER B1K_PYTHON
export OMNIGIBSON_ROOT BDDL_ROOT BEHAVIOR_DATA RUN_TAG

# 用系统 Python 替换当前 Bash 进程。-u 表示无缓冲输出；- 从标准输入读取下方嵌入的 Python；"$@" 原样传递命令行参数。
# <<'PY' 是带引号的 here-document，Bash 不展开其中的 $ 或反斜杠等内容；直到末尾独立的 PY 标记才结束。
exec /usr/bin/python3 -u - "$@" <<'PY'
# 下方为主控 Python：只调度任务、检查资源和汇总结果；实际的 Isaac Sim 评测在另一个子进程执行。
import argparse
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(os.environ["LAUNCHER_PATH"]).parent))
from groot_eval_results import finish_run, write_json, write_manifest
from groot_tasks import TASK_NAMES, missing_instances, print_tasks


# Task IDs match the fixed adapter. Timeout hints never override native limits.
TASKS = {task_id: {"name": name, "timeout_hint": {21: 38372, 22: 15384}.get(task_id)}
         for task_id, name in TASK_NAMES.items()}
# 策略服务器地址：仿真在本机 4090 上跑，模型推理由远端策略服务提供。请按实际网络修改。
POLICY_HOST = "172.28.62.17"
# 远端 PolicyClient 连接端口。
POLICY_PORT = 5556
# OmniGibson 使用的 GPU 索引；它通过 OMNIGIBSON_GPU_ID 下发，并不等价于设置 CUDA_VISIBLE_DEVICES。
GPU_INDEX = "1"
# 共卡准入检查使用剩余显存，单位 MiB；不更改仿真参数。
GPU_MIN_FREE_MIB = {1: 10240, 2: 24576}
# 默认每个任务测试 10 条 episode（独立回合）；注意回合数不是仿真步数。
DEFAULT_EPISODES = 10
# 每次向策略服务器请求动作后，在环境中连续执行的动作步数；是否每 8 步才再次推理由 rollout_policy.py 的实现决定。
DEFAULT_ACTION_STEPS = 8
# 传给外部 rollout 的极大回合步数上限：意图是不让 runner 提前截断；真正停止时间仍取决于环境结束条件、环境自带超时及 runner 实现。
# 重要：这个数不是预期执行约 10 亿步，也不是本脚本保证采用的 BEHAVIOR 原生超时。
RUNNER_MAX_EPISODE_STEPS = 999999999

# 命令行配置入口，可传 --tasks、--episodes、--action-steps、--dry-run。
parser = argparse.ArgumentParser(
    description="Run the GR00T N1.6 BEHAVIOR priority-task queue on the 4090."
)
parser.add_argument("--preset", choices=["fast-debug", "official-eval"], default="official-eval")
parser.add_argument("--debug-max-steps", type=int, help="fast-debug only: runner action-step cap (default: 1200); never modifies the native task timeout")
parser.add_argument("--tasks", type=int, nargs="+", default=None, help="task IDs to run in order")
parser.add_argument("--list-tasks", action="store_true", help="list all 50 tasks and local instance availability without launching")
parser.add_argument("--episodes", type=int, default=None)
parser.add_argument("--n-envs", type=int, choices=[1, 2, 4], default=1)
parser.add_argument("--action-steps", type=int, default=DEFAULT_ACTION_STEPS)
parser.add_argument("--gpu", "--gpu-index", dest="gpu", choices=["0", "1"], default=GPU_INDEX, help="simulation GPU index (default: 1)")
parser.add_argument("--policy-host", default=POLICY_HOST)
parser.add_argument("--policy-port", type=int, default=POLICY_PORT)
parser.add_argument("--run-tag", default=os.environ["RUN_TAG"])
parser.add_argument("--output-root", type=pathlib.Path, help="directory containing this GPU queue log and task outputs")
parser.add_argument("--public-output-dir", type=pathlib.Path, help=argparse.SUPPRESS)
parser.add_argument("--instance-ids", "--instance-id", dest="instance_ids", type=int, nargs="+", help="explicit BEHAVIOR test instance IDs (0-9); use disjoint lists for concurrent same-task runs")
parser.add_argument("--seed", type=int, default=0, help="base seed reused for matched task instances and environment slots")
parser.add_argument("--seed-mode", choices=["sequence", "fixed", "random"], default="sequence", help="sequence: base+env_idx+episode; fixed: identical seed; random: per-episode seeds derived from base")
parser.add_argument("--limit-env-episodes", action="store_true", help="stop each vector slot at its assigned valid episode quota; leave finished slots idle")
parser.add_argument("--episode-layout", help=argparse.SUPPRESS)
parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=None, help="record video (enabled by default; disabled by default for --profile-cycles)")
parser.add_argument("--disable-viewer", action="store_true", help="pause only the viewer's render products after camera initialization/reset; keep robot cameras and graph creation order")
parser.add_argument("--audit-semantics", action="store_true", help="record render-product ownership, policy observations, actions and window validity")
parser.add_argument("--diagnostic-only", action="store_true", help="stop the process after the profile window; do not count it as a completed episode")
parser.add_argument("--monitor-resources", action="store_true", help="sample GPU/CPU/RAM and I/O throughout full episodes as well")
parser.add_argument("--video-steps-per-render", type=int, default=24, help="action steps per saved frame (24 gives 16x at 30 Hz and 20 FPS)")
parser.add_argument("--profile-cycles", type=int, choices=range(100, 201), help="measure the first 100-200 policy/env cycles; finish the episode normally")
parser.add_argument("--appdata-path", help="reuse an existing Isaac Sim appdata/cache directory (single process only)")
parser.add_argument("--wait-for-gpu", action="store_true", help="wait for free GPU memory: >10 GiB for one env, >24 GiB for two envs")
parser.add_argument("--gpu-wait-interval", type=float, default=30.0, help="seconds between GPU availability checks when --wait-for-gpu is set")
parser.add_argument("--dry-run", action="store_true", help="print the plan without starting or writing anything")
args = parser.parse_args()
if args.list_tasks:
    print_tasks(os.environ["GROOT_REPO"])
    raise SystemExit(0)
if args.record_video is None:
    args.record_video = args.profile_cycles is None
if args.debug_max_steps is not None and args.preset != "fast-debug":
    parser.error("--debug-max-steps requires --preset fast-debug")
if args.debug_max_steps is not None and args.debug_max_steps < 1:
    parser.error("--debug-max-steps must be positive")
if args.preset == "fast-debug":
    RUNNER_MAX_EPISODE_STEPS = args.debug_max_steps or 1200
if args.diagnostic_only and args.profile_cycles is None:
    parser.error("--diagnostic-only requires --profile-cycles")
if (args.disable_viewer or args.audit_semantics) and args.n_envs != 1:
    parser.error("--disable-viewer and --audit-semantics require --n-envs 1")
if args.tasks is None:
    args.tasks = [21] if args.preset == "fast-debug" else [21, 22, 26]
if args.public_output_dir is not None and len(args.tasks) != 1:
    parser.error("a compact GPU queue must evaluate exactly one task")
if args.episodes is None:
    args.episodes = 1 if args.preset == "fast-debug" else DEFAULT_EPISODES
GPU_INDEX = args.gpu
POLICY_HOST = args.policy_host
POLICY_PORT = args.policy_port
if not 1 <= POLICY_PORT <= 65535:
    parser.error("--policy-port must be between 1 and 65535")

# 输入校验：禁止任务白名单以外的 ID，并限制回合数/动作步数必须为正整数。
for task_id in args.tasks:
    if task_id not in TASKS:
        parser.error(f"unsupported task id: {task_id}")
if args.episodes < 1 or args.action_steps < 1 or args.video_steps_per_render < 1:
    parser.error("--episodes, --action-steps, and --video-steps-per-render must be positive")
if args.gpu_wait_interval <= 0:
    parser.error("--gpu-wait-interval must be positive")
if args.profile_cycles and (args.n_envs != 1 or args.episodes != 1 or args.record_video):
    parser.error("--profile-cycles requires --n-envs 1, --episodes 1, and video disabled")
if args.episodes < args.n_envs:
    parser.error("--episodes must be at least --n-envs (the GR00T runner raises its effective count to n_envs)")
if args.instance_ids is not None:
    if len(set(args.instance_ids)) != len(args.instance_ids):
        parser.error("--instance-ids must not contain duplicates")
    if any(instance_id not in range(10) for instance_id in args.instance_ids):
        parser.error("BEHAVIOR test instance IDs must be between 0 and 9")
    if len(args.instance_ids) < args.n_envs:
        parser.error("--instance-ids must provide at least one instance per environment")
episode_layout = None
if args.episode_layout:
    try:
        episode_layout = json.loads(args.episode_layout)
        ids = args.instance_ids if args.instance_ids is not None else list(range(10))
        width = len(ids) // args.n_envs
        assert isinstance(episode_layout, list) and len(episode_layout) == args.n_envs
        for i, slot in enumerate(episode_layout):
            expected = ids[i*width:(i+1)*width] if i < args.n_envs-1 else ids[i*width:]
            assert slot['instances'] == expected
            assert type(slot['quota']) is int and slot['quota'] >= 0
            assert type(slot['offset']) is int and slot['offset'] >= 0
        assert sum(slot['quota'] for slot in episode_layout) == args.episodes
        assert args.limit_env_episodes
    except (ValueError, AssertionError, KeyError, TypeError):
        parser.error('--episode-layout must match native instance partitions and the requested episode budget')

# 把从 Bash 导入的路径转换为 pathlib.Path，便于用 / 拼接目录、检查文件是否存在。
project_root = pathlib.Path(os.environ["PROJECT_ROOT"])
groot_repo = pathlib.Path(os.environ["GROOT_REPO"])
b1k_python_launcher = pathlib.Path(os.environ["B1K_PYTHON_LAUNCHER"])
b1k_python = pathlib.Path(os.environ["B1K_PYTHON"])
omnigibson_root = pathlib.Path(os.environ["OMNIGIBSON_ROOT"])
bddl_root = pathlib.Path(os.environ["BDDL_ROOT"])
behavior_data = pathlib.Path(os.environ["BEHAVIOR_DATA"])
run_tag = args.run_tag + f"_gpu{GPU_INDEX}"
# 限制 RUN_TAG 的字符集，避免生成不期望的目录/日志路径。
if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_tag):
    parser.error("RUN_TAG may contain only letters, digits, dot, underscore, and hyphen")

# 底层诊断入口也默认归入 baseline；日常评估由统一入口分配版本与运行目录。
eval_root = args.output_root.resolve() if args.output_root is not None else project_root / "evaluation/baseline"
appdata_path = pathlib.Path(args.appdata_path) if args.appdata_path else pathlib.Path("/nvme3/tmp/memory_vla") / ("groot_runtime_gpu" + GPU_INDEX) / "appdata"
if not appdata_path.is_absolute():
    parser.error("--appdata-path must be absolute")
# 队列级日志：记录不同任务的起止、预检查、成功率等，而非 Isaac Sim 全量输出。
public_output = args.public_output_dir.resolve() if args.public_output_dir is not None else None
queue_log = (public_output / f"gpu{GPU_INDEX}.log" if public_output else
             eval_root / f"groot_n1d6_priority_{run_tag}.queue.log")
# 按用户传入顺序建立任务队列，例如 [21,22,26]；任务是串行而非并行执行。
selected = [(task_id, TASKS[task_id]) for task_id in args.tasks]
# 为每个任务预计算独立结果目录，名称包含任务 ID、英文名和 RUN_TAG。
outputs = [
    eval_root / f"groot_n1d6_task{task_id:03d}_{task['name']}_{run_tag}"
    for task_id, task in selected
]

# 这是一段以字符串形式构造的“子进程启动代码”，而非立即执行的代码；稍后交给 b1k_python.sh -c。
# 用 repr（!r）把真实文件路径安全嵌入 Python 源码。
child_code = f'''import os, sys, runpy, traceback
repo={str(groot_repo)!r}
omnigibson={str(omnigibson_root)!r}
bddl={str(bddl_root)!r}
script=repo+"/gr00t/eval/rollout_policy.py"
try:
    # 必须先导入 isaacsim，再导入 OmniGibson/GR00T 相关路径，以满足当前仿真运行环境的初始化顺序。
    import isaacsim
    # 把三个源码目录插入模块搜索路径前端，优先导入这几份源码。
    sys.path[:0]=[repo,omnigibson,bddl]
    sys.path.insert(0, {str(project_root / 'scripts')!r})
    sys.path.insert(0, {str(pathlib.Path(os.environ['LAUNCHER_PATH']).parent)!r})
    from groot_eval_results import install_episode_recording
    install_episode_recording(os.environ["GROOT_EVAL_OUTPUT_DIR"])
    # 模拟直接执行 rollout_policy.py：把 argv[0] 替换为目标脚本，其余参数沿用启动命令参数。
    sys.argv=[script]+sys.argv[1:]
    # 在当前 Python 解释器内执行目标脚本，效果近似 python rollout_policy.py ...。
    runpy.run_path(script,run_name="__main__")
# 捕获子程序包括 SystemExit 在内的异常并输出堆栈。注意：这也会把目标脚本正常抛出的 SystemExit(0) 当作异常并以状态 1 退出。
except BaseException:
    traceback.print_exc()
    # 直接退出子进程，绕过 Python 正常清理；这是强制退出方式，异常时不会继续运行主控代码。
    os._exit(1)
# 子脚本正常返回时立即以 0 退出进程。
os._exit(0)
'''


# 构造传给 GR00T rollout_policy.py 的命令行参数；这里的参数名来自外部 runner，并非由当前脚本定义。
def runner_args(task_name):
    return [
        "--max_episode_steps", str(RUNNER_MAX_EPISODE_STEPS),
        "--n_episodes", str(args.episodes),
        # model_path 传空字符串：该模式依赖外部 policy_server，而不是本地加载模型。是否允许空值应以目标 runner 源码为准。
        "--model_path", "",
        "--policy_client_host", POLICY_HOST,
        "--policy_client_port", str(POLICY_PORT),
        # 环境 ID：sim_behavior_r1_pro/任务英文名；对应 BEHAVIOR R1 Pro 机器人。
        "--env_name", f"sim_behavior_r1_pro/{task_name}",
        # 并行环境数由 --n-envs 指定；当前 runner 在该环境数下执行同步策略查询。
        "--n_envs", str(args.n_envs),
        "--n_action_steps", str(args.action_steps),
        "--video-steps-per-render", str(args.video_steps_per_render),
        *( ["--profile-cycles", str(args.profile_cycles)] if args.profile_cycles else [] ),
        *( ["--record-video"] if args.record_video else [] ),
    ]


# 只打印配置和每个任务的实际 runner 参数，不建立日志/输出目录，也不启动仿真。
if args.dry_run:
    print("DRY RUN: no evaluation process started")
    print(f"PRESET: {args.preset}")
    if args.preset == "fast-debug":
        print("FAST DEBUG: runner-capped window; not an official success-rate sample")
    print(f"PROJECT_ROOT={project_root}")
    print(f"GROOT_REPO={groot_repo}")
    print(f"B1K_PYTHON_LAUNCHER={b1k_python_launcher}")
    print(f"B1K_PYTHON={b1k_python}")
    print(f"QUEUE_LOG={queue_log}")
    for outdir in outputs:
        print(f"RESULTS_JSON={outdir / 'results.json'}")
    print("ENV: OMNI_KIT_ACCEPT_EULA=YES LIBRARY_PATH=/usr/local/cuda/lib64/stubs")
    print(f"ENV: OMNIGIBSON_DATA_PATH={behavior_data} OMNIGIBSON_GPU_ID={GPU_INDEX} OMNIGIBSON_HEADLESS=true")
    print(f"ENV: OMNIGIBSON_APPDATA_PATH={appdata_path}")
    print(f"VIEWER_ENABLED: {not args.disable_viewer}; DIAGNOSTIC_ONLY: {args.diagnostic_only}")
    print("ENV: CUDA_VISIBLE_DEVICES unset; use the existing b1k2025-sim runtime wrapper")
    print("BOOTSTRAP: import isaacsim, then add GR00T/OmniGibson/BDDL without filtering the simulator site-packages")
    for task_id, task in selected:
        print(f"TASK {task_id}: {task['name']} episodes={args.episodes} n_envs={args.n_envs} n_action_steps={args.action_steps} record_video={args.record_video}")
        print(f"INSTANCE_IDS: {args.instance_ids if args.instance_ids is not None else list(range(10))}")
        print(f"SEED: {args.seed}")
        print(f"SEED_MODE: {args.seed_mode}; LIMIT_ENV_EPISODES: {args.limit_env_episodes}")
        print("RUNNER_ARGV: " + shlex.join(runner_args(task["name"])))
    raise SystemExit(0)


# 防止同一 RUN_TAG 覆盖已存在的队列日志或任务目录。
if (public_output is None and queue_log.exists()) or any(path.exists() for path in outputs):
    raise SystemExit(f"Refusing to overwrite an existing queue or task output for RUN_TAG={run_tag}")
# 逐项检查源码、环境启动脚本与资源目录是否存在；这里只检查“存在”，不验证版本或完整性。
for path, label in [
    (groot_repo, "GR00T N1.6 checkout"),
    (b1k_python_launcher, "BEHAVIOR simulator Python launcher"),
    (b1k_python, "b1k2025-sim Python"),
    (omnigibson_root, "BEHAVIOR OmniGibson source"),
    (bddl_root, "BEHAVIOR BDDL source"),
    (behavior_data, "BEHAVIOR assets"),
]:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
# Catch missing per-task TRO states before the expensive simulator startup.
# Fixture runners without a BEHAVIOR adapter are used by launcher tests.
if (groot_repo / 'gr00t/eval/sim/BEHAVIOR/behavior_env.py').is_file():
    for task_id, _ in selected:
        missing = missing_instances(groot_repo, task_id, args.instance_ids if args.instance_ids is not None else range(10))
        if missing:
            raise SystemExit(
                f"Missing Task{task_id} test instance TRO states: {missing}. "
                f"Prepare the official gr00t/eval/sim/BEHAVIOR/test_instances assets first. "
                f"No simulator started; use --list-tasks to check availability."
            )
# 检查能否调用 nvidia-smi，以便读取 GPU 显存占用。
if not shutil.which("nvidia-smi"):
    raise SystemExit("nvidia-smi was not found; refusing to start the simulation")

# 先创建 evaluation 目录。parents=True 可自动创建父目录。
eval_root.mkdir(parents=True, exist_ok=True)
if public_output is None:
    try:
        with queue_log.open("x", encoding="utf-8"):
            pass
    except FileExistsError:
        raise SystemExit(f"Refusing to overwrite existing queue log: {queue_log}")


# 写队列日志并同步打印终端。flush=True 可降低输出滞后。
def log(message):
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + str(message)
    if public_output is None:
        with queue_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print(line, flush=True)


# 先复制当前环境，避免直接修改主控进程环境变量。
common_env = os.environ.copy()
# 清除可能污染 Isaac Sim/B1K 子进程的 Python、动态库、Isaac 和 CUDA 环境变量。
for name in (
    "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH", "CUDA_VISIBLE_DEVICES",
):
    common_env.pop(name, None)
# 设置仿真运行时环境：同意 EULA、CUDA stub 搜索路径、数据集目录、无界面模式及 GPU 编号。
common_env.update({
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs",
    "OMNIGIBSON_DATA_PATH": str(behavior_data),
    "OMNIGIBSON_HEADLESS": "true",
    "OMNIGIBSON_GPU_ID": GPU_INDEX,
    "OMNIGIBSON_APPDATA_PATH": str(appdata_path),
    "GROOT_EVAL_TRACE": "1" if args.profile_cycles else "0",
    "GROOT_EVAL_DISABLE_VIEWER": "1" if args.disable_viewer else "0",
    "GROOT_EVAL_AUDIT_SEMANTICS": "1" if args.audit_semantics else "0",
})
# 独立准备策略服务健康检查使用的环境。
ping_env = common_env.copy()
# 构造一行 Python 代码，借助 GR00T PolicyClient 远程检查 A100 上的推理服务是否联通及模态配置是否可读取。
ping_code = (
    "from gr00t.policy.server_client import PolicyClient; "
    f"c=PolicyClient(host={POLICY_HOST!r},port={POLICY_PORT},timeout_ms=5000); "
    "print('PING',c.ping()); "
    "print('MODALITY_KEYS',sorted(c.get_modality_config().keys()))"
)

# 保存正在运行的子进程句柄，用于等待与处理中断信号。
active_process = None
received_signal = None


# Let the runner close its environments and encoders before workers exit.
def handle_signal(signum, _frame):
    global received_signal
    received_signal = signum
    log(f"QUEUE_SIGNAL signal={signum}; stopping active runner and not starting later tasks")
    if active_process is not None and active_process.poll() is None:
        try:
            active_process.terminate()
        except ProcessLookupError:
            pass


# 安装信号处理器，使主控能结束正在运行的仿真进程组。
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

# 记录/打印当前队列状态；此处及后文复用同名日志函数。
log(
    f"QUEUE_START pid={os.getpid()} tasks={','.join(str(i) for i, _ in selected)}; "
    f"GPU={os.environ.get('GROOT_DEVICE_ID', '4090')}:{GPU_INDEX}; policy={POLICY_HOST}:{POLICY_PORT}; episodes={args.episodes}; "
    f"runner_max_episode_steps={RUNNER_MAX_EPISODE_STEPS}; launcher={os.environ['LAUNCHER_PATH']}"
)

# Only count tasks that reach a successful terminal summary. A precheck stop
# must return failure instead of silently exiting with code zero.
completed_tasks = 0
# 主任务循环：按顺序依次执行，每完成一个任务才启动下一个。任何预检失败、运行失败或结果解析失败都会停止后续任务。
for (task_id, task), outdir in zip(selected, outputs):
    # 如果主控收到中断信号，保留已有部分日志/文件并以 128+信号编号退出。
    if received_signal:
        break
    log(f"TASK_PRECHECK id={task_id} name={task['name']}")
    # 每个任务开始前重新确认策略服务器可用，最多重试 3 次。
    ping_ok = False
    for attempt in range(1, 4):
        try:
            # 短进程健康检查：capture_output=True 捕获 stdout/stderr；timeout=15 表示每次最多等待 15 秒。
            result = subprocess.run(
                [str(b1k_python_launcher), "-c", ping_code],
                cwd=str(groot_repo), env=ping_env, capture_output=True,
                text=True, timeout=15,
            )
            # 同时要求检查脚本退出码为 0 且输出包含 PING True，才认定服务正常。
            if result.returncode == 0 and "PING True" in result.stdout:
                ping_ok = True
                log(f"POLICY_PING_OK task={task_id} attempt={attempt} {result.stdout.strip().replace(chr(10), ' ')}")
                break
            log(
                f"POLICY_PING_FAIL task={task_id} attempt={attempt} rc={result.returncode} "
                f"stdout={result.stdout[-300:]} stderr={result.stderr[-400:]}"
            )
        except Exception as error:
            log(f"POLICY_PING_FAIL task={task_id} attempt={attempt} error={error!r}")
        if attempt < 3:
            # 前两次失败后等待 10 秒再重试；第 3 次失败则停止队列。
            time.sleep(10)
    # 如果主控收到中断信号，保留已有部分日志/文件并以 128+信号编号退出。
    if received_signal:
        break
    # 网络/策略协议预检没通过，直接结束任务队列。
    if not ping_ok:
        log(f"QUEUE_STOP policy protocol ping failed before task {task_id}")
        break

    def gpu_memory():
        raw = subprocess.check_output(
            ["nvidia-smi", "-i", GPU_INDEX, "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        free, used = map(int, raw.split(','))
        # The diagnostic-only n_envs=4 entry retains its previous admission rule.
        required = GPU_MIN_FREE_MIB.get(args.n_envs, free + used - 10240 - 1)
        return free, used, required
    try:
        free_mib, used_mib, required_mib = gpu_memory()
    except Exception as error:
        log(f"QUEUE_STOP could not verify GPU{GPU_INDEX}: {error!r}")
        break
    while free_mib <= required_mib and args.wait_for_gpu and not received_signal:
        log(f"GPU_WAIT GPU{GPU_INDEX} free={free_mib} MiB; required_free>{required_mib} MiB; used={used_mib} MiB; n_envs={args.n_envs}; waiting")
        # Respond promptly to a pause even when the configured poll interval is long.
        deadline = time.monotonic() + args.gpu_wait_interval
        while not received_signal and time.monotonic() < deadline:
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        try:
            free_mib, used_mib, required_mib = gpu_memory()
        except Exception as error:
            log(f"GPU_WAIT_CHECK_FAILED GPU{GPU_INDEX}: {error!r}; retrying")
    if received_signal:
        break
    if free_mib <= required_mib:
        log(f"QUEUE_STOP GPU{GPU_INDEX} free {free_mib} MiB does not exceed required {required_mib} MiB for n_envs={args.n_envs} (used={used_mib} MiB); use --wait-for-gpu to wait without starting a simulator")
        break
    # 再次确认任务输出目录尚不存在，避免覆盖。
    if outdir.exists():
        log(f"QUEUE_STOP output path unexpectedly exists: {outdir}")
        break

    # 为当前任务创建专属目录，用来保存运行日志、结果、CSV 和视频。
    outdir.mkdir(parents=True)
    task_env = common_env.copy()
    task_env["GROOT_EVAL_OUTPUT_DIR"] = str(outdir)
    task_env["GROOT_EVAL_VIDEO_DIR"] = str(outdir / "videos")
    task_env["GROOT_EVAL_PROCESS_STARTED_AT"] = str(time.perf_counter())
    task_env["GROOT_EVAL_SEED"] = str(args.seed)
    task_env["GROOT_EVAL_INSTANCE_IDS"] = ",".join(
        str(instance_id) for instance_id in (args.instance_ids if args.instance_ids is not None else range(10))
    )
    env_name = f"sim_behavior_r1_pro/{task['name']}"
    # 构建 run_meta.json：记录本轮评估的任务、环境、策略服务、硬件和参数，以便后续实验复现。
    metadata = {
        "task_id": task_id,
        "gpu_index": int(GPU_INDEX),
        "public_output_dir": str(public_output) if public_output else None,
        "task_name": task["name"],
        "preset": args.preset,
        "env_name": env_name,
        "baseline": "NVIDIA Isaac-GR00T N1.6 BEHAVIOR1k checkpoint",
        "runner_source": str(groot_repo / "gr00t/eval/rollout_policy.py"),
        "runtime": str(b1k_python),
        # 这里的版本号是硬编码文本，不是运行时自动检测得到的资产版本。
        "assets_version": "3.7.2rc1",
        "policy_server": f"{POLICY_HOST}:{POLICY_PORT}",
        "sim_host_gpu": f"{os.environ.get('GROOT_DEVICE_ID', '4090')} GPU{GPU_INDEX}",
        "omnigibson_appdata_path": common_env["OMNIGIBSON_APPDATA_PATH"],
        "profile_cycles": args.profile_cycles,
        "profile_detail": common_env.get("GROOT_EVAL_DETAIL_PROFILE") == "1",
        "profile_simulator": common_env.get("GROOT_EVAL_SIM_PROFILE") == "1",
        "camera_graph_counters": common_env.get("GROOT_EVAL_CAMERA_COUNTERS") == "1",
        "episodes": args.episodes,
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "limit_env_episodes": args.limit_env_episodes,
        "episode_layout": episode_layout,
        "instance_ids": args.instance_ids if args.instance_ids is not None else list(range(10)),
        "n_envs": args.n_envs,
        "n_action_steps": args.action_steps,
        "record_video": args.record_video,
        "viewer_enabled": not args.disable_viewer,
        "audit_semantics": args.audit_semantics,
        "diagnostic_only": args.diagnostic_only,
        "video_steps_per_render": args.video_steps_per_render,
        "runner_max_episode_steps": RUNNER_MAX_EPISODE_STEPS,
        # 这里只是记录作者预期的超时语义；本脚本没有读取环境内部 timeout，也不能据此证明 runner 实际按该规则终止。
        "horizon_semantics": (
            "debug runner cap only; native BEHAVIOR timeout unchanged; excluded from official success rate"
            if args.preset == "fast-debug" else
            "native BEHAVIOR task timeout (2x human-demo length); no practical runner cap"
        ),
        "task_timeout_hint_steps": task["timeout_hint"],
        "launcher": os.environ["LAUNCHER_PATH"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(outdir / "run_meta.json", metadata)
    run_log_path = queue_log if public_output else outdir / "run.log"
    # 为子进程创建行缓冲日志文件；接下来其 stdout/stderr 都重定向进这里。
    run_log = run_log_path.open("a" if public_output else "w", encoding="utf-8", buffering=1)
    # 根据任务名称产生实际 rollout 命令行参数。
    command_args = runner_args(task["name"])
    log(
        f"TASK_START id={task_id} env={env_name} gpu{GPU_INDEX}_mem_before={used_mib} MiB "
        f"max_episode_steps={RUNNER_MAX_EPISODE_STEPS} n_episodes={args.episodes} "
        f"n_envs={args.n_envs} n_action_steps={args.action_steps} record_video={args.record_video}"
    )
    # 正式启动仿真子进程：主控只负责等待和日志管理，Isaac Sim 与 GR00T rollout 在子进程内运行。
    runner_command = [str(b1k_python_launcher), "-u", "-c", child_code, *command_args]
    active_process = subprocess.Popen(
        # 等价于：b1k_python.sh -u -c "<child_code>" --max_episode_steps ...；包装脚本需能转发这些 Python 参数。
        runner_command,
        cwd=str(groot_repo), env=task_env, stdout=run_log,
        # 把标准错误合并写入 run.log；独立会话使停止队列时可以批量终止子进程组。
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    (outdir / "runner.pid").write_text(str(active_process.pid) + "\n", encoding="utf-8")
    log(f"RUNNER_STARTED task={task_id} pid={active_process.pid} log={run_log_path}")
    monitor_process = None
    if args.profile_cycles or args.monitor_resources:
        monitor_process = subprocess.Popen(
            [
                "/usr/bin/python3", str(project_root / "scripts/groot_eval_monitor.py"),
                "--pid", str(active_process.pid), "--gpu", GPU_INDEX,
                "--output", str(outdir / "system_metrics.csv"),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log(f"RESOURCE_MONITOR_STARTED pid={monitor_process.pid} output={outdir / 'system_metrics.csv'}")
    # 阻塞直到当前任务退出；这正是队列串行执行的关键。
    diagnostic_completed = False
    if args.diagnostic_only:
        while active_process.poll() is None and not received_signal:
            if (outdir / "diagnostic_profile.json").exists():
                profile = json.loads((outdir / "diagnostic_profile.json").read_text())
                if profile["cycles"] == args.profile_cycles:
                    diagnostic_completed = True
                    active_process.terminate()
                    break
            try:
                active_process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
    return_code = active_process.wait()
    if received_signal or diagnostic_completed:
        # A stop during construction can predate the runner's handler. Reap
        # any remaining workers only after the main process has exited.
        try:
            os.killpg(active_process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    active_process = None
    if monitor_process is not None:
        monitor_process.wait(timeout=10)
    run_log.close()
    summary = finish_run(outdir, return_code, interrupted=bool(received_signal))
    raw_videos = sorted((outdir / "videos").glob("*.mp4"))
    summary["saved_video_count"] = len(raw_videos)
    write_json(outdir / "results.json", summary)
    if diagnostic_completed and not received_signal:
        summary.update(status="diagnostic_complete", diagnostic_only=True)
        write_json(outdir / "results.json", summary)
        log(f"DIAGNOSTIC_COMPLETE task={task_id} window={args.profile_cycles} output={outdir}; unfinished episode excluded")
        completed_tasks += 1
        continue
    # 如果主控收到中断信号，保留已有部分日志/文件并以 128+信号编号退出。
    if received_signal:
        log(f"QUEUE_INTERRUPTED signal={received_signal} task={task_id}; partial run retained at {outdir}")
        raise SystemExit(128 + received_signal)

    # 若子进程非正常退出，标记当前任务失败并停止剩余任务。
    if return_code != 0:
        log(f"TASK_FAILED id={task_id} rc={return_code}; stopping queue; inspect {run_log_path}")
        break
    if summary["status"] != "complete":
        log(f"TASK_FAILED id={task_id} expected {args.episodes} valid episodes, recorded {summary['completed_episodes']}; stopping queue")
        break
    successes = summary["successes"]
    progress = summary["task_progress"]

    if raw_videos:
        log(f"VIDEO_NOTE task={task_id} videos={len(raw_videos)}; original filenames retained")

    records = [json.loads(line) for line in (outdir / "episodes.jsonl").read_text().splitlines()]
    write_manifest(outdir, records)
    summary["saved_video_count"] = len(raw_videos)
    write_json(outdir / "results.json", summary)
    outcome = (
        "FAST_DEBUG_COMPLETE official_success_rate=null "
        if args.preset == "fast-debug" else
        f"TASK_COMPLETE success={sum(successes)}/{args.episodes} rate={sum(successes)/len(successes):.4f} "
    )
    log(
        f"{outcome}id={task_id} "
        f"mean_q_score={summary['mean_q_score']} "
        f"q_score_episodes={summary['q_score_episodes']}/{summary['completed_episodes']} "
        f"mean_task_progress={summary['mean_task_progress']} "
        f"videos={len(raw_videos)} output={outdir}"
    )
    completed_tasks += 1

# 主循环结束后写入队列终止标记。注意：即使因为失败而 break，也会写 QUEUE_FINISHED；它不代表所有任务都成功。
log("QUEUE_FINISHED")
raise SystemExit(128 + received_signal if received_signal else (0 if completed_tasks == len(selected) else 1))
PY
