import argparse
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


TASKS = {
    "train": ("training", "训练", "a100", ("a100", "a6000")),
    "infer": ("inference", "推理", "a100", ("a100", "a6000")),
    "eval": ("launch", "评估", "local", ("local", "a6000")),
}
DEVICE_LABELS = {"local": "4090", "remote": "A100", "a6000": "RTX 6000 Ada"}


def parser_for_cli():
    parser = argparse.ArgumentParser(prog="memory-workbench", description="MemoryVLA 工作台")
    parser.add_argument("--url", default=os.getenv("MEMORY_WORKBENCH_URL", "http://127.0.0.1:8765"),
                        help="工作台地址，也可通过 MEMORY_WORKBENCH_URL 设置")
    parser.add_argument("--timeout", type=int, default=180, help="请求超时秒数")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="查看设备和 GPU 状态")
    status.add_argument("--json", action="store_true", help="输出原始 JSON")
    commands.add_parser("results", help="查看评估结果")
    resource = commands.add_parser("resource", help="统一资源管理：先查询，再预检，最后操作")
    actions = resource.add_subparsers(dest="action", required=True)
    for action in ("hosts","disks","browse","stat","size","register","mkdir","trash","restore"):
        ap = actions.add_parser(action)
        if action != "hosts":
            ap.add_argument("--host", required=True)
        if action not in ("hosts","disks"):
            ap.add_argument("--path", required=True)
        if action == "register":
            ap.add_argument("--kind", choices=("dataset","weights"), required=True)
            ap.add_argument("--name")
        if action == "mkdir":
            ap.add_argument("--name", required=True)
        if action == "restore":
            ap.add_argument("--token", required=True)
        if action == "trash":
            ap.add_argument("--confirm", action="store_true", required=True)
    for action in ("download-options","download-status","download-preview","download-start"):
        ap = actions.add_parser(action)
        if action != "download-options":
            ap.add_argument("recipe_id")
    serve = commands.add_parser("serve", help="从项目启动工作台服务")
    serve.add_argument("--repo", type=Path, default=Path.cwd(), help="MemoryVLA 仓库根目录")
    serve.add_argument("--host", default="127.0.0.1", help="监听地址")
    serve.add_argument("--port", type=int, default=8765, help="监听端口")
    for command, (_, title, _, devices) in TASKS.items():
        task = commands.add_parser(command, help=title)
        actions = task.add_subparsers(dest="action", required=True)
        for action, help_text in (
            ("options", "查看设备支持的配置"),
            ("jobs", "查看已有作业及状态"),
            ("preview", "检查 JSON 配置"),
            ("start", "检查配置后提交作业"),
        ):
            action_parser = actions.add_parser(action, help=help_text)
            if command != "eval" or action != "jobs":
                action_parser.add_argument("--device", choices=devices, help="设备")
            if action in ("preview", "start"):
                action_parser.add_argument("-f", "--file", required=True, help="JSON 文件，使用 - 从 stdin 读取")
    return parser


def request_json(base_url, route, timeout, payload=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(base_url + route, data=body, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        with error:
            try:
                response = json.load(error)
            except (ValueError, UnicodeError):
                response = None
        message = response.get("error") if isinstance(response, dict) else None
        raise ValueError(f"HTTP {error.code} {route} — {message or error.reason}") from error


def print_status(snapshot):
    for device, state in snapshot.items():
        label = DEVICE_LABELS.get(device, device)
        status = {True: "在线", False: "不可达"}.get(state.get("ok"), "未知")
        checked_at = state.get("checked_at") or "—"
        print(f"{label} · {status} · 采样 {checked_at}")
        if state.get("ok") is not True or not state.get("gpus"):
            print(f"  {state.get('gpus_error') or 'GPU 状态未知'}")
            continue
        for gpu in state["gpus"]:
            usage = gpu.get("utilization")
            utilization = f"{usage}%" if usage is not None else "未知"
            print(f"  GPU {gpu['index']}  {gpu['used_mb']}/{gpu['total_mb']} MiB  利用率 {utilization}")


def task_request(args, base_url):
    prefix, _, default_device, devices = TASKS[args.command]
    path = f"/api/{prefix}"
    if args.action in ("options", "jobs"):
        device = getattr(args, "device", default_device) or default_device
        query = "" if args.command == "eval" and args.action == "jobs" else "?" + urlencode({"device": device})
        return request_json(base_url, f"{path}/{args.action}{query}", args.timeout)

    payload = json.load(sys.stdin) if args.file == "-" else json.loads(Path(args.file).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置必须是 JSON 对象")
    device = args.device or payload.get("device", default_device)
    if device not in devices:
        raise ValueError(f"设备必须是 {' 或 '.join(devices)}")
    payload["device"] = device
    if args.command == "train":
        payload.setdefault("initiator","agent")
        if payload["initiator"] == "agent" and not str(payload.get("notes") or "").strip():
            raise ValueError("Agent training requires concise notes")
    preview = request_json(base_url, f"{path}/preview", args.timeout, payload)
    if args.action == "preview":
        return preview
    try:
        return request_json(base_url, f"{path}/start", args.timeout, payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"提交响应中断，请先查询 {args.command} jobs 核对是否已创建作业。{error}") from error


def resource_request(args, base_url):
    if args.action.startswith("download-"):
        action = args.action.removeprefix("download-")
        path = "/api/resources/downloads"
        payload = {"action":action}
        if action != "options":
            payload["recipe_id"] = args.recipe_id
        if action in ("options","status"):
            return request_json(base_url,path+"?"+urlencode(payload),args.timeout)
        if action == "start":
            request_json(base_url,path,args.timeout,{**payload,"action":"preview"})
        return request_json(base_url,path,args.timeout,payload)
    payload = {"action":args.action}
    for key in ("host","path","kind","name","token"):
        value = getattr(args,key,None)
        if value is not None:
            payload[key]=value
    path = "/api/registry/storage"
    if args.action in ("hosts","disks","browse","stat","size"):
        return request_json(base_url,path+"?"+urlencode(payload),args.timeout)
    if args.action == "trash":
        request_json(base_url,path+"?"+urlencode({**payload,"action":"stat"}),args.timeout)
    return request_json(base_url,path,args.timeout,payload)


def main():
    parser = parser_for_cli()
    args = parser.parse_args()
    try:
        if args.command == "serve":
            script = args.repo.expanduser().resolve() / "evaluation" / "scripts" / "view_results.py"
            if not script.is_file():
                raise ValueError("找不到工作台入口，请在仓库根目录运行或用 --repo 指定目录")
            os.execv(sys.executable, [sys.executable, str(script), "--host", args.host,
                                     "--port", str(args.port), "--no-browser"])

        url = urlsplit(args.url)
        if url.scheme not in ("http", "https") or not url.hostname or url.path.rstrip("/") or url.query or url.fragment:
            raise ValueError("--url 应为工作台服务地址，例如 http://127.0.0.1:8765")
        if args.timeout <= 0:
            raise ValueError("--timeout 必须大于 0")
        base_url = args.url.rstrip("/")
        if args.command == "status":
            result = request_json(base_url, "/api/services", args.timeout)
            if not args.json:
                print_status(result)
                return 0
        elif args.command == "resource":
            result = resource_request(args, base_url)
        elif args.command == "results":
            result = request_json(base_url, "/api/results", args.timeout)
        else:
            result = task_request(args, base_url)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, URLError) as error:
        print(f"memory-workbench — {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
