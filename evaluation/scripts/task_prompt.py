"""从 B1K BDDL 官方定义生成任务说明，供视觉模型判断任务进度。

只读官方 bddl 文件，不依赖模型知识、不手写任务内容。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BDDL_ROOT = Path("/nvme3/shared/b1k2025/BEHAVIOR-1K/bddl/bddl/activity_definitions")
CACHE_PATH = Path(__file__).with_name("task_prompts.json")

PREDICATES = {
    "real": "{a} 已经真实存在",
    "contains": "{a} 里面装着 {b}",
    "ontop": "{a} 在 {b} 上表面",
    "inside": "{a} 在 {b} 内部",
    "under": "{a} 在 {b} 下面",
    "nextto": "{a} 紧挨着 {b}",
    "inroom": "{a} 在 {b} 房间里",
    "onfloor": "{a} 在地板上",
    "cooked": "{a} 已经做熟",
    "toggled_on": "{a} 处于开启/工作状态",
    "open": "{a} 处于打开状态",
    "closed": "{a} 处于关闭状态",
    "filled": "{a} 里装满了东西",
    "covered": "{a} 被覆盖/沾上 {b}",
    "sliced": "{a} 被切开",
    "diced": "{a} 被切丁",
    "folded": "{a} 被折叠",
    "soaked": "{a} 被浸泡过",
    "dusty": "{a} 有灰尘需要清理",
    "stained": "{a} 有污渍需要清理",
    "broken": "{a} 是损坏的",
    "burned": "{a} 被烧焦",
    "frozen": "{a} 被冷冻",
    "saturated": "{a} 被浸透",
    "empty": "{a} 是空的",
    "toggled_off": "{a} 处于关闭状态",
    "assembled": "{a} 被组装好",
    "attached": "{a} 被固定到 {b}",
}

SKIP_OBJECTS = {"agent", "floor", "room", "water", "surface", "structure"}


def _balanced(text: str, anchor: str) -> str:
    """取出 anchor 之后第一个括号配平的表达式。"""
    start = text.find(anchor)
    if start < 0:
        return ""
    start = text.find("(", start)
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return ""


def _clean(token: str) -> str:
    value = token.lstrip("?").strip()
    value = re.sub(r"_\d+$", "", value)
    value = re.sub(r"\.[a-z]{1,3}\.\d+$", "", value)
    return value.replace("__", " ").replace("_", " ")


def parse_bddl(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    objects_block = _balanced(text, "(:objects")
    goal_block = _balanced(text, "(:goal")

    objects: list[str] = []
    for line in objects_block.splitlines():
        match = re.match(r"\s*([\w.\-]+)\s+-\s+([\w.\-]+)", line)
        if not match:
            continue
        name = _clean(match.group(1))
        kind = _clean(match.group(2))
        label = name if name == kind else f"{name}（{kind}）"
        if any(word in label for word in SKIP_OBJECTS):
            continue
        if label not in objects:
            objects.append(label)

    goals: list[str] = []
    for match in re.finditer(r"\(([a-z_]+)((?:\s+\?[\w.\-]+)+)\)", goal_block):
        predicate = match.group(1)
        args = [_clean(token) for token in match.group(2).split()]
        template = PREDICATES.get(predicate)
        if template:
            filled = template.format(a=args[0], b=args[1] if len(args) > 1 else "")
            goals.append(filled)
        else:
            goals.append(f"{predicate}({', '.join(args)})")

    return {"objects": objects, "goals": goals}


def build_cache() -> dict:
    cache: dict[str, dict] = {}
    if not BDDL_ROOT.is_dir():
        return cache
    for folder in sorted(BDDL_ROOT.iterdir()):
        problem = folder / "problem0.bddl"
        if not problem.is_file():
            continue
        try:
            parsed = parse_bddl(problem)
        except OSError:
            continue
        if parsed["goals"]:
            cache[folder.name] = parsed
    return cache


def _load() -> dict:
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    cache = build_cache()
    if cache:
        try:
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass
    return cache


def describe(task_name: str) -> dict:
    """返回任务的物体与成功条件；查不到就返回空。"""
    if not task_name:
        return {}
    return _load().get(task_name, {})


def instruction_for(task_name: str) -> str:
    """把任务名扩成一段有实质内容的指令，喂给视觉模型。"""
    info = describe(task_name)
    human = (task_name or "").replace("_", " ").strip()
    if not info:
        return f"任务是 {human}。"
    goals = "；".join(info.get("goals", [])[:6])
    objects = "、".join(info.get("objects", [])[:8])
    parts = [f"任务是 {human}。"]
    if goals:
        parts.append(f"任务成功的条件是：{goals}。")
    if objects:
        parts.append(f"场景里与此任务相关的物体有：{objects}。")
    parts.append("请只围绕上述目标和物体判断机械臂的进展，不要描述与任务无关的物体或场景。")
    return "".join(parts)
