"""A6000 视觉诊断服务：Qwen3-VL-8B 看评估视频关键帧，输出结构化诊断。

启动： source venv/bin/activate && python service.py --port 8901
接口：
  GET  /health
  POST /analyze  {"task":..., "episode":..., "frames":[b64...], "instruction":"...", ...}
"""
import argparse, base64, io, json, re, threading, time
from typing import Any, Dict, List, Optional

import torch
from vlm_stage_protocol import prompt as stage_prompt
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_DIR = "/home/1024201092WYH/vlm_service/models/Qwen3-VL-8B-Instruct"

BEHAVIOR = ["chaotic_motion", "repetitive_stuck", "goal_directed_incomplete",
            "near_complete", "completed", "unclear"]
FAILURE = ["not_started", "approach", "grasp", "transport", "place", "other", "none"]

PROMPT = """你是机器人操作任务的表现评估员。下面是同一个 episode 按时间顺序均匀采样的一组画面（第 1 张是开始，最后 1 张是结束）。
任务指令：{instruction}
场景提示：{notes}

分段观察记录（由逐段细看得到，可靠性高，可作为主要依据）：
{history}

画面布局：每张图是横向拼接的三路相机画面——左边是左手腕相机（贴近左夹爪），中间是右手腕相机（贴近右夹爪），右边是头部相机（能看到机械臂整体和环境）。腕部相机看抓取、接触和物体细节，头部相机看整体位置与物体关系，三者结合判断。
画面底部有一条文字栏（任务名、success=0/1、episode 编号等），那是录像自带的调试标注，不是场景内容，可能和真实进展不一致。完全不要参考它，也不要把它当作判断依据，只依据画面上机械臂和物体的实际状态判断。

请只依据画面判断，输出严格 JSON（不要 markdown 代码块、不要解释），字段：
{{
  "behavior_class": "从 {behaviors} 中选一个",
  "visual_progress": 0-100 的整数，画面上看任务完成了多少,
  "trend_correct": true/false，动作方向是否在朝任务目标推进（即使最终没完成也算 true）,
  "stuck": true/false，是否明显卡住（原地抖动、重复同一动作、撞住不动、乱动）,
  "failure_stage": "从 {failures} 中选一个，即最早明显失败的环节",
  "target_touched": true/false，目标物体是否被机械臂接触/移动过,
  "phases": [{{"frames": "1-3", "action": "这 3 帧里机械臂在做什么"}}],
  "evidence": "一句话说明你的判断依据（看到什么）",
  "confidence": 0-1 的小数
}}
phases 用 2-4 段覆盖整段过程。"""

SEGMENT_PROMPT = """
你在分析机器人操作录像里的一小段连续画面（{label}，共 {n} 帧，按时间顺序）。
任务指令：{instruction}
场景提示：{notes}

画面布局：每张图是横向拼接的三路相机画面——左边是左手腕相机（贴近左夹爪），中间是右手腕相机（贴近右夹爪），右边是头部相机（能看到机械臂整体和环境）。腕部相机看抓取、接触和物体细节，头部相机看整体位置与物体关系，三者结合判断。
画面底部有一条文字栏（任务名、success=0/1、episode 编号等），那是录像自带的调试标注，不是场景内容，可能和真实进展不一致。完全不要参考它，只依据画面上机械臂和物体的实际状态判断。

只描述这一小段里实际发生的事，不要推测整个任务成败，不要输出 JSON，用中文直接写三行：
动作：按顺序写这一小段里机械臂和夹爪做了什么（接近哪个物体、抓没抓住、往哪移动、有没有空抓、碰撞、掉物）
接触：碰到或抓住的物体名，没有就写 none
判断：是否与任务目标相关；是否卡住或重复同一动作；这一小段推进了多少进度（0-100 整数）
"""



def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i:j + 1]
    return json.loads(text)


class Frame(BaseModel):
    task: str = ""
    episode: str = ""
    group: str = ""
    instruction: str = "unknown"
    notes: str = ""
    frames: List[str] = []
    history: str = ""
    mode: str = "final"
    label: str = ""


app = FastAPI()
_lock = threading.Lock()
_state: Dict[str, Any] = {}


@app.on_event("startup")
def load():
    t0 = time.time()
    _state["proc"] = AutoProcessor.from_pretrained(MODEL_DIR)
    _state["model"] = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    _state["load_s"] = round(time.time() - t0, 1)
    print(f"[ready] loaded in {_state['load_s']}s", flush=True)


@app.get("/health")
def health():
    return {"ok": "model" in _state, "model": MODEL_DIR,
            "load_s": _state.get("load_s"),
            "gpu_mem_gb": round(torch.cuda.memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None}


@app.post("/analyze")
def analyze(req: Frame):
    t0 = time.time()
    imgs = []
    for b in req.frames:
        raw = base64.b64decode(b.split(",")[-1])
        imgs.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    if not imgs:
        return {"error": "no frames"}
    if req.mode == "task40_stage":
        text = stage_prompt(req.label or "Task40候选窗口", req.notes or "", len(imgs))
        max_new = 1200
    elif req.mode == "segment":
        text = SEGMENT_PROMPT.format(instruction=req.instruction or "unknown",
                                     notes=req.notes or "",
                                     label=req.label or "片段",
                                     n=len(imgs))
        max_new = 420
    else:
        text = PROMPT.format(instruction=req.instruction or "unknown",
                             notes=req.notes or "",
                             history=req.history or "（无分段记录，请直接依据画面判断）",
                             behaviors=", ".join(BEHAVIOR),
                             failures=", ".join(FAILURE))
        max_new = 900
    content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    inputs = _state["proc"].apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to("cuda")
    with _lock, torch.inference_mode():
        out = _state["model"].generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    raw = _state["proc"].batch_decode(gen, skip_special_tokens=True)[0]
    if req.mode == "segment":
        return {"_ok": bool(raw.strip()), "raw": raw.strip()[:1200],
                "task": req.task, "episode": req.episode, "label": req.label,
                "n_frames": len(imgs), "latency_s": round(time.time() - t0, 1)}
    try:
        data = extract_json(raw)
        data["_ok"] = True
    except Exception as e:
        data = {"_ok": False, "raw": raw[:1500], "error": str(e)}
    data.update({"task": req.task, "episode": req.episode, "group": req.group,
                 "n_frames": len(imgs), "latency_s": round(time.time() - t0, 1)})
    # 规范化 + 派生执行质量分
    if data.get("_ok") and req.mode not in ("segment", "task40_stage"):
        bc = data.get("behavior_class", "unclear")
        data["behavior_class"] = bc if bc in BEHAVIOR else "unclear"
        fs = data.get("failure_stage", "other")
        data["failure_stage"] = fs if fs in FAILURE else "other"
        vp = float(data.get("visual_progress") or 0)
        data["visual_progress"] = max(0, min(100, int(vp)))
        quality = data["visual_progress"]
        if data.get("trend_correct"):
            quality += 15
        if data.get("stuck"):
            quality -= 25
        if data["behavior_class"] == "chaotic_motion":
            quality -= 20
        data["execution_quality"] = max(0, min(100, int(quality)))
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="info")
