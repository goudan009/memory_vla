#!/usr/bin/env python3
"""Task40 根因诊断 v2：聚焦三类根本失败原因，原生视频输入，不做事件序列标注。

根因分类（来自人工核对）：
  decision_first_step_bag  — 决策问题：第一步先去拿袋子，而不是先开微波炉门
  decision_skip_to_start   — 决策问题：开门失败几次后直接跳到启动微波炉，跳过所有中间步骤
  execution_close_door     — 执行问题：前面决策全对（开门→拿袋→放入），但关门阶段失败
  success                  — 成功完成

诊断逻辑（给模型的三个问题）：
  Q1  机器人从厨房出发后，第一个靠近并尝试操作的目标是什么？
       → 袋子/爆米花 → decision_first_step_bag
       → 微波炉门把手 / 控制面板 → 继续 Q2
  Q2  开门失败（或未尝试）之后，机器人的下一个动作是什么？
       → 直接启动微波炉（面板操作，此时没有袋子在炉内）→ decision_skip_to_start
       → 去拿袋子 / 重试开门 → 继续 Q3
  Q3  机器人是否成功打开了门、成功把袋子放入炉内？
       → 是，但关门/启动阶段失败 → execution_close_door
       → 否（其他情况）→ decision_first_step_bag 或 decision_skip_to_start（回 Q1/Q2）

Usage（在 A6000 上运行）：
  python task40_root_cause_v2.py \\
      --video-root /path/to/full_episodes \\
      --output /path/to/task40_root_cause_v2.jsonl \\
      [--model-path /home/1024201092WYH/vlm_service/models/Qwen3-VL-8B-Instruct] \\
      [--fps 2.0] [--max-new-tokens 600] [--resume]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT_CAUSES = ["decision_first_step_bag", "decision_skip_to_start",
               "execution_close_door", "success"]

TASK_DESCRIPTION = (
    "Task40 make_microwave_popcorn. "
    "Correct step order: (1) go to kitchen, (2) open microwave door, "
    "(3) go to bag, (4) grasp bag, (5) carry bag to microwave, "
    "(6) place bag inside microwave, (7) close door, (8) start microwave. "
    "Steps 2-3 must come in this order: door must be open before the bag can be placed inside."
)

PROMPT = (
    "You are diagnosing a robot's failure at Task40 (make_microwave_popcorn). "
    "ASSUME FAILURE unless you can prove success with explicit visual evidence.\n"
    "Correct order: (1) open microwave door, (2) get popcorn bag, "
    "(3) place bag INSIDE microwave cavity, (4) close door, (5) press start button.\n"
    "Video: horizontal composite — left-wrist | right-wrist | head camera (rightmost, full scene).\n\n"
    "Answer each step carefully with what you actually observe in the video.\n\n"
    "--- STEP 1: VERY FIRST ACTION (look only at the first 20% of the video) ---\n"
    "In the OPENING SECONDS of the video, before the robot does anything else, "
    "which object does the robot arm move toward FIRST?\n"
    "  bag_area: the robot reaches toward the popcorn bag or the basket/shelf where the bag sits\n"
    "  microwave_door: the robot reaches toward the microwave door handle or door panel\n"
    "  microwave_panel: the robot reaches toward the control keypad on the microwave front face\n"
    "Focus on the head camera (rightmost panel) for the robot initial direction of movement.\n\n"
    "--- STEP 2: BAG GRASP ---\n"
    "At any point in the video, do you see the robot gripper close around or hold the popcorn bag? "
    "Look at both wrist cameras for a close-up of the gripper gripping the bag.\n\n"
    "--- STEP 3: BAG PLACEMENT ---\n"
    "If the robot grasped the bag: do you then see the bag carried to the microwave opening and "
    "deposited INSIDE the cavity while the microwave door is open? "
    "The bag must visibly cross into the microwave interior. "
    "If the robot never grasped the bag, answer false.\n\n"
    "--- STEP 4: PANEL PRESS ---\n"
    "Does the robot press the microwave CONTROL PANEL (the flat face with number/start buttons)? "
    "Look for the robot arm pressing flat buttons on the front face of the microwave, "
    "distinct from pulling the door handle.\n\n"
    "--- STEP 5: DOOR CLOSE ---\n"
    "If the bag was placed inside: did the robot push the microwave door closed afterward? "
    "Answer null if bag was never placed.\n\n"
    "--- CLASSIFICATION (apply top-down) ---\n"
    "  A. step1_first_object = bag_area  ->  decision_first_step_bag\n"
    "  B. step1_first_object = microwave_door  AND  bag_placed = false  AND  panel_pressed = true  "
    "->  decision_skip_to_start\n"
    "  C. step1_first_object = microwave_door  AND  bag_grasped = false  AND  panel_pressed = false  "
    "->  decision_skip_to_start  "
    "(robot attempted door, never grasped bag, task could not complete)\n"
    "  D. step1_first_object = microwave_door  AND  bag_grasped = true  AND  bag_placed = false  "
    "->  decision_first_step_bag  (robot grasped bag but could not place it)\n"
    "  E. bag_placed = true  AND  door_closed = false  ->  execution_close_door\n"
    "  F. bag_placed = true  AND  door_closed = true  AND  panel_pressed = true  ->  success\n"
    "  G. otherwise  ->  use closest category, explain in evidence\n\n"
    "Return ONLY a JSON object (no markdown):\n"
    "{\n"
    "  \"step1_first_object\": \"bag_area | microwave_door | microwave_panel | other | unclear\",\n"
    "  \"step1_evidence\": \"one sentence: what you saw in the opening frames\",\n"
    "  \"step2_bag_grasped\": true/false,\n"
    "  \"step2_evidence\": \"one sentence: did the gripper hold the bag?\",\n"
    "  \"step3_bag_placed\": true/false,\n"
    "  \"step3_evidence\": \"one sentence: did the bag enter the microwave cavity?\",\n"
    "  \"step4_panel_pressed\": true/false,\n"
    "  \"step4_evidence\": \"one sentence: did the robot press the control panel?\",\n"
    "  \"step5_door_closed\": true/false/null,\n"
    "  \"step5_evidence\": \"one sentence or not_applicable\",\n"
    "  \"root_cause\": \"decision_first_step_bag | decision_skip_to_start | execution_close_door | success\",\n"
    "  \"root_cause_evidence\": \"one sentence: strongest observation supporting this classification\",\n"
    "  \"confidence_0_to_1\": 0.0\n"
    "}"
)


def parse_json_response(text: str) -> dict | None:
    """Find and parse the first JSON object in the response."""
    import json as _json
    decoder = _json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            if isinstance(value, dict) and "root_cause" in value:
                return value
        except _json.JSONDecodeError:
            continue
    return None


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True,
                        help="Directory containing sequence_C01.mp4 ... sequence_C18.mp4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/1024201092WYH/vlm_service/models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--fps", type=float, default=2.0,
                        help="FPS to pass to processor (matches recording FPS)")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quantize-4bit", action="store_true",
                        help="Load model in 4-bit (nf4) via bitsandbytes — required for 32B on 48GB GPU")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable Qwen3-VL thinking mode (longer reasoning, more tokens)")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

    video_paths = sorted(args.video_root.glob("sequence_C*.mp4"))
    if not video_paths:
        raise SystemExit(f"No sequence_C*.mp4 found in {args.video_root}")
    print(f"Found {len(video_paths)} videos", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and args.output.is_file():
        for row in read_jsonl(args.output):
            if not row.get("error"):
                completed.add(row["case_id"])
        print(f"Resuming: {len(completed)} already done", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {args.model_path.name} (4bit={args.quantize_4bit}, thinking={args.thinking})...", flush=True)
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)

    load_kwargs: dict = dict(device_map="auto", local_files_only=True, low_cpu_mem_usage=True)
    if args.quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = dtype

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_path), **load_kwargs
    ).eval()
    print(f"Model loaded. dtype={dtype}", flush=True)

    mode = "a" if args.resume and args.output.is_file() else "w"
    with args.output.open(mode, encoding="utf-8") as stream:
        for idx, video in enumerate(video_paths, 1):
            case_id = video.stem.replace("sequence_", "")  # C01 ... C18
            if case_id in completed:
                print(f"{idx}/{len(video_paths)} {case_id} skipped (cached)", flush=True)
                continue

            started = time.time()
            base = {"case_id": case_id, "video_file": video.name,
                    "model_name": args.model_path.name, "model_path": str(args.model_path),
                    "fps": args.fps, "prompt_version": "task40_root_cause_v4"}
            try:
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": str(video)},
                    {"type": "text", "text": PROMPT},
                ]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", fps=args.fps,
                ).to(model.device)
                sampled_frames = int(inputs["video_grid_thw"][0, 0].item()) * 2
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                            do_sample=False)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(generated, skip_special_tokens=True)[0]
                parsed = parse_json_response(text)
                row = {
                    **base,
                    "raw_response": text,
                    "prediction": parsed,
                    "sampled_frame_count": sampled_frames,
                    "prompt_token_count": int(inputs["input_ids"].shape[-1]),
                    "elapsed_seconds": round(time.time() - started, 2),
                    "error": None if parsed is not None else "response_not_parseable",
                }
                del inputs, output_ids, generated
                torch.cuda.empty_cache()
            except Exception as exc:
                row = {**base, "raw_response": None, "prediction": None,
                       "sampled_frame_count": None, "prompt_token_count": None,
                       "elapsed_seconds": round(time.time() - started, 2),
                       "error": f"{type(exc).__name__}: {exc}"}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rc = (row.get("prediction") or {}).get("root_cause", "?")
            print(f"{idx}/{len(video_paths)} {case_id} root_cause={rc} "
                  f"frames={row.get('sampled_frame_count')} "
                  f"elapsed={row['elapsed_seconds']}s error={row['error']}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
