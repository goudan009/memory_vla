#!/usr/bin/env python3
"""Prepare blind full-episode VLM inputs, run inference, and score event sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT / "outputs/overnight_memory_task40_20261114/verification/task40_vlm_blind_review"
DEFAULT_ANNOTATIONS = ROOT / "outputs/task40_human_sequence_annotations (1).json"
TASK_DESCRIPTION = (
    "Task40 make_microwave_popcorn. Intended task: go to the kitchen, open the microwave, "
    "get the popcorn bag, put it inside, close the door, and start the microwave. "
    "The video may show a wrong order or failed attempts; report what actually happened, not the intended plan."
)
FAILURE_CLASSES = [
    "high_level_decision", "low_level_execution", "perception_state_judgment",
    "insufficient_evidence", "none",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def compact_vector(value, dimensions: int):
    if not isinstance(value, list) or not value:
        return None
    vectors = value if isinstance(value[0], list) else [value]
    vectors = [v for v in vectors if isinstance(v, list) and len(v) >= dimensions]
    if not vectors:
        return None
    try:
        return [round(sum(float(v[i]) for v in vectors) / len(vectors), 4)
                for i in range(dimensions)]
    except (TypeError, ValueError):
        return None


def compact_action(row: dict) -> dict:
    action = row.get("action", {})
    output = {"action_start": row.get("action_start"),
              "action_end": row.get("action_end")}
    for key, dims in (("base", 3), ("left_arm", 3), ("right_arm", 3),
                      ("left_gripper", 1), ("right_gripper", 1)):
        value = compact_vector(action.get(key), dims)
        if value is not None:
            output[key] = value
    return output


def build_log_context(run_name: str, episode_index: int) -> dict:
    sys.path.insert(0, str(ROOT / "evaluation/scripts"))
    import task_stage_diagnostics as task40

    run = ROOT / "evaluation" / run_name
    episode_rows = task40.read_jsonl(run / "episodes.jsonl")
    matches = [row for row in episode_rows if row.get("episode_index") == episode_index]
    if len(matches) != 1:
        raise ValueError(f"{run_name} episode {episode_index}: expected one episode row, got {len(matches)}")
    episode = matches[0]
    traces, trace_warning = task40._select_traces(run, episode)
    if trace_warning:
        traces = []
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    cfg = result.get("configuration", {})
    scale = (cfg.get("video_steps_per_render") or 24) * (cfg.get("video_fps") or 20)

    # Retain evenly spaced state samples and every sampled physical transition.
    state_rows = []
    prior = None
    for row in traces:
        facts = task40.physics(row)
        changed = prior is not None and facts != prior
        state_rows.append((row, facts, changed))
        prior = facts
    keep = set()
    if state_rows:
        count = min(22, len(state_rows))
        keep.update(round(i * (len(state_rows) - 1) / max(count - 1, 1)) for i in range(count))
        keep.update(i for i, item in enumerate(state_rows) if item[2])
    physical = []
    for i in sorted(keep):
        row, facts, _ = state_rows[i]
        physical.append({
            "action_step": int(row["action_steps"]),
            "video_seconds_approx": round(float(row["action_steps"]) / scale, 2),
            "facts": facts,
        })

    action_path = run / "stages" / f"actions_task40_gpu{episode.get('gpu_index')}_env{episode.get('env_idx')}.jsonl"
    actions = [row for row in task40.read_jsonl(action_path)
               if all(row.get(key) == episode.get(key)
                      for key in ("instance_id", "seed", "env_idx"))]
    if len({row.get("episode_number") for row in actions}) > 1:
        actions = []
    action_samples = []
    if actions:
        actions.sort(key=lambda row: row.get("action_start", -1))
        count = min(20, len(actions))
        indices = sorted(set(round(i * (len(actions) - 1) / max(count - 1, 1))
                             for i in range(count)))
        for i in indices:
            row = actions[i]
            action_samples.append({
                "video_seconds_approx": round(float(row.get("action_start", 0)) / scale, 2),
                **compact_action(row),
            })
    return {
        "action_commands": action_samples,
        "physical_state_samples": physical,
        "time_mapping": "video seconds are approximate conversions from action_steps",
        "missing_log_reason": trace_warning,
    }


def make_prompt(actions: list[dict], context: dict) -> str:
    compact_actions = []
    for action in actions:
        compact_actions.append({
            "action_id": action["id"],
            "action": action["label"],
            "results": [{"state_id": state[0], "result": state[1]}
                        for state in action["states"]],
            "visible_details": [{"detail_id": detail[0], "detail": detail[1]}
                                for detail in action.get("details", [])],
        })
    return (
        "你正在审查Task40一个完整机器人回合。三路相机被拼接在同一视频画面中；"
        "这是完整视频，不是定位短片。按时间顺序列出视频中实际可见的任务相关动作。\n"
        f"任务目标（仅用于判断决策是否偏离）：{TASK_DESCRIPTION}\n"
        "动作控制日志是稀疏数值采样，不能仅凭数值猜测语义动作。物理状态是稀疏采样；"
        "video_seconds_approx是近似时间。证据缺失时写unknown，不要根据正常流程补全。"
        "动作及物理日志用于核对尝试和结果，不得据此声称视频没有显示的动作。\n"
        "事件切分规则（严格遵守）：一个持续移动/接近目标的过程只记一个事件；"
        "同一物体/部件上的连续抓取、开门、放置等重试合并成一个事件，状态写最终可见结果；"
        "仅当中间发生了另一个不同任务动作，才把之后再次尝试另记为新事件。"
        "绝不能按抽样帧重复记录同一动作/状态；没有可见变化的等待、停留、视角变化不单独列事件。"
        "只列任务相关的不同动作，通常4-12条，最多18条；不足以确认时写unknown或省略，不要凑数。"
        "time_seconds只有在能可靠定位时才填，否则必须为null，禁止默认填0。\n"
        f"动作指令日志：{json.dumps(context['action_commands'], ensure_ascii=False, separators=(',', ':'))}\n"
        f"物理状态日志：{json.dumps(context['physical_state_samples'], ensure_ascii=False, separators=(',', ':'))}\n"
        f"允许的事件动作和结果枚举：{json.dumps(compact_actions, ensure_ascii=False, separators=(',', ':'))}\n"
        "输出一个JSON对象，不要markdown，字段必须为：\n"
        "{\"events\":[{\"order\":1,\"action_id\":\"枚举id\",\"state_id\":\"枚举state_id\","
        "\"detail_id\":\"枚举detail_id或unspecified\",\"time_seconds\":数值或null,"
        "\"evidence\":\"画面中直接可见的简短证据\"}],"
        "\"episode_status\":\"success|failed|unclear\","
        f"\"failure_class\":\"{'|'.join(FAILURE_CLASSES)}\","
        "\"failure_evidence\":\"可观察证据；不能归因时说明不足\","
        "\"confidence_0_to_1\":0.0}\n"
        "把同一动作中的连续重试记成一个事件并选择重试后的最终结果；若重试之间出现其他动作，"
        "按真实顺序分别记录。unknown是合法结果。high_level_decision仅用于错误的动作/目标选择有直接证据；"
        "low_level_execution仅用于目标正确但执行可见失败；perception_state_judgment仅用于有证据显示"
        "机器人按错误状态行动；否则用insufficient_evidence。视频显示成功且无失败时用none。"
    )


def prepare(args) -> None:
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    source_key = json.loads((args.review_root / "source_key.json").read_text(encoding="utf-8"))
    schema = json.loads((args.review_root / "sequence_schema.json").read_text(encoding="utf-8"))
    sources = {row["segment_id"]: row for row in source_key["segments"]}
    videos = {row["video_id"]: row for row in annotations["videos"]}
    if set(videos) != set(sources):
        raise ValueError("annotation IDs do not match the 18 source episodes")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for case_id in sorted(videos):
            human = videos[case_id]
            source = sources[case_id]
            source_hash = human.get("source", {}).get("source_video_sha256")
            if source_hash != source["source_video_sha256"]:
                raise ValueError(f"{case_id}: source video hash mismatch")
            video = args.review_root / "full_episodes" / f"sequence_{case_id}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            run_name = source["source_run"]
            episode_index = int(source["episode_index"])
            context = build_log_context(run_name, episode_index)
            prompt = make_prompt(schema["actions"], context)
            item = {
                "case_id": case_id,
                "episode_id": human["source"]["source_episode_id"],
                "video_file": f"sequence_{case_id}.mp4",
                "review_video_sha256": sha256_bytes(video.read_bytes()),
                "source_video_sha256": source_hash,
                "video_duration_seconds": human["source"]["source_duration_seconds"],
                "video_fps": args.fps,
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "log_context": context,
                "label_count": len(human.get("steps", [])),
            }
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"prepared {count} blind full-episode inputs -> {output}")


def parse_json_response(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("events"), list):
            return value
    return None


def infer(args) -> None:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    manifest = read_jsonl(args.manifest)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.resume and output.is_file():
        for row in read_jsonl(output):
            if row.get("model_path") == str(args.model_path) and not row.get("error"):
                completed[row["case_id"]] = row
    if len(completed) == len(manifest):
        print(f"already complete: {len(completed)} cases")
        return

    visible_devices = torch.cuda.device_count()
    if not torch.cuda.is_available() or visible_devices < 1:
        raise RuntimeError("CUDA is unavailable in this inference environment")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    max_memory = {i: args.max_memory for i in range(visible_devices)}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_path), dtype=dtype, device_map="auto", max_memory=max_memory,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    print(f"loaded {args.model_name}; torch={torch.__version__}; dtype={dtype}; gpus={visible_devices}", flush=True)

    mode = "a" if args.resume and output.is_file() else "w"
    with output.open(mode, encoding="utf-8") as stream:
        for index, item in enumerate(manifest, 1):
            if item["case_id"] in completed:
                continue
            video = args.video_root / item["video_file"]
            started = time.time()
            base = {
                "case_id": item["case_id"], "episode_id": item["episode_id"],
                "model_name": args.model_name, "model_path": str(args.model_path),
                "prompt_sha256": item["prompt_sha256"],
                "review_video_sha256": item["review_video_sha256"],
                "source_video_sha256": item["source_video_sha256"],
                "video_fps": item["video_fps"],
            }
            try:
                if sha256_bytes(video.read_bytes()) != item["review_video_sha256"]:
                    raise ValueError("staged review-video SHA256 mismatch")
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": str(video)},
                    {"type": "text", "text": item["prompt"]},
                ]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", fps=item["video_fps"],
                ).to(model.device)
                sampled_frames = int(inputs["video_grid_thw"][0, 0].item()) * 2
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                            do_sample=False)
                generated = output_ids[:, inputs["input_ids"].shape[1]:]
                text = processor.batch_decode(generated, skip_special_tokens=True)[0]
                parsed = parse_json_response(text)
                row = {
                    **base, "raw_response": text, "prediction": parsed,
                    "sampled_frame_count": sampled_frames,
                    "prompt_token_count": int(inputs["input_ids"].shape[-1]),
                    "elapsed_seconds": round(time.time() - started, 2),
                    "error": None if parsed is not None else "response_not_parseable_as_event_json",
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
            print(f"{index}/{len(manifest)} {item['case_id']} error={row['error'] is not None} "
                  f"elapsed={row['elapsed_seconds']}s", flush=True)
    del model, processor
    torch.cuda.empty_cache()


def lcs_pairs(left: list, right: list, key) -> list[tuple[int, int]]:
    rows, cols = len(left), len(right)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if key(left[i]) == key(right[j]):
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    pairs = []
    i = j = 0
    while i < rows and j < cols:
        if key(left[i]) == key(right[j]):
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def safe_events(row: dict, schema: dict) -> tuple[list[dict], int]:
    prediction = row.get("prediction") or {}
    valid_actions = {action["id"]: {state[0] for state in action["states"]}
                     for action in schema["actions"]}
    events, invalid = [], 0
    for event in prediction.get("events", []):
        if not isinstance(event, dict):
            invalid += 1
            continue
        action = event.get("action_id")
        state = event.get("state_id")
        if action not in valid_actions or state not in valid_actions[action]:
            invalid += 1
            continue
        events.append(event)
    return events, invalid


def score_case(reference: list[dict], prediction_row: dict, schema: dict) -> dict:
    predicted, invalid = safe_events(prediction_row, schema)
    action_pairs = lcs_pairs(reference, predicted, lambda x: x["action"] if "action" in x else x["action_id"])
    # Score event identity as an order-preserving exact action+result match.
    event_pairs = lcs_pairs(reference, predicted,
                            lambda x: (x["action"], x["state"]) if "action" in x
                            else (x["action_id"], x["state_id"]))
    action_tp = len(action_pairs)
    event_tp = len(event_pairs)
    ref_n, pred_n = len(reference), len(predicted) + invalid
    p = action_tp / pred_n if pred_n else (1.0 if not ref_n else 0.0)
    r = action_tp / ref_n if ref_n else (1.0 if not pred_n else 0.0)
    event_p = event_tp / pred_n if pred_n else (1.0 if not ref_n else 0.0)
    event_r = event_tp / ref_n if ref_n else (1.0 if not pred_n else 0.0)
    states_correct = sum(reference[i]["state"] == predicted[j]["state_id"]
                         for i, j in action_pairs)
    detail_pairs = [(i, j) for i, j in event_pairs
                    if reference[i].get("detail", "unspecified") != "unspecified"]
    details_correct = sum(reference[i].get("detail") == predicted[j].get("detail_id", "unspecified")
                          for i, j in detail_pairs)
    edit = [[0] * (len(predicted) + 1) for _ in range(ref_n + 1)]
    for i in range(ref_n + 1):
        edit[i][0] = i
    for j in range(len(predicted) + 1):
        edit[0][j] = j
    for i in range(1, ref_n + 1):
        for j in range(1, len(predicted) + 1):
            cost = 0 if (reference[i - 1]["action"], reference[i - 1]["state"]) == (
                predicted[j - 1]["action_id"], predicted[j - 1]["state_id"]) else 1
            edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + cost)
    denom = max(ref_n, len(predicted), 1)
    diagnosis = (prediction_row.get("prediction") or {}).get("failure_class", "unparseable")
    return {
        "reference_events": ref_n, "predicted_events": pred_n,
        "invalid_predictions": invalid,
        "action_tp": action_tp, "action_precision": p, "action_recall": r,
        "action_f1": 2 * p * r / (p + r) if p + r else 0.0,
        "missed_events": max(0, ref_n - event_tp),
        "extra_or_wrong_result_events": max(0, pred_n - event_tp),
        "event_precision": event_p, "event_recall": event_r,
        "event_f1": 2 * event_p * event_r / (event_p + event_r) if event_p + event_r else 0.0,
        "state_correct": states_correct, "state_accuracy_given_action_match":
            states_correct / action_tp if action_tp else None,
        "detail_correct": details_correct, "detail_scored": len(detail_pairs),
        "sequence_edit_similarity": 1 - edit[ref_n][len(predicted)] / denom,
        "failure_class_unscored": diagnosis,
        "prediction_error": prediction_row.get("error"),
    }


def macro(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def score(args) -> None:
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    schema = json.loads((args.review_root / "sequence_schema.json").read_text(encoding="utf-8"))
    references = {row["video_id"]: row["steps"] for row in annotations["videos"]}
    manifest_rows = read_jsonl(args.review_root / "model_outputs/task40_vlm_inputs_blind.jsonl")
    manifest = {row["case_id"]: row for row in manifest_rows}
    if set(manifest) != set(references):
        raise ValueError("blind manifest case IDs do not match the annotated videos")
    prediction_sets = {"Qwen3-VL-8B-Instruct": read_jsonl(args.pred8),
                       "Qwen3-VL-32B-Instruct": read_jsonl(args.pred32)}
    by_model = {}
    raw_predictions = {}
    for model_name, outputs in prediction_sets.items():
        indexed = {}
        for row in outputs:
            case_id = row.get("case_id")
            expected = manifest.get(case_id)
            if expected is None:
                raise ValueError(f"{model_name}: unexpected prediction case {case_id!r}")
            for output_key, manifest_key in (("prompt_sha256", "prompt_sha256"),
                                             ("review_video_sha256", "review_video_sha256"),
                                             ("source_video_sha256", "source_video_sha256")):
                if row.get(output_key) != expected.get(manifest_key):
                    raise ValueError(f"{model_name} {case_id}: {output_key} does not match blind manifest")
            prior = indexed.get(case_id)
            if prior is None or (prior.get("error") and not row.get("error")) or (
                    bool(prior.get("error")) == bool(row.get("error"))):
                indexed[case_id] = row
        raw_predictions[model_name] = {
            case_id: {"prediction": indexed.get(case_id, {}).get("prediction"),
                      "error": indexed.get(case_id, {}).get("error")}
            for case_id in references
        }
        scored = []
        for case_id in sorted(references):
            raw = indexed.get(case_id, {"case_id": case_id, "prediction": None,
                                        "error": "missing_prediction"})
            metrics = score_case(references[case_id], raw, schema)
            scored.append({"case_id": case_id, **metrics})
        by_model[model_name] = scored

    per_model = {}
    for model, rows in by_model.items():
        per_model[model] = {
            "n_videos": len(rows),
            "reference_events": sum(x["reference_events"] for x in rows),
            "predicted_events": sum(x["predicted_events"] for x in rows),
            "action_precision_macro": macro(rows, "action_precision"),
            "action_recall_macro": macro(rows, "action_recall"),
            "action_f1_macro": macro(rows, "action_f1"),
            "event_precision_macro": macro(rows, "event_precision"),
            "event_recall_macro": macro(rows, "event_recall"),
            "event_f1_macro": macro(rows, "event_f1"),
            "state_accuracy_given_action_match": (
                sum(x["state_correct"] for x in rows) /
                sum(x["action_tp"] for x in rows) if sum(x["action_tp"] for x in rows) else None),
            "sequence_edit_similarity_macro": macro(rows, "sequence_edit_similarity"),
            "videos_with_inference_error": sum(bool(x["prediction_error"]) for x in rows),
            "failure_class_distribution_unscored": {},
            "cases": rows,
        }
        for row in rows:
            label = row["failure_class_unscored"]
            per_model[model]["failure_class_distribution_unscored"][label] = (
                per_model[model]["failure_class_distribution_unscored"].get(label, 0) + 1)

    paired = []
    left = {x["case_id"]: x for x in by_model["Qwen3-VL-8B-Instruct"]}
    right = {x["case_id"]: x for x in by_model["Qwen3-VL-32B-Instruct"]}
    for case_id in sorted(set(left) & set(right)):
        paired.append({"case_id": case_id,
                       "event_f1_delta_32b_minus_8b": right[case_id]["event_f1"] - left[case_id]["event_f1"],
                       "action_f1_delta_32b_minus_8b": right[case_id]["action_f1"] - left[case_id]["action_f1"],
                       "state_accuracy_8b": left[case_id]["state_accuracy_given_action_match"],
                       "state_accuracy_32b": right[case_id]["state_accuracy_given_action_match"]})
    summary = {
        "schema_version": 1,
        "annotations_sha256": sha256_bytes(args.annotations.read_bytes()),
        "annotation_count": len(references),
        "taxonomy_version": annotations["taxonomy_version"],
        "metrics": per_model,
        "paired_case_deltas": paired,
        "raw_predictions": raw_predictions,
        "failure_attribution_accuracy": None,
        "failure_attribution_note": (
            "Human JSON contains event sequence/outcomes, not episode-level causal labels. "
            "Failure-class predictions are reported but not scored as accuracy."),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, summary)


def fmt_pct(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def write_report(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = summary["metrics"]
    names = list(models)
    lines = [
        "# Task40：18条完整视频 VL 事件级对照",
        "",
        f"标注数：{summary['annotation_count']} 段；人工事件数：{sum(x['reference_events'] for x in models.values()) // len(models)}。",
        f"人工标注 SHA256：`{summary['annotations_sha256']}`；taxonomy：`{summary['taxonomy_version']}`。",
        "输入协议：每个模型单次收到一个完整回合视频（Qwen3-VL 原生 video 输入，均匀 2 fps）与相同的动作/物理日志；未拆分窗口，未向模型提供人工事件答案。",
        "",
        "## 事件识别",
        "",
        "事件精确匹配要求动作、结果状态均一致且顺序一致；动作指标单独用动作类别做顺序匹配。指标按视频宏平均，状态准确率只在动作类别匹配的事件上计算。",
        "",
        "|模型|事件F1|事件精确率|事件召回率|动作F1|动作召回率|状态准确率*|序列相似度|推理错误视频|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        data = models[name]
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            name, fmt_pct(data["event_f1_macro"]), fmt_pct(data["event_precision_macro"]),
            fmt_pct(data["event_recall_macro"]), fmt_pct(data["action_f1_macro"]),
            fmt_pct(data["action_recall_macro"]), fmt_pct(data["state_accuracy_given_action_match"]),
            fmt_pct(data["sequence_edit_similarity_macro"]), data["videos_with_inference_error"]))
    lines += [
        "",
        "*状态准确率是动作匹配子集上的条件准确率；`detail` 只在人工标注了具体 detail 的事件上另行统计。漏报率 = 1 − 事件召回率；虚构/多报率 = 1 − 事件精确率。",
        "",
        "## 逐视频结果",
        "",
        "|视频|人工事件|8B预测|8B事件F1|8B状态准确率|32B预测|32B事件F1|32B状态准确率|归因输出（8B / 32B，未评分）|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    case_index = {model: {row["case_id"]: row for row in values["cases"]}
                  for model, values in models.items()}
    outputs = summary.get("raw_predictions", {})
    for case_id in sorted(case_index[names[0]]):
        a, b = case_index[names[0]][case_id], case_index[names[1]][case_id]
        pred8 = outputs.get(names[0], {}).get(case_id, {})
        pred32 = outputs.get(names[1], {}).get(case_id, {})
        cause8 = (pred8.get("prediction") or {}).get("failure_class", "unparseable")
        cause32 = (pred32.get("prediction") or {}).get("failure_class", "unparseable")
        lines.append(f"|{case_id}|{a['reference_events']}|{a['predicted_events']}|{fmt_pct(a['event_f1'])}|"
                     f"{fmt_pct(a['state_accuracy_given_action_match'])}|{b['predicted_events']}|"
                     f"{fmt_pct(b['event_f1'])}|{fmt_pct(b['state_accuracy_given_action_match'])}|"
                     f"{cause8} / {cause32}|")
    model8, model32 = models[names[0]], models[names[1]]
    event_delta = model32["event_f1_macro"] - model8["event_f1_macro"]
    action_delta = model32["action_f1_macro"] - model8["action_f1_macro"]
    state_delta = (model32["state_accuracy_given_action_match"] -
                   model8["state_accuracy_given_action_match"])
    over_limit = [row["case_id"] for row in model32["cases"] if row["predicted_events"] > 18]
    lines += [
        "",
        "## 结果结论",
        "",
        f"32B相对8B的事件F1提升{100 * event_delta:.1f}个百分点、动作F1提升{100 * action_delta:.1f}个百分点；"
        f"匹配动作后的状态准确率变化{100 * state_delta:+.1f}个百分点。",
        f"32B共预测{model32['predicted_events']}个事件，对照人工128个；事件精确率为{fmt_pct(model32['event_precision_macro'])}，"
        f"说明仍有明显多报/切分误差。8B有{sum(row['predicted_events'] == 0 for row in model8['cases'])}/18条输出空事件序列。",
        (f"32B在提示要求最多18条的情况下，{', '.join(over_limit)}超过上限；" if over_limit else
         "32B没有视频超过提示设定的18条上限；")
        + "本轮可支持‘32B事件/动作识别优于8B’的初步结论，但还不能据此判定其稳定可靠。",
        "32B的C03首次输出触及token上限并截断，随后用相同视频与提示提高输出上限重跑成功；原始JSONL保留两次尝试，指标使用成功响应。",
        "",
        "## 失败归因边界",
        "",
        "模型会输出 high-level decision、low-level execution、perception/state judgment、insufficient evidence 或 none，并融合动作指令与物理状态采样。但现有人工 JSON 没有逐回合因果标准答案，因此本报告不计算归因准确率；不能把模型自述或日志规则冒充人工真值。",
        "",
        "## 可复核文件",
        "",
        "- `task40_vlm_inputs_blind.jsonl`：发送给模型的盲测提示与日志摘要，不包含人工事件答案。",
        "- `qwen3_vl_8b_full_video.jsonl`、`qwen3_vl_32b_full_video.jsonl`：逐回合原始响应、解析结果、SHA256、采样帧数、推理时长及错误。",
        "- `task40_vlm_event_metrics.json`：机器可读的逐视频与汇总指标。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    p.add_argument("--output", type=Path, default=REVIEW_ROOT / "model_outputs/task40_vlm_inputs_blind.jsonl")
    p.add_argument("--fps", type=float, default=2.0)
    p.set_defaults(func=prepare)
    p = sub.add_parser("infer")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-memory", default="72GiB")
    p.add_argument("--max-new-tokens", type=int, default=1800)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=infer)
    p = sub.add_parser("score")
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    p.add_argument("--pred8", type=Path, required=True)
    p.add_argument("--pred32", type=Path, required=True)
    p.add_argument("--summary", type=Path, default=REVIEW_ROOT / "model_outputs/task40_vlm_event_metrics.json")
    p.add_argument("--report", type=Path, default=REVIEW_ROOT / "TASK40_18_VL_EVENT_COMPARISON.md")
    p.set_defaults(func=score)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
