#!/usr/bin/env python3
"""Run fixed-frame, event-level VLM inference against an OpenAI-compatible API."""
import argparse
import base64
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import cv2


PROMPT = """Analyze only the visible evidence in this short video excerpt. It is a horizontal composite of left-wrist, right-wrist, and head cameras. The supplied frames are timestamped and ordered. Do not infer events outside the excerpt or use an unstated task goal. If the visible evidence does not establish a field, use unknown; use insufficient_evidence when a failure cause cannot be supported. Distinguish a visible attempt from successful execution and visible physical result. Return exactly one JSON object with these fields: visible_behavior (string), object_or_target (string or unknown), operation_attempted (yes/no/unknown), execution_observed (string or unknown), physical_result_visible (string or unknown), failure_class (high_level_decision/low_level_execution/perception_state_judgment/insufficient_evidence/none), evidence_time_seconds_in_clip (number or unknown), confidence_0_to_1 (number), unknown_reason_or_notes (string). Use high_level_decision only when an incorrect goal or operation choice is directly supported; low_level_execution only when the intended operation is clear and its execution visibly fails; perception_state_judgment only when a mistaken visible-state judgment is evidenced. Otherwise choose insufficient_evidence."""


def encode_frames(path: Path, max_frames: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 1:
        capture.release()
        raise RuntimeError(f"video has no frames: {path}")
    sample_count = min(max_frames, count)
    denominator = max(sample_count - 1, 1)
    indices = sorted(set(round(i * (count - 1) / denominator) for i in range(sample_count)))
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"failed reading frame {index} from {path}")
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            capture.release()
            raise RuntimeError(f"failed encoding frame {index} from {path}")
        timestamp = index / fps
        frames.append((timestamp, base64.b64encode(jpeg).decode("ascii")))
    capture.release()
    return frames


def infer(api_base: str, model: str, frames, max_tokens: int):
    content = [{"type": "text", "text": PROMPT}]
    for timestamp, encoded in frames:
        content.append({"type": "text", "text": f"Frame at t={timestamp:.3f}s:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    text = result["choices"][0]["message"]["content"]
    parsed = None
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        pass
    return text, parsed, result.get("usage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8900/v1")
    parser.add_argument("--model", default="qwen3-vl-32b-instruct")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clip_paths = sorted(args.clips.glob("E[0-9][0-9].mp4"))
    if not clip_paths:
        raise SystemExit(f"no blinded clips found in {args.clips}")

    prompt_hash = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    with args.output.open("w", encoding="utf-8") as stream:
        for index, path in enumerate(clip_paths, start=1):
            started = time.time()
            frames = encode_frames(path, args.max_frames)
            try:
                raw, parsed, usage = infer(args.api_base, args.model, frames, args.max_tokens)
                error = None
            except Exception as exc:
                raw, parsed, usage = None, None, None
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "case_id": path.stem,
                "model": args.model,
                "prompt_sha256": prompt_hash,
                "clip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frame_timestamps_seconds": [round(t, 6) for t, _ in frames],
                "max_frames": args.max_frames,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "raw_response": raw,
                "parsed_response": parsed,
                "usage": usage,
                "error": error,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index}/{len(clip_paths)} {path.stem} error={error is not None}", flush=True)


if __name__ == "__main__":
    main()
