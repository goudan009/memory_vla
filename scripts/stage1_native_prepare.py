"""Prepare a leakage-free native GR00T Stage1 dataset and guarded action bounds.

No model training or source dataset modifications. All statistics fit train only.
Relative arm/torso statistics are 32 x D, one current-state anchor per chunk.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

TAG = "behavior_r1_pro"
RELATIVE = {"torso": "trunk_qpos", "left_arm": "arm_left_qpos",
            "right_arm": "arm_right_qpos"}
H = 32
FIELDS = ("min", "max", "mean", "std", "q01", "q99")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stat(x):
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("empty/nonfinite data")
    return dict(zip(FIELDS, (v.tolist() for v in
        (x.min(0), x.max(0), x.mean(0, dtype=np.float64),
         x.std(0, dtype=np.float64), np.quantile(x, .01, axis=0),
         np.quantile(x, .99, axis=0)))))


def bounds(x, mode):
    """Per physical channel: preserve ordinary actions and zero; conservative tail gate.

    A bound is accepted only with >=4x span improvement, <=2.1% clipped
    elements, <=0.1% clipped by >0.1 rad and <=0.001 rad mean truncation
    error. Minimum width 0.02 rad (prevents near-constant saturation).
    These are engineering choices, not learned thresholds.
    """
    s = stat(x)
    lo, hi, q01, q99 = (np.asarray(s[k]) for k in ("min", "max", "q01", "q99"))
    mid = (q01 + q99) / 2
    l = np.maximum(lo, np.minimum(np.minimum(q01, mid-.01), 0.))
    u = np.minimum(hi, np.maximum(np.maximum(q99, mid+.01), 0.))
    span, robust_span = hi-lo, u-l
    # Quantiles can be equal; never narrow a genuinely constant channel.
    error = np.maximum(l-x, 0.) + np.maximum(x-u, 0.)
    clip = (error > 1e-7).mean(0)
    severe = (error > .1).mean(0)
    mae = error.mean(0)
    accept = ((robust_span >= .019999) & (span >= 4*robust_span) &
              (clip <= .021) & (severe <= .001) & (mae <= .001))
    chosen = accept if mode == "guarded" else np.zeros_like(accept)
    # Both upstream and repository processors read relative min/max even
    # with use_percentiles=True. Store effective bounds here and raw bounds
    # separately, retain minmax codec / raw state bounds for Stage2.
    effective = dict(s, min=np.where(chosen,l,lo).tolist(),
                     max=np.where(chosen,u,hi).tolist())
    report = dict(raw=s, candidate_min=l.tolist(), candidate_max=u.tolist(),
                  span_ratio=(span/np.maximum(robust_span,1e-12)).tolist(),
                  candidate_clip_fraction=clip.tolist(),
                  candidate_severe_fraction=severe.tolist(),
                  candidate_mae_rad=mae.tolist(), accepted=chosen.tolist())
    return effective, report


def parse_split(path, ids):
    meta = read(path)
    split = meta.get("split", meta)
    tr, va = split["train"], split["validation"]
    if (split.get("task") != 40 or split.get("seed") != 42 or
        len(tr) != 180 or len(va) != 20 or len(set(tr+va)) != 200 or
        set(tr+va) != set(ids)):
        raise ValueError("expected original exact Task40 180/20 split, seed42")
    return split


def resolve_one(src, relative, fallback):
    p = src / relative
    if p.is_file():
        return p.resolve()
    matches = list(src.glob(fallback))
    if len(matches) != 1:
        raise FileNotFoundError(f"{p}: {len(matches)} fallback matches")
    return matches[0].resolve()


def link(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(source.resolve())


def view(src, dst, episodes, info, modality):
    dst.mkdir(parents=True, exist_ok=False)
    new_info = dict(info, total_episodes=len(episodes),
                    total_frames=sum(int(e["length"]) for e in episodes),
                    total_videos=3*len(episodes),
                    splits={"train": f"0:{len(episodes)}"},
                    data_path="data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
                    video_path="videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4")
    chunk_size = info["chunks_size"]
    records = {}
    for e in episodes:
        eid = int(e["episode_index"])
        fields = dict(episode_index=eid, episode_chunk=eid//chunk_size)
        p = resolve_one(src, info["data_path"].format(**fields),
                        f"data/*/episode_{eid:08d}.parquet")
        dest = dst/new_info["data_path"].format(**fields)
        link(p, dest)
        records[eid] = dest
        for key, spec in modality["video"].items():
            original = spec.get("original_key", key)
            vfields = dict(fields, video_key=original)
            v = resolve_one(src, info["video_path"].format(**vfields),
                            f"videos/*/{original}/episode_{eid:08d}.mp4")
            link(v, dst/new_info["video_path"].format(**vfields))
    write(dst/"meta/info.json", new_info)
    write(dst/"meta/modality.json", modality)
    (dst/"meta/episodes.jsonl").write_text("".join(json.dumps(e)+"\n" for e in episodes))
    shutil.copyfile(src/"meta/tasks.jsonl", dst/"meta/tasks.jsonl")
    if set(records) != {int(e["episode_index"]) for e in episodes}:
        raise ValueError("duplicate episode IDs")
    return records


def load_arrays(records, lengths):
    out = []
    for eid, path in sorted(records.items()):
        df = pd.read_parquet(path, columns=["observation.state", "action",
                                           "episode_index", "frame_index", "task_index"])
        q = np.stack(df["observation.state"]).astype(np.float32)
        a = np.stack(df["action"]).astype(np.float32)
        if (q.shape != (lengths[eid], 82) or a.shape != (len(q), 23) or len(q) < H
            or not np.isfinite(q).all() or not np.isfinite(a).all()
            or set(df.episode_index) != {eid} or set(df.task_index) != {40}
            or not np.array_equal(df.frame_index, np.arange(len(q)))):
            raise ValueError(f"bad episode identity/state/action: {eid}")
        out.append((eid, q, a))
    return out


def deltas(arrays, qslice, aslice, h):
    return np.concatenate([a[h:h+len(a)-H+1, aslice] -
                           q[:len(q)-H+1, qslice] for _,q,a in arrays])


def slc(m, key):
    return slice(m[key]["start"], m[key]["end"])


def fit(train, validation, modality, mode):
    q = np.concatenate([x[1] for x in train])
    a = np.concatenate([x[2] for x in train])
    absolute = {"observation.state": stat(q), "action": stat(a)}
    relative, audit = {}, {}
    for key, sk in RELATIVE.items():
        qs, ac = slc(modality["state"], sk), slc(modality["action"], key)
        rows, reports = [], []
        for h in range(H):
            x = deltas(train, qs, ac, h)
            s, r = bounds(x, mode)
            vx = deltas(validation, qs, ac, h)
            lo, hi = np.asarray(s["min"]), np.asarray(s["max"])
            error = np.maximum(lo-vx,0.) + np.maximum(vx-hi,0.)
            r["validation_clip_fraction"] = (error > 1e-7).mean(0).tolist()
            r["validation_mae_rad"] = error.mean(0).tolist()
            # Largest train h0 delta identity; never silently delete/relabel it.
            if h == 0:
                offsets = np.cumsum([0]+[len(t[1])-H+1 for t in train])
                ii = np.abs(x).argmax(0)
                r["h0_extremes"] = [
                    dict(episode=int(train[np.searchsorted(offsets,i,side="right")-1][0]),
                         frame=int(i-offsets[np.searchsorted(offsets,i,side="right")-1]),
                         channel=j, delta=float(x[i,j])) for j,i in enumerate(ii)]
            rows.append(s)
            reports.append(r)
        relative[key] = {k:[s[k] for s in rows] for k in FIELDS}
        audit[key] = reports
    return absolute, relative, audit


def prepare(dataset, split_path, base, out, mode):
    dataset, base, out = map(lambda p: Path(p).resolve(), (dataset,base,out))
    if base.name != "GR00T-N1.6-3B" or (base/"v2_1_stage1.json").exists():
        raise ValueError("start from original generic GR00T-N1.6-3B, not Task40/B1K weights")
    cfg = read(base/"config.json")
    pc = read(base/"processor_config.json")["processor_kwargs"]
    if (cfg.get("memory_enabled") or cfg.get("finetuning_lora") or
        cfg.get("action_horizon") != 50 or cfg.get("max_action_dim") != 128 or
        cfg.get("num_inference_timesteps") != 4 or pc.get("max_action_horizon") != 50 or
        pc.get("max_action_dim") != 128 or pc.get("max_state_dim") != 128 or
        not pc.get("use_relative_action") or pc.get("use_percentiles") or
        not pc.get("clip_outliers")):
        raise ValueError("unexpected generic checkpoint / codec contract")
    info, modality = read(dataset/"meta/info.json"), read(dataset/"meta/modality.json")
    if (info["fps"] != 30 or len(modality["video"]) != 3 or
        (info["features"]["observation.state"]["shape"],info["features"]["action"]["shape"]) != ([82],[23])):
        raise ValueError("expected converted 30Hz R1Pro data")
    episodes = [json.loads(l) for l in (dataset/"meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    meta = {int(e["episode_index"]):e for e in episodes}
    if len(episodes)!=200 or len(meta)!=200:
        raise ValueError("expected 200 unique episodes")
    split = parse_split(split_path, meta)
    out.mkdir(parents=True, exist_ok=False)
    train = view(dataset,out/"train",[meta[e] for e in split["train"]],info,modality)
    val = view(dataset,out/"validation",[meta[e] for e in split["validation"]],info,modality)
    lengths = {e:int(m["length"]) for e,m in meta.items()}
    tr, va = load_arrays(train,lengths), load_arrays(val,lengths)
    absolute, relative, audit = fit(tr,va,modality,mode)
    # Train stats are copied to validation, never fitted on validation.
    for partition in ("train","validation"):
        write(out/partition/"meta/stats.json",absolute)
        write(out/partition/"meta/relative_stats.json",relative)
    write(out/"data_split.json",split)
    write(out/"normalization_audit.json",audit)
    view_base = out/"generic_base"
    view_base.mkdir()
    for f in base.iterdir():
        if f.name not in ("config.json",) and f.is_file():
            link(f,view_base/f.name)
    # Upstream pipeline doesn't override tune_top_llm_layers when loading.
    # Only change freeze flags on the private view; not original weights.
    cfg.update(tune_llm=False,tune_visual=False,tune_top_llm_layers=0,
               tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
               backbone_trainable_params_fp32=False)
    write(view_base/"config.json",cfg)
    files = ["data_split.json","normalization_audit.json","generic_base/config.json"]
    for partition in ("train","validation"):
        files.extend(f"{partition}/meta/{name}" for name in
                     ("stats.json","relative_stats.json","episodes.jsonl","info.json","modality.json","tasks.jsonl"))
    report = dict(dataset=str(dataset),base=str(base),mode=mode,
                  source_split_sha256=sha(split_path),base_config_sha256=sha(base/"config.json"),
                  train_frames=sum(lengths[e] for e in split["train"]),
                  val_frames=sum(lengths[e] for e in split["validation"]),
                  train_valid_anchors=sum(lengths[e]-H+1 for e in split["train"]),
                  accepted_cells={k:sum(sum(r["accepted"]) for r in v) for k,v in audit.items()},
                  raw_lo=absolute["observation.state"]["min"],raw_hi=absolute["observation.state"]["max"],
                  hashes={f:sha(out/f) for f in files})
    write(out/"PREPARED.json",report)
    return report


def verify_prepared(path):
    path=Path(path)
    m=read(path/"PREPARED.json")
    if sha(Path(m["base"])/"config.json")!=m["base_config_sha256"]:
        raise ValueError("original base config changed")
    for f,digest in m["hashes"].items():
        if sha(path/f)!=digest:
            raise ValueError(f"prepared artifact changed: {f}")
    # The official stats generator globs data/*/*.parquet; ensure the
    # physical view cannot accidentally contain held-out files.
    split=read(path/"data_split.json")
    for partition, key in (("train","train"),("validation","validation")):
        paths=list((path/partition).glob("data/*/*.parquet"))
        ids=[int(p.stem.rsplit("_",1)[1]) for p in paths]
        if len(ids)!=len(split[key]) or set(ids)!=set(split[key]) or any(not p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError("prepared parquet view changed: "+partition)
    return m


if __name__=="__main__":
    raise SystemExit("BLOCKED_WORKBENCH_INTEGRATION: dataset preparation belongs to the registered job; do not create a separate output directory.")
