"""Render pre-registered validation candidates, without consulting model errors."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from v1_groot_memoryvla.data.convert_b1k import CAMERAS


def frames(path, wanted):
    images = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                images[i] = frame.to_image().resize((224, 224))
            if i >= max(wanted):
                break
    assert len(images) == len(wanted)
    return images


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    a = p.parse_args()
    split = json.loads((a.run / "config/data_split.json").read_text())
    root = a.run / "diagnostics/segments"
    root.mkdir(exist_ok=True)
    index = []
    for g in split["groups"]:
        if g["split"] != "validation" or g["stratum"] != "event_candidate":
            continue
        eid, start, event = g["episode"], g["start"], g["event"]
        wanted = [start, start+16*8, start+24*8, start+31*8]
        canvas = Image.new("RGB", (4*224, 3*248+70), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), f"Episode {eid}, group {g['group_id']}: {event['object']} {event['event']}", fill="black")
        draw.text((5, 25), f"event frame {event['frame']}; candidates at decisions 25--32; observational proxy only", fill="black")
        for row, camera in enumerate(CAMERAS):
            path = a.dataset / f"videos/task-0021/observation.images.rgb.{camera}_256_256/episode_{eid:08d}.mp4"
            images = frames(path, wanted)
            for col, frame in enumerate(wanted):
                y = 70+row*248
                draw.text((col*224+2, y), f"{camera} f{frame} t{frame/30:.1f}s", fill="black")
                canvas.paste(images[frame], (col*224, y+24))
        name = f"episode{eid}_group{g['group_id']}.jpg"
        canvas.save(root/name, quality=92)
        index.append(dict(g, contact_sheet=name, sampled_frames=wanted))
        print(name, flush=True)
    (root / "index.json").write_text(json.dumps(index, indent=2)+"\n")


if __name__ == "__main__":
    main()
