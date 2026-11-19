"""Check two live REQ routes without running inference or changing server state."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

sys.path.insert(0, os.environ.get("GROOT_REPO", "/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1"))
from gr00t.policy.server_client import MsgSerializer
import zmq


def probe(host, port):
    barrier = threading.Barrier(2)
    tokens = ["groot_route_probe_" + uuid.uuid4().hex for _ in range(2)]

    def worker(index):
        with zmq.Context() as context:
            with context.socket(zmq.REQ) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 15000)
                socket.setsockopt(zmq.SNDTIMEO, 15000)
                socket.connect(f"tcp://{host}:{port}")
                barrier.wait()
                started = time.monotonic()
                # Existing server echoes unknown endpoint names in its error reply.
                # This is an expected protocol probe, not a model error.
                socket.send(MsgSerializer.to_bytes({"endpoint": tokens[index]}))
                result = MsgSerializer.from_bytes(socket.recv())
                expected = "Unknown endpoint: " + tokens[index]
                return {"client": index, "token": tokens[index], "reply": result,
                        "matched": result == {"error": expected}, "elapsed_s": time.monotonic() - started}

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(worker, range(2)))
    return {"host": host, "port": port, "all_matched": all(row["matched"] for row in rows),
            "note": "two intentional unknown-endpoint replies verify live REP routing; model inputs and server state untouched", "rows": rows}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.28.62.17")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.host, args.port)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["all_matched"] else 1)
