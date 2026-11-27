"""Load a real saved LIBERO policy; serve only its paired simulator on localhost."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import threading

import numpy as np
import torch

from .protocol import CONTRACT, HORIZONS, KEYS, ROOT, read_json, state_dict
from v1_groot_memoryvla.data.embodiment_tags import EmbodimentTag
from v1_groot_memoryvla.policy.gr00t_policy import Gr00tPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--suite', choices=HORIZONS, required=True)
    parser.add_argument('--tasks', type=int, nargs='+', required=True)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=0, help='0 uses the suite horizon; shorter windows are debug only')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = read_json(args.checkpoint/'libero.json')
    if not metadata or metadata['contract'] != CONTRACT:
        raise ValueError('Select a checkpoint trained with the LIBERO contract, not a B1K checkpoint')
    torch.set_num_threads(4)
    policy = Gr00tPolicy(EmbodimentTag.LIBERO_PANDA, str(args.checkpoint), device='cuda:0')
    token = secrets.token_hex(24)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get('Authorization') != token:
                self.send_error(403); return
            try:
                value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/reset':
                    torch.manual_seed(value['seed']); torch.cuda.manual_seed_all(value['seed'])
                    policy.reset()
                    result = dict(reset=True)
                elif self.path == '/act':
                    videos = {key: np.frombuffer(base64.b64decode(image),dtype=np.uint8).reshape(1,1,256,256,3)
                              for key,image in value['images'].items()}
                    states = state_dict(np.asarray(value['eef'])[None,None,:],np.asarray(value['gripper'])[None,None,:])
                    observation = dict(video=videos,state=states,
                        language={policy.language_key:[[value['language']]]})
                    actions, _ = policy.get_action(observation, options=dict(session_id='libero',env_ids=[0],episode_ids=[value['episode']]))
                    chunk = np.concatenate([actions[key][0] for key in KEYS],axis=-1)[:CONTRACT['action_steps']]
                    if chunk.shape != (CONTRACT['action_steps'],7) or not np.isfinite(chunk).all():
                        raise ValueError('Invalid policy action chunk')
                    result = dict(actions=chunk.tolist())
                else:
                    self.send_error(404); return
                body = json.dumps(result).encode()
                self.send_response(200); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
            except Exception as error:
                body = json.dumps(dict(error=str(error))).encode()
                self.send_response(500); self.end_headers(); self.wfile.write(body)
        def log_message(self,*args): pass
    server = HTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    print(json.dumps(dict(policy='ready',checkpoint=str(args.checkpoint),arm=metadata['arm'])),flush=True)
    # Same job/process group. Stopping it terminates both policy and simulator.
    argv = [str(ROOT/'env/libero-smoke/bin/python'), str(ROOT/'scripts/libero_rollout.py'),
        '--policy-port',str(server.server_port),'--policy-token',token,
        '--checkpoint',str(args.checkpoint),'--suite',args.suite,'--tasks',*[str(t) for t in args.tasks],
        '--trials',str(args.trials),'--max-steps',str(args.max_steps),'--seed',str(args.seed),'--output',str(args.output)]
    try:
        result = subprocess.run(argv,cwd=ROOT)
        if result.returncode: raise SystemExit(result.returncode)
    finally:
        server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': main()
