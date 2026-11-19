"""Download the unmodified NVIDIA N1.6 base checkpoint and transfer it to A100."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'checkpoints/groot_n1d6_base'
REMOTE='/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B'
REPO='nvidia/GR00T-N1.6-3B'


def main():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    info=json.load(opener.open('https://hf-mirror.com/api/models/'+REPO+'?blobs=true',timeout=30))
    revision=info['sha']
    TARGET.mkdir(parents=True,exist_ok=True)
    def download(entry):
        name=entry['rfilename']
        path=TARGET/name
        if path.is_file() and path.stat().st_size==entry.get('size'):
            return
        if name.endswith('.safetensors'):
            url=f'https://hf-mirror.com/{REPO}/resolve/{revision}/{name}?download=true'
        else:
            url=f'https://hf-mirror.com/{REPO}/raw/{revision}/{name}'
        partial=path.with_name(name+'.partial')
        subprocess.run(['curl','--noproxy','*','--fail','--location','--retry','3',
            '--connect-timeout','20','--output',str(partial),url],check=True)
        if partial.stat().st_size!=entry['size']:
            raise RuntimeError('File size mismatch: '+name)
        if entry.get('lfs',{}).get('sha256'):
            digest=hashlib.sha256()
            with partial.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):
                    digest.update(block)
            if digest.hexdigest()!=entry['lfs']['sha256']:
                raise RuntimeError('SHA256 mismatch: '+name)
        partial.replace(path)
        print('Verified '+name,flush=True)
    entries=[entry for entry in info['siblings'] if entry['rfilename'] in {
        'config.json','embodiment_id.json','processor_config.json','statistics.json',
        'model.safetensors.index.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','LICENSE'}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download,entries))
    (TARGET/'download_provenance.json').write_text(json.dumps(dict(repo=REPO,revision=revision,files=entries),indent=2))
    subprocess.run(['ssh','-o','BatchMode=yes','172.28.62.17','mkdir','-p',REMOTE],check=True)
    subprocess.run(['rsync','-a','--exclude=*.partial',str(TARGET)+'/',f'172.28.62.17:{REMOTE}/'],check=True)
    print('A100 checkpoint ready: '+REMOTE,flush=True)


if __name__=='__main__':
    main()
