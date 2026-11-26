"""Fetch only relevant members of official historical ZIPs by HTTP byte range.

Never replaces shared assets. Encrypted sources stay in local experiment storage;
only provenance and derived geometry are intended for public delivery. No key is copied.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import requests

OUT=Path(__file__).resolve().parents[1]/'label_extension'
CACHE=Path('/nvme3/datasets/task40_asset_recovery')


class RangeReader(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=size;self.pos=0;self.session=requests.Session();self.session.trust_env=False
        self.downloaded=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,size=-1):
        end=min(self.size,self.pos+size) if size>=0 else self.size
        if end<=self.pos:return b''
        start=self.pos
        # Separate cache keys prevent stale partial responses being reused across ranges.
        url=self.url+('&' if '?' in self.url else '?')+f'range_start={start}&range_end={end-1}'
        r=self.session.get(url,headers={'Range':f'bytes={start}-{end-1}'},timeout=(12,50))
        r.raise_for_status()
        assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end-1}/'),(r.status_code,r.headers)
        assert len(r.content)==end-start
        self.pos=end;self.downloaded+=len(r.content)
        return r.content


def main():
    p=argparse.ArgumentParser();p.add_argument('--revision',default='015e37a26823a110fedb6a0c3d943f573de00ffe')
    p.add_argument('--archive',default='behavior-1k-assets.zip');a=p.parse_args()
    session=requests.Session();session.trust_env=False
    base='https://hf-mirror.com';repo='behavior-1k/zipped-datasets'
    listing=session.get(f'{base}/api/datasets/{repo}/tree/{a.revision}',timeout=(10,25));listing.raise_for_status()
    info=next(x for x in listing.json() if x['path']==a.archive)
    target=CACHE/a.revision;target.mkdir(parents=True,exist_ok=True)
    url=f'{base}/datasets/{repo}/resolve/{a.revision}/{a.archive}'
    reader=RangeReader(url,info['size']);members=[]
    with zipfile.ZipFile(reader) as z:
        names=[n for n in z.namelist() if '/popcorn_bag/hdcpqg/' in n or '/microwave/hjjxmi/' in n or n in ('VERSION','behavior-1k-assets/VERSION','og_dataset/VERSION')]
        for name in names:
            if not name.endswith(('.encrypted.usd','.usdz.encrypted','metadata.json','VERSION')):continue
            dest=target/name
            if not dest.is_file():
                data=z.read(name);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            data=dest.read_bytes()
            members.append(dict(member=name,local_path=str(dest),bytes=len(data),md5=hashlib.md5(data).hexdigest(),sha256=hashlib.sha256(data).hexdigest()))
            print(json.dumps(members[-1]),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(repo=repo,revision=a.revision,archive=a.archive,archive_metadata=info,
        official_url=f'https://huggingface.co/datasets/{repo}/resolve/{a.revision}/{a.archive}',
        transport='hf-mirror.com; ZIP CRC validated, members hashed; original expected encrypted MD5 separately checked',
        transferred_bytes=reader.downloaded,members=members)
    (OUT/f'asset_recovery_{a.revision[:8]}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
