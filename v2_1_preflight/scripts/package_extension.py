"""Versioned full delivery, manifest and archive verification; old release intact."""
import hashlib
import json
from pathlib import Path
import tarfile
import pandas as pd
from package import validate as validate_original

OUT=Path(__file__).resolve().parents[1]
EXT=OUT/'label_extension'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    original=validate_original()
    validation=json.loads((EXT/'validation.json').read_text())
    source=json.loads((EXT/'inside_source_validation.json').read_text())
    assert validation['passed'] and source['passed'] and source['matches']==600
    assert validation['states']==21 and validation['episodes']==200 and validation['frames']==647555
    frame=pd.read_parquet(EXT/'task40_labels.parquet')
    stats=pd.read_csv(EXT/'label_statistics.csv',dtype={'episode_index':str})
    masks=pd.read_parquet(EXT/'task40_masks.parquet')
    pd.testing.assert_frame_equal(masks,frame[masks.columns])
    states=json.loads((EXT/'summary.json').read_text())['states']
    for name in states:
        row=stats[(stats.episode_index=='ALL_200')&(stats.state==name)].iloc[0]
        assert row.positive==int(frame[name].sum()) and row.negative==int((~frame[name]).sum()) and row.unknown==int(frame[name].isna().sum())
        assert row.certified_positive+row.certified_negative==int(frame[name+'_certified_valid'].sum())
        assert row.weak_positive+row.weak_negative==int(frame[name+'_weak_valid'].sum())
    for path in OUT.rglob('*.json'):json.loads(path.read_text())
    for field in ('bag_inside','bag_grasped'):assert not frame[field+'_certified_valid'].any()
    for item in validation['images']:assert (EXT/item['image']).is_file()
    records=[]
    for path in sorted(OUT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path==OUT/'MANIFEST.json':continue
        assert not path.is_symlink()
        assert path.suffix not in ('.usd','.encrypted','.key'),path
        records.append(dict(path=str(path.relative_to(OUT)),bytes=path.stat().st_size,sha256=sha(path)))
    manifest=dict(algorithm='sha256',version='task40-labels-v002',scope='all files except root MANIFEST and __pycache__',
        original_validation=original,extension_validation={k:v for k,v in validation.items() if k not in ('images','panels')},files=records)
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    archive=OUT.parent/'v2_1_preflight_task40_labels_v002.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for relative in [r['path'] for r in records]+['MANIFEST.json']:
            tar.add(OUT/relative,arcname='v2_1_preflight/'+relative,recursive=False)
    expected={'v2_1_preflight/'+r['path']:r['sha256'] for r in records};expected['v2_1_preflight/MANIFEST.json']=sha(OUT/'MANIFEST.json')
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected
            h=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==expected.pop(member.name)
    assert not expected
    digest=sha(archive);archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps(dict(passed=True,archive=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(records)+1)))


if __name__=='__main__':main()
