"""Download the official radio checkpoint with curl range resume and ZIP/hash evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

from download_v12_resources import ROOT, atomic_json

URL = 'https://drive.usercontent.google.com/download?id=1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789&export=download&confirm=t'
TARGET = Path('/nvme3/checkpoints/memory_vla/official-n17-radio')
SIZE = 6917651266


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    archive = TARGET / 'turning_on_radio_GR00T-checkpoint-150000.zip'
    partial = archive.with_suffix('.zip.part')
    state = {'source': URL, 'target': str(archive), 'expected_bytes': SIZE, 'status': 'downloading'}
    def save():
        state['updated_at'] = time.time()
        state['downloaded_bytes'] = archive.stat().st_size if archive.exists() else (partial.stat().st_size if partial.exists() else 0)
        atomic_json(ROOT / 'radio-progress.json', state)
    try:
        if not archive.exists():
            p = subprocess.Popen(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                                  '--speed-time', '120', '--speed-limit', '1024', '--continue-at', '-',
                                  '--output', str(partial), URL])
            while p.poll() is None:
                save()
                time.sleep(10)
            if p.returncode:
                raise RuntimeError('curl exit ' + str(p.returncode))
            assert partial.stat().st_size == SIZE
            partial.replace(archive)
        assert archive.stat().st_size == SIZE
        state['status'] = 'verifying'
        save()
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        state['sha256'] = digest.hexdigest()
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            state['archive_files'] = [{'path': f.filename, 'bytes': f.file_size} for f in z.infolist()]
        state['status'] = 'verified_archive'
    except Exception as error:
        state['status'] = 'failed'
        state['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
