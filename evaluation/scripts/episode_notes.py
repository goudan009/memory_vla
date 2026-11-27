"""Free-form notes on a recorded episode, independent from human scoring."""
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import human_review


def read(run, episode_index):
    path = Path(run)/'episode_notes'/f'episode_{episode_index}.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def save(run, episode_index, notes):
    identity = human_review.target(Path(run), episode_index=episode_index)
    if not isinstance(notes, str) or len(notes) > 2000:
        raise ValueError('回合备注最多 2000 个字符')
    record = dict(**identity, notes=notes.strip(), updated_at=datetime.now(timezone.utc).isoformat())
    folder = Path(run)/'episode_notes'
    folder.mkdir(exist_ok=True)
    path = folder/f'episode_{episode_index}.json'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                     prefix='.', suffix='.pending', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write('\n'); handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return record
