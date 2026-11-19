"""Human judgements persist by real Episode without changing official/VLM scores."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation/scripts'))
import human_review

spec = importlib.util.spec_from_file_location('human_review_viewer', ROOT/'evaluation/scripts/view_results.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


@pytest.fixture
def dashboard(tmp_path):
    for name in ('first', 'second'):
        run = tmp_path/name
        (run/'videos/gpu0').mkdir(parents=True)
        (run/'results.json').write_text(json.dumps(dict(status='complete', mean_q_score=.2)))
        (run/'episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=i, instance_id=i+3,
            seed=42, gpu_index=0, env_idx=0, video_file=f'videos/gpu0/episode{i}.mp4' if i<3 else None))+'\n'
            for i in (1,2,3)))
        for i in (1,2):
            (run/f'videos/gpu0/episode{i}.mp4').write_bytes(b'video fixture')
        (run/'vlm_diagnosis').mkdir()
        (run/'vlm_diagnosis/episode1.json').write_text('{"visual_progress":25}')
    class Handler(viewer.Handler):
        root = tmp_path
    server = viewer.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield tmp_path, f'http://127.0.0.1:{server.server_port}'
    server.shutdown();server.server_close();worker.join(timeout=3)


def request(base, method='GET', **payload):
    url = base+'/api/human-review'
    if method == 'GET':
        req = Request(url+'?'+urlencode(payload))
    else:
        req = Request(url, method=method, data=json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json'})
    with urlopen(req, timeout=3) as response:
        assert response.headers['Cache-Control'] == 'no-store'
        return json.load(response)


def test_real_http_save_edit_reload_delete_and_episode_run_isolation(dashboard):
    root, base = dashboard
    originals = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    assert request(base, run='first', episode_index=1)['review'] is None
    fields = dict(execution_quality=0, visual_progress=12.5, behavior_class='goal_directed_incomplete',
        failure_stage='grasp', target_touched=False, stuck=True, vlm_agreement='disagree', notes='手未抓稳 <script>')
    saved = request(base, 'POST', run='first', episode_index=1, review=fields)['review']
    assert saved['execution_quality'] == 0 and saved['target_touched'] is False
    assert saved['source'] == 'human' and saved['instance_id'] == 4 and saved['seed'] == 42
    assert request(base, run='first', path='videos/gpu0/episode1.mp4')['review'] == saved
    assert request(base, run='first', episode_index=2)['review'] is None
    assert request(base, run='second', episode_index=1)['review'] is None
    edited = request(base, 'POST', run='first', episode_index=1,
                     review=dict(fields, execution_quality=65, notes='补充观察'))['review']
    assert edited['created_at'] == saved['created_at'] and edited['updated_at'] >= saved['updated_at']
    assert edited['execution_quality'] == 65
    runs = {r['_folder']:r for r in viewer.load_runs(root)}
    assert runs['first']['_human_reviews']['episode:1'] == edited
    assert runs['first']['mean_q_score'] is None  # never replaced by human score
    # A new module instance reads the same disk data; no process-local persistence.
    identity = human_review.target(root/'first', 1)
    assert json.loads(next((root/'first/human_reviews').glob('*.json')).read_text()) == edited
    assert human_review.read(root/'first', identity) == edited
    request(base, 'POST', run='first', episode_index=2, review=dict(notes='第二回合'))
    assert request(base, 'DELETE', run='first', episode_index=1)['deleted'] is True
    assert request(base, run='first', episode_index=1)['review'] is None
    assert request(base, run='first', episode_index=2)['review']['notes'] == '第二回合'
    assert request(base, 'DELETE', run='first', episode_index=1)['deleted'] is False
    assert all(path.read_bytes()==before for path,before in originals.items())


def test_finished_episode_without_video_and_legacy_same_basename_videos(dashboard):
    root, base = dashboard
    request(base, 'POST', run='first', episode_index=3, review=dict(notes='没有录像仍可评价'))
    assert request(base, run='first', episode_index=3)['review']['video_file'] is None
    for gpu in (0,1):
        path = root/f'first/videos/gpu{gpu}/legacy.mp4'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'video')
        request(base, 'POST', run='first', path=f'videos/gpu{gpu}/legacy.mp4',
                review=dict(execution_quality=gpu*100))
    assert request(base, run='first', path='videos/gpu0/legacy.mp4')['review']['execution_quality'] == 0
    assert request(base, run='first', path='videos/gpu1/legacy.mp4')['review']['execution_quality'] == 100


def test_secondary_reason_is_optional_saved_with_its_root_and_can_be_cleared(dashboard, monkeypatch):
    root, base = dashboard
    monkeypatch.setattr(human_review, 'STAGE_OPTIONS_PATH', root/'taxonomy.json')
    journal = root/'first/episodes.jsonl'
    episodes = [json.loads(line) for line in journal.read_text().splitlines()]
    for episode in episodes:
        episode['task_id'] = 40
    journal.write_text(''.join(json.dumps(episode)+'\n' for episode in episodes))

    stage = human_review.stage_options(40)[0]
    review = dict(reached_stage=stage['id'], root_cause='D', specific_reason='D1',
                  secondary_root_cause='M', secondary_specific_reason='M1')
    saved = request(base, 'POST', run='first', episode_index=1, review=review)
    assert saved['secondary_root_cause'] == 'M'
    assert saved['secondary_root_cause_label'] == 'M. 操作执行错误'
    assert saved['secondary_specific_reason'] == 'M1'
    assert request(base, run='first', episode_index=1)['review'] == saved

    cleared = request(base, 'POST', run='first', episode_index=1, review=dict(
        review, secondary_root_cause=None, secondary_specific_reason=None))
    assert cleared['secondary_root_cause'] is None
    assert cleared['secondary_root_cause_label'] == ''
    assert cleared['secondary_specific_reason'] is None

    with pytest.raises(HTTPError) as error:
        request(base, 'POST', run='first', episode_index=1,
                review=dict(review, secondary_specific_reason=None))
    assert error.value.code == 400


def test_stage_supplement_saves_without_root_cause_link(dashboard, monkeypatch):
    root, base = dashboard
    monkeypatch.setattr(human_review, 'STAGE_OPTIONS_PATH', root/'taxonomy.json')
    journal = root/'first/episodes.jsonl'
    episodes = [json.loads(line) for line in journal.read_text().splitlines()]
    for episode in episodes:
        episode['task_id'] = 40
    journal.write_text(''.join(json.dumps(episode)+'\n' for episode in episodes))

    stage = human_review.stage_options(40)[0]
    options = human_review.update_annotation_options(
        40, 'supplement', 'add', stage['id'], label='额外观察到抓取滑脱')
    supplement = options[0]['supplements'][0]
    saved = request(base, 'POST', run='first', episode_index=1, review=dict(
        reached_stage=stage['id'], root_cause='D', specific_reason='D1',
        secondary_root_cause=None, secondary_specific_reason=supplement['id']))
    assert saved['secondary_root_cause'] is None
    assert saved['secondary_specific_reason'] == supplement['id']
    assert saved['secondary_specific_reason_label'] == '额外观察到抓取滑脱'


def test_supplement_reason_options_can_be_added_renamed_and_deleted(dashboard, monkeypatch):
    root, _ = dashboard
    monkeypatch.setattr(human_review, 'STAGE_OPTIONS_PATH', root/'taxonomy.json')
    stage = human_review.stage_options(40)[0]
    stage_id = stage['id']
    cause = next(item for item in stage['causes'] if item['id'] == 'M')

    notes_before = list(cause['notes'])
    assert stage['supplements'] == []
    options = human_review.update_annotation_options(
        40, 'supplement', 'add', stage_id, label='补充：接触角度不佳')
    added = options[0]['supplements'][-1]
    assert added['label'] == '补充：接触角度不佳'
    assert next(item for item in options[0]['causes'] if item['id'] == 'M')['notes'] == notes_before
    assert all(item['id'] != added['id'] for item in options[1]['supplements'])

    options = human_review.update_annotation_options(
        40, 'supplement', 'rename', stage_id,
        label='补充：夹持位置偏移', option_id=added['id'])
    renamed = options[0]['supplements'][-1]
    assert renamed == {'id': added['id'], 'label': '补充：夹持位置偏移'}

    options = human_review.update_annotation_options(
        40, 'supplement', 'delete', stage_id, option_id=added['id'])
    supplements = options[0]['supplements']
    assert all(item['id'] != added['id'] for item in supplements)


@pytest.mark.parametrize('bad', [dict(execution_quality=-1), dict(execution_quality=101),
    dict(execution_quality=True), dict(visual_progress=float('nan')), dict(stuck='false'),
    dict(behavior_class='invented'), dict(failure_stage=[]), dict(vlm_agreement='yes'),
    dict(notes='x'*10001), dict(official_success=True), {}, dict(notes='  ')])
def test_invalid_scores_and_fields_do_not_write_files(dashboard, bad):
    root, base = dashboard
    with pytest.raises(HTTPError) as error:
        request(base, 'POST', run='first', episode_index=1, review=bad)
    assert error.value.code == 400
    assert not (root/'first/human_reviews').exists()


@pytest.mark.parametrize('target', [dict(run='../outside', episode_index=1),
    dict(run='first',episode_index=999),dict(run='first',episode_index=True),
    dict(run='first',episode_index=1,path='videos/gpu0/episode2.mp4'),
    dict(run='first',path='../outside.mp4')])
def test_missing_or_mismatched_identity_is_rejected(dashboard, target):
    root, base = dashboard
    with pytest.raises(HTTPError) as error:
        request(base, 'POST', **target, review=dict(notes='bad identity'))
    assert error.value.code == 400 and not list(root.rglob('human_reviews'))
