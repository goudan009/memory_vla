import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evaluation/scripts'))
from workbench_annotations import AnnotationStore, episode_keys

def test_isolation_persistence_and_clear(tmp_path):
    path = tmp_path/'notes.json'
    store = AnnotationStore(path)
    value = dict(notes='观察', stage='接近', cause='阶段决策', reason='重复动作')
    keys = {'json/radio_319_0.json'}
    key = next(iter(keys))
    store.save('b1k2026', 'run', keys, key, value)
    assert AnnotationStore(path).read('b1k2026','run',keys)[key]['notes'] == '观察'
    assert store.read('libero','run',keys) == {}
    assert store.read('b1k2026','other',keys) == {}
    with pytest.raises(ValueError):
        store.save('b1k2026','run',keys,'missing',value)
    store.save('b1k2026','run',keys,key,{k:'' for k in value})
    assert store.read('b1k2026','run',keys) == {}

def test_episode_keys_are_native_ids():
    class Launch:
        def snapshot(self):
            return [{'id':'a','kind':'b1k2026_websocket','native_episode_keys':['json/a.json']}]
    class Bench:
        def detail(self, request):
            return {'job': {'kind':'libero_eval'}, 'reports':[{'data':{'type':'libero_eval','episodes':[{'task':1,'trial':2,'initial_state_index':3}]}}]}
    assert episode_keys('b1k2026','a',Launch(),Bench()) == {'json/a.json'}
    assert episode_keys('libero','b',Launch(),Bench()) == {'task:1:trial:2:initial:3'}
    with pytest.raises(ValueError):
        episode_keys('b1k2026','b',Launch(),Bench())
