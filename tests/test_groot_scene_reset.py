"""A new task instance must not adopt toppled background furniture."""
import copy
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation/scripts'))


def test_settling_drift_does_not_replace_background_baseline(monkeypatch):
    from groot_scene_reset import install_scene_reset

    class Scene:
        def __init__(self):
            self._initial_file = {'metadata': {'instance': -1}, 'state': {'registry': {
                'object_registry': {'chair': {'pose': [0]}, 'bag': {'pose': [1]},
                                    'robot': {'pose': [2]}},
                'system_registry': {'water': [0]}}}}
            self.live = copy.deepcopy(self._initial_file)

        def update_initial_file(self, scene_file=None):
            self._initial_file = copy.deepcopy(self.live if scene_file is None else scene_file)

    class Env:
        def __init__(self, task_name):
            self.env = SimpleNamespace(scene=Scene(), robots=[SimpleNamespace(name='robot')],
                task=SimpleNamespace(object_scope={
                    'bag.n.01_1': SimpleNamespace(name='bag', exists=True, is_system=False)}))

    def load_instance(env, robot, instance_id):
        env.scene.live['state']['registry']['object_registry'].update(
            chair={'pose': [99]}, bag={'pose': [instance_id]}, robot={'pose': [instance_id + 10]})
        env.scene.live['state']['registry']['system_registry']['water'] = [instance_id]
        env.scene.live['metadata']['instance'] = instance_id
        env.scene.update_initial_file()
        return 'loaded'

    behavior = SimpleNamespace(BEHAVIORGr00tEnv=Env, load_task_instance_for_env=load_instance)
    monkeypatch.setitem(sys.modules, 'gr00t.eval.sim.BEHAVIOR.behavior_env', behavior)
    # A spawned worker first creates its environment, then installs the hook.
    wrapped = Env('make_microwave_popcorn')
    install_scene_reset(wrapped)
    install_scene_reset()  # The eval factory may install the hook more than once.
    env = wrapped.env
    original_update = env.scene.update_initial_file
    for instance in (1, 4):
        assert behavior.load_task_instance_for_env(env, env.robots[0], instance) == 'loaded'
        state = env.scene._initial_file['state']['registry']
        assert state['object_registry'] == {
            'chair': {'pose': [0]}, 'bag': {'pose': [instance]}, 'robot': {'pose': [instance + 10]}}
        assert state['system_registry']['water'] == [instance]
        assert env.scene._initial_file['metadata']['instance'] == instance
        assert env.scene.update_initial_file == original_update
        state['object_registry']['chair']['pose'][0] = 77

    other = Env('another_task').env
    behavior.load_task_instance_for_env(other, other.robots[0], 3)
    assert other.scene._initial_file['state']['registry']['object_registry']['chair']['pose'] == [99]
