"""Keep Task40 background objects out of the mutable per-instance baseline."""
import copy
from functools import wraps
import importlib


def install_scene_reset(environment=None):
    # AsyncVectorEnv spawns fresh interpreters. The recording factory installs
    # here in each worker, after construction but before its first reset.
    if environment is not None:
        capture_baseline(environment)
    behavior = importlib.import_module('gr00t.eval.sim.BEHAVIOR.behavior_env')
    cls = behavior.BEHAVIORGr00tEnv
    if getattr(cls, '_groot_scene_reset_installed', False):
        return
    original_init = cls.__init__
    original_load = behavior.load_task_instance_for_env

    @wraps(original_init)
    def initialize(self, task_name, *args, **kwargs):
        original_init(self, task_name, *args, **kwargs)
        if task_name == 'make_microwave_popcorn':
            capture_baseline(self)

    @wraps(original_load)
    def load_instance(env, robot, instance_id):
        scene = env.scene
        baseline = getattr(scene, '_groot_background_baseline', None)
        if baseline is None:
            return original_load(env, robot, instance_id)
        instance_objects = {entity.name for entity in env.task.object_scope.values()
                            if not entity.is_system and entity.exists}
        instance_objects.update(robot.name for robot in env.robots)
        original_update = scene.update_initial_file

        @wraps(original_update)
        def update_instance(*args, **kwargs):
            result = original_update(*args, **kwargs)
            saved = scene._initial_file['state']['registry']['object_registry']
            for name, state in baseline.items():
                if name not in instance_objects:
                    saved[name] = copy.deepcopy(state)
            return result

        # The upstream loader saves after 25 physics steps, then immediately
        # resets from that save. Replace background states before that reset,
        # keeping the current TRO, robot, systems and instance metadata intact.
        scene.update_initial_file = update_instance
        try:
            return original_load(env, robot, instance_id)
        finally:
            scene.update_initial_file = original_update

    cls.__init__ = initialize
    cls._groot_scene_reset_installed = True
    behavior.load_task_instance_for_env = load_instance


def capture_baseline(environment):
    scene = environment.env.scene
    if not hasattr(scene, '_groot_background_baseline'):
        scene._groot_background_baseline = copy.deepcopy(
            scene._initial_file['state']['registry']['object_registry'])
        print('TASK40_SCENE_RESET baseline=constructor background=immutable', flush=True)
