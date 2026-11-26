import gymnasium as gym
from omnigibson.envs import EnvironmentWrapper
from omnigibson.learning.utils.task_progress_utils import CHALLENGE_TASKS_PROGRESS_APPROXIMATION

try:
    from groot_eval_timing import profile_scope
except ImportError:
    from contextlib import nullcontext

    def profile_scope(_name):
        return nullcontext()


class TaskProgressWrapper(EnvironmentWrapper, gym.Wrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def step(self, action, n_render_iterations=1):
        # Run super first
        obs, reward, terminated, truncated, info = super().step(action, n_render_iterations=n_render_iterations)

        assert (
            self.env.task.activity_name in CHALLENGE_TASKS_PROGRESS_APPROXIMATION
        ), f"Task {self.env.task.activity_name} not supported in TaskProgressWrapper!"

        # Get approximated task progress
        progress_fn = CHALLENGE_TASKS_PROGRESS_APPROXIMATION[self.env.task.activity_name]
        with profile_scope("task_progress.update"):
            checks = progress_fn(self.env)

        # Add to info
        info["task_progress"] = checks

        return obs, reward, terminated, truncated, info
