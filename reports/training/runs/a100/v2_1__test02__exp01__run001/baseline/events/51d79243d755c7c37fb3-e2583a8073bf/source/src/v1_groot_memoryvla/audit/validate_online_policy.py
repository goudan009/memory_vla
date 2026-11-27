"""Actual saved v1 Policy: BF16 inference, batched environments, partial Reset."""

import argparse
import json
from pathlib import Path

import numpy as np

from v1_groot_memoryvla.audit.train_official_batch import load_real_step
from v1_groot_memoryvla.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from v1_groot_memoryvla.data.types import EmbodimentTag
from v1_groot_memoryvla.policy.gr00t_policy import Gr00tPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path,
        default=Path("outputs/v1_groot_memoryvla/validation/acceptance/online_policy.json"))
    args = parser.parse_args()
    policy = Gr00tPolicy(EmbodimentTag.BEHAVIOR_R1_PRO, str(args.checkpoint), device="cuda")
    loader = LeRobotEpisodeLoader(args.dataset, policy.modality_configs, video_backend="pyav")
    episode = loader.episodes_metadata[0]["episode_index"]
    step = load_real_step(loader, episode, 0)
    options = {"session_id": "online-test", "env_ids": [0, 1], "episode_ids": ["e0", "e0"]}
    observation = {"video": {k: np.stack([np.asarray(v)] * 2) for k, v in step.images.items()},
                   "state": {k: np.stack([v] * 2) for k, v in step.states.items()},
                   "language": {policy.language_key: [[step.text], [step.text]]}}
    shapes = {}
    for decision in range(2):
        actions, _ = policy.get_action(observation, options)
        for name, action in actions.items():
            assert np.isfinite(action).all() and action.shape[0] == 2
            shapes[name] = list(action.shape)
        for bank in (policy.model.memory.perceptual, policy.model.memory.cognitive):
            assert len(bank.bank[("online-test", "0", "e0")]) == decision + 1
            assert len(bank.bank[("online-test", "1", "e0")]) == decision + 1
    policy.reset({"session_id": "online-test", "env_ids": [0], "episode_ids": ["e1"]})
    for bank in (policy.model.memory.perceptual, policy.model.memory.cognitive):
        assert ("online-test", "0", "e0") not in bank.bank
        assert len(bank.bank[("online-test", "1", "e0")]) == 2
    options["episode_ids"][0] = "e1"
    policy.get_action(observation, options)
    for bank in (policy.model.memory.perceptual, policy.model.memory.cognitive):
        assert len(bank.bank[("online-test", "0", "e1")]) == 1
        assert len(bank.bank[("online-test", "1", "e0")]) == 3
    policy.reset()
    assert not policy.model.memory.perceptual.bank and not policy.memory_state.slots
    result = {"real_bf16_policy_inference": True, "partial_environment_reset": True,
              "one_write_per_decision": True, "full_reset": True, "action_shapes": shapes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
