"""2025 data migration audit and explicit native N1.7 modality candidate.

Do not pad/truncate or reinterpret the 82-D 2025 vector as the 61-D 2026 one.
Preserve all existing named state groups and absolute 23-D targets for a
separate 2025 fine-tune. This is not compatible with the 2026 radio weights.
"""
from pathlib import Path
from .contracts import read, ACTION_GROUPS

STATE_WIDTHS = {"robot_pos":3,"robot_ori_cos":3,"robot_ori_sin":3,
                "robot_2d_ori":1,"robot_2d_ori_cos":1,"robot_2d_ori_sin":1,
                "robot_lin_vel":3,"robot_ang_vel":3,"arm_left_qpos":7,
                "arm_left_qpos_sin":7,"arm_left_qpos_cos":7,"eef_left_pos":3,
                "eef_left_quat":4,"gripper_left_qpos":2,"arm_right_qpos":7,
                "arm_right_qpos_sin":7,"arm_right_qpos_cos":7,"eef_right_pos":3,
                "eef_right_quat":4,"gripper_right_qpos":2,"trunk_qpos":4}
RELATIVE = {"torso":"trunk_qpos","left_arm":"arm_left_qpos","right_arm":"arm_right_qpos"}
CAMERAS = ["observation.images.rgb."+k+"_256_256" for k in ("head","left_wrist","right_wrist")]
LANGUAGE = "annotation.human.coarse_action"


def audit_2025(dataset):
    dataset=Path(dataset)
    info=read(dataset/"meta/info.json")
    modality=read(dataset/"meta/modality.json")
    if info["fps"]!=30 or info["features"]["observation.state"]["shape"]!=[82] or info["features"]["action"]["shape"]!=[23]:
        raise ValueError("expected original converted B1K2025 30Hz 82/23")
    expected={};offset=0
    for key,width in STATE_WIDTHS.items():
        expected[key]=(offset,offset+width);offset+=width
    assert offset==82
    for group,layout in (("state",expected),("action",ACTION_GROUPS)):
        if set(modality[group])!=set(layout):
            raise ValueError("unexpected 2025 named group: "+group)
        for key,(lo,hi) in layout.items():
            spec=modality[group][key]
            if (spec["start"],spec["end"])!=(lo,hi):
                raise ValueError("2025 group offset mismatch: "+key)
    if set(modality["video"])!=set(CAMERAS) or "human.coarse_action" not in modality["annotation"]:
        raise ValueError("2025 camera/language key mismatch")
    return dict(state_dim=82,action_dim=23,fps=30,state_groups=list(expected),
                action_groups=list(ACTION_GROUPS),video_keys=CAMERAS,language_key=LANGUAGE,
                relative_state_keys=RELATIVE,format="lerobot_v2_1_b1k2025",
                needs_new_statistics=True,compatible_with_official_2026_radio=False,
                warning="82-D packed state lacks the explicitly verified 2026 robot-local base_qvel group")


def modality_source(dataset, *, horizon):
    """Produce a native N1.7 NEW_EMBODIMENT config, with no disk side effects.

    The chosen action horizon is a scientific choice, not a default.
    The workbench must save this exact source inside the job snapshot and
    recompute train-only stats using it. No old N1.6 statistics reuse.
    """
    c=audit_2025(dataset)
    if type(horizon) is not int or not 1<=horizon<=40:
        raise ValueError("explicit supervised horizon within N1.7 base 40 required")
    return (
        "from gr00t.configs.data.embodiment_configs import register_modality_config\n"
        "from gr00t.data.embodiment_tags import EmbodimentTag\n"
        "from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig\n"
        "config = {\n"
        f"    'video': ModalityConfig(delta_indices=[0], modality_keys={c['video_keys']!r}),\n"
        f"    'state': ModalityConfig(delta_indices=[0], modality_keys={c['state_groups']!r}),\n"
        f"    'language': ModalityConfig(delta_indices=[0], modality_keys={[c['language_key']]!r}),\n"
        f"    'action': ModalityConfig(delta_indices=list(range({horizon})), modality_keys={c['action_groups']!r}, action_configs=[\n"
        + "".join(
            "        ActionConfig(rep=ActionRepresentation."
            +("RELATIVE" if key in RELATIVE else "ABSOLUTE")
            +", type=ActionType.NON_EEF, format=ActionFormat.DEFAULT"
            +(f", state_key={RELATIVE[key]!r}" if key in RELATIVE else "")
            +(", is_gripper=True" if "gripper" in key else "")+"),\n"
            for key in ACTION_GROUPS)
        +"    ]),\n}\nregister_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)\n"
    )
