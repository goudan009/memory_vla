"""Thin native model adapters; no private server or scheduler.

The workbench must own the process, allocated GPU, source snapshot and
checkpoint binding. Call these functions only inside that process.
"""
from pathlib import Path
from .contracts import inspect_upstream, inspect_checkpoint, checkpoint_digest, B1K2025
from .data_contract import STATE_WIDTHS, CAMERAS, LANGUAGE, ACTION_GROUPS


def _native_import(upstream):
    import sys
    root=Path(inspect_upstream(upstream)["root"])
    # Never reuse an already imported N1.6 gr00t package.
    if "gr00t" in sys.modules:
        module=sys.modules["gr00t"]
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("wrong gr00t runtime; use the isolated N1.7 environment")
    sys.path.insert(0,str(root))
    return root


def official_radio_policy(upstream,checkpoint,device,*,verified_radio_sha256):
    """Unmodified official 2026 wrapper; temporal ensembling is not RTC."""
    root=_native_import(upstream)
    inspect_checkpoint(checkpoint,role="official_2026_radio")
    if checkpoint_digest(checkpoint)!=verified_radio_sha256:
        raise ValueError("official radio download identity mismatch")
    from gr00t.eval.eval_b1k_wrapper import B1KPolicyWrapper,load_modality_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from .contracts import read
    modality=root/"examples/b1k/r1pro.py"
    load_modality_config(str(modality))
    tag=EmbodimentTag.NEW_EMBODIMENT
    policy=Gr00tPolicy(embodiment_tag=tag,model_path=str(checkpoint),device=device,strict=True)
    return B1KPolicyWrapper(policy=policy,embodiment_tag=tag,
        modality_config=read(modality.with_suffix(".json")),control_mode="temporal_ensemble")


def b1k2025_policy(upstream,checkpoint,device,*,horizon):
    """Candidate native flat-observation adapter for a separately trained 2025 model.

    Requires its own explicit lineage marker and matching processor. Never
    load the 2026 radio checkpoint into the existing B1K2025 evaluator.
    Transport and full simulator compatibility must be locally verified.
    """
    root=_native_import(upstream)
    from .contracts import read
    checkpoint=Path(checkpoint)
    lineage=read(checkpoint/"v1_2_baseline.json")
    if (lineage.get("protocol")!=B1K2025 or lineage.get("version")!="V_1_2"
        or lineage.get("horizon")!=horizon):
        raise ValueError("not a compatible V1.2 B1K2025 checkpoint")
    inspect_checkpoint(checkpoint,role="family_only")  # model-family + required artifacts only
    from gr00t.policy.gr00t_policy import Gr00tPolicy,Gr00tSimPolicyWrapper
    from gr00t.data.embodiment_tags import EmbodimentTag
    policy=Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                       model_path=str(checkpoint),device=device,strict=True)
    expected={"state":list(STATE_WIDTHS),"video":CAMERAS,
              "action":list(ACTION_GROUPS),"language":[LANGUAGE]}
    for kind,keys in expected.items():
        config=policy.modality_configs[kind]
        if list(config.modality_keys)!=keys or list(config.delta_indices)!=(list(range(horizon)) if kind=="action" else [0]):
            raise ValueError("incompatible 2025 native processor: "+kind)
    return Gr00tSimPolicyWrapper(policy,strict=True)
