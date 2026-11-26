"""Real-model equivalence and reload checks used by the disposable training gate."""
from pathlib import Path
import torch
from .model.integration import Integrated
from .model.velocity import inference_velocity
from v1_groot_memoryvla.experiment.first_training import seeded

@torch.no_grad()
def equivalence(base,model,ds,output,seed):
    if model is None:return {"not_applicable":"no residual branch"}
    from copy import deepcopy
    old_banks=model.memory._banks
    prior=model.training
    model.memory._banks=None;model.eval();base.eval()
    try:
        # Real current + previous decision, fixed source episode and original processor.
        g=next(ds.get_group(i) for i in ds.order(seed) if len(ds.frame_indices(i))>=2 and ds.position[i]==0)
        inputs=ds.processor.collator(g["steps"][:2])["inputs"]
        bk,act=base.prepare_input(inputs)
        raw=torch.from_numpy(g["raw_q"][:2]).cuda()
        ctx={"episode_keys":[g["episode_key"]]*2,"timesteps":[0,1],"sequence_start":True}
        head=base.action_head
        with torch.autocast("cuda",dtype=torch.bfloat16):
            feat=base.backbone(bk)
            cache=model.prepare_cache(feat,bk["input_ids"],raw,ctx)
            assert cache.active.tolist()==[False,True]
            enc=head._encode_features(feat,act)
            def original():
                seeded(seed);return head.get_action(feat,act)["action_pred"]
            def corrected(m,c,gain):
                seeded(seed)
                return inference_velocity(head,enc.backbone_features,enc.state_features,act.embodiment_id,
                                          feat,m.refinement,c,m.memory.robot,gain)["action_pred"]
            ref=original()
            bypass=corrected(model,cache,0.)
            if not torch.equal(ref,bypass):raise RuntimeError("gamma=0 differs from original integration")
            zero=corrected(model,cache,1.)
            # At initialization the output head is all zero. At trained checkpoints
            # gamma=1 may differ; do not assert equality then.
            is_zero=all(torch.count_nonzero(p)==0 for p in model.refinement.output.parameters())
            if is_zero and not torch.equal(ref,zero):raise RuntimeError("zero-initialized residual differs")
            model.reset()
            empty=model.prepare_cache(feat,bk["input_ids"],raw,{**ctx,"history_mode":"empty"})
            if empty.active.any() or not torch.equal(ref,corrected(model,empty,1.)):
                raise RuntimeError("empty history does not match base")
            model.reset()
            cached=model.prepare_cache(feat,bk["input_ids"],raw,ctx)
            original_out=corrected(model,cached,1.)
            directory=Path(output)/"reload_sidecar"
            model.save_sidecar(directory)
            loaded=Integrated.load_sidecar(base,directory,"cuda")
            reloaded=loaded.prepare_cache(feat,bk["input_ids"],raw,ctx)
            reload_out=corrected(loaded,reloaded,1.)
            if not torch.equal(original_out,reload_out):raise RuntimeError("sidecar reload output mismatch")
            loaded.reset()
            del loaded
        return dict(gamma0_matches_original=True,zero_init_checked=is_zero,
                    zero_init_matches_original=bool(torch.equal(ref,zero)) if is_zero else None,
                    empty_matches_original=True,reload_matches=True,episode=g["episode"],frames=g["frame_indices"][:2])
    finally:
        model.memory._banks=old_banks;model.train(prior)
