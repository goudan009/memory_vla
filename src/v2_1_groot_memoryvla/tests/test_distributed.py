"""Real four-process Gloo: global denominators, idle ranks and unused AdamW parameters."""
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous):
    from v2_1_groot_memoryvla.distributed import sum_gradients
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4)
    try:
        p=torch.nn.Parameter(torch.tensor([.13,-.27,.41]))
        unused=torch.nn.Parameter(torch.tensor([.7]))
        ref=torch.nn.Parameter(p.detach().clone())
        ref_unused=torch.nn.Parameter(unused.detach().clone())
        opt=torch.optim.AdamW([p,unused],lr=1e-5)
        expected=torch.optim.AdamW([ref,ref_unused],lr=1e-5)
        # First step all ranks; tail step only rank0; empty state masks.
        for samples in (7,1,5):
            opt.zero_grad(set_to_none=True);expected.zero_grad(set_to_none=True)
            for i in range(samples):
                x=torch.tensor([i+1.,2.,-.5])
                y=torch.tensor(float(i%2))
                loss=(p.dot(x)-y).square()/samples
                if i%4==rank:loss.backward()
                (ref.dot(x)-y).square().div(samples).backward()
            sum_gradients([p,unused],bucket_bytes=8)
            assert unused.grad is None
            torch.testing.assert_close(p.grad,ref.grad,rtol=1e-6,atol=1e-6)
            torch.nn.utils.clip_grad_norm_([p,unused],1.)
            torch.nn.utils.clip_grad_norm_([ref,ref_unused],1.)
            opt.step();expected.step()
            torch.testing.assert_close(p,ref,rtol=1e-6,atol=1e-7)
            assert torch.equal(unused,ref_unused)
            # A resumed optimizer must make the next update identical.
            restored=torch.optim.AdamW([p,unused],lr=1e-5)
            restored.load_state_dict(opt.state_dict());opt=restored
    finally:
        dist.destroy_process_group()


def test_four_rank_global_update_and_idle_tail(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),),nprocs=4,join=True)
