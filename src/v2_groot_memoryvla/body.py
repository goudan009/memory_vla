"""Variable-size joint tokens with local topology attention and URDF FK."""
import torch
from torch import nn


class BodyEncoder(nn.Module):
    def __init__(self, dim=128, layers=2, heads=4):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(27, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dynamic = nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.Linear(dim, dim))
        self.base = nn.Linear(6, dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, dim*2,
            dropout=0., batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, graphs, body):
        """One graph per batch member; body arrays padded on the node axis.

        Padding is inferred from graph length. Missing measured joints remain
        graph nodes with availability bits, distinct from padded nodes.
        """
        if not isinstance(graphs, (tuple, list)): graphs = [graphs]*len(body["q"])
        if len(graphs) != len(body["q"]): raise ValueError("Graph batch mismatch")
        width = body["q"].shape[1]
        encoded, masks, poses = [], [], []
        # Independent graphs make cross-robot attention impossible, and avoid
        # all-masked attention rows for padded nodes.
        for b, graph in enumerate(graphs):
            n = len(graph.names)
            if n > width: raise ValueError("Body padding is too short")
            q = body["q"][b:b+1,:n].float()
            qm = body["q_mask"][b:b+1,:n].float()
            transforms, valid = graph.fk(q, qm)
            angular = q.new_tensor([False]+[j.kind in {"revolute","continuous"} for j in graph.joints])
            continuous = q.new_tensor([False]+[j.kind == "continuous" for j in graph.joints]).bool()
            stat = graph.static.to(q.device)
            # Continuous joints have no finite position interval. Fixed nodes
            # have structural transforms, never invented measured motion.
            low, high, bounded = stat[:,13], stat[:,14], stat[:,15].bool()
            scaled = torch.where(bounded, (q-(low+high)/2)/((high-low)/2).clamp_min(1e-6), q)
            scaled = torch.where(continuous, torch.zeros_like(scaled), scaled)*qm
            velocity = body["qvel"][b:b+1,:n].float()*body["velocity_mask"][b:b+1,:n]
            # Invalid FK branches carry zero pose, with an explicit validity bit.
            pos = transforms[:,:,:3,3]*valid.unsqueeze(-1)
            rot = transforms[:,:,:3,:2].reshape(1,n,6)*valid.unsqueeze(-1)
            dynamic = torch.cat((scaled.unsqueeze(-1), (q.sin()*angular*qm).unsqueeze(-1),
                (q.cos()*angular*qm).unsqueeze(-1), velocity.unsqueeze(-1), qm.unsqueeze(-1),
                body["velocity_mask"][b:b+1,:n].unsqueeze(-1),pos,rot,valid.unsqueeze(-1)), -1)
            x = self.static(stat).unsqueeze(0) + self.dynamic(dynamic)
            x = x + self.base(body["base"][b:b+1,:6].float()).unsqueeze(1)
            for layer in self.layers:
                x = layer(x, src_mask=~graph.allowed.to(q.device))
            x = self.norm(x)
            encoded.append(torch.nn.functional.pad(x, (0,0,0,width-n)))
            masks.append(torch.arange(width,device=q.device)>=n)
            poses.append(transforms)
        tokens = torch.cat(encoded)
        padding = torch.stack(masks)
        pooled = (tokens*~padding.unsqueeze(-1)).sum(1)/(~padding).sum(1,keepdim=True)
        return {"tokens":tokens,"padding_mask":padding,"pooled":pooled,"poses":poses}
