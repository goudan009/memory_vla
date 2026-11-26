"""R1Pro physical mapping, normalized encoding, audited USD FK."""
import json
from importlib.resources import files
import torch
from torch import nn
import torch.nn.functional as F

def load_asset():
    return json.loads(files("v2_1_groot_memoryvla").joinpath("assets/r1pro.json").read_text())

def quat_matrix(q):
    x,y,z,w=F.normalize(q.float(),dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)

def transform(p,q):
    t=torch.eye(4);t[:3,:3]=quat_matrix(torch.tensor(q));t[:3,3]=torch.tensor(p)
    return t

def mlp(din,dout):
    return nn.Sequential(nn.Linear(din,256),nn.GELU(),nn.Linear(256,dout))

class R1Pro(nn.Module):
    def __init__(self,lo,hi,prior):
        super().__init__()
        self.prior=prior;asset=load_asset();channels=asset["channels"]
        lo=torch.as_tensor(lo,dtype=torch.float32);hi=torch.as_tensor(hi,dtype=torch.float32)
        if lo.shape!=(82,) or hi.shape!=(82,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (hi<lo).any():
            raise ValueError("expected training-only physical state bounds")
        self.register_buffer("lo",lo);self.register_buffer("hi",hi)
        if [c["column"] for c in channels]!=list(range(23)):raise ValueError("bad control mapping")
        self.register_buffer("columns",torch.arange(23))
        adj=torch.eye(23,dtype=torch.bool)
        for a,b in asset["edges"]:adj[a,b]=adj[b,a]=True
        self.register_buffer("local_mask",~adj)
        attrs=[]
        for c in channels:
            typ=0 if c["index"]<2 else 1 if c["index"]==2 else 3 if "gripper" in c["group"] else 2
            axis=c["axis"]; bounds=c["range"]
            valid=isinstance(bounds,list) and len(bounds)==2 and all(isinstance(x,(int,float)) for x in bounds)
            attrs.append([float(k==typ) for k in range(4)]+[float(axis==k) for k in "XYZ"]+[float(axis in ("X","Y","Z"))]+(bounds if valid else [0.,0.])+[float(valid)])
        self.register_buffer("attrs",torch.tensor(attrs,dtype=torch.float32))
        self.id_embed=nn.Embedding(23,16);self.whole=mlp(82,128)
        self.body=mlp(128+16+11+4+(10 if prior=="fk" else 0),256)
        lookup={name:i for c in channels if c["index"]>=3 for name,i in zip(c["joints"],c["q_indices"])}
        self.fk_nodes=[];pending=list(asset["joints"]);known={"base_link"}
        while True:
            ready=[a for a in pending if a["parent"] in known and a["child"] not in known]
            if not ready:break
            for a in ready:
                k=len(self.fk_nodes)
                self.register_buffer("fa"+str(k),transform(a["pos0"],a["quat0_xyzw"]))
                self.register_buffer("fb"+str(k),torch.linalg.inv(transform(a["pos1"],a["quat1_xyzw"])))
                idx=lookup.get(a["name"])
                if a["type"]!="PhysicsFixedJoint" and idx is None:raise ValueError("unmapped joint "+a["name"])
                self.fk_nodes.append((a["parent"],a["child"],a["type"],a["axis"],idx))
                known.add(a["child"]);pending.remove(a)
        self.fk_links=["base_link" if c["index"]<3 else c["children"][0] for c in channels]
        self.fk_links[14]="left_gripper_link";self.fk_links[22]="right_gripper_link"
        if not set(self.fk_links)<=known:raise ValueError("missing FK link")
    def normalize(self,q):
        span=self.hi-self.lo
        return torch.where(span>1e-6,2*(q.float()-self.lo)/span.clamp_min(1e-6)-1,0.).clamp(-1,1)
    def fk(self,q):
        with torch.autocast(device_type=q.device.type,enabled=False):
            q=q.float();eye=torch.eye(4,device=q.device).expand(len(q),4,4)
            poses={"base_link":eye}
            for k,(parent,child,kind,axis,idx) in enumerate(self.fk_nodes):
                motion=eye.clone()
                if idx is not None:
                    v=q[:,idx];a="XYZ".index(axis)
                    if kind=="PhysicsRevoluteJoint":
                        i,j=(a+1)%3,(a+2)%3
                        motion[:,i,i]=v.cos();motion[:,j,j]=v.cos()
                        motion[:,i,j]=-v.sin();motion[:,j,i]=v.sin()
                    elif kind=="PhysicsPrismaticJoint":motion[:,a,3]=v
                    else:raise ValueError(kind)
                poses[child]=poses[parent]@getattr(self,"fa"+str(k)).float()@motion@getattr(self,"fb"+str(k)).float()
            return poses
    def forward(self,q):
        q=q.float();n=self.normalize(q);b=len(q);loc=q.new_zeros((b,23,4))
        yaw=q[:,9];vx,vy=q[:,12],q[:,13]
        loc[:,0,0]=(yaw.cos()*vx+yaw.sin()*vy)/.75
        loc[:,1,0]=(-yaw.sin()*vx+yaw.cos()*vy)/.75
        loc[:,2,0]=q[:,17];loc[:,:3,2]=1
        for start,indices in [(3,range(78,82)),(7,range(18,25)),(15,range(48,55))]:
            for k,i in enumerate(indices):loc[:,start+k,0]=n[:,i];loc[:,start+k,2]=1
        for col,i in [(14,46),(22,76)]:
            loc[:,col,:2]=n[:,i:i+2];loc[:,col,2:]=1
        parts=[loc,self.attrs.expand(b,-1,-1),self.id_embed.weight.expand(b,-1,-1),self.whole(n)[:,None].expand(-1,23,-1)]
        if self.prior=="fk":
            poses=self.fk(q);t=torch.stack([poses[k] for k in self.fk_links],1)
            parts.append(torch.cat((t[:,:,:3,3],t[:,:,:3,:2].transpose(-2,-1).flatten(-2),torch.ones(b,23,1,device=q.device)),-1))
        return self.body(torch.cat(parts,-1))
