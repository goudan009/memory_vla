"""URDF graph, geometry descriptors, and differentiable FK in base_link frame.

No simulator import or world-state dependency. Missing movable joints retain an
explicit validity mask; their default-zero transforms never masquerade as data.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


def numbers(element, key, default):
    return [float(x) for x in element.get(key, default).split()] if element is not None else [float(x) for x in default.split()]


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = roll.cos(), roll.sin(), pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin()
    return torch.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                        sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                        -sp, cp*sr, cp*cr), -1).reshape(*rpy.shape[:-1], 3, 3)


def quat_matrix(xyzw):
    q = xyzw / xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    xyz: list
    rpy: list
    axis: list
    limits: list
    has_limits: bool


class RobotGraph:
    kinds = ("fixed", "revolute", "continuous", "prismatic")

    def __init__(self, urdf, extra_fixed=()):
        self.urdf = str(Path(urdf).resolve())
        self.sha256 = hashlib.sha256(Path(urdf).read_bytes()).hexdigest()
        xml = ET.parse(urdf).getroot()
        links = {e.attrib["name"]: e for e in xml.findall("link")}
        joints = []
        for e in xml.findall("joint"):
            if e.attrib["type"] not in self.kinds or e.find("mimic") is not None:
                raise ValueError("Unsupported joint type/mimic must be explicitly resolved")
            origin, limit = e.find("origin"), e.find("limit")
            bounded = e.attrib["type"] in {"revolute", "prismatic"} and limit is not None
            joints.append(Joint(e.attrib["name"], e.find("parent").attrib["link"],
                                e.find("child").attrib["link"], e.attrib["type"],
                                numbers(origin, "xyz", "0 0 0"), numbers(origin, "rpy", "0 0 0"),
                                numbers(e.find("axis"), "xyz", "1 0 0"),
                                [float(limit.get("lower", 0)), float(limit.get("upper", 0))] if bounded else [0, 0], bounded))
        for e in extra_fixed:
            if e["child"] in links:
                raise ValueError("Extra link already exists in URDF")
            links[e["child"]] = ET.Element("link")
            joints.append(Joint(e["name"], e["parent"], e["child"], "fixed", e["xyz"], e["rpy"], [1,0,0], [0,0], False))
        roots = set(links) - {j.child for j in joints}
        if len(roots) != 1:
            raise ValueError("Expected a connected URDF tree")
        self.root = roots.pop()
        self.joints, self.links = [], [self.root]
        pending = list(joints)
        while pending:
            ready = [j for j in pending if j.parent in self.links]
            if not ready:
                raise ValueError("Cycle or disconnected URDF")
            for j in ready:
                if j.child in self.links:
                    raise ValueError("Multiple parents")
                self.joints.append(j)
                self.links.append(j.child)
                pending.remove(j)
        self.names = ["__root__"] + [j.name for j in self.joints]
        self.index = {name: i for i, name in enumerate(self.names)}
        self.link_index = {name: i for i, name in enumerate(self.links)}
        self.parents = [-1] + [self.link_index[j.parent] for j in self.joints]
        descriptions = []
        for i, link in enumerate(self.links):
            j = self.joints[i-1] if i else Joint("__root__", "", link, "fixed", [0]*3, [0]*3, [0]*3, [0]*2, False)
            e = links[link]
            mass = e.find("inertial/mass")
            inertia = e.find("inertial/inertia")
            geom = e.find("visual/geometry")
            primitive = next(iter(geom), None) if geom is not None else None
            geometry = [0.,0.,0.]
            if primitive is not None:
                if primitive.tag == "mesh": geometry = numbers(primitive, "scale", "1 1 1")
                elif primitive.tag == "box": geometry = numbers(primitive, "size", "0 0 0")
                elif primitive.tag == "sphere": geometry = [float(primitive.get("radius"))]*3
                elif primitive.tag == "cylinder": geometry = [float(primitive.get("radius")),float(primitive.get("length")),0.]
            kind = [float(j.kind == k) for k in self.kinds]
            descriptors = kind + j.axis + j.xyz + j.rpy + j.limits + [float(j.has_limits)]
            descriptors += [float(mass.get("value")) if mass is not None else 0.]
            descriptors += [float(inertia.get(k, 0)) if inertia is not None else 0. for k in ("ixx","ixy","ixz","iyy","iyz","izz")]
            descriptors += geometry + [float(primitive is not None)]
            descriptions.append(descriptors)
        self.static = torch.tensor(descriptions, dtype=torch.float32)
        n = len(self.names)
        self.allowed = torch.eye(n, dtype=torch.bool)
        for child, parent in enumerate(self.parents):
            if parent >= 0:
                self.allowed[child,parent] = self.allowed[parent,child] = True

    def fk(self, q, observed=None):
        """Return [B,N,4,4], valid [B,N], each child link in URDF root frame."""
        if q.shape[-1] != len(self.names): raise ValueError("Joint count mismatch")
        observed = torch.ones_like(q, dtype=torch.bool) if observed is None else observed.bool()
        transforms = [torch.eye(4, device=q.device, dtype=q.dtype).expand(q.shape[0],4,4)]
        validity = [torch.ones(q.shape[0], device=q.device, dtype=torch.bool)]
        for i, j in enumerate(self.joints, 1):
            axis = q.new_tensor(j.axis)
            axis = axis/axis.norm().clamp_min(1e-8)
            xyz = q.new_tensor(j.xyz).expand(q.shape[0],3)
            rotation = rpy_matrix(q.new_tensor(j.rpy)).expand(q.shape[0],3,3)
            if j.kind in {"revolute","continuous"}:
                x,y,z = axis.unbind()
                zero = x*0
                skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
                angle = q[:,i,None,None]
                motion = torch.eye(3, device=q.device, dtype=q.dtype) + angle.sin()*skew + (1-angle.cos())*(skew@skew)
                rotation = rotation @ motion
            elif j.kind == "prismatic":
                xyz = xyz + (rotation @ (q[:,i,None]*axis).unsqueeze(-1)).squeeze(-1)
            top = torch.cat((rotation,xyz.unsqueeze(-1)), -1)
            bottom = q.new_tensor([0,0,0,1]).expand(q.shape[0],1,4)
            local = torch.cat((top,bottom), -2)
            transforms.append(transforms[self.parents[i]] @ local)
            validity.append(validity[self.parents[i]] & (observed[:,i] | (j.kind == "fixed")))
        return torch.stack(transforms,1), torch.stack(validity,1)
