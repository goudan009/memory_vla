"""Task-independent nullable predicate algebra and causal history.

Quality: 0 unknown, 1 certified recorded/source predicate, 2 weak inference.
No action-type or episode-time heuristic is used. Histories reset per episode.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation


@dataclass
class Label:
    value: np.ndarray
    quality: np.ndarray

    def __post_init__(self):
        self.value=np.asarray(self.value,dtype=bool)
        self.quality=np.broadcast_to(np.asarray(self.quality,dtype=np.uint8),self.value.shape).copy()
        assert self.value.shape==self.quality.shape and np.isin(self.quality,[0,1,2]).all()

    def columns(self,name):
        return {name:pd.arrays.BooleanArray(self.value,self.quality==0),
            name+'_quality':self.quality,name+'_valid':self.quality>0,
            name+'_certified_valid':self.quality==1,name+'_weak_valid':self.quality==2}


def negate(label):
    return Label(~label.value,label.quality)


def conjunction(*labels):
    """Kleene AND, preserving strongest sufficient evidence for false."""
    values=np.stack([s.value for s in labels]);q=np.stack([s.quality for s in labels])
    known=q>0;false=known&~values
    result=np.all(values,axis=0)
    quality=np.zeros(result.shape,dtype=np.uint8)
    all_known=known.all(0)
    quality[all_known]=np.where((q[:,all_known]==2).any(0),2,1)
    quality[false.any(0)]=2
    quality[((q==1)&~values).any(0)]=1
    result[false.any(0)]=False
    return Label(result,quality)


def disjunction(*labels):
    return negate(conjunction(*(negate(s) for s in labels)))


def ever(label):
    """Certified positive witness dominates unknown/weak past; never future fills."""
    value=np.zeros(len(label.value),bool);quality=np.zeros(len(value),np.uint8)
    certified_positive=False;weak_positive=False;all_known=True;any_weak=False
    for i,(v,q) in enumerate(zip(label.value,label.quality)):
        certified_positive|=bool(v and q==1);weak_positive|=bool(v and q==2)
        all_known&=q>0;any_weak|=q==2
        if certified_positive:value[i]=True;quality[i]=1
        elif weak_positive:value[i]=True;quality[i]=2
        elif all_known:quality[i]=2 if any_weak else 1
    return Label(value,quality)


def transform_points(points,pose,scale):
    rotation=Rotation.from_quat(pose[:,3:7]).as_matrix()
    return np.einsum('tij,pj->tpi',rotation,points*np.asarray(scale))+pose[:,None,:3]


def inside_candidate(inner_pose,outer_pose,inner_points,container_points,inner_scale,outer_scale,boundary_m):
    """Actual OG Inside definition: inner world-AABB center in container volume.

    Caller must prove container convex hull lies inside outer collision convex hull
    (making outer-AABB test redundant), or separately evaluate that AABB gate.
    Asset identity controls certification outside this numerical adapter.
    """
    world=transform_points(inner_points,inner_pose,inner_scale)
    center=(world.min(1)+world.max(1))/2
    outer_rotation=Rotation.from_quat(outer_pose[:,3:7]).as_matrix()
    local=np.einsum('tji,tj->ti',outer_rotation,center-outer_pose[:,:3])/outer_scale
    volume=container_points*np.asarray(outer_scale)
    scaled_local=local*np.asarray(outer_scale)
    planes=ConvexHull(volume).equations
    # Positive is minimum inward signed distance, in physical metres.
    margin=-(scaled_local@planes[:,:3].T+planes[:,3]).max(1)
    candidate=Delaunay(container_points).find_simplex(local)>=0
    assert np.array_equal(candidate,margin>=-1e-10)
    quality=np.where(np.abs(margin)>boundary_m,2,0)
    return Label(candidate,quality),center,local,margin


def grasp_candidate(bag_pose,eef_pose,finger_positions,config):
    """Causal, geometry/motion based weak evidence; NOT AG runtime truth.

    A positive requires past bag+EEF motion with a stable relative transform.
    It can persist at rest only while that witnessed transform and closed jaws
    persist. Negative requires far distance or both fully open fingers.
    """
    bag_rotation=Rotation.from_quat(bag_pose[:,3:7])
    eef_rotation=Rotation.from_quat(eef_pose[:,3:7])
    relative_position=eef_rotation.inv().apply(bag_pose[:,:3]-eef_pose[:,:3])
    relative_rotation=eef_rotation.inv()*bag_rotation
    distance=np.linalg.norm(relative_position,axis=1)
    # Contacts can stop the independently simulated fingers asymmetrically.
    # ep400450: [0.04205m, ~0m] while bag and EEF move rigidly together.
    # Requiring BOTH fingers below 4cm incorrectly rejects this real sample.
    closed=finger_positions.min(1)<config['closed_finger_m']
    open_jaw=finger_positions.min(1)>config['open_finger_m']
    far=distance>config['far_distance_m']
    n=len(distance);window=config['motion_window_frames']
    quality=np.zeros(n,np.uint8);value=np.zeros(n,bool);reason=np.zeros(n,np.uint8)
    relative_error=np.full(n,np.nan);angular_error=np.full(n,np.nan);bag_motion=np.full(n,np.nan);eef_motion=np.full(n,np.nan)
    latch=False;anchor_pos=None;anchor_rot=None
    for t in range(n):
        if t>=window:
            relative_error[t]=np.max(np.linalg.norm(relative_position[t-window:t+1]-relative_position[t],axis=1))
            angular_error[t]=np.max((relative_rotation[t].inv()*relative_rotation[t-window:t+1]).magnitude())
            bag_motion[t]=np.linalg.norm(bag_pose[t,:3]-bag_pose[t-window,:3])
            eef_motion[t]=np.linalg.norm(eef_pose[t,:3]-eef_pose[t-window,:3])
        if latch:
            latch=bool(closed[t] and distance[t]<config['near_distance_m'] and
                np.linalg.norm(relative_position[t]-anchor_pos)<config['hold_relative_error_m'] and
                (anchor_rot.inv()*relative_rotation[t]).magnitude()<config['hold_relative_error_rad'])
        witness=bool(t>=window and closed[t-window:t+1].all() and distance[t]<config['near_distance_m'] and
            relative_error[t]<config['relative_error_m'] and angular_error[t]<config['relative_error_rad'] and
            bag_motion[t]>config['min_motion_m'] and eef_motion[t]>config['min_motion_m'])
        if witness:
            latch=True;anchor_pos=relative_position[t].copy();anchor_rot=relative_rotation[t];reason[t]=1
        elif latch:reason[t]=2
        if latch:value[t]=True;quality[t]=2
        elif far[t] or open_jaw[t]:quality[t]=2;reason[t]=3 if far[t] else 4
    return Label(value,quality),dict(distance_m=distance,relative_error_m=relative_error,
        relative_error_rad=angular_error,bag_motion_m=bag_motion,eef_motion_m=eef_motion,reason=reason)
