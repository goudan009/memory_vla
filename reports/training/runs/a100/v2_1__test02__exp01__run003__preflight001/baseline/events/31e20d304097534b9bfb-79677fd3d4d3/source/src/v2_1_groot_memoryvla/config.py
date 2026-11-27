from dataclasses import dataclass, asdict
import math

CORE_STATES = ("door_open", "door_ever_opened", "bag_grasped", "bag_inside",
               "bag_ever_grasped", "bag_ever_inside", "started", "cooked_popcorn_exists")
ALL_STATES = ("door_open", "started", "cooked_popcorn_exists", "raw_popcorn_exists",
              "bag_inside", "bag_grasped_left", "bag_grasped_right", "bag_grasped",
              "door_closed", "door_ever_opened", "bag_ever_inside", "bag_ever_grasped",
              "microwave_ever_started", "cooked_popcorn_ever_observed", "door_reclosed_after_open",
              "door_ever_reclosed_after_open", "bag_inside_with_door_closed",
              "bag_released_inside_candidate", "microwave_on_with_closed_door",
              "microwave_on_with_open_door", "bag_inside_while_on")

def check_scale(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1:
        raise ValueError("residual_scale must be a finite number in [0,1]")
    return float(x)

@dataclass(frozen=True)
class Config:
    prior: str = "topology"
    fusion_layers: int = 0
    states: tuple = CORE_STATES
    capacity: int = 64
    alpha: float = .2
    retrieval_lambda: float = 1.
    state_weight: float = .01
    weak_weight: float = .5
    residual_scale: float = 1.
    def __post_init__(self):
        object.__setattr__(self,"states",tuple(self.states))
        if self.prior not in ("topology","fk","none") or self.fusion_layers not in (0,1,2) or (self.prior!="none" and self.fusion_layers):
            raise ValueError("invalid prior")
        if not self.states or len(set(self.states))!=len(self.states) or not set(self.states)<=set(ALL_STATES):
            raise ValueError("unknown or duplicate labels")
        if self.capacity<1 or not 0<=self.alpha<=1 or not math.isfinite(self.retrieval_lambda) or self.retrieval_lambda<0:
            raise ValueError("invalid retrieval configuration")
        if not 0<=self.weak_weight<=1 or not math.isfinite(self.state_weight) or self.state_weight<0:
            raise ValueError("invalid loss weight")
        check_scale(self.residual_scale)
    def as_dict(self):
        return asdict(self)
