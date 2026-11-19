from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class V2Config:
    variant: str = "v2c"
    body_dim: int = 128
    body_layers: int = 2
    body_heads: int = 4
    memory_dim: int = 256
    memory_capacity: int = 64
    # None inherits trained V1; a no-memory GR00T starts with direct / 64.
    memory_conditioning: str | None = None
    memory_length: int | None = None
    # b and c instantiate the same modules; only query's body input differs.
    injection_scale: float = 0.01

    def __post_init__(self):
        if self.variant not in {"baseline", "v1", "v2a", "v2b", "v2c"}:
            raise ValueError(self.variant)
        if self.body_dim % self.body_heads or self.memory_capacity < 1:
            raise ValueError("Invalid body heads or memory capacity")
        if self.memory_conditioning not in {None,"direct","residual"} or (self.memory_length is not None and self.memory_length<1):
            raise ValueError("Invalid V1 conditioning or memory length")

    @property
    def body(self):
        return self.variant in {"v2a", "v2b", "v2c"}

    @property
    def history(self):
        return self.variant in {"v2b", "v2c"}

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        values=json.loads(Path(path).read_text())
        # Checkpoints written before this synchronization used capacity 16.
        values.setdefault("memory_capacity",16)
        return cls(**values)
