# fusion.py — 静态融合策略（numpy → numpy）

import numpy as np
from abc import ABC, abstractmethod


class BaseFusion(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fuse(self, dynamic_embs, static_embs):
        raise NotImplementedError

    def output_dim(self, d1: int, d2: int) -> int:
        dummy = self.fuse(np.zeros((1, d1)), np.zeros((1, d2)))
        return dummy.shape[-1]


class ConcatFusion(BaseFusion):
    @property
    def name(self):
        return "concat"

    def fuse(self, dynamic_embs, static_embs):
        return np.concatenate([dynamic_embs, static_embs], axis=-1).astype(np.float32)


class WeightedFusion(BaseFusion):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    @property
    def name(self):
        return f"weighted_a{self.alpha}"

    def fuse(self, dynamic_embs, static_embs):
        assert dynamic_embs.shape[-1] == static_embs.shape[-1]
        return (self.alpha * dynamic_embs + (1 - self.alpha) * static_embs).astype(np.float32)


class GatedFusion(BaseFusion):
    @property
    def name(self):
        return "gated_norm"

    def fuse(self, dynamic_embs, static_embs):
        assert dynamic_embs.shape[-1] == static_embs.shape[-1]
        eps = 1e-8
        d_norm = np.linalg.norm(dynamic_embs, axis=-1, keepdims=True)
        s_norm = np.linalg.norm(static_embs, axis=-1, keepdims=True)
        gate = d_norm / (d_norm + s_norm + eps)
        return (gate * dynamic_embs + (1 - gate) * static_embs).astype(np.float32)


FUSION_REGISTRY = {
    "concat":    ConcatFusion,
    "weighted":  WeightedFusion,
    "gated_norm": GatedFusion,
}


def get_fusion(method: str, **kwargs) -> BaseFusion:
    if method not in FUSION_REGISTRY:
        raise ValueError(f"未知融合方法 '{method}'，可选: {list(FUSION_REGISTRY.keys())}")

    cls = FUSION_REGISTRY[method]

    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered_kwargs)


def list_fusion_methods():
    for name, cls in FUSION_REGISTRY.items():
        print(f"  {name:<15} → {cls.__name__}")
