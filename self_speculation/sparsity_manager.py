# sparsity_manager.py
import contextlib, functools, torch, os
from torch.nn.utils import prune
import pytest

# ────────────────────────────────
# 1️⃣  Different pruning algorithms
#     - input: torch.nn.Module
# ────────────────────────────────

def prune_magnitude_4to2(module: torch.nn.Module) -> torch.Tensor:
    """ use 4:2 pruning """
    w = module.weight.detach()
    grouped = w.view(-1, 4)
    top2 = grouped.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(1, top2, 1)
    return mask.view_as(w)

def prune_nvidia_asp(module: torch.nn.Module) -> torch.Tensor:
    """
    called by NVIDIA's ASP library
    """
    import asp  # need pip install nvidia-asp
    mask = asp.prune_magnitude(module.weight, N=4, M=2)  # ex: 4:2
    return mask.bool()

# ────────────────────────────────
# 2️⃣  Registry
# ────────────────────────────────
ALGO_REGISTRY = {
    "oneshot_4to2": prune_magnitude_4to2,
    "nvidia_asp": prune_nvidia_asp,          
}

# ────────────────────────────────
# 3️⃣  SparsityManager
# ────────────────────────────────
class SparsityManager:
    def __init__(self, model: torch.nn.Module, algo: str):
        if algo not in ALGO_REGISTRY:
            raise ValueError(f"[SparsityManager] Unknown algo '{algo}'. "
                             f"Available: {list(ALGO_REGISTRY)}")
        self.model = model
        self.algo_name = algo
        self.prune_fn = ALGO_REGISTRY[algo]
        self.masks: dict[str, torch.Tensor] = {}
        self._orig: dict[str, torch.Tensor] = {}
        self._enabled = False

    # ---------- generate mask ----------
    def build_masks(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                self.masks[name] = self.prune_fn(mod).to(mod.weight.device)

    # ---------- Hook management ----------
    def _apply(self):
        for name, mod in self.model.named_modules():
            if name in self.masks:
                self._orig[name] = mod.weight.data.clone()
                mod.weight.data *= self.masks[name]

    def _restore(self):
        for name, mod in self.model.named_modules():
            if name in self._orig:
                mod.weight.data.copy_(self._orig[name])
        self._orig.clear()

    @contextlib.contextmanager
    def enable(self):
        if not self.masks:
            self.build_masks()
        if not self._enabled:
            self._apply()
            self._enabled = True
        try:
            yield
        finally:
            if self._enabled:
                self._restore()
                self._enabled = False
