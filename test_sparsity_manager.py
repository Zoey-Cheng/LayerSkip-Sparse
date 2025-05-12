# test_sparsity_manager.py
import copy
import torch
import transformers
import pytest

from self_speculation.sparsity_manager import SparsityManager

MODEL_ID = "facebook/layerskip-llama3.2-1B"
ALGO     = "oneshot_4to2"         # can be "oneshot_4to2" or "nvidia_asp" ...

@pytest.fixture(scope="session")
def model():
    """ load model once for all tests """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float16
    )
    model.eval()
    return model

def test_mask_and_mul(model):
    sm = SparsityManager(model, ALGO)
    sm.build_masks()

    # 1) mask only 0/1
    for name, mask in sm.masks.items():
        assert mask.dtype == torch.bool or mask.min() >= 0 and mask.max() <= 1

    # deeep copy of original weights
    orig_weights = {
        name: mod.weight.data.clone() 
        for name, mod in model.named_modules() 
        if isinstance(mod, torch.nn.Linear)
    }

    # 2) enter with block: weights * masked
    with sm.enable():
        for name, mod in model.named_modules():
            if name in sm.masks:
                w_masked = orig_weights[name] * sm.masks[name].to(orig_weights[name].device)
                assert torch.allclose(mod.weight, w_masked, atol=0, rtol=0), f"Mismatch in layer {name}"

    # 3) exit with block: weights are restored
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            assert torch.allclose(mod.weight, orig_weights[name], atol=0, rtol=0), f"Restore failed in {name}"
