# test_sparsity_manager.py
import copy
import torch
import transformers
import pytest

from self_speculation.sparsity_manager import SparsityManager

MODEL_ID = "facebook/layerskip-llama3.2-1B"
ALGO     = "oneshot_4to2"         # 可换成 random_50 / nvidia_asp 等

@pytest.fixture(scope="session")
def model():
    """一次加载模型，供整个测试文件复用"""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float16
    )
    model.eval()
    return model

def test_mask_and_mul(model):
    sm = SparsityManager(model, ALGO)
    sm.build_masks()

    # 1) mask 只含 0/1
    for name, mask in sm.masks.items():
        assert mask.dtype == torch.bool or mask.min() >= 0 and mask.max() <= 1

    # 深拷贝原始权重，用于比较
    orig_weights = {
        name: mod.weight.data.clone() 
        for name, mod in model.named_modules() 
        if isinstance(mod, torch.nn.Linear)
    }

    # 2) 进入 with 块：权重 == 原始 * mask
    with sm.enable():
        for name, mod in model.named_modules():
            if name in sm.masks:
                w_masked = orig_weights[name] * sm.masks[name].to(orig_weights[name].device)
                assert torch.allclose(mod.weight, w_masked, atol=0, rtol=0), f"Mismatch in layer {name}"

    # 3) 退出 with 块：权重恢复原状
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            assert torch.allclose(mod.weight, orig_weights[name], atol=0, rtol=0), f"Restore failed in {name}"
