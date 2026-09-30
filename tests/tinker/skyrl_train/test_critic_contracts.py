"""CPU regressions for the FSDP critic paths used by PPO pretraining."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from skyrl.backends.skyrl_train_backend import FSDPBackendOverrides, SkyRLTrainBackend
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    get_llm_for_sequence_regression,
)


def test_value_head_is_trainable_with_lora_and_gets_a_gradient(tmp_path):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    model = get_llm_for_sequence_regression(
        str(tmp_path),
        "critic",
        bf16=False,
        lora_rank=2,
        target_modules=["q_proj", "v_proj"],
        init_value_head=True,
    )
    assert model.value_head.weight.requires_grad
    ids = torch.tensor([[1, 2, 3, 4]])
    values = model(ids, num_actions=3, attention_mask=torch.ones_like(ids))
    assert values.shape == (1, 3)
    values.square().mean().backward()
    assert model.value_head.weight.grad is not None
    assert torch.isfinite(model.value_head.weight.grad).all()
    assert model.value_head.weight.grad.abs().sum() > 0


def test_cache_release_only_on_shared_training_role_changes():
    backend = SimpleNamespace(config=SimpleNamespace(policy_gpu_fraction=0.55), _dispatch=Mock())
    for role in ["critic", "critic", "policy", "policy", "critic"]:
        SkyRLTrainBackend._release_shared_gpu_cache(backend, role)
    assert backend._dispatch.empty_cache.call_count == 3
    backend = SimpleNamespace(config=SimpleNamespace(policy_gpu_fraction=None), _dispatch=Mock())
    SkyRLTrainBackend._release_shared_gpu_cache(backend, "critic")
    backend._dispatch.empty_cache.assert_not_called()


def test_critic_cannot_share_policy_and_inference_simultaneously():
    with pytest.raises(ValueError, match="cannot be combined"):
        FSDPBackendOverrides(
            critic_with_inference=True,
            policy_gpu_fraction=0.55,
            critic_gpu_fraction=0.45,
        )
    assert FSDPBackendOverrides(critic_with_inference=True).critic_with_inference


def test_single_example_critic_unpadding_preserves_values_and_gradients(tmp_path):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    model = get_llm_for_sequence_regression(
        str(tmp_path),
        "critic",
        bf16=False,
        lora_rank=2,
        target_modules=["q_proj", "v_proj"],
        init_value_head=True,
    )
    # Exercise the FA4 padding path with a CPU attention implementation.
    model.get_base_model().fa4_unpad = True
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    plain = model(ids, num_actions=3, attention_mask=torch.ones_like(ids))
    plain.square().sum().backward()
    grad = model.value_head.weight.grad.clone()
    model.zero_grad()
    padded = torch.tensor([[0, 0, 0, 1, 2, 3, 4]])
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    values = model(padded, num_actions=6, attention_mask=mask)
    torch.testing.assert_close(values[:, 3:], plain)
    assert torch.count_nonzero(values[:, :3]) == 0
    values.square().sum().backward()
    torch.testing.assert_close(model.value_head.weight.grad, grad)
    with pytest.raises(ValueError, match="single-example"):
        model(ids.expand(2, -1), num_actions=3, attention_mask=torch.ones(2, 4))
