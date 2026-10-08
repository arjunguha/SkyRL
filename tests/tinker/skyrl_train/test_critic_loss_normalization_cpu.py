"""CPU contracts for request/microbatch normalization; no distributed runtime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from skyrl.backends.skyrl_train.workers import worker as worker_module
from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_critic_loss
from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend
from skyrl.tinker import types


@pytest.mark.parametrize("token_batching", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2, 3])
@pytest.mark.parametrize("chunk_sizes", [[5], [2, 3], [1, 1, 3]])
def test_loss_and_gradient_match_full_sequence_mean(
    monkeypatch, token_batching, dp_size, chunk_sizes
):
    monkeypatch.setattr(
        worker_module,
        "BaseBatchIterator",
        SimpleNamespace(batch_to_experience=lambda batch: batch),
    )
    monkeypatch.setattr(
        worker_module, "all_reduce_metrics", lambda metrics, strategy: metrics
    )
    # Unequal masks make sequence means different from a global token mean.
    features = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 1.0],
            [3.0, 2.0, 4.0],
            [4.0, 1.0, 2.0],
            [2.0, 3.0, 5.0],
        ]
    )
    masks = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    config = SimpleNamespace(value_clip=None)
    baseline_param = torch.tensor(0.4, requires_grad=True)
    loss_fn = ppo_critic_loss
    baseline, _ = loss_fn(
        baseline_param * features,
        torch.zeros_like(features),
        torch.ones_like(features),
        config,
        masks,
    )
    baseline.backward()
    reported_loss = 0.0
    gradient = 0.0
    start = 0
    for count in chunk_sizes:
        rows = list(range(start, start + count))
        start += count
        rows += [-1] * (-len(rows) % dp_size)  # backend DP padding
        shard_size = len(rows) // dp_size
        for rank in range(dp_size):
            param = torch.tensor(0.4, requires_grad=True)
            shard = rows[rank * shard_size : (rank + 1) * shard_size]
            # Include unequal microbatches and a loss-neutral synchronization batch.
            batches = [shard[:1]] + ([shard[1:]] if len(shard) > 1 else [])
            if token_batching:
                batches.append([-1])
            monkeypatch.setattr(
                worker_module,
                "get_microbatch_iterator",
                lambda *args, **kwargs: batches,
            )

            def backward_micro(indices, microbatch_weight=None):
                x = torch.stack(
                    [features[i] if i >= 0 else torch.zeros(3) for i in indices]
                )
                mask = torch.stack(
                    [masks[i] if i >= 0 else torch.zeros(3) for i in indices]
                )
                loss, _ = loss_fn(
                    param * x, torch.zeros_like(x), torch.ones_like(x), config, mask
                )
                if microbatch_weight is not None:
                    loss = loss * microbatch_weight
                loss.backward()
                return {"critic_loss": loss.item()}

            worker = SimpleNamespace(
                cfg=SimpleNamespace(
                    max_tokens_per_microbatch=16 if token_batching else -1,
                    micro_train_batch_size_per_gpu=2,
                ),
                mesh_rank=SimpleNamespace(dp_size=dp_size),
                strategy=None,
                _micro_batches_accumulated=0,
                _forward_backward_micro=backward_micro,
            )
            result = worker_module.CriticWorkerBase.forward_backward(
                worker, shard, normalization_num_sequences=5
            )
            reported_loss += result.metrics["critic_loss"] / dp_size
            gradient += param.grad.item() / dp_size
            assert worker._micro_batches_accumulated == 0
    assert reported_loss == pytest.approx(baseline.item())
    assert gradient == pytest.approx(baseline_param.grad.item())


@pytest.mark.parametrize("role", ["policy", "critic"])
def test_coalesced_requests_keep_configs_and_metrics_separate(role):
    seen = []

    def single_request(batch):
        seen.append(batch)
        request_id = batch.request_batch_slices[0][0]
        return {request_id: sum(row[0] for row in batch.all_targets)}

    backend = SimpleNamespace(
        _get_batch_role=lambda _: role,
        _forward_backward_single_model_batch=single_request,
    )
    fields = {
        field: [[], [], []] for field in types.PreparedModelPassBatch.model_fields
    }
    fields.update(
        all_model_inputs=[types.ModelInput(chunks=[])] * 3,
        all_targets=[[1], [2], [4]],
        all_model_ids=["model"] * 3,
        all_loss_fns=["ppo"] * 3,
        all_loss_fn_configs=[{"normalization_num_sequences": 3}] * 2
        + [{"normalization_num_sequences": 7}],
        request_batch_slices=[("a", "model", 0, 2), ("b", "model", 2, 3)],
    )
    batch = types.PreparedModelPassBatch(**fields)
    assert SkyRLTrainBackend._forward_backward_single_model_batch(backend, batch) == {
        "a": 3,
        "b": 4,
    }
    assert seen[0].all_loss_fn_configs == [{"normalization_num_sequences": 3}] * 2
    assert seen[1].all_loss_fn_configs == [{"normalization_num_sequences": 7}]
    assert seen[1].request_batch_slices == [("b", "model", 0, 1)]


@pytest.mark.parametrize(
    "denominator", [None, 64, 0, -1, 1.5, float("nan"), float("inf")]
)
def test_backend_passes_caller_count_before_dp_padding(denominator):
    fields = {field: [[]] for field in types.PreparedModelPassBatch.model_fields}
    config = {} if denominator is None else {"normalization_num_sequences": denominator}
    fields.update(
        all_model_inputs=[types.ModelInput(chunks=[])],
        all_model_ids=["critic"],
        all_loss_fns=["ppo_critic"],
        all_loss_fn_configs=[config],
        request_batch_slices=[("request", "critic", 0, 1)],
    )
    batch = types.PreparedModelPassBatch(**fields)
    dispatch = Mock()
    dispatch.forward_backward.return_value = SimpleNamespace(
        metrics={"critic_loss": 0.5},
        loss_fn_outputs=[],
        loss_fn_output_type="ppo_critic",
    )
    backend = SimpleNamespace(
        _get_batch_role=lambda _: "critic",
        _release_shared_gpu_cache=Mock(),
        _validate_batch_role_and_loss=Mock(),
        _to_training_batch=lambda *args: "unpadded",
        _pad_batch=lambda *args, **kwargs: ("padded", 1),
        _normalize_policy_loss_request=lambda role, loss, config: (loss, config),
        _cfg=SimpleNamespace(
            trainer=SimpleNamespace(
                strategy="fsdp", algorithm=SimpleNamespace(value_clip=0.2)
            )
        ),
        _dispatch=dispatch,
        _extract_metrics=lambda data: {"critic_loss:sum": data["critic_loss"]},
    )
    if denominator is not None and (
        not denominator > 0 or denominator in (1.5, float("inf"))
    ):
        with pytest.raises(ValueError, match="normalization_num_sequences"):
            SkyRLTrainBackend._forward_backward_single_model_batch(backend, batch)
        dispatch.forward_backward.assert_not_called()
    else:
        result = SkyRLTrainBackend._forward_backward_single_model_batch(backend, batch)
        dispatch.forward_backward.assert_called_once_with(
            "critic",
            "padded",
            model_id="critic",
            normalization_num_sequences=denominator or 1,
        )
        assert result["request"].metrics["critic_loss:sum"] == 0.5
