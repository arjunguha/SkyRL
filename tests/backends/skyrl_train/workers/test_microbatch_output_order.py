"""Forward/backward token packing preserves the caller's sample order."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase


@pytest.mark.parametrize("budget,expected_microbatches", [(28, 1), (14, 2), (-1, 2)])
def test_forward_backward_returns_outputs_in_input_order(budget, expected_microbatches):
    lengths = torch.tensor([4, 8, 6, 10])
    mask = torch.arange(10)[None] >= 10 - lengths[:, None]
    batch = TrainingInputBatch(
        {
            "sequences": torch.arange(4)[:, None].expand(4, 10),
            "attention_mask": mask,
            "loss_mask": mask.float(),
        }
    )
    batch.metadata = {"response_length": 10}
    executed_sizes = []

    def forward_backward_micro(experience, *args, **kwargs):
        indices = experience.sequences[:, -1].tolist()
        executed_sizes.append(len(indices))
        return {"loss": 0.0, "loss_fn_outputs": [{"sample": index} for index in indices]}

    worker = SimpleNamespace(
        cfg=SimpleNamespace(
            micro_train_batch_size_per_gpu=2, max_tokens_per_microbatch=budget, remove_microbatch_padding=True
        ),
        device_mesh=SimpleNamespace(get_group=lambda _: None),
        strategy=None,
        _forward_backward_micro=forward_backward_micro,
    )
    with patch(
        "skyrl.backends.skyrl_train.workers.worker.all_reduce_metrics",
        side_effect=lambda metrics, *args, **kwargs: metrics,
    ):
        output = PolicyWorkerBase.forward_backward(worker, batch)
    assert [item["sample"] for item in output.loss_fn_outputs] == [0, 1, 2, 3]
    assert len(executed_sizes) == expected_microbatches
    assert all(size in (2, 4) for size in executed_sizes)
    if budget > 0:
        assert output.metrics["num_microbatches"] == expected_microbatches
        assert output.metrics["num_padding_microbatches"] == 0
