import torch

from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
    move_fsdp2_model_buffers_to_device,
)


def test_move_fsdp2_model_buffers_does_not_move_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))

    move_fsdp2_model_buffers_to_device(model, torch.device("meta"))

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(buffer.device.type == "meta" for buffer in model.buffers())
