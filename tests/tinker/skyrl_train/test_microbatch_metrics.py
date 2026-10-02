"""Check that token-batching diagnostics reach Tinker responses."""

import pytest

from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend


def test_microbatch_counts_are_exposed_without_changing_loss_metrics():
    metrics = SkyRLTrainBackend._extract_metrics(
        None,
        {
            "loss": 2.5,
            "num_microbatches": 4,
            "num_padding_microbatches": 0.5,
        },
    )
    assert metrics == {
        "total_loss:sum": 2.5,
        "num_microbatches:sum": 4.0,
        "num_padding_microbatches:sum": 0.5,
    }


def test_counts_are_absent_when_worker_does_not_report_them():
    assert SkyRLTrainBackend._extract_metrics(None, {"loss": 2.5}) == {"total_loss:sum": 2.5}


def test_sdk_adds_microbatch_counts_across_separate_chunks():
    from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results
    from tinker.types import ForwardBackwardOutput

    outputs = [
        ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[],
            metrics=SkyRLTrainBackend._extract_metrics(None, {"num_microbatches": count}),
        )
        for count in (1, 2)
    ]
    result = combine_fwd_bwd_output_results(outputs)
    assert result.metrics["num_microbatches:sum"] == pytest.approx(3.0)
