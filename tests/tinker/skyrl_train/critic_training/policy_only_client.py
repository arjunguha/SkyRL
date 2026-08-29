"""Exercise policy-only FSDP training through the stock async Tinker API.

The driver gives the policy one full GPU and configures no critic. This client
runs one cross-entropy forward/backward pass, checks its finite per-token loss,
then checks that the optimizer reports a positive finite gradient norm.
"""

import asyncio
import math

import tinker
from tinker import types


async def main() -> None:
    service = tinker.ServiceClient(base_url="http://127.0.0.1:18080", api_key="tml-unused")
    policy = await service.create_lora_training_client_async(base_model="Qwen/Qwen3-0.6B", rank=8)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]},
    )

    train_future = await policy.forward_backward_async([datum], "cross_entropy")
    train_result = await train_future
    losses = train_result.loss_fn_outputs[0]["elementwise_loss"].data
    assert len(losses) == 3 and all(math.isfinite(loss) for loss in losses)
    assert math.isfinite(train_result.metrics["total_loss:sum"])

    step_future = await policy.optim_step_async(types.AdamParams(learning_rate=1e-5))
    step_result = await step_future
    grad_norm = step_result.metrics["skyrl.ai/grad_norm"]
    assert math.isfinite(grad_norm) and grad_norm > 0


if __name__ == "__main__":
    asyncio.run(main())
