"""Exercise fractional policy/critic GPU sharing through stock async Tinker.

The driver assigns 0.55 GPU to the policy and 0.45 GPU to the critic, leaving a
second GPU for vLLM. This client checks critic forward/backward and optimizer
results, then samples once to force the inference engine to start. NumPy arrays
keep the critic-only ``values`` and ``returns`` on Tinker's generic tensor path.
"""

import asyncio
import math

import numpy as np
import tinker
from tinker import types


async def main() -> None:
    service = tinker.ServiceClient(base_url="http://127.0.0.1:18080", api_key="tml-unused")
    policy = await service.create_lora_training_client_async(base_model="Qwen/Qwen3-0.6B", rank=8)
    critic = await service.create_lora_training_client_async(
        base_model="Qwen/Qwen3-0.6B",
        rank=8,
        user_metadata={"model_role": "critic"},
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": [2, 3, 4],
            "weights": [1.0, 1.0, 1.0],
            "values": np.zeros(3, dtype=np.float32),
            "returns": np.ones(3, dtype=np.float32),
        },
    )

    train_future = await critic.forward_backward_async([datum], "ppo_critic", {"value_clip": 0.2})
    train_result = await train_future
    critic_loss = train_result.metrics["critic_loss:sum"]
    assert math.isfinite(critic_loss) and critic_loss > 0

    step_future = await critic.optim_step_async(types.AdamParams(learning_rate=1e-5))
    step_result = await step_future
    grad_norm = step_result.metrics["skyrl.ai/grad_norm"]
    assert math.isfinite(grad_norm) and grad_norm > 0

    sampler = await policy.save_weights_and_get_sampling_client_async()
    sample = await sampler.sample_async(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2, temperature=0.0),
    )
    assert len(sample.sequences) == 1


if __name__ == "__main__":
    asyncio.run(main())
