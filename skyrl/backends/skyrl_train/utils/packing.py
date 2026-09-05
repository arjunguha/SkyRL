"""Padding helpers used by the FSDP model wrapper.

These are the small PyTorch equivalents of the helpers historically imported
from FlashAttention 2. Keeping them local lets FSDP attention implementations
choose their own kernel package without forcing FA2 and FA4 to share a Python
namespace.
"""

import torch
import torch.nn.functional as F


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Remove padding and return the selected-token metadata."""

    all_masks = (
        attention_mask + unused_mask if unused_mask is not None else attention_mask
    )
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    flattened = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return (
        flattened[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """Restore selected tokens to their padded batch positions."""

    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])
