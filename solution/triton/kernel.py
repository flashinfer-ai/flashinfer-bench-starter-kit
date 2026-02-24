"""
Triton Kernel Template for FlashInfer Competition.

Implement your kernel logic here. The entry point function name should match
the `entry_point` setting in config.toml.

See the track definition for required function signature and semantics.
"""

import triton
import triton.language as tl
from . import dsa_sparse_attention


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    """
    Proxy kernel implementation that calls the DSA Sparse Attention implementation.
    """
    dsa_sparse_attention.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
