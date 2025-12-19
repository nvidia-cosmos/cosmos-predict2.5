## Causal Mask

NOTE: Flash only implements bottom-right-aligned causal mask, but the default
in SDPA, CUTLASS/NATTEN, cuDNN is top-left.
To get the same behavior, we __might__ be able to implement top-left-aligned with
the sliding window argument, but some of Flash's overrides prevent this...

```python
seqlen_q = query.shape[1]
seqlen_k = key.shape[1]

# From Flash Attn readme:
#     Query at position i will only attend to keys between
#     [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
#
# so our window_size when doing top-left causal masking should satisfy:
#  i + seqlen_k - seqlen_q - window_size[0] = 0
#  i + seqlen_k - seqlen_q + window_size[1] = i
# =>
#  seqlen_k - seqlen_q + window_size[1] = 0 ==>
#  window_size[1] = seqlen_q - seqlen_k
#
# and:
#
#  i + seqlen_k - seqlen_q = window_size[0]
#
# which has to be satisfied for all 0 <= i < seqlen_q:
#  seqlen_k - seqlen_q = window_size[0]
#
#  seqlen_q - 1 + seqlen_k - seqlen_q = window_size[0] ==>
#   seqlen_k - 1 = window_size[0]
#
# which means ...
#
#
# Other Flash overrides:
#      if (window_size_left >= seqlen_k) { window_size_left = -1; }
#      if (window_size_right >= seqlen_k) { window_size_right = -1; }
#
#      params.is_causal = window_size_left < 0 && window_size_right == 0;
#
#      if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
#      if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
#      params.window_size_left = window_size_left;
#      params.window_size_right = window_size_right;
#
# scheduler:
#      n_block_min = max(0, (m_block * kBlockM + seqlen_k - seqlen_q - window_size_left) / kBlockN);
#      n_block_max = min(n_block_max, ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q + window_size_right, kBlockN));
#

flash_causal = False
window_size = (-1, -1) if not is_causal else (seqlen_k, seqlen_q - seqlen_k)
if is_causal and seqlen_k < seqlen_q:
    window_size = (-1, 0)

    padding_KV = seqlen_q - seqlen_k
    old_shape = key.shape
    key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, padding_KV), "constant", 0)
    value = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, padding_KV), "constant", 0)
    log.debug(f"Flash Attention: padded KV from {old_shape} to {key.shape}.")

print(f"{window_size=}")

#window_size = (-1, -1) if not is_causal else (-1, 0)
#window_size = (-1, -1) if not is_causal else (-1, seqlen_q - seqlen_k)

# seqlen_q=7688, seqlen_kv=2048, is_causal=True
#      n_block_min = max(0, (q_start - 7688) / kBlockN);
#      n_block_max = min(n_block_max, ceil_div(q_end, kBlockN));
```
