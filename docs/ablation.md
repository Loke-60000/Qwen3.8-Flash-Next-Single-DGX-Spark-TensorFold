# Per-request abliteration: one checkpoint, stock and abliterated

`patches/0018-ablate.patch` teaches the Zig engine to apply an abliteration at run time, per request;
`patches/0019-identities.patch` makes the list of them data: every model id in `TENSORFOLD_IDENTITIES` is an identity
with a directions file of its own. The served checkpoint is the stock one; a request that names an identity gets that
abliterated model's reply from the same weights, in the same decode round as everyone else.

## Why it is exact

An abliterated build edits writers: `W' = W - R diag(lam) R^T W`, with `R` the refusal directions (orthonormal rows)
and `lam` the strengths. The writer's output is therefore `y - R diag(lam) (R^T y)`. The engine subtracts that term
from each edited writer's output, for the rows whose request asked for it, before the hyper-connection write-back
(the layer loop of `Pass.chunk` in `cuda_batch.zig`):

- the token mixers' output projections (`linear_attn.out_proj`, `self_attn.o_proj`) with `lam_attn`: the batched pass
  holds them as fp32 K slices, so the kernel projects their sum and lands the correction in the first slice
- the shared expert's down projection, its own last slot of the MoE output before the router's gate (or its output
  read in place on prompt chunks), with `lam_shared`

from the identity's `first_layer` up. The MTP draft head is untouched; the main forward verifies its drafts.

A baked checkpoint requantises the edited writers (MXFP8 here), so it is close to, not bit-equal with, the run-time
correction on the stock weights.

## Settings

| Setting | Meaning |
|---|---|
| `TENSORFOLD_IDENTITIES` | the identities, `id=path` entries separated by commas (at most 32): the model id a request names and the path inside the container of its safetensors file, `directions` [k, d] fp32 (orthonormal rows, k <= 4), `lam_attn` [k], `lam_shared` [k] fp32, `first_layer` [1] int64. `/v1/models` lists every id (no `--alias` needed) |
| `TENSORFOLD_ABLATE`, `TENSORFOLD_ABLATE_MODEL` | one identity the short way (the file, the model id); read only while `TENSORFOLD_IDENTITIES` is unset |
| `TENSORFOLD_ABLATE_SCALE` | multiplies every identity's strengths (default 1), for tuning runs |

`start.sh` passes every `TENSORFOLD_*` variable into the container and mounts the Hugging Face cache at
`/root/.cache/huggingface`, so a file under `~/.cache/huggingface/ablate/` needs no extra mount:

```bash
TENSORFOLD_IDENTITIES="abliterated=/root/.cache/huggingface/ablate/lr7-skuld.safetensors,abliterated-lr7=/root/.cache/huggingface/ablate/lr7.safetensors" \
  ./start.sh
```

The engine numbers the identities in the list's order (1..n; 0 is the stock model) and logs one `identity i id: ...`
line each at load; a file that fails its checks stops the load with the reason.

## Behaviour

- Requests naming no identity are unchanged byte for byte: the kernel returns before reading a row whose identity is
  0. With no identity listed nothing is launched.
- Each row's identity is uploaded with the token ids before each pass, from the pinned staging the ids use; every
  identity's directions sit in one device table (about 40 KiB each), so adding one costs nothing per request.
- Each identity has prompt reuse of its own: the lane host keys kept states by the prompt's tokens, and a request's
  key carries its identity in its first word (marked the way an image frame marks it), so a conversation's next abliterated
  turn resumes its last abliterated state and no two identities, nor an identity and the stock model, ever resume each
  other's states (they differ). `zig/src/core/lane_host_reuse_test.zig` covers it.
- Two ranks (TP=2) are refused at load. `--parallel 1` works.
- The server refuses to start when identities are listed but the engine did not load them, and when the list is
  malformed, names an id twice, or has more than 32 entries.

## Files

- `lr7.safetensors`: two directions recovered from leimroth-lab/Unbound-LR7, LR7's fitted strengths (1.32/1.27
  attention, 0.176/0.161 shared expert), from layer 8.
- `lr7-skuld.safetensors`: the same directions at 1.75 attention / 0.5 shared expert, the strengths baked into
  Loke-60000/Qwen3.8-Flash-Next-FP4-Local-abliterated (realised 1.72 / 0.46 after its requantisation).

## Checks

`tools/verify_ablate.py` sends one long prompt twice to each model id it is given (default: `abliterated` and the
stock `SERVED_NAME`) against the running server: the second pass resumes that id's own kept state and its greedy
reply matches the cold one.
