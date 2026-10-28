# Per-request abliteration: one checkpoint, stock and abliterated

`patches/0018-ablate.patch` teaches the Zig engine to apply an abliteration at run time, per request. The served
checkpoint is the stock one; a request that names a second model id gets the abliterated model's reply from the same
weights, in the same decode round as everyone else.

## Why it is exact

An abliterated build edits writers: `W' = W - R diag(lam) R^T W`, with `R` the refusal directions (orthonormal rows)
and `lam` the strengths. The writer's output is therefore `y - R diag(lam) (R^T y)`. The engine subtracts that term
from each edited writer's output, for the rows whose request asked for it, before the hyper-connection write-back
(the layer loop of `Pass.chunk` in `cuda_batch.zig`):

- the token mixers' output projections (`linear_attn.out_proj`, `self_attn.o_proj`) with `lam_attn`: the batched pass
  holds them as fp32 K slices, so the kernel projects their sum and lands the correction in the first slice
- the shared expert's down projection, its own last slot of the MoE output before the router's gate (or its output
  read in place on prompt chunks), with `lam_shared`

from `first_layer` up. The MTP draft head is untouched; the main forward verifies its drafts.

A baked checkpoint requantises the edited writers (MXFP8 here), so it is close to, not bit-equal with, the run-time
correction on the stock weights.

## Settings

| Setting | Meaning |
|---|---|
| `TENSORFOLD_ABLATE` | path inside the container of a safetensors file: `directions` [k, d] fp32 (orthonormal rows, k <= 4), `lam_attn` [k], `lam_shared` [k] fp32, `first_layer` [1] int64 |
| `TENSORFOLD_ABLATE_MODEL` | the model id that turns the ablation on for a request; `/v1/models` lists it (no `--alias` needed) |
| `TENSORFOLD_ABLATE_SCALE` | multiplies both strengths (default 1), for tuning runs |

`start.sh` passes every `TENSORFOLD_*` variable into the container and mounts the Hugging Face cache at
`/root/.cache/huggingface`, so a file under `~/.cache/huggingface/ablate/` needs no extra mount:

```bash
TENSORFOLD_ABLATE=/root/.cache/huggingface/ablate/lr7-skuld.safetensors TENSORFOLD_ABLATE_MODEL=abliterated ./start.sh
```

## Behaviour

- Requests without the model id are unchanged byte for byte: the kernel returns before reading a row whose mask is
  0. With `TENSORFOLD_ABLATE` unset nothing is launched.
- The per-row mask is uploaded with the token ids before each pass, from the pinned staging the ids use.
- Ablated requests have prompt reuse of their own: the lane host keys kept states by the prompt's tokens, and an
  ablated request's key carries the mode in its first word (marked the way an image frame marks it), so a
  conversation's next abliterated turn resumes its last abliterated state and the two modes never resume each other's (their
  states differ). `zig/src/core/lane_host_reuse_test.zig` covers it.
- Two ranks (TP=2) are refused at load. `--parallel 1` works.
- The server refuses to start when `TENSORFOLD_ABLATE_MODEL` is set but the engine did not load `TENSORFOLD_ABLATE`.

## Files

- `lr7.safetensors`: two directions recovered from leimroth-lab/Unbound-LR7, LR7's fitted strengths (1.32/1.27
  attention, 0.176/0.161 shared expert), from layer 8.
- `lr7-skuld.safetensors`: the same directions at 1.75 attention / 0.5 shared expert, the strengths baked into
  Loke-60000/Qwen3.8-Flash-Next-FP4-Local-abliterated (realised 1.72 / 0.46 after its requantisation).

## Checks

`tools/verify_ablate.py` sends one long prompt twice to each model id it is given (default: `abliterated` and the
stock `SERVED_NAME`) against the running server: the second pass resumes that id's own kept state and its greedy
reply matches the cold one.
