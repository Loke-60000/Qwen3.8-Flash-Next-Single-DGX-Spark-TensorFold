# Per-request abliteration: one checkpoint, stock and abliterated

Patches 0006 to 0008 let the server apply an abliteration at run time, per request. The served checkpoint is the stock
one, and a request that names an identity in `TENSORFOLD_IDENTITIES` gets that abliterated model's reply from the same
weights, in the same decode round as every other request. Requests that name no identity are unchanged, byte for
byte.

## How it works

An abliterated build edits the residual-stream writers with `W' = W - R diag(lam) R^T W`, where `R` is the refusal
directions as orthonormal rows and `lam` is the strengths. The writer's output is then `y - R diag(lam) (R^T y)`.

The engine subtracts that term from the writers' outputs for the rows of requests that named an identity, from the
identity's `first_layer` up. The token mixers' output projections (`linear_attn.out_proj`, `self_attn.o_proj`) use
`lam_attn`, and the shared expert's down projection uses `lam_shared`. The MTP draft head isn't changed, and the main
forward pass verifies its drafts.

A baked checkpoint requantises the edited weights, so it's close to the run-time correction but not bit-equal with it.

## Settings

| Setting | Meaning |
|---|---|
| `TENSORFOLD_IDENTITIES` | `id=path` entries separated by commas, at most 32. `id` is the model id a request names, and `path` is its directions file, either a path inside the container or `hf:owner/repo/file[@revision]` on Hugging Face. `/v1/models` lists every id. |
| `TENSORFOLD_ABLATE`, `TENSORFOLD_ABLATE_MODEL` | One identity the short way, with the file first and the model id second. They're read only while `TENSORFOLD_IDENTITIES` is unset. |
| `TENSORFOLD_ABLATE_SCALE` | Multiplies every identity's strengths, for tuning runs. The default is 1. |

`start.sh` passes every `TENSORFOLD_*` variable into the container and mounts the Hugging Face cache at
`/root/.cache/huggingface`, so a file under `~/.cache/huggingface/ablate/` needs no extra mount:

```bash
TENSORFOLD_IDENTITIES="Qwen3.8-Flash-Next-abliterated=/root/.cache/huggingface/ablate/abliterated.safetensors" ./start.sh
```

`scripts/prepare.sh` downloads a file on Hugging Face into the same cache, like the checkpoint, and `start.sh` then gives
the engine its path. For a gated repo, the account of `HF_TOKEN` (or `hf auth login`) must accept its terms. This
example loads drowzeys's house-qsa abliteration, the overlay that `ABLIT=1` grafts, as a directions file:

```bash
TENSORFOLD_IDENTITIES="Qwen3.8-Flash-Next-abliterated=hf:Loke-60000/Qwen3.8-Flash-Next-FP4-Local-house-qsa-directions/house-qsa.safetensors" ./start.sh
```

At load, the engine logs one `identity i id: ...` line for each identity. If a file fails its checks, the load stops
and gives the reason.
`ABLIT=2` sets this up for drowzeys's house-qsa abliteration in one switch. It serves the stock model as `SERVED_NAME`
and the abliterated one as `SERVED_NAME-abliterated`, and downloads the directions file at a pinned revision
(`ABLIT_DIRECTIONS`). A `TENSORFOLD_IDENTITIES` of your own takes precedence. Don't list identities with `ABLIT=1`,
because every request then already runs on the overlay's abliterated weights, the stock id included.

## Directions files

A global file applies the same directions at every layer from `first_layer` up:

| Tensor | Meaning |
|---|---|
| `directions` [k, d] fp32 | orthonormal rows, k <= 4 |
| `lam_attn` [k], `lam_shared` [k] fp32 | the strengths for the token mixers and the shared expert |
| `first_layer` [1] int64 | the first layer edited |

A per-layer file (patch 0008) edits only the layers it lists, each with a direction of its own. The edit
works like `W' = D (W - lam v v^T W)`, where `D` is a per-row rescale, as after a row-norm restore. At a listed layer,
the token mixer's output becomes `D (y - lam (v . y) v)`, while other layers and the shared expert are left alone:

| Tensor | Meaning |
|---|---|
| `layers` [n] int64 | the layers it edits, in increasing order |
| `directions` [n, d] fp32 | one unit direction per listed layer |
| `lam` [n] fp32 | each layer's strength |
| `row_scale` [n, d] fp32 | optional. Each layer's `D`, all ones when absent |

`tools/ablate_from_overlay.py` writes a per-layer file from a baked overlay of `self_attn.o_proj` weights, in MXFP8
like the stock checkpoint, and each layer's `lam`. One example is drowzeys's house-qsa sidecar (the one `ABLIT=1`
downloads) and its `HOUSE_QSA_MXFP8.json`. The script prints each layer's reconstruction error, which should match the
overlay's own requantisation error. It needs torch, which the image's base has:

```bash
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface -v "$PWD":/w nvcr.io/nvidia/pytorch:26.07-py3 \
  python3 /w/tools/ablate_from_overlay.py /root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/<rev> \
  /w/house-qsa-oproj-mxfp8-L3-47.safetensors /w/HOUSE_QSA_MXFP8.json /root/.cache/huggingface/ablate/house-qsa.safetensors
```

## Behaviour

- Each identity keeps prompt states of its own, so a conversation's next turn resumes its own identity's state, never the
  stock model's or another identity's. The server marks the request with the identity that its `model` names
  (`Request.named`, kept in `Request.identity`), and that mark also changes the request's prompt-cache key.
  `zig/src/core/lane_host_identity_test.zig` covers it.
- A pass whose rows name no identity launches nothing extra, and an identity's kernels run only at the layers it edits.
- The server refuses two ranks (TP=2) at load.
- The server refuses to start when the list is malformed, names an id twice, has more than 32 entries, or uses the
  served name or an alias as an identity id. The load also stops on a `first_layer` past the last layer (47).

## Checks

`tools/verify_ablate.py` sends one long prompt twice to each model id it's given, by default the stock
`SERVED_NAME` and `SERVED_NAME-abliterated`. The second pass resumes that id's own kept state, and its greedy reply
matches the cold one.
