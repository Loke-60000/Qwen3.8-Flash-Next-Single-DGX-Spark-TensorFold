# Configuration

Settings go in `.env` beside `start.sh` (`KEY=value` lines, read, never run), then `./start.sh restart`. MiaAI-Lab's
settings (window, sampling, vision, video, PDF, KV format) are in [README-upstream.md](../README-upstream.md#configuration).
This branch adds:

| Setting | Default | What it does |
| --- | --- | --- |
| `ABLIT` | `0` | `2` also serves `<SERVED_NAME>-abliterated`, abliterated per request from the same weights (patches `0006` to `0008`). `1` grafts an overlay into the checkpoint instead. |
| `TENSORFOLD_IDENTITIES` | off | More abliterated ids, `id=file` entries. A file is a local safetensors path or `hf:owner/repo/path[@revision]`. |
| `PROMPT_CACHE_GIB` | engine's | The prompt cache's cap. The engine gives it at most half of the memory free after the page pool. |
| `TENSORFOLD_PROMPT_CACHE_PARTIAL` | `1` | A conversation past the cache keeps the furthest state that fits (patch `0013`). |
| `SPILL_GIB` | off | The disk tier's size in GiB, under `SPILL_DIR` (patch `0012`). |
| `SPILL_DIR` | `~/.cache/tensorfold-qwen38/spilled` | Where the disk tier writes. |
| `SPILL_DAILY_GIB` | `24` | The most the disk tier writes in a day. |
| `SPILL_MIN_FREE_GIB` | `200` | Free disk the disk tier always leaves. |
| `TENSORFOLD_ROWS_MIN_TOKENS` | `16384` | Shorter states stay off the disk, since prefilling them costs less than writing them. |
| `DRAFT_VOCAB` | built-in | The tokens the draft head can propose (patch `0014`). Empty keeps the built-in list, `all` allows every token, or give a list file. `scripts/draft-vocab-fr-de.txt` adds French and German, for +10% and +18% decode with English and code unchanged. |
| `TENSORFOLD_DECISION_IDS` | off | Decision model ids, `id=base` entries (patches `0015` to `0017`). |
| `TENSORFOLD_DECISION_SHARE` | `0.1` | The share of the engine's time decisions may take while other requests decode. |
| `TENSORFOLD_DECISION_ROWS` | `256` | Prompt rows a decision feeds in one step beside decoding requests. |
| `TENSORFOLD_DECISION_MAX_TOKENS` | `16384` | The longest decision prompt. |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `9` | Memory the engine leaves free. Raise it when other services share the Spark. The lab runs 15 beside about 8 GiB of speech and embedding services. |

## Settings behind the benchmarks

The [README](../README.md)'s numbers were measured with:

```bash
PARALLEL=15
KV_DTYPE=fp8
TF_FLASHNEXT_KV4=values
TF_FLASHNEXT_KV_PAGES=256
TENSORFOLD_MEMORY_RESERVE_GIB=15
PROMPT_CACHE_GIB=4
SPILL_GIB=64
DRAFT_VOCAB=/home/<you>/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/scripts/draft-vocab-fr-de.txt
TENSORFOLD_DECISION_IDS=Qwen3.8-Flash-Next-decision=Qwen3.8-Flash-Next
VISION=1
PDF=1
VIDEO=1
```
