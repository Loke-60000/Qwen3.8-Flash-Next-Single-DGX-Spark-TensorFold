# Long conversations

A conversation's next turn resumes from a kept state of its earlier turns, so only the new part is prefilled. Three
layers keep those states. Their settings are in [docs/configuration.md](configuration.md), and the timings in the
[README](../README.md#long-prompts-and-conversations).

## Prompt cache

`PROMPT_CACHE_GIB` caps the memory for whole states of recent conversations, about 16 KiB a token. The engine gives the
cache at most half of the memory free once the page pool is taken. With pages, the pool takes its room at start and the
cache fills as it's used, so set this and `TENSORFOLD_MEMORY_RESERVE_GIB` together for the memory you can give.

## Partial keep

Upstream keeps nothing of a prompt whose states are all bigger than the cache. With
`TENSORFOLD_PROMPT_CACHE_PARTIAL=1` (patch `0013`, on by default), it keeps the furthest state that fits beside the
pass's other states, on a 256-token block edge. Its next turn prefills only the rest.

## Disk tier

`SPILL_GIB` turns on TensorFold's own disk tier (`--spill-gib`) under `SPILL_DIR`. A state pushed out of memory is
written to the SSD and read back when its conversation returns, token for token the same as a fresh prompt pass.

It's built to spare the SSD:

- It writes only states pushed out of memory, and states still in memory at a planned stop.
- Rows that several states share are written once.
- States under `TENSORFOLD_ROWS_MIN_TOKENS` (16,384, patch `0012`) stay off the disk, because prefilling them costs less than
  writing their ~110 MiB of DeltaNet state.
- It stops writing at `SPILL_DAILY_GIB` (24 GiB a day) and always leaves `SPILL_MIN_FREE_GIB` (200 GiB) free.

It reads a state back only when that's faster than prefilling it, and checks every piece against its hash.

`TENSORFOLD_ROWS_SYNC=1` writes a block or a 16 MiB piece at a time straight from the kept state, with no staged copy in
memory. The cost is that an eviction writes on the engine thread, which takes about 1 to 3 s for a 1.6 GiB state.
