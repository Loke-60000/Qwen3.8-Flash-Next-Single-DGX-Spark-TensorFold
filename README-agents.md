# How AI agents deploy Qwen3.8 Flash Next

You're an AI agent asked to serve Qwen3.8 Flash Next on a DGX Spark with this repository. Follow these steps in order.
Deploy only when no server is running. The README for people is [README.md](README.md).

## Rules

- Do not stop, restart or replace a running server unless the user asks for it. A restart drops every request in flight
  and takes the model offline while the weights load.
- Do not delete `~/.cache/huggingface` (the 96 GiB checkpoint), the Docker image `tensorfold-qwen38:*` or `SPILL_DIR`.
- Do not edit `patches/` or `scripts/`. Settings go in `.env`.
- Do not commit `.env`, and do not put a Hugging Face token in any file of the repository.
- Do not run load tests or 100K-token prompts on a Spark that runs other services. GPU memory is the system's RAM, so the
  kernel's OOM killer can stop other processes while `MemAvailable` still looks fine.

## 1. Check whether the model is already served

```bash
curl -sf --max-time 5 http://127.0.0.1:8888/v1/models
docker ps --filter name=qwen38-flash-next-tf --format '{{.Names}} {{.Status}} {{.Image}}'
```

- **If `/v1/models` answers with a model list**, the model is served. Do not deploy. Report the ids and the port to the
  user, then stop here.
- **If the container runs but `/v1/models` doesn't answer**, the server is still loading, which takes about a minute
  after a start. Watch `docker logs -f qwen38-flash-next-tf` for up to 10 minutes, then check again. If it never answers,
  report the log's last lines to the user. Do not restart it yourself.
- **If nothing answers and no container runs**, continue with step 2.

If the user gave another port, or the repository's `.env` sets `PORT=`, check that port instead of 8888.

## 2. Check the machine

Run each check. If one fails, stop and tell the user which one and its value.

```bash
nvidia-smi --query-gpu=name --format=csv,noheader          # must name a GB10 (DGX Spark)
awk '/^MemAvailable:/ {print int($2/1048576) " GiB"}' /proc/meminfo   # needs about 107 GiB
docker info --format '{{json .Runtimes}}' | grep -q nvidia && echo "nvidia runtime ok"
docker ps >/dev/null && echo "docker access ok"            # fails if the user is not in the docker group
df -BG --output=avail,target "$HOME" "$(docker info --format '{{.DockerRootDir}}')"   # about 160 GB free on a fresh machine
ss -ltn 'sport = :8888'                                    # must print no listener
```

- **If less than 107 GiB is available**, something else holds memory, often another model server. List it with
  `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`, then ask the user what to stop. Do not
  stop it yourself.
- **If something else uses port 8888**, ask the user for another port and put `PORT=<port>` in `.env`.
- **If there's less free disk than needed**, check `~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local`.
  If the checkpoint is already there, the deploy needs 96 GiB less. Otherwise, ask the user where to free space.

## 3. Get the repository

```bash
cd ~
if [ -d Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/.git ]; then
  git -C Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold status --short --branch
else
  git clone https://github.com/Loke-60000/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
fi
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
```

The default branch is `main-gl`. If a checkout exists, use it as it is, and report its branch if it isn't `main-gl`.
Do not pull, reset or switch branches unless the user asks.

## 4. Settings

The defaults work without a `.env` and serve the model id `Qwen3.8-Flash-Next` on port 8888, with up to 5 requests at
once. Write `.env` only for what the user asked. Common lines:

```bash
SERVED_NAME=Qwen3.8-Flash-Next                                      # the model id clients use
PARALLEL=5                                                          # requests at once, 1 to 16
ABLIT=2                                                             # also serve Qwen3.8-Flash-Next-abliterated
TENSORFOLD_DECISION_IDS=Qwen3.8-Flash-Next-decision=Qwen3.8-Flash-Next   # the decision API
SPILL_GIB=64                                                        # the disk tier for long conversations
```

If `.env` exists, keep its lines and add or change only what the user asked. `ABLIT=2` serves a model with its safety
refusals removed, so set it only when the user asks for the abliterated model.

## 5. Start

Check what a start would do first:

```bash
DRY_RUN=1 ./start.sh
```

It prints the model, the memory it needs and the docker command, and changes nothing. Read its warnings and fix them,
then start:

```bash
./start.sh
```

The first run builds the image and downloads the 96 GiB checkpoint, so it takes as long as that download plus the
build. Run it in the background with a timeout of at least 2 hours, and send its output to a log file. A later start
loads the weights in about a minute.
The start is done when the log shows `is now LIVE! on port 8888`, and `start.sh` then exits 0.

| `start.sh` says | Do this |
| --- | --- |
| `... is already running ...: nothing to do.` | The server runs. Go to step 6. |
| `another ./start.sh is already running` | Someone else is deploying. Wait for it, then go to step 1. |
| `port 8888 is already in use` | Go back to the port check in step 2. |
| `only N GiB memory available` | Go back to the memory check in step 2. Do not use `MEM_CHECK=0`. |
| `not ready after 1800s` | The server is still loading. Watch `docker logs -f qwen38-flash-next-tf`. |
| A download error from Hugging Face | Check the network. `ABLIT=1` needs a token, so ask the user for one. |

## 6. Check it works

```bash
curl -s http://127.0.0.1:8888/v1/models
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "max_tokens": 1000}'
```

If `.env` sets `SERVED_NAME`, use that name. The reply must have non-empty `content`. The model thinks first, so keep
`max_tokens` at 1000 or more. If decision ids are on, also check one:

```bash
curl -s http://127.0.0.1:8888/v1/decisions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-decision",
  "input": "The sky is blue.",
  "questions": [{"id": "q", "type": "yes_no", "question": "Is this statement true?"}]}'
```

The answer must have `answers.q.probabilities` with `yes` and `no`.

Then run the exactness check with `python3 tools/exact.py`. It takes under a minute, puts little load on the server,
and must print `exact: PASS`.

## 7. Report to the user

Tell the user:

- the model ids from `/v1/models` and the address, `http://<host>:8888/v1`,
- the image tag (`docker ps --format '{{.Image}}' --filter name=qwen38-flash-next-tf`),
- each line you added or changed in `.env`,
- the result of `tools/exact.py`.

`./stop.sh` stops the server, and `./start.sh restart` applies a changed `.env`. Both take the model offline, so run them
only when the user asks.
