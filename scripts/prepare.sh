#!/usr/bin/env bash
# Prepare this Spark to serve Qwen3.8-Flash-Next ($QUANT_LABEL; see CHECKPOINT in scripts/config.sh) with TensorFold's Zig engine (tensorfold-native, one GPU):
#   1. preflight: docker, the GPU, disk
#   2. the image: tensorfold-native built from TensorFold (TF_REF) plus patches/*.patch with Zig, on NVIDIA's
#      PyTorch container (kernels are embedded in the binary), with libjpeg-turbo, libwebp's decoder and the wheel's
#      libpdfium (PDF parts, PDF=1) and libtfvideo (video, VIDEO=1) built over a pinned static FFmpeg; pulled prebuilt when a matching tag is reachable (PULL=0 skips
#      that), else built locally
#   3. the checkpoint in the Hugging Face cache (~96 GiB for the NVIDIA FP4 default, ~122 GiB for CHECKPOINT=int4), at its pinned revision
#   4. verify it: every shard its index names is there, and it is a checkpoint format this engine serves (ModelOpt NVFP4 or GPTQ int4)
#   5. ABLIT=1 only: the Keys overlay's sidecar (~190 MB, gated) and the checkpoint folder with it grafted in (scripts/ablit_view.py)
#   6. hf: identities in TENSORFOLD_IDENTITIES (ABLIT=2 sets one): their directions files (KB, gated or not)
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

mkdir -p "$KERNEL_CACHE" "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running; wait for it or stop it"

log "Preflight checks"
_mem_here=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
if (( _mem_here < MEM_FLOOR_GIB )); then
  _msg="only ${_mem_here} GiB MemAvailable (the floor is MEM_FLOOR_GIB=$MEM_FLOOR_GIB): stop other work first"
  if [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$_msg"; else die "$_msg (MEM_CHECK=0 continues anyway)"; fi
fi
log "Memory: ${_mem_here} GiB MemAvailable (floor $MEM_FLOOR_GIB)"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found"
elif ! nvidia-smi -L >/dev/null 2>&1; then warn "nvidia-smi failed: is the NVIDIA driver working?"; fi
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime; --gpus all may fail"

PATCHES_HASH=$(image_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

HUB_MISSING_PY='
import math, os, sys
from huggingface_hub import HfApi
hub, repo, rev = sys.argv[1], sys.argv[2], sys.argv[3]
blobs = os.path.join(hub, "models--" + repo.replace("/", "--"), "blobs")
missing = 0
for f in HfApi().model_info(repo, revision=rev or None, files_metadata=True).siblings:
    lfs = f.lfs
    name = (getattr(lfs, "sha256", None) or lfs["sha256"]) if lfs else f.blob_id
    size = (getattr(lfs, "size", None) or lfs["size"]) if lfs else (f.size or 0)
    p = os.path.join(blobs, name)
    missing += 0 if os.path.isfile(p) and os.path.getsize(p) == size else size
print(math.ceil(missing / 2**30))'
hub_missing_gb() {
  local py out tried=""
  for py in "$(command -v python3)" "$(sed -n '1s/^#! *\(\/[^ ]*python[0-9.]*\)$/\1/p' "$(command -v hf || echo /dev/null)" 2>/dev/null)"; do
    [[ -n "$py" && -x "$py" && "$py" != "${tried:-}" ]] || continue
    tried=$py
    out=$(timeout 30 "$py" -c "$HUB_MISSING_PY" "$HF_CACHE/hub" "$MODEL_ID" "$MODEL_REVISION" 2>/dev/null) && [[ "$out" =~ ^[0-9]+$ ]] &&
      { echo "$out"; return 0; }
  done
  return 1
}

DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
rev=$(snapshot_rev)
if missing=$(hub_missing_gb); then
  need_ckpt=0; (( missing == 0 )) || need_ckpt=$((missing + 5))
  ckpt_what="the download needs ~${missing} GB that the cache does not hold yet"
else
  need_ckpt=0
  [[ -f "$(model_cache_dir)/snapshots/$rev/config.json" ]] || need_ckpt=$MIN_FREE_GB
  ckpt_what="the checkpoint needs MIN_FREE_GB (the Hub's file list could not be read)"
fi
(( need_ckpt )) || ckpt_what="nothing to download"
need_img=0; [[ $REBUILD -eq 0 && "$built_hash" == "$PATCHES_HASH" ]] || need_img=$IMAGE_FREE_GB
if [[ "$(stat -c %d "$HF_CACHE")" == "$(stat -c %d "$DOCKER_ROOT" 2>/dev/null)" ]]; then
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt + need_img )) ||
    die "only ${have} GB free under $HF_CACHE (also Docker's root), ~$((need_ckpt + need_img)) GB needed: $ckpt_what; the image needs ${need_img} GB"
else
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt )) || die "only ${have} GB free under $HF_CACHE, ~${need_ckpt} GB needed: $ckpt_what"
  have=$(free_gb "$DOCKER_ROOT"); (( have >= need_img )) ||
    die "only ${have} GB free under Docker's root ($DOCKER_ROOT); the image needs ~${need_img} GB"
fi
log "Disk: $(free_gb "$HF_CACHE") GB free under $HF_CACHE ($ckpt_what)"

prebuilt=$(prebuilt_image)
if [[ $REBUILD -eq 0 && "${PULL:-1}" == 1 && "$built_hash" != "$PATCHES_HASH" ]]; then
  log "Pulling the prebuilt image $prebuilt (PULL=0 builds instead)"
  if docker pull "$prebuilt" &&
     [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$prebuilt")" == "$PATCHES_HASH" ]]; then
    docker tag "$prebuilt" "$IMAGE"; built_hash=$PATCHES_HASH
    log "Using $prebuilt as $IMAGE"
  else
    warn "could not pull $prebuilt: building it locally"
  fi
fi
if [[ $REBUILD -eq 1 || "$built_hash" != "$PATCHES_HASH" ]]; then
  REBUILD=$REBUILD scripts/build-image.sh
else
  log "Image $IMAGE already built with patches $PATCHES_HASH"
fi
docker run --rm --network none --entrypoint test "$IMAGE" -x /opt/tensorfold/native/bin/tensorfold-native ||
  die "$IMAGE has no tensorfold-native"
log "Image $IMAGE: tensorfold-native, kernel set present"

command -v hf >/dev/null || warn "host 'hf' CLI not found, downloading from inside the container"
download() {   # download <repo> [revision] [include pattern]
  if command -v hf >/dev/null; then
    hf download "$1" ${2:+--revision "$2"} --cache-dir "$HF_CACHE/hub" ${3:+--include "$3"} >/dev/null
  else
    docker run --rm --user "$(id -u):$(id -g)" --network host --entrypoint python ${HF_TOKEN:+-e HF_TOKEN} \
      ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} \
      -v "$HF_CACHE":/hf -e HF_HOME=/hf -e HOME=/tmp "$IMAGE" -c \
      'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None, allow_patterns=sys.argv[3] or None)' "$1" "${2:-}" "${3:-}"
  fi
}
dir=$(model_cache_dir)
pin=$MODEL_REVISION
had=0; [[ -n "$pin" && -f "$dir/snapshots/$pin/config.json" ]] && had=1
log "Downloading $MODEL_ID${pin:+ @ ${pin:0:8}} into $HF_CACHE/hub"
if ! download "$MODEL_ID" "$pin"; then
  [[ -n "$pin" && -f "$dir/snapshots/$pin/config.json" ]] || die "$MODEL_ID: the download failed"
  warn "$MODEL_ID: could not reach Hugging Face; using the snapshot already here (${pin:0:8})"
fi
(( had )) || [[ -z "$pin" || -f "$dir/refs/main" ]] || { mkdir -p "$dir/refs"; printf %s "$pin" > "$dir/refs/main"; }
rev=$(snapshot_rev)
[[ -n "$rev" && -d "$dir/snapshots/$rev" ]] || die "$MODEL_ID: no snapshot after the download"
log "Checkpoint: $dir/snapshots/$rev ($(du -shL "$dir/snapshots/$rev" | cut -f1))"

log "Verifying the checkpoint"
CHECK_PY='
import glob, json, os, sys
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
if cfg.get("model_type") != "qwen4_exp":
    sys.exit("model_type is %r, not qwen4_exp" % cfg.get("model_type"))
q = cfg.get("quantization_config") or {}
kind = str(q.get("quant_method", "")).lower()
if kind == "modelopt":
    algos = {str(v.get("quant_algo")) for v in (q.get("quantized_layers") or {}).values()}
    if "NVFP4" not in algos:
        sys.exit("ModelOpt checkpoint without NVFP4 layers (quant_algo %s)" % sorted(algos))
    what = "ModelOpt %s (%s)" % (q.get("quant_algo"), "+".join(sorted(algos)))
elif kind == "gptq" and q.get("bits") == 4:
    table = [f for f in glob.glob(os.path.join(d, "ple-table", "*.safetensors")) if os.path.getsize(f) > 0]
    if not table:
        sys.exit("no n-gram table files in ple-table/")
    what = "AutoRound int4 g%s, %d n-gram table files" % (q.get("group_size"), len(table))
else:
    sys.exit("neither a ModelOpt NVFP4 nor a GPTQ-format int4 checkpoint (quant_method %r, bits %r)" % (kind, q.get("bits")))
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
bad = [s for s in shards if not os.path.isfile(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
if bad:
    sys.exit("missing or empty shards: " + ", ".join(bad[:5]))
print("qwen4_exp, %s, %d tensors in %d shards" % (what, len(idx["weight_map"]), len(shards)))'
snap="$dir/snapshots/$rev"
info=$(python3 -I -c "$CHECK_PY" "$snap" 2>&1) || die "the checkpoint at $snap is not ready: $info"
log "Checkpoint OK: $info"

if [[ "$ABLIT" == 1 ]]; then
  warn "ABLIT=1: $ABLIT_ID is gated and its model has had safety refusals removed. Accept its terms at" \
       "https://huggingface.co/$ABLIT_ID while logged in, with a read token for that account: export HF_TOKEN=hf_... or hf auth login."
  side_snap="$(ablit_cache_dir)/snapshots/$ABLIT_REVISION"
  log "Downloading the overlay's sidecar ($ABLIT_SIDECAR, ~190 MB) from $ABLIT_ID @ ${ABLIT_REVISION:0:8}"
  if ! download "$ABLIT_ID" "$ABLIT_REVISION" "$ABLIT_SIDECAR"; then
    [[ -s "$side_snap/$ABLIT_SIDECAR" ]] || die "$ABLIT_ID: the download failed (is its gate accepted by the account of this HF_TOKEN?)"
    warn "$ABLIT_ID: could not reach Hugging Face; using the sidecar already here (${ABLIT_REVISION:0:8})"
  fi
  [[ -s "$side_snap/$ABLIT_SIDECAR" ]] || die "$ABLIT_ID: no sidecar at $side_snap/$ABLIT_SIDECAR after the download"
  python3 -I scripts/ablit_view.py "$snap" "$side_snap/$ABLIT_SIDECAR" "$HF_CACHE/$ABLIT_VIEW_SUB" || die "could not build the ABLIT=1 checkpoint folder"
  info=$(python3 -I -c "$CHECK_PY" "$HF_CACHE/$ABLIT_VIEW_SUB" 2>&1) || die "the checkpoint at $HF_CACHE/$ABLIT_VIEW_SUB is not ready: $info"
  log "ABLIT=1 checkpoint OK ($ABLIT_VIEW_SUB): $info"
fi

while read -r repo rev path; do
  [[ -n "$repo" ]] || continue
  [[ -n "$path" ]] || die "TENSORFOLD_IDENTITIES: hf:$repo names no file (the form is hf:owner/repo/file[@revision])"
  log "Downloading the identity file $path from $repo @ $rev"
  idfile() { echo "$HF_CACHE/hub/models--${repo//\//--}/snapshots/$(hf_snapshot "$repo" "$rev")/$path"; }
  if ! download "$repo" "$rev" "$path"; then
    [[ -s "$(idfile)" ]] || die "$repo: the download of $path failed (a gated repo needs its terms accepted by the account of HF_TOKEN)"
    warn "$repo: could not reach Hugging Face; using the $path already here"
  fi
  [[ -s "$(idfile)" ]] || die "$repo: no $path at $(idfile) after the download"
done < <(hf_identities)
prepared_state > "$PREPARED_MARKER"
log "Done. Start the server with ./start.sh (port $PORT)."
