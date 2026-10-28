#!/bin/bash
# GL-Zero restart onto the .env IMAGE: small services paused for the load, memguard off meanwhile.
cd ~/stacks/gl-zero
log(){ echo "$(date +%T) $*"; }
avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
systemctl --user stop memguard gl-embed qwen-asr gl-tempo
./stop.sh >/dev/null 2>&1; docker rm -f qwen38-flash-next-tf >/dev/null 2>&1
sleep 5; log "services and GL-Zero stopped, MemAvailable $(avail) GiB"
( while sleep 1; do f=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    [ "$f" -lt 6000000 ] && { log "GUARD: MemAvailable $f kB, killing the server"; docker kill qwen38-flash-next-tf; }
  done ) & G=$!
./start.sh; rc=$?
kill $G
log "start.sh rc=$rc, MemAvailable $(avail) GiB"
for s in gl-tempo gl-embed qwen-asr; do systemctl --user start $s; sleep 20; log "$s started, MemAvailable $(avail) GiB"; done
systemctl --user start memguard
docker logs qwen38-flash-next-tf 2>&1 | grep -E "vision:|ablation:|identity [0-9]" | cut -c1-200
curl -sf -m 5 localhost:8888/v1/models | cut -c1-200
log RESTART_DONE
