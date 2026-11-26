#!/usr/bin/env bash
# Bring the server back after a crash or memguard's last step: restart.sh with the speech services paused too, so the
# load sees the room it sizes its pool for, then /v1/models must list SERVED_NAME; one retry. memguard runs this as
# its own systemd unit, because restart.sh stops and starts memguard.
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
answers(){ for _ in $(seq 1 40); do curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$SERVED_NAME\"" && return 0; sleep 15; done; return 1; }
systemctl --user stop christina-tts-shim christina-tts 2>/dev/null
ok=0
for attempt in 1 2; do
  ./restart.sh && answers && { ok=1; break; }
  warn "attempt $attempt: $SERVED_NAME does not answer on port $PORT"
done
systemctl --user start christina-tts christina-tts-shim 2>/dev/null
(( ok )) && log "RECOVERED: $SERVED_NAME answers" || { warn "RECOVER_FAILED: $SERVED_NAME is down"; exit 1; }
