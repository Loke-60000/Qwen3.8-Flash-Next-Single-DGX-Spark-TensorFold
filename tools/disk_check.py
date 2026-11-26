"""Disk tier exactness: prompt A cold, A pushed out of RAM by two long prompts, A again from disk; replies must match.

usage: python3 tools/disk_check.py [A_TOKENS] [FILLER_TOKENS]
"""
import json
import sys
import time

from client import model_id
from session_bench import doc, turn

MODEL = model_id()


def ask(text, seed):
    msgs = [{"role": "system", "content": "You answer in one short sentence."},
            {"role": "user", "content": doc(int(text * 0.55), seed) + "\n\nWhich word appears on line 3?"}]
    ttft, total, reply, usage = turn(MODEL, msgs)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(json.dumps({"prompt": usage.get("prompt_tokens"), "cached": cached, "ttft_s": round(ttft, 2), "reply": reply}), flush=True)
    return reply, cached or 0, usage.get("prompt_tokens") or 0


def main():
    a_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    filler = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
    seed = int(time.time())
    first, _, _ = ask(a_tokens, seed)
    ask(filler, seed + 1)
    ask(filler, seed + 2)
    time.sleep(5)  # idle: let the writer finish
    again, cached, prompt = ask(a_tokens, seed)
    ok = again == first and cached > prompt // 2
    print("DISK_CHECK", "PASS" if ok else "FAIL", f"cached={cached}/{prompt}", "same_reply=" + str(again == first))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
