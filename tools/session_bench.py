"""Growing-session test: a long cold prompt, then turns that append a reply and a short user message.

Prints time to first token and the server's cached count per turn. Standard library only.
usage: python3 tools/session_bench.py TOKENS_APPROX TURNS [MODEL]
"""
import json
import random
import sys
import time
import urllib.request

from client import model_id

URL = "http://127.0.0.1:8888/v1/chat/completions"


def doc(words, seed):
    rnd = random.Random(seed)
    vocab = "alpha beta gamma delta river stone cloud paper engine signal market garden copper silver window ledger".split()
    lines = []
    for i in range(words // 12):
        lines.append(f"{i}: " + " ".join(rnd.choice(vocab) for _ in range(11)) + ".")
    return "\n".join(lines)


def turn(model, messages):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 64, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                d = c.get("delta", {}).get("content")
                if d:
                    first = first or time.time()
                    text += d
    return (first or time.time()) - t0, time.time() - t0, text, usage


def main():
    tokens, turns = int(sys.argv[1]), int(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else model_id()
    seed = int(time.time())
    msgs = [{"role": "system", "content": "You answer in one short sentence."},
            {"role": "user", "content": doc(int(tokens * 0.55), seed) + "\n\nWhich word appears on line 3?"}]
    for t in range(turns):
        ttft, total, text, usage = turn(model, msgs)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        print(json.dumps({"turn": t, "prompt": usage.get("prompt_tokens"), "cached": cached,
                          "ttft_s": round(ttft, 2), "total_s": round(total, 2)}), flush=True)
        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": f"And line {t + 5}?"}]


if __name__ == "__main__":
    main()
