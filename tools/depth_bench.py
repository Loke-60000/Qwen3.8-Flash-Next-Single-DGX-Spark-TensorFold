"""Decode speed against context depth: a code task after N tokens of filler, greedy, thinking off.

usage: python3 tools/depth_bench.py DEPTH... (approximate prompt tokens)   prints one JSON line a depth
"""
import json, sys, time, urllib.request
from session_bench import doc, URL

TASK = ("Write a Python module with a class LRUCache (get, put, capacity, O(1) operations) and pytest tests for it. "
        "Output only Python source.")


def run(depth, seed):
    body = json.dumps({"model": "gl-zero", "max_tokens": 400, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": (doc(int(depth * 0.68), seed) + "\n\n" if depth > 600 else "") + TASK}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            if any(c.get("delta", {}).get("content") for c in ev.get("choices", [])):
                first = first or time.time()
    end = time.time()
    n = usage.get("completion_tokens", 0)
    return {"depth": usage.get("prompt_tokens"), "tokens": n, "ttft_s": round(first - t0, 2),
            "decode_tok_s": round((n - 1) / (end - first), 1) if n > 1 else None}


if __name__ == "__main__":
    for d in map(int, sys.argv[1:]):
        print(json.dumps(run(d, int(time.time()))), flush=True)
