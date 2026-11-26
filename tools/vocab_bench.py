"""Decode speed and draft acceptance by language: greedy, thinking off, one request at a time.

Acceptance comes from /metrics (tensorfold:mtp_accepted_total over mtp_drafted_total) around each request, so run it
on an otherwise idle server. usage: python3 tools/vocab_bench.py [LABEL]
"""
import json
import re
import sys
import time
import urllib.request

from client import model_id

URL = "http://127.0.0.1:8888"
PROMPTS = {
    "french": ["Explique en détail, en français, comment on prépare un budget familial mensuel.",
               "Raconte en français l'histoire d'un phare sur la côte bretonne pendant une tempête.",
               "Rédige en français un guide pour bien choisir un vélo électrique."],
    "german": ["Erkläre ausführlich auf Deutsch, wie man einen Garten im Frühling vorbereitet.",
               "Schreibe auf Deutsch einen Leitfaden zur Auswahl einer Waschmaschine."],
    "english": ["Explain in detail how a household should plan a monthly budget.",
                "Write a guide to choosing an electric bicycle."],
    "code": ["Write a Python module with a class LRUCache (get, put, O(1)) and pytest tests for it. Output only Python source.",
             "Write a Go HTTP handler that validates a JSON body and stores it in a map with a mutex. Output only Go source."],
}


def metric(name):
    text = urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode()
    m = re.search(rf"^tensorfold:{name} (\S+)$", text, re.M)
    return float(m.group(1)) if m else 0.0


def decode(prompt):
    body = json.dumps({"model": model_id(), "max_tokens": 400, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    first = None; usage = {}
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            if any(c.get("delta", {}).get("content") for c in ev.get("choices", [])):
                first = first or time.time()
    n = usage.get("completion_tokens", 0)
    return n, (n - 1) / (time.time() - first)


label = sys.argv[1] if len(sys.argv) > 1 else "run"
for lang, prompts in PROMPTS.items():
    d0, a0 = metric("mtp_drafted_total"), metric("mtp_accepted_total")
    speeds = [decode(p)[1] for p in prompts]
    time.sleep(1)  # the counters count finished requests
    drafted, accepted = metric("mtp_drafted_total") - d0, metric("mtp_accepted_total") - a0
    print(json.dumps({"label": label, "lang": lang, "decode_tok_s": round(sum(speeds) / len(speeds), 1),
                      "accept": round(accepted / drafted, 3) if drafted else None, "drafted": int(drafted)}), flush=True)
