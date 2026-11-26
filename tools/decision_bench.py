"""A normal decode stream's speed alone, then with /v1/decisions requests arriving back to back beside it.

usage: python3 tools/decision_bench.py [MODEL]   (the decision model id, default gl-zero)
"""
import json, sys, threading, time, urllib.request
URL = "http://127.0.0.1:8888"
TEXT = " ".join(f"Line {i}: the shipment for order {i} left the warehouse on day {i % 28 + 1} and the client confirmed receipt." for i in range(70))
def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
def decode():
    body = {"model": "gl-zero", "max_tokens": 400, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": "Write a Python LRU cache class with tests. Output only Python source."}]}
    first = None; n = 0
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            ev = json.loads(line[5:])
            if ev.get("usage"): n = ev["usage"]["completion_tokens"]
            if any(c.get("delta", {}).get("content") for c in ev.get("choices", [])): first = first or time.time()
    return (n - 1) / (time.time() - first)
def decisions(stop, lat):
    while not stop.is_set():
        t = time.time()
        post("/v1/decisions", {"model": sys.argv[1] if len(sys.argv) > 1 else "gl-zero", "input": TEXT, "questions": [
            {"id": "late", "type": "yes_no", "question": "Was any shipment late?"},
            {"id": "tone", "type": "choice", "question": "Tone of the log?", "options": [{"name": "neutral"}, {"name": "urgent"}, {"name": "angry"}]}]})
        lat.append(time.time() - t)
alone = decode()
stop = threading.Event(); lat = []
th = threading.Thread(target=decisions, args=(stop, lat)); th.start(); time.sleep(1)
beside = decode(); stop.set(); th.join()
print(json.dumps({"decode_alone_tok_s": round(alone, 1), "decode_beside_decisions_tok_s": round(beside, 1),
                  "decisions_done": len(lat), "decision_latency_s": round(sum(lat) / len(lat), 2) if lat else None}))
