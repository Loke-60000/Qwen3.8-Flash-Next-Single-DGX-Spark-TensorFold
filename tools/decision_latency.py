"""Decision latency on GL-Zero's decision ids: time from request to answer, median of --reps runs, printed as CSV blocks.

usage: python3 tools/decision_latency.py [--reps 3]
Sweeps: input length x questions (yes/no), options per choice question, question type, endpoint and identity, and
idle against one request decoding beside it. Inputs are generated shipment logs; the server must be otherwise idle.
"""
import argparse, json, statistics, threading, time, urllib.request

URL = "http://127.0.0.1:8888"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return time.time() - t, r


def tokens(text):
    return post("/tokenize", {"model": "gl-zero", "prompt": text})[1]["count"]


def log(lines):
    return " ".join(f"Line {i}: the shipment for order {i} left the warehouse on day {i % 28 + 1} and the client confirmed receipt." for i in range(lines))


def yes_no(n):
    return [{"id": f"q{i}", "type": "yes_no", "question": f"Did order {i * 7} leave on day {i % 28 + 1}?"} for i in range(n)]


def timed(reps, path, body):
    runs = [post(path, body)[0] for _ in range(reps)]
    return statistics.median(runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    reps = a.reps
    inputs = {n: log(n) for n in (1, 12, 50, 200)}
    sizes = {n: tokens(text) for n, text in inputs.items()}

    print("```csv\ninput_tokens,questions,latency_s,per_question_s")
    for n, text in inputs.items():
        for q in (1, 2, 4, 8, 16):
            s = timed(reps, "/v1/decisions", {"model": "gl-zero-decision", "input": text, "questions": yes_no(q)})
            print(f"{sizes[n]},{q},{s:.3f},{s / q:.3f}", flush=True)
    print("```\n")

    text = inputs[50]
    print("```csv\noptions,input_tokens,latency_s")
    for k in (2, 3, 5, 10, 26):
        opts = [{"name": f"option{j}"} for j in range(k)]
        s = timed(reps, "/v1/decisions", {"model": "gl-zero-decision", "input": text, "questions": [{"id": "c", "type": "choice", "question": "Which option fits the log best?", "options": opts}]})
        print(f"{k},{sizes[50]},{s:.3f}", flush=True)
    print("```\n")

    print("```csv\nquestion_type,input_tokens,latency_s")
    kinds = {
        "yes_no": {"id": "q", "type": "yes_no", "question": "Was any shipment late?"},
        "choice_3": {"id": "q", "type": "choice", "question": "Tone of the log?", "options": [{"name": "neutral"}, {"name": "urgent"}, {"name": "angry"}]},
        "score_5": {"id": "q", "type": "score", "question": "How complete is the log?", "levels": ["very poor", "poor", "fair", "good", "very good"]},
    }
    for name, q in kinds.items():
        s = timed(reps, "/v1/decisions", {"model": "gl-zero-decision", "input": text, "questions": [q]})
        print(f"{name},{sizes[50]},{s:.3f}", flush=True)
    print("```\n")

    print("```csv\nendpoint,model,questions,input_tokens,latency_s")
    for model in ("gl-zero-decision", "gl-zero-skuld-decision"):
        for q in (1, 4):
            s = timed(reps, "/v1/decisions", {"model": model, "input": text, "questions": yes_no(q)})
            print(f"/v1/decisions,{model},{q},{sizes[50]},{s:.3f}", flush=True)
            qs = {f"q{i}": {"type": "noul", "instructions": f"Did order {i * 7} leave on day {i % 28 + 1}?", "criteria": {"true": "yes", "false": "no"}} for i in range(q)}
            s = timed(reps, "/v1/systemone", {"model": model, "state": text, "questions": qs})
            print(f"/v1/systemone,{model},{q},{sizes[50]},{s:.3f}", flush=True)
    print("```\n")

    # beside one request decoding: a code task decoding 2,000 tokens while the decisions run
    stop = threading.Event()

    def decode():
        body = {"model": "gl-zero", "max_tokens": 2000, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "Write a long Python module with many small utility functions and tests. Output only Python source."}]}
        while not stop.is_set():
            post("/v1/chat/completions", body)

    th = threading.Thread(target=decode, daemon=True)
    th.start()
    time.sleep(3)
    print("```csv\nserver,input_tokens,questions,latency_s")
    for n in (12, 200):
        for q in (1, 4):
            s = timed(reps, "/v1/decisions", {"model": "gl-zero-decision", "input": inputs[n], "questions": yes_no(q)})
            print(f"one_request_decoding,{sizes[n]},{q},{s:.3f}", flush=True)
    stop.set()
    th.join(timeout=300)
    time.sleep(2)
    for n in (12, 200):
        for q in (1, 4):
            s = timed(reps, "/v1/decisions", {"model": "gl-zero-decision", "input": inputs[n], "questions": yes_no(q)})
            print(f"idle,{sizes[n]},{q},{s:.3f}", flush=True)
    print("```")


if __name__ == "__main__":
    main()
