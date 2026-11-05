#!/usr/bin/env python3
"""KV cache quality: the same greedy requests against servers with different --kv-cache formats, compared with a bf16 run.

Usage: tools/kvq.py run LABEL OUT.json          (API_URL / PORT / MODEL as in client.py)
       tools/kvq.py compare REF.json OTHER.json ...
run sends, greedy with thinking off and a fixed seed:
  replies   short questions alone, then after ~32K and ~120K tokens of prose (one prose text per length, so the prompt
            cache can resume it between questions); a reply is compared with the reference's text
  needle    a passphrase in ~128K tokens (at 25% and 75%) and ~250K tokens (at 50%), as tools/needle.py builds it
  lookup    20 keys looked up among ~24K and ~100K tokens of key: value lines
compare prints, per file against REF: replies equal to the reference's and the mean share of the reference's text a
reply repeats before it differs, needles found, lookups right.
"""
import json
import os
import random
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client  # noqa: E402
import needle  # noqa: E402

QUESTIONS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain in four sentences why the sky is blue.",
    "A train leaves at 14:05 and arrives at 17:50. How long is the trip? Show the arithmetic.",
    "List five prime numbers between 100 and 150, comma separated.",
    "Translate into French: The meeting is moved to Thursday because the room is booked.",
    "What is the derivative of x^3 * sin(x)? Give the result and one line of reasoning.",
    "Write a SQL query that returns the three customers with the highest total order value from orders(customer_id, amount).",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Give a bash one-liner that counts the lines of every .py file under the current directory.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer and explain briefly.",
    "Write a haiku about a lighthouse in winter.",
    "What does HTTP status 409 mean, and when should an API return it?",
]
PADS = [(0, QUESTIONS), (32_000, QUESTIONS[:6]), (120_000, QUESTIONS[:4])]
NEEDLES = [(128_000, 0.25), (128_000, 0.75), (250_000, 0.5)]
LOOKUPS = [(2_000, 10), (8_000, 10)]          # key: value lines, keys asked
MAX_TOKENS = 160


def ask(prompt, max_tokens=MAX_TOKENS):
    r = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, thinking=False, temperature=0.0, seed=1234)
    return r.content, r.prompt_tokens


def prose(tokens, seed):
    if tokens == 0:
        return ""
    cpt, _ = client.calibrate(client.sentence, seed)
    return " ".join(client.filler_units(client.sentence, tokens, seed, cpt)) + "\n\n"


def lookup_table(lines, seed):
    rng = random.Random(f"kvq-{seed}")
    keys = rng.sample(range(100_000, 999_999), lines)
    table = {f"K{k}": f"{rng.randint(10_000, 99_999)}-{rng.choice(client._NOUNS)}" for k in keys}
    return table, "\n".join(f"{k}: {v}" for k, v in table.items())


def run(label, out):
    res = {"label": label, "replies": {}, "needle": {}, "lookup": {}}
    for pad, qs in PADS:
        text = prose(pad, seed=11)
        for i, q in enumerate(qs):
            content, n = ask(text + q)
            res["replies"][f"{pad}/{i}"] = content
            print(f"{label}: reply pad {pad} #{i} ({n} tokens): {content[:60]!r}", flush=True)
    for size, depth in NEEDLES:
        prompt, secret, _, _ = needle.build(size, 7, depth)
        content, n = ask(prompt, 48)
        found = secret.lower() in content.lower()
        res["needle"][f"{size}@{depth}"] = found
        print(f"{label}: needle {n} tokens at {depth:.0%}: {'found' if found else 'MISSED'} ({content.strip()[:60]!r})", flush=True)
    for lines, asks in LOOKUPS:
        table, text = lookup_table(lines, seed=lines)
        rng = random.Random(lines)
        right = 0
        for k in rng.sample(sorted(table), asks):
            content, n = ask(f"{text}\n\nWhat is the value of {k} in the list above? Reply with the value only.", 32)
            right += table[k].lower() in content.lower()
        res["lookup"][str(lines)] = [right, asks]
        print(f"{label}: lookup {lines} lines ({n} tokens): {right}/{asks}", flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return client.PASS


def shared(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), 1)


def compare(ref_path, paths):
    ref = json.load(open(ref_path))
    print(f"{'format':<14} {'same reply':>11} {'shared text':>12} {'pad 0/32K/120K same':>20} {'needles':>8} {'lookups':>8}")
    for p in [ref_path] + paths:
        r = json.load(open(p))
        keys = sorted(ref["replies"])
        same = [r["replies"][k] == ref["replies"][k] for k in keys]
        share = [shared(ref["replies"][k], r["replies"][k]) for k in keys]
        by_pad = "/".join(f"{sum(s for k, s in zip(keys, same) if k.startswith(f'{pad}/'))}of{len(qs)}" for pad, qs in PADS)
        needles = f"{sum(r['needle'].values())}/{len(r['needle'])}"
        right = sum(v[0] for v in r["lookup"].values())
        asked = sum(v[1] for v in r["lookup"].values())
        print(f"{r['label']:<14} {sum(same):>5}/{len(same):<5} {sum(share) / len(share):>11.1%} {by_pad:>20} {needles:>8} {right:>4}/{asked:<3}")
    return client.PASS


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "run":
        return run(sys.argv[2], sys.argv[3])
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        return compare(sys.argv[2], sys.argv[3:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    client.run(main)
