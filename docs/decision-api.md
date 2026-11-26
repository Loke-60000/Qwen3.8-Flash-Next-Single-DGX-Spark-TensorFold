# Decision API

Decision ids answer choice, score and yes/no questions from the model's own label probabilities. They generate no text.
Speed numbers are in the [README](../README.md#decision-api).

## Turning it on

Add one line to `.env`, then run `./start.sh restart`:

```bash
TENSORFOLD_DECISION_IDS=Qwen3.8-Flash-Next-decision=Qwen3.8-Flash-Next
```

Each entry is `id=base`, where the base is the served name, an alias or an identity id; several entries are separated by
commas. `GET /v1/models` lists the ids, and a chat request to one is refused. An abliterated id can have its own
decision id:

```bash
TENSORFOLD_DECISION_IDS=Qwen3.8-Flash-Next-decision=Qwen3.8-Flash-Next,Qwen3.8-Flash-Next-abliterated-decision=Qwen3.8-Flash-Next-abliterated
```

## Requests

Two request shapes are served:

```bash
# OpenAI's shape: choice, score and yes_no questions
curl -s http://<spark-address>:8888/v1/decisions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-decision",
  "input": "The invoice is 3 months overdue and the client stopped answering emails.",
  "questions": [
    {"id": "risk", "type": "yes_no", "question": "Is this client a payment risk?"},
    {"id": "next", "type": "choice", "question": "Best next step?",
     "options": [{"name": "wait"}, {"name": "call"}, {"name": "legal"}]},
    {"id": "urgency", "type": "score", "question": "How urgent is this?", "levels": ["low", "medium", "high"]}
  ]}'

# Jev's System One shape: noul, choice and score questions
curl -s http://<spark-address>:8888/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-decision",
  "state": "The invoice is 3 months overdue and the client stopped answering emails.",
  "questions": {
    "risk": {"type": "noul", "instructions": "Is this client a payment risk?",
             "criteria": {"true": "the client may not pay", "false": "the client will pay"}},
    "next": {"type": "choice", "instructions": "Best next step?",
             "criteria": {"wait": "do nothing", "call": "phone them", "legal": "send to a lawyer"}}
  }}'
```

## Answers

The first answers (trimmed):

```json
{"object": "decisions", "model": "Qwen3.8-Flash-Next-decision",
 "answers": {
   "risk":    {"type": "yes_no", "probabilities": {"yes": 0.989, "no": 0.011}},
   "next":    {"type": "choice", "probabilities": {"wait": 0.004, "call": 0.991, "legal": 0.005}, "choice": "call"},
   "urgency": {"type": "score", "probabilities": {"0": 0.002, "1": 0.004, "2": 0.994}, "score": 1.99}},
 "usage": {"prompt_tokens": 158, "completion_tokens": 0, "total_tokens": 158}}
```

A score is the expected level index (0 = `low`, 2 = `high`). The second answers `"risk": {"type": "noul", "noul": 0.924}`
and `"next": {"type": "choice", "choice": "call", "confidence": 0.942, ...}`.

## Under load

Decisions run between decode rounds and take at most `TENSORFOLD_DECISION_SHARE` of the engine's time while other
requests decode (0.1 by default), so those requests keep about 96% of their speed. With nothing decoding, a decision
runs at full speed. Raise the share for faster decisions under load. A decision prompt is capped at
`TENSORFOLD_DECISION_MAX_TOKENS` (16,384) because its cache sits outside the streams' memory plan.
