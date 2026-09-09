# Unified v2 architecture

ToxPred owns weights, tokenization, raw/calibrated probabilities, thresholds,
decisions, applicability, uncertainty and attribution. ToxAgent Control owns
durable cases, plans, evidence reasoning, budgets, coverage, numeric
compilation and deterministic answer validation. Runtime adapters own only
session transport, normalized events, cancellation and cleanup.

Allowed dependency direction:

```text
frontend -> control -> ToxPred HTTP API
                    -> AgentRuntime -> capability-token MCP -> control tools
```

The product database and runtime transcript never become sources of predictor
truth. Runtime credentials are stored behind opaque secret references and are
not placed in rows, prompts, events or transcripts.
