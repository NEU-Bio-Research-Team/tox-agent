# Unified v2 baseline

The machine-readable source of truth is
`benchmarks/manifests/unified-baseline-v1.json`. It links the frozen numerical
panel, scientific report, API/tool contracts, artifact hashes and runtime
versions. A missing measurement remains explicitly pending and cannot satisfy a
release gate.

Numerical parity uses a maximum absolute probability delta of `1e-6` on the
same CPU runtime. Contract changes require a reviewed snapshot diff. Scientific
behavior changes require a separately versioned artifact and cannot be hidden
inside a repository move.

The reference agent manifests under `toxagent-control/evals/manifests/` are
retained as the OpenCode live baseline. Re-running a credentialed matrix is a
release operation because it can consume provider quota.
