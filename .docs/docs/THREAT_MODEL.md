# Threat Model

## Scope

The service exposes an internal API for tracking readiness evidence. The most important assets are:

- service ownership metadata
- readiness evidence and audit history
- deployment integrity
- availability of health and readiness endpoints

## Trust boundaries

- API clients to application server
- application server to future persistent storage
- CI/CD system to production runtime
- operators to incident tooling and logs

## Primary threats and mitigations

### Unauthorized modification of readiness state

Risk:

- Attackers or accidental misuse could mark a service as ready without real evidence.

Mitigations:

- Require authentication and authorization in production
- Restrict mutating endpoints to service owners and platform operators
- Maintain immutable audit logs for write operations

### Injection or malformed input

Risk:

- Malformed payloads or future database queries could trigger unsafe behavior.

Mitigations:

- Strict request validation
- JSON-only API surface
- Parameterized queries when a database is introduced

### Sensitive data leakage

Risk:

- Ownership data, internal URLs, or incident evidence could leak through logs or public endpoints.

Mitigations:

- Avoid logging raw request bodies by default
- Keep evidence links internal
- Review logs for secrets before enabling broad access

### Denial of service

Risk:

- Unbounded requests or large payloads could exhaust CPU or memory.

Mitigations:

- Rate limiting at the edge
- Maximum request body size enforcement
- Horizontal scaling and autoscaling in production

### Supply-chain or deployment compromise

Risk:

- A malicious artifact or unauthorized deployment changes readiness state or exposes data.

Mitigations:

- Protected branches
- Reviewed pull requests
- Signed artifacts when supported
- Least-privilege deploy credentials

## Residual risk

The current reference implementation does not include auth, persistent audit logs, or rate limiting. Those gaps are acceptable for a teaching repository but must be closed before using this service in a real production environment.
