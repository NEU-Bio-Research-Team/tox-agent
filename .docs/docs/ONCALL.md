# On-Call Guide

## Coverage model

- Primary on-call: responds to alerts and customer-impacting incidents
- Secondary on-call: joins after 15 minutes or immediately for sev-1 incidents
- Engineering manager: informed for sev-1 and prolonged sev-2 incidents

## Escalation targets

| Level | Role | When to page |
| --- | --- | --- |
| 1 | Primary on-call | Any firing production alert |
| 2 | Secondary on-call | No acknowledgement within 15 minutes |
| 3 | Team lead | Incident exceeds 30 minutes or needs product decision |
| 4 | Engineering manager | Sev-1, external communication, or extended outage |

## Expected response times

- Sev-1: acknowledge within 5 minutes
- Sev-2: acknowledge within 15 minutes
- Sev-3: acknowledge within 1 business hour

## Minimum incident handling expectations

- Open an incident channel
- Assign an incident commander
- Post status updates every 15 minutes for active sev-1 or sev-2 incidents
- Record start time, impact, and mitigation steps

## Useful links

- Runbook: `docs/RUNBOOK.md`
- Threat model: `docs/THREAT_MODEL.md`
- Deployment guide: `docs/DEPLOYMENT.md`
