# Data Model

## Overview

The domain is intentionally small:

- `services` represent deployable systems or major components
- `readiness_checks` represent evidence that a service has passed or failed a release gate

## Entities

### services

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Primary key |
| `name` | text | Human-readable service identifier, unique in production |
| `owner` | text | Team alias or email group |
| `tier` | text | Criticality classification such as `tier-1` |
| `description` | text | Short operational context |
| `lifecycle` | text | `active`, `deprecated`, or `experimental` |
| `created_at` | timestamptz | Creation time |
| `updated_at` | timestamptz | Last metadata update |

### readiness_checks

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Primary key |
| `service_id` | UUID | Foreign key to `services.id` |
| `category` | text | Example: `security`, `operability`, `compliance` |
| `status` | text | One of `pass`, `warn`, `fail` |
| `summary` | text | Concise explanation of the result |
| `evidence_url` | text | Optional link to dashboards, tickets, or documents |
| `checked_at` | timestamptz | When the check result was established |
| `created_at` | timestamptz | Record creation time |

## Relationships

- One service has many readiness checks
- Checks are append-heavy and should be queryable by `service_id`, `category`, and latest `checked_at`

## Suggested production indexes

- `services(name)` unique
- `readiness_checks(service_id, checked_at desc)`
- `readiness_checks(service_id, category, checked_at desc)`

## Retention guidance

- Keep service records indefinitely
- Keep readiness checks for at least 13 months to cover audit and retrospective needs
- Archive or compact superseded checks only if volume becomes operationally expensive
