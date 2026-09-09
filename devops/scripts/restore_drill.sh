#!/usr/bin/env bash
# Prove a backup restores onto a *different* database, that what comes back is
# what went in, and that a container recreate leaves the state volume alone
# (K07 / I16).
#
# The operations runbook describes backup and restore. A runbook is not a
# drill: `toxagent backup` piping into `toxagent restore` on the same database
# proves neither that the dump is complete nor that it can be read by an
# instance that was never the source. This runs the real thing — pg_dump from
# a migrated, populated database into a second, empty PostgreSQL container,
# then compares every table row-for-row.
#
#   devops/scripts/restore_drill.sh
#
# Needs Docker and a PostgreSQL image. It creates and removes its own two
# containers and touches nothing else; the ports are unusual on purpose so a
# running stack is never the subject.
set -euo pipefail

IMAGE="${DRILL_POSTGRES_IMAGE:-postgres:16}"
SOURCE="toxagent-drill-source"
TARGET="toxagent-drill-target"
SOURCE_PORT="${DRILL_SOURCE_PORT:-55440}"
TARGET_PORT="${DRILL_TARGET_PORT:-55441}"
WORK="$(mktemp -d)"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${DRILL_PYTHON:-python3}"

cleanup() { docker rm -f "$SOURCE" "$TARGET" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

start() {
  local name="$1" port="$2"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" -e POSTGRES_PASSWORD=toxagent -e POSTGRES_USER=toxagent \
    -e POSTGRES_DB=toxagent -p "$port:5432" "$IMAGE" >/dev/null
  for _ in $(seq 1 60); do
    docker exec "$name" pg_isready -U toxagent -q 2>/dev/null && return 0
    sleep 1
  done
  echo "drill: $name never became ready" >&2; exit 1
}

# Every public table with its row count, so an empty table restored as empty
# and a populated one restored as populated are distinguishable. Counting has
# to be per table, which needs the query built per table — the shape below is
# the standard `query_to_xml` way of doing that in one statement.
# `alembic_version` is included deliberately: a restore that loses the schema
# version leaves an instance that would try to migrate itself again. Contents
# beyond the counts are checked by --verify, through the repositories.
fingerprint() {
  docker exec -i "$1" psql -qtAX -U toxagent -d toxagent <<'SQL'
SELECT string_agg(format('%s=%s', table_name, rows), E'\n' ORDER BY table_name)
FROM (
  SELECT table_name,
         (xpath('/row/c/text()',
           query_to_xml(format('SELECT count(*) AS c FROM %I.%I', table_schema, table_name),
                        false, true, '')))[1]::text::bigint AS rows
  FROM information_schema.tables
  WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
) t;
SQL
}

echo "drill: starting source and target instances ($IMAGE)"
start "$SOURCE" "$SOURCE_PORT"
start "$TARGET" "$TARGET_PORT"

echo "drill: migrating the source from empty"
( cd "$REPO/backend/control" && PYTHONPATH=src \
  TOXAGENT_ALEMBIC_URL="postgresql+psycopg://toxagent:toxagent@localhost:$SOURCE_PORT/toxagent" \
  "$PY" -m alembic -c alembic.ini upgrade head >/dev/null )

echo "drill: writing product data through the repositories"
( cd "$REPO/backend/control" && PYTHONPATH=src \
  TOXAGENT_DRILL_URL="postgresql+asyncpg://toxagent:toxagent@localhost:$SOURCE_PORT/toxagent" \
  "$PY" "$REPO/devops/scripts/restore_drill_seed.py" )

echo "drill: pg_dump from the source"
docker exec -i "$SOURCE" pg_dump -U toxagent toxagent | gzip > "$WORK/backup.sql.gz"
gzip -t "$WORK/backup.sql.gz"
echo "drill: backup is $(stat -c%s "$WORK/backup.sql.gz") bytes"

# The target has never seen this application. This is the part a same-database
# restore cannot test.
echo "drill: restoring into an instance that was never the source"
set -o pipefail
gzip -dc "$WORK/backup.sql.gz" \
  | docker exec -i "$TARGET" psql -v ON_ERROR_STOP=1 -q -U toxagent -d toxagent >/dev/null
set +o pipefail

echo "drill: comparing"
before="$(fingerprint "$SOURCE")"
after="$(fingerprint "$TARGET")"
if [[ "$before" != "$after" ]]; then
  echo "drill: FAILED — restored contents differ" >&2
  diff <(echo "$before") <(echo "$after") >&2 || true
  exit 1
fi

( cd "$REPO/backend/control" && PYTHONPATH=src \
  TOXAGENT_DRILL_URL="postgresql+asyncpg://toxagent:toxagent@localhost:$TARGET_PORT/toxagent" \
  "$PY" "$REPO/devops/scripts/restore_drill_seed.py" --verify )

# ------------------------------------------------------------------ volumes
#
# The other half of I16. `test_the_compose_stack_mounts_a_volume_over_both
# _directories` asserts the compose file declares the volume; it cannot assert
# that a container recreate leaves the bytes behind, which is the thing an
# operator actually needs to be true. Attachment bytes and AI-profile
# credentials live under one mount point, and losing them leaves the rows in
# `attachments` and `model_connections` pointing at nothing.
#
# The volume name and mount path are read out of compose.yaml rather than
# typed here, so renaming either in the deployment and not in the drill fails
# instead of silently testing a path nothing uses.
# The volume name, its mount path and the two directories under it are read
# out of compose.yaml rather than typed here, so renaming any of them in the
# deployment and not in the drill fails instead of silently exercising a path
# nothing uses.
read -r VOLUME_NAME VOLUME_PATH ATTACH_DIR SECRETS_DIR <<<"$("$PY" - "$REPO/devops/compose/compose.yaml" <<'PYEOF'
import sys, yaml
compose = yaml.safe_load(open(sys.argv[1]))
declared = compose.get("volumes", {})
service = next(
    s for s in compose["services"].values()
    if any(str(m).split(":")[0] in declared for m in (s.get("volumes") or []))
    and "TOXAGENT_SECRETS_DIR" in (s.get("environment") or {})
)
name, path = next(
    (str(m).split(":")[0], str(m).split(":")[1])
    for m in service["volumes"] if str(m).split(":")[0] in declared
)
env = service["environment"]
attachments, secrets = env["TOXAGENT_OBJECT_STORE_DIR"], env["TOXAGENT_SECRETS_DIR"]
for directory in (attachments, secrets):
    assert directory.startswith(path.rstrip("/") + "/"), (
        f"{directory} is not under the {name} mount at {path}: a rebuild would destroy it"
    )
print(name, path, attachments, secrets)
PYEOF
)"
DRILL_VOLUME="toxagent-drill-${VOLUME_NAME}"
echo "drill: state volume '$VOLUME_NAME' at $VOLUME_PATH covers $ATTACH_DIR and $SECRETS_DIR"

docker volume rm -f "$DRILL_VOLUME" >/dev/null 2>&1 || true
docker volume create "$DRILL_VOLUME" >/dev/null
trap 'cleanup; docker volume rm -f "$DRILL_VOLUME" >/dev/null 2>&1 || true' EXIT

# An attachment and a credential, written the way the stores write them: the
# secret directory closed, because a credential store that inherits a
# world-readable directory is one in name only.
docker run --rm -v "$DRILL_VOLUME:$VOLUME_PATH" "$IMAGE" sh -c "
  mkdir -p '$ATTACH_DIR' '$SECRETS_DIR'
  chmod 700 '$SECRETS_DIR'
  printf 'attachment-bytes' > '$ATTACH_DIR/att_drill'
  printf 'sk-drill-not-a-real-key' > '$SECRETS_DIR/sec_drill'
  chmod 600 '$SECRETS_DIR/sec_drill'
" >/dev/null

echo "drill: recreating the container over the same volume"
survived="$(docker run --rm -v "$DRILL_VOLUME:$VOLUME_PATH" "$IMAGE" sh -c "
  cat '$ATTACH_DIR/att_drill'
  printf ' '
  cat '$SECRETS_DIR/sec_drill'
  printf ' '
  stat -c '%a' '$SECRETS_DIR/sec_drill'
")"
expected="attachment-bytes sk-drill-not-a-real-key 600"
if [[ "$survived" != "$expected" ]]; then
  echo "drill: FAILED — state did not survive the recreate" >&2
  echo "  expected: $expected" >&2
  echo "  got:      $survived" >&2
  exit 1
fi

echo "drill: PASSED — the restored instance holds the same data and the same schema version,"
echo "drill:          and attachments and credentials survived a container recreate"
