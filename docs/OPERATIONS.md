# Operations

Use `status` for Compose health and `logs [service]` for recent logs. `down`
stops containers but retains PostgreSQL data. `backup` writes a compressed SQL
dump under ignored `backups/`; `restore <file>` requires typing `RESTORE` and
replaces the current database.

Before an upgrade: run `backup`, record the image/tag version, then run
`down`, update to a tagged release and run `up`. Roll back by checking out the
previous release tag and running `up`; restore the matching backup if a schema
change requires it. Keep `.env` and `.artifacts/` outside source-control.

The normal first run downloads approximately 1.1 GB of OCR weights and builds
CPU images. Do not run benchmark suites or parallel image builds on a small
machine. Subsequent offline starts work when images and `.artifacts/` are
already cached.
