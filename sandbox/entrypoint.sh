#!/usr/bin/env bash
# Container entrypoint: overlay generated files (/gen, read-only) onto the
# pre-installed template (/app), then run the checks into /out.
set -eu
if [ -d /gen/src/app ]; then rm -rf /app/app; fi   # src/ layout replaces root app/
cp -R /gen/. /app/
exec /usr/local/bin/check.sh /app /out
