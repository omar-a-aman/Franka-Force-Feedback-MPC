#!/usr/bin/env bash
set -e
unset PYTHONPATH
export PYTHONNOUSERSITE=1
exec "$@"
