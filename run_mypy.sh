#!/usr/bin/env bash
# https://jaredkhan.com/blog/mypy-pre-commit

set -o errexit

cd "$(dirname "$0")"

pip install --editable ".[dev]" \
  --retries 1 \
  --no-input \
  --quiet

mypy multitasking
