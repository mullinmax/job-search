#!/bin/bash
set -euo pipefail
FILE="build_number.txt"
if [ ! -f "$FILE" ]; then
  echo "0" > "$FILE"
fi
NUM=$(cat "$FILE")
NUM=$((NUM + 1))
echo "$NUM" > "$FILE"

docker build --build-arg BUILD_NUMBER=$NUM -t job-ranker:$NUM -t job-ranker:latest .
