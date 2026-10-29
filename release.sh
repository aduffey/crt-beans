#!/bin/sh

# Usage: sh release.sh crt-beans-snapshot-20241024

BASE_PATH="$(realpath "$(dirname "$0")")"
TEMP_DIR="$(mktemp -d)"
mkdir "$TEMP_DIR/$1"
cp -r "$BASE_PATH/slang-shaders/." "$TEMP_DIR/$1/"
cd "$TEMP_DIR"
zip -r "$BASE_PATH/$1.zip" .
