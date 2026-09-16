#!/usr/bin/env bash
set -euo pipefail

readonly PROTOC_VERSION="36.1"
readonly PROTOC_SHA256="c4bc672d9d49214dc8cafdceadf4df92182d6ca8e3ec65a56b2d7de5602669b4"
readonly PROTOC_URL="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip"

temporary_directory="$(mktemp -d)"
trap 'rm -rf -- "$temporary_directory"' EXIT

archive_path="$temporary_directory/protoc.zip"
# The manylinux_2_28 image ships an older curl without --retry-all-errors.
curl --fail --location --retry 3 --silent --show-error \
    "$PROTOC_URL" --output "$archive_path"
echo "$PROTOC_SHA256  $archive_path" | sha256sum --check --status
unzip -q "$archive_path" -d /usr/local
protoc --version
