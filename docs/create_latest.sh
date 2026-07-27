#!/usr/bin/env bash
set -euo pipefail

default_sdk="$(awk -F, '
/^[[:space:]]*($|#)/ { next }
{
    sdk = $1
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", sdk)
    for (i = 2; i <= NF; i++) {
    flag = $i
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", flag)
    if (tolower(flag) == "default") {
        print sdk
        exit
    }
    }
}
' docs/advertised_sdk.txt)"

if [[ -z "$default_sdk" ]]; then
    echo "::error::No default SDK found in docs/advertised_sdk.txt"
    exit 1
fi

default_sdk_dir="docs/$default_sdk"
if [[ ! -d "$default_sdk_dir" ]]; then
    echo "::error::Default SDK directory not found: $default_sdk_dir"
    exit 1
fi

for package_dir in "$default_sdk_dir"/*/; do
    [[ -d "$package_dir" ]] || continue
    package_name="$(basename "$package_dir")"

    echo "Copying $package_name from $default_sdk_dir to docs/$package_name"
    cp -a "$package_dir" "docs/$package_name"
done
