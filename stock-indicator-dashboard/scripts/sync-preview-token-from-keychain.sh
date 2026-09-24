#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
variables_file=${1:-"$repository_root/.dev.vars"}
keychain_service=${MOOTDX_CF_COLLECTOR_KEYCHAIN_SERVICE:-mootdx-cf-preview-collector-api-token}
keychain_account=${MOOTDX_CF_COLLECTOR_KEYCHAIN_ACCOUNT:-deployment-01}

collector_token=${MOOTDX_DATA_API_BEARER_TOKEN:-}
if [ -z "$collector_token" ]; then
    collector_token=$(
        security find-generic-password \
            -s "$keychain_service" \
            -a "$keychain_account" \
            -w
    )
fi
if [ -z "$collector_token" ]; then
    printf 'Keychain returned an empty deployment-01 Collector Token\n' >&2
    exit 1
fi

umask 077
temporary_file=$(mktemp "$variables_file.tmp.XXXXXX")
cleanup() {
    rm -f "$temporary_file"
}
trap cleanup EXIT HUP INT TERM

found=false
if [ -f "$variables_file" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            MOOTDX_DATA_API_BEARER_TOKEN=*)
                printf 'MOOTDX_DATA_API_BEARER_TOKEN=%s\n' "$collector_token" >> "$temporary_file"
                found=true
                ;;
            *)
                printf '%s\n' "$line" >> "$temporary_file"
                ;;
        esac
    done < "$variables_file"
fi

if [ "$found" != true ]; then
    printf 'MOOTDX_DATA_API_BEARER_TOKEN=%s\n' "$collector_token" >> "$temporary_file"
fi

chmod 600 "$temporary_file"
mv "$temporary_file" "$variables_file"
trap - EXIT HUP INT TERM

printf 'synchronized deployment-01 Collector Token into %s\n' "$variables_file"
