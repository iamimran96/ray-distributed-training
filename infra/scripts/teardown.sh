#!/usr/bin/env bash
# Delete the Ray kind cluster (and everything running on it).
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-ray}"

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  echo "==> Deleting kind cluster '$CLUSTER_NAME'"
  kind delete cluster --name "$CLUSTER_NAME"
else
  echo "==> kind cluster '$CLUSTER_NAME' not found, nothing to do"
fi
