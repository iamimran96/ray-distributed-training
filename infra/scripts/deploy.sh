#!/usr/bin/env bash
# Create a local kind cluster, install the KubeRay operator, and start a Ray
# cluster whose dashboard / job API is reachable on http://127.0.0.1:8265.
#
# Usage:
#   ./infra/scripts/deploy.sh         # CPU-only: 1 head + 2 CPU workers
#   GPU=1 ./infra/scripts/deploy.sh   # with the local NVIDIA GPU: 1 head + 1 GPU worker
#
# Configuration (environment variables, all optional):
#   CLUSTER_NAME        kind cluster name                      (default: ray)
#   NAMESPACE           namespace for the operator and Ray     (default: ray)
#   KUBERAY_VERSION     kuberay-operator chart version         (default: 1.7.1)
#   GPU                 1 = GPU cluster via setup-gpu.sh       (default: 0)
#   MONITORING          1 = Prometheus + Grafana via setup-monitoring.sh (default: 1)
#   WAIT_TIMEOUT        how long to wait for Ray pods          (default: 20m)
#   MEMORY_BUDGET_GB    max memory for the whole setup         (default: 16)
#   SHARED_DIR          folder mounted as /mnt/shared in Ray pods (default: <repo>/shared)
set -euo pipefail

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(dirname "$INFRA_DIR")"
# Datasets, checkpoints, Ray results and the Hugging Face cache all live here, in
# the repo (gitignored), instead of inside Docker's VM.
SHARED_DIR="${SHARED_DIR:-${REPO_DIR}/shared}"

CLUSTER_NAME="${CLUSTER_NAME:-ray}"
NAMESPACE="${NAMESPACE:-ray}"
KUBERAY_VERSION="${KUBERAY_VERSION:-1.7.1}"
GPU="${GPU:-0}"
MONITORING="${MONITORING:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-20m}"
MEMORY_BUDGET_GB="${MEMORY_BUDGET_GB:-16}"
# Kubernetes system pods (API server, etcd, ...) have no limits; measured ~1.5 GiB on kind.
SYSTEM_RESERVE_MIB=1536
KUBE_CONTEXT="kind-${CLUSTER_NAME}"
DASHBOARD="http://127.0.0.1:8265"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
kc()   { kubectl --context "$KUBE_CONTEXT" "$@"; }

check_prereqs() {
  log "Checking prerequisites"
  local missing=()
  for tool in docker kind kubectl helm curl; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  [[ ${#missing[@]} -eq 0 ]] || fail "Missing tools: ${missing[*]}"
  docker info >/dev/null 2>&1 || fail "Docker daemon is not running. Start Docker Desktop and retry."
}

create_cluster() {
  if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    log "kind cluster '$CLUSTER_NAME' already exists, reusing it"
  else
    local template="${INFRA_DIR}/kind/cluster.yaml"
    [[ "$GPU" == "1" ]] && template="${INFRA_DIR}/kind/cluster-gpu.yaml"
    # Fill in the absolute path of the shared folder. Docker Desktop on Windows
    # needs a Windows-style path (C:/...); cygpath exists only in Git Bash/MSYS.
    mkdir -p "$SHARED_DIR"
    local host_path
    host_path="$(cd "$SHARED_DIR" && pwd)"
    command -v cygpath >/dev/null 2>&1 && host_path="$(cygpath -m "$host_path")"
    local config
    config="$(mktemp)"
    sed "s|__SHARED_DIR__|${host_path}|g" "$template" > "$config"
    log "Creating kind cluster '$CLUSTER_NAME' from $(basename "$template") (shared storage: ${host_path})"
    kind create cluster --name "$CLUSTER_NAME" --config "$config" --wait 5m
    rm -f "$config"
  fi
  kc wait --for=condition=Ready nodes --all --timeout=5m >/dev/null

  # Ray containers run as uid 1000; make the shared checkpoint dir writable.
  # (Windows bind mounts ignore chmod and are already writable, hence `|| true`.)
  for node in $(kind get nodes --name "$CLUSTER_NAME"); do
    docker exec "$node" sh -c 'mkdir -p /shared && chmod 1777 /shared' 2>/dev/null || true
  done

  if [[ "$GPU" == "1" ]]; then
    CLUSTER_NAME="$CLUSTER_NAME" "${INFRA_DIR}/scripts/setup-gpu.sh"
    build_gpu_image
  fi
}

# GPU pods use the Ray image plus gcc (Triton needs a C compiler); build it
# locally and load it into the kind nodes instead of pushing to a registry.
build_gpu_image() {
  local image="ray-gpu:2.58.0-py312"
  log "Building ${image} and loading it into kind"
  docker build -q -f "${INFRA_DIR}/docker/Dockerfile.gpu" -t "$image" "${INFRA_DIR}/docker" >/dev/null
  kind load docker-image "$image" --name "$CLUSTER_NAME" >/dev/null
}

install_kuberay() {
  log "Installing KubeRay operator ${KUBERAY_VERSION}"
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update >/dev/null
  helm repo update kuberay >/dev/null
  helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --version "$KUBERAY_VERSION" \
    --kube-context "$KUBE_CONTEXT" \
    --namespace "$NAMESPACE" --create-namespace \
    --set resources.requests.memory=64Mi,resources.limits.memory=256Mi \
    --wait --timeout 5m >/dev/null
}

# Sum every container's memory limit and check the setup stays within the budget.
# Kubernetes system pods have no limits, so a measured reserve is added for them.
check_memory_budget() {
  local limits unlimited limits_mib budget_mib total_mib
  limits="$(kc get pods -A -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.resources.limits.memory}{"\n"}{end}{end}')"
  limits_mib="$(echo "$limits" | awk '
    /Gi$/ {t += $0 * 1024; next}
    /Mi$/ {t += $0; next}
    /Ki$/ {t += $0 / 1024; next}
    /G$/  {t += $0 * 1e9 / 1048576; next}
    /M$/  {t += $0 * 1e6 / 1048576; next}
    /^[0-9]+$/ {t += $0 / 1048576}
    END {printf "%d", t}')"
  unlimited="$(echo "$limits" | grep -c '^$' || true)"
  budget_mib=$((MEMORY_BUDGET_GB * 1024))
  total_mib=$((limits_mib + SYSTEM_RESERVE_MIB))
  log "Memory: limits $(gib "$limits_mib") GiB + system reserve $(gib "$SYSTEM_RESERVE_MIB") GiB (${unlimited} system containers without limits) = $(gib "$total_mib") GiB of ${MEMORY_BUDGET_GB} GB budget"
  (( total_mib <= budget_mib )) || fail "Memory limits exceed the ${MEMORY_BUDGET_GB} GB budget; lower the resources in infra/k8s or infra/monitoring"
}
gib() { awk -v m="$1" 'BEGIN {printf "%.1f", m / 1024}'; }

deploy_ray() {
  local manifest="${INFRA_DIR}/k8s/raycluster-cpu.yaml"
  [[ "$GPU" == "1" ]] && manifest="${INFRA_DIR}/k8s/raycluster-gpu.yaml"
  log "Applying $(basename "$manifest") (first run pulls the ~1 GB Ray image)"
  kc -n "$NAMESPACE" apply -f "$manifest" -f "${INFRA_DIR}/k8s/head-service.yaml" >/dev/null

  log "Waiting for Ray pods to be ready (timeout ${WAIT_TIMEOUT})"
  # Pods are created by the operator, so wait until they exist before `kubectl wait`.
  for _ in $(seq 1 60); do
    [[ -n "$(kc -n "$NAMESPACE" get pods -l ray.io/cluster=ray -o name 2>/dev/null)" ]] && break
    sleep 2
  done
  kc -n "$NAMESPACE" wait pod -l ray.io/cluster=ray --for=condition=Ready --timeout="$WAIT_TIMEOUT" >/dev/null
  kc -n "$NAMESPACE" get pods -l ray.io/cluster=ray -o wide
}

verify() {
  log "Checking Ray job API at ${DASHBOARD}"
  for _ in $(seq 1 30); do
    if version="$(curl -fsS "${DASHBOARD}/api/version" 2>/dev/null)"; then
      log "Ray is up: ${version}"
      kc -n "$NAMESPACE" exec "$(kc -n "$NAMESPACE" get pod -l ray.io/node-type=head -o name)" -- ray status 2>/dev/null \
        | sed -n '/Resources/,$p' || true
      cat <<EOF

  Dashboard : ${DASHBOARD}   (Metrics tab shows Grafana panels when MONITORING=1)
  Grafana   : http://127.0.0.1:3000   (anonymous viewer; admin / prom-operator)
  Prometheus: http://127.0.0.1:9090
  Submit    : ray job submit --address ${DASHBOARD} --working-dir jobs/hello -- python hello.py
  (set up the CLI first: ./infra/scripts/setup-venv.sh, see README)

EOF
      return 0
    fi
    sleep 5
  done
  fail "Ray job API did not respond. Check: kubectl --context $KUBE_CONTEXT -n $NAMESPACE get pods"
}

check_prereqs
create_cluster
install_kuberay
deploy_ray
if [[ "$MONITORING" == "1" ]]; then
  CLUSTER_NAME="$CLUSTER_NAME" RAY_NAMESPACE="$NAMESPACE" "${INFRA_DIR}/scripts/setup-monitoring.sh"
fi
check_memory_budget
verify
