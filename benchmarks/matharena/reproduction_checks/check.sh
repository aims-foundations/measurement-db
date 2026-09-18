#!/usr/bin/env bash
# Custom checks.py selects frozen tasks, invokes native code, and exports native results.
# Shared chat_budget.py adds dollar accounting; upstream captures are never edited.
set -euo pipefail
check_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
benchmark_dir=$(dirname -- "$check_dir")
repo_dir=$(cd -- "$check_dir/../../.." && pwd)
export REPRO_BUDGET_FILE="${REPRO_BUDGET_FILE:-$check_dir/run_results/api-budget.json}"
source "$repo_dir/scripts/reproduce_evaluations/procedure.sh"
capture_upstream() {
    local name url revision archive captured
    while read -r name url revision; do
        archive="$benchmark_dir/raw/reproducibility/$name-$revision.tar.gz"
        captured="$benchmark_dir/raw/reproducibility/$name-$revision"
        if [[ ! -f "$archive" ]]; then
            git clone --filter=blob:none --no-checkout "$url.git" "$REPRO_WORK_DIR/checkout-$name"
            git -C "$REPRO_WORK_DIR/checkout-$name" fetch --depth=1 origin "$revision"
            git -C "$REPRO_WORK_DIR/checkout-$name" checkout --detach "$revision"
            test "$(git -C "$REPRO_WORK_DIR/checkout-$name" rev-parse HEAD)" = "$revision"
            mkdir -p "$(dirname -- "$archive")"
            git -C "$REPRO_WORK_DIR/checkout-$name" archive --format=tar.gz --prefix="$name-$revision/" HEAD > "$archive"
        fi
        if [[ ! -d "$captured" ]]; then
            mkdir "$captured"
            tar -xzf "$archive" --strip-components=1 -C "$captured"
        fi
        "$REPRO_PYTHON" "$repo_dir/scripts/reproduce_evaluations/capture_receipt.py" "$benchmark_dir" "$archive" "$url" "$revision"
        cp -a -- "$captured" "$REPRO_WORK_DIR/$name"
        # Restore Git metadata only in the writable copy, for native version reporting.
        git -C "$REPRO_WORK_DIR/$name" init --quiet
        git -C "$REPRO_WORK_DIR/$name" remote add origin "$url.git"
        git -C "$REPRO_WORK_DIR/$name" fetch --depth=1 origin "$revision"
        git -C "$REPRO_WORK_DIR/$name" reset --mixed "$revision"
        printf '%s %s %s\n' "$name" "$url" "$revision" >> "$REPRO_RUN_DIR/upstream-commits.txt"
    done <<'SOURCES'
source https://github.com/eth-sri/matharena b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b
SOURCES
    "$REPRO_PYTHON" "$check_dir/helpers/checks.py" inputs
    "$REPRO_PYTHON" "$bookkeeping" seal
}
prepare() {
    "$REPRO_PYTHON" -m venv "$REPRO_WORK_DIR/environment"
    # API-only path; no local model or Lean agent is used for these final-answer tasks.
    "$REPRO_WORK_DIR/environment/bin/pip" install 'openai==2.48.0' 'httpx==0.28.1' 'anthropic>=0.84.0' 'together' 'transformers' 'loguru' 'sympy' 'regex' 'antlr4-python3-runtime==4.11' 'pyyaml' 'numpy' 'datasets' 'pymupdf' 'docker' 'modal' 'axiom-axle' 'aristotlelib' 'zstandard' 'json5' 'pillow' "$REPRO_SOURCE_DIR/harness_wrapper"
    "$REPRO_WORK_DIR/environment/bin/pip" install --no-deps -e "$REPRO_SOURCE_DIR"
    "$REPRO_WORK_DIR/environment/bin/pip" freeze > "$REPRO_RUN_DIR/environment.freeze.txt"
}
exercise() { "$REPRO_WORK_DIR/environment/bin/python" "$check_dir/helpers/checks.py" exercise; }
rerun() {
    local status=0
    "$REPRO_PYTHON" "$bookkeeping" credential-run "$REPRO_WORK_DIR/environment/bin/python" "$check_dir/helpers/checks.py" rerun || status=$?
    if [[ -d "$REPRO_RUN_DIR/native/sources" ]]; then
        "$REPRO_PYTHON" "$benchmark_dir/build.py" --source "$REPRO_RUN_DIR/native" --output "$REPRO_RUN_DIR/tables"
    fi
    return "$status"
}
stage capture capture_upstream
stage prepare prepare
stage exercise exercise
if (( ! prepare_only )); then stage rerun rerun; fi
