#!/usr/bin/env bash
# Complete reproduction procedure. Upstream code is captured unchanged in raw/.
# Custom helpers: prepare_inputs.py selects the frozen tasks and records inputs;
# checks.py invokes the upstream agent/grader and exports their native results.
# chat_budget.py meters provider requests without changing the agent or grader.
# capture_receipt.py verifies code archive hashes. Shared bookkeeping records each stage.
# Any provider compatibility adaptation is explicit in helpers/, never in raw/.
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
source https://github.com/agi-inc/REAL 56d8f1643bdaa5e662086b1a357a9693c9ededf6
SOURCES
    "$REPRO_PYTHON" "$check_dir/helpers/prepare_inputs.py"
    "$REPRO_PYTHON" "$bookkeeping" seal
}
prepare() {
    "$REPRO_PYTHON" -m venv "$REPRO_WORK_DIR/environment"
    "$REPRO_WORK_DIR/environment/bin/pip" install -e "$REPRO_SOURCE_DIR" 'httpx==0.28.1'
    export PLAYWRIGHT_BROWSERS_PATH="$REPRO_WORK_DIR/browsers"
    "$REPRO_WORK_DIR/environment/bin/playwright" install chromium
    "$REPRO_WORK_DIR/environment/bin/pip" freeze > "$REPRO_RUN_DIR/environment.freeze.txt"
}
exercise() { "$REPRO_WORK_DIR/environment/bin/python" "$check_dir/helpers/checks.py" exercise; }
rerun() { "$REPRO_PYTHON" "$bookkeeping" credential-run "$REPRO_WORK_DIR/environment/bin/python" "$check_dir/helpers/checks.py" rerun; }
stage capture capture_upstream
stage prepare prepare
stage exercise exercise
if (( ! prepare_only )); then
    stage rerun rerun
fi
