#!/usr/bin/env bash
# Complete reproduction procedure. Upstream code is captured unchanged in raw/.
# Custom helpers: prepare_inputs.py selects the frozen tasks and records inputs;
# checks.py invokes the upstream agent/grader and exports their native results.
# model_entry.py (coding tasks) registers model metadata; run_agent.py invokes native CLIs.
# export_native.py places completed native results in the builder input layout.
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
source https://github.com/multi-swe-bench/MSWE-agent 88217624b637646b886cb0462995c07559e96f58
grader https://github.com/multi-swe-bench/multi-swe-bench 24f493f8a103e72312ded4f6b9c89f081d69cb09
SOURCES
    "$REPRO_PYTHON" "$check_dir/helpers/prepare_inputs.py"
    "$REPRO_PYTHON" "$bookkeeping" seal
}
prepare() {
    "$REPRO_PYTHON" -m venv "$REPRO_WORK_DIR/environment"
    "$REPRO_WORK_DIR/environment/bin/pip" install -r "$REPRO_SOURCE_DIR/requirements.txt" -e "$REPRO_WORK_DIR/grader" 'config==0.5.1' 'anthropic[bedrock]==0.49.0' 'swebench==1.1.5' 'httpx==0.28.1'
    "$REPRO_WORK_DIR/environment/bin/pip" freeze > "$REPRO_RUN_DIR/environment.freeze.txt"
    while read -r reference; do docker pull "$reference"; done < "$REPRO_RUN_DIR/selected-images.txt"
}
exercise() { "$REPRO_WORK_DIR/environment/bin/python" "$check_dir/helpers/checks.py" exercise; }
rerun() {
    local status=0
    "$REPRO_PYTHON" "$bookkeeping" credential-run "$REPRO_WORK_DIR/environment/bin/python" "$check_dir/helpers/checks.py" rerun || status=$?
    if "$REPRO_PYTHON" "$check_dir/helpers/export_native.py" "$REPRO_RUN_DIR"; then
        "$REPRO_PYTHON" "$benchmark_dir/build.py" --source "$REPRO_RUN_DIR/native" --output "$REPRO_RUN_DIR/tables" || return $?
    elif (( status == 0 )); then
        status=1
    fi
    return "$status"
}
stage capture capture_upstream
stage prepare prepare
stage exercise exercise
if (( ! prepare_only )); then
    stage rerun rerun
fi
