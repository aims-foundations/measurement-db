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
source https://github.com/PatrickHua/ResearchCodeBench 2758001c2ff84fc25c546339d65479ed058b0265
mmcv https://github.com/open-mmlab/mmcv d409eedc816fccfb1c8d57e5eed5f03bd075f327
mmgeneration https://github.com/Lakonik/mmgeneration 500a93e474638c87dcaf8fad94cfbe1ef29acd1f
diffusers https://github.com/huggingface/diffusers 57084dacc5275d7212513b24837b60a28e55603d
SOURCES
    "$REPRO_PYTHON" "$check_dir/helpers/checks.py" inputs
    "$REPRO_PYTHON" "$bookkeeping" seal
}
prepare() {
    # Native Python and CPU Torch versions; only dependencies of selected papers.
    "$REPRO_PYTHON" -m venv "$REPRO_WORK_DIR/bootstrap"
    "$REPRO_WORK_DIR/bootstrap/bin/pip" install uv
    "$REPRO_WORK_DIR/bootstrap/bin/uv" venv --python 3.11 --seed "$REPRO_WORK_DIR/environment"
    "$REPRO_WORK_DIR/environment/bin/pip" install 'numpy==1.26.4' 'torch==2.1.2+cpu' 'torchvision==0.16.2+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
    "$REPRO_WORK_DIR/environment/bin/pip" install 'setuptools==69.5.1' 'wheel' 'openai==2.48.0' 'httpx==0.28.1' 'pydantic' 'pyyaml' 'scipy' 'pandas' 'matplotlib' 'tqdm' 'huggingface-hub==0.25.2' 'transformers==4.40.2' 'addict' 'yapf==0.40.1' 'tensorboard' 'prettytable' 'lpips' 'click' 'pyarrow' 'einops' 'regex' 'scikit-image' 'torchsde' 'ninja' 'opencv-python-headless<4.12'
    "$REPRO_WORK_DIR/environment/bin/pip" install --no-deps 'mmcls==0.25.0'
    "$REPRO_WORK_DIR/environment/bin/pip" install --no-build-isolation --no-deps "$REPRO_WORK_DIR/mmcv" "$REPRO_WORK_DIR/mmgeneration" "$REPRO_WORK_DIR/diffusers"
    "$REPRO_WORK_DIR/environment/bin/pip" freeze > "$REPRO_RUN_DIR/environment.freeze.txt"
}
export CUDA_VISIBLE_DEVICES=""
export PYTHONBREAKPOINT=0
exercise() { PATH="$REPRO_WORK_DIR/environment/bin:$PATH" "$REPRO_WORK_DIR/environment/bin/python" "$check_dir/helpers/checks.py" exercise; }
rerun() {
    local status=0
    PATH="$REPRO_WORK_DIR/environment/bin:$PATH" "$REPRO_PYTHON" "$bookkeeping" credential-run "$REPRO_WORK_DIR/environment/bin/python" "$check_dir/helpers/checks.py" rerun || status=$?
    if [[ -f "$REPRO_RUN_DIR/native/overall_stats.json" ]]; then
        "$REPRO_PYTHON" "$benchmark_dir/build.py" --source "$REPRO_RUN_DIR/native" --output "$REPRO_RUN_DIR/tables"
    fi
    return "$status"
}
stage capture capture_upstream
stage prepare prepare
stage exercise exercise
if (( ! prepare_only )); then stage rerun rerun; fi
