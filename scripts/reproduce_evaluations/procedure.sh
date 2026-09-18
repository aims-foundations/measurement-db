#!/usr/bin/env bash
# Shared bookkeeping only. Each check.sh contains its downloads, setup and run.
# Caller sets check_dir, benchmark_dir, repo_dir; no secrets are logged.
export REPRO_BENCHMARK_DIR="$benchmark_dir"
export REPRO_AGENT_CONFIG="$check_dir/luna.json"
export REPRO_PYTHON="${REPRO_PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
prepare_only=0
if [[ ${1:-} == --prepare-only ]]; then prepare_only=1; shift; fi
if (( $# )); then echo 'Usage: check.sh [--prepare-only]' >&2; exit 2; fi
export REPRO_RUN_DIR="${REPRO_RUN_DIR:-$check_dir/run_results/$(date -u +%Y%m%dT%H%M%S)-bash-pilot}"
# Compatibility with the older shared runner: keep this complete procedure in
# a distinct child run instead of overwriting that runner's own evidence.
if [[ -n ${REPRO_ARTIFACTS:-} && -d "$REPRO_RUN_DIR" ]]; then
    export REPRO_RUN_DIR="$REPRO_RUN_DIR/bash-$(date -u +%Y%m%dT%H%M%S)"
fi
mkdir -p -- "$(dirname -- "$REPRO_RUN_DIR")"
mkdir -- "$REPRO_RUN_DIR" # Never overwrite an earlier experiment.
export REPRO_WORK_DIR="$REPRO_RUN_DIR/work"
export REPRO_SOURCE_DIR="$REPRO_WORK_DIR/source"
mkdir -p -- "$REPRO_WORK_DIR" "$REPRO_WORK_DIR/tmp" "$REPRO_WORK_DIR/cache"
export TMPDIR="$REPRO_WORK_DIR/tmp"
export XDG_CACHE_HOME="$REPRO_WORK_DIR/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$XDG_CACHE_HOME/uv-python"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export PYTHONPATH="$repo_dir/scripts/reproduce_evaluations${PYTHONPATH:+:$PYTHONPATH}"
bookkeeping="$repo_dir/scripts/reproduce_evaluations/bash_pilot.py"
"$REPRO_PYTHON" "$bookkeeping" begin
export REPRO_TASKS
REPRO_TASKS=$("$REPRO_PYTHON" "$bookkeeping" tasks)
finish_procedure() {
    local code=$?
    trap - EXIT
    set +e
    "$REPRO_PYTHON" "$bookkeeping" finish "$code" "$prepare_only"
    exit "$?"
}
trap finish_procedure EXIT

stage() {
    local name=$1 code=0
    shift
    "$REPRO_PYTHON" "$bookkeeping" stage "$name" running
    set +e
    ( set -e; "$@" ) > >(tee "$REPRO_RUN_DIR/$name.stdout.log") 2> >(tee "$REPRO_RUN_DIR/$name.stderr.log" >&2)
    code=$?
    set -e
    if (( code )); then
        "$REPRO_PYTHON" "$bookkeeping" stage "$name" failed
        return "$code"
    fi
    "$REPRO_PYTHON" "$bookkeeping" stage "$name" passed
}
