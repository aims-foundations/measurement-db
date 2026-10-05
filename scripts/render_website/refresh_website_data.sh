#!/usr/bin/env bash
set -euo pipefail

# Rebuild the "Frontier Model Coverage" leaderboard end to end.
# Run beside the shared .hf-cache used by the generators.
cd "$(dirname "${BASH_SOURCE[0]}")"

python3 generate_benchmark_gallery.py all  # download new benchmark data + publish cards
python3 generate_benchmark_attributions.py # validate producer credit metadata
python3 generate_model_statistics.py       # regenerate model statistics CSV
python3 generate_chart_marginals.py        # regenerate chart marginals
python3 generate_model_timeline.py         # regenerate model timeline
