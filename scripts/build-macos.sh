#!/usr/bin/env bash
# Build and test a native macOS wheel without GitHub Actions or publication.
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
  echo "Run this script on macOS to build a native macOS wheel." >&2
  exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null
command -v cargo >/dev/null

if [[ ! -x .venv-macos/bin/python ]]; then
  uv venv --python "${PYTHON_VERSION:-3.13}" .venv-macos
fi
python="$PWD/.venv-macos/bin/python"
uv pip install --python "$python" 'maturin>=1.12,<2.0' pytest 'rouge-score==0.1.2'
mkdir -p dist
output_dir="$(mktemp -d "$PWD/dist/local-macos.XXXXXX")"
"$python" -m maturin build --release --locked --interpreter "$python" --out "$output_dir"
uv pip install --python "$python" --no-deps --force-reinstall "$output_dir"/*.whl
"$python" -m pytest -q
echo "Built and tested native macOS wheel in $output_dir"
