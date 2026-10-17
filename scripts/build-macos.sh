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

# Pass multiple Python versions to build the release matrix on this Mac.
if [[ "$#" -eq 0 ]]; then
  set -- "${PYTHON_VERSION:-3.13}"
fi
mkdir -p dist
output_dir="$(mktemp -d "$PWD/dist/local-macos.XXXXXX")"
for python_version in "$@"; do
  case "$python_version" in
    3.[89]|3.1[0-4]) ;;
    *) echo "Expected a supported Python version (3.8–3.14): $python_version" >&2; exit 1 ;;
  esac
  venv_dir="$PWD/.venv-macos-$python_version"
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    uv venv --python "$python_version" "$venv_dir"
  fi
  python="$venv_dir/bin/python"
  uv pip install --python "$python" 'maturin>=1.12,<2.0' pytest 'rouge-score==0.1.2'
  "$python" -m maturin build --release --locked --interpreter "$python" --out "$output_dir"
  # Select this interpreter's wheel, even when the output contains other versions.
  python_tag="cp${python_version//./}"
  uv pip install --python "$python" --no-deps --force-reinstall "$output_dir"/*-"$python_tag"-"$python_tag"-*.whl
  "$python" -m pytest -q
done
echo "Built and tested native macOS wheels in $output_dir"
