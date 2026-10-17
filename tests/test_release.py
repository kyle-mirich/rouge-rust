"""Check hosted and combined release matrices without compiling foreign wheels."""

import io
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import fast_rouge

ROOT = Path(__file__).resolve().parents[1]
HOSTED = ["linux-x86_64", "linux-aarch64", "windows-x64"]
WHEEL_PLATFORMS = {
    "linux-x86_64": "manylinux_2_17_x86_64",
    "linux-aarch64": "manylinux_2_17_aarch64",
    "windows-x64": "win_amd64",
    "macos-arm64": "macosx_11_0_arm64",
    "macos-x86_64": "macosx_10_13_x86_64",
}


def make_distributions(directory, targets):
    version = fast_rouge.__version__
    with tarfile.open(directory / f"rouge_rust-{version}.tar.gz", "w:gz") as archive:
        sources = [p for p in ROOT.iterdir() if p.is_file()]
        for folder in ["docs", "tests", "examples", "scripts", "src"]:
            sources.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
        for source in sources:
            data = source.read_bytes()
            info = tarfile.TarInfo(f"rouge_rust-{version}/{source.relative_to(ROOT)}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    for target in targets:
        for minor in range(8, 15):
            python = f"cp3{minor}"
            wheel = directory / (
                f"rouge_rust-{version}-{python}-{python}-{WHEEL_PLATFORMS[target]}.whl"
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    f"rouge_rust-{version}.dist-info/METADATA",
                    f"Name: rouge-rust\nVersion: {version}\nRequires-Python: >=3.8\n",
                )
                archive.writestr("fast_rouge/__init__.pyi", "")
                archive.writestr("fast_rouge/py.typed", "")


def check(directory, *arguments):
    env = {key: value for key, value in os.environ.items() if key != "GITHUB_REF"}
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_release.py"),
            "--dist",
            str(directory),
            "--complete",
            *arguments,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.skipif(sys.version_info < (3, 11), reason="release checker requires Python 3.11+")
@pytest.mark.parametrize("case", ["hosted", "combined", "missing", "extra"])
def test_release_matrix(tmp_path, case):
    targets = list(WHEEL_PLATFORMS) if case in {"combined", "extra"} else HOSTED
    make_distributions(tmp_path, targets)
    if case == "missing":
        next(tmp_path.glob("*-cp314-cp314-win_amd64.whl")).unlink()
    arguments = [] if case == "combined" else ["--platforms", *HOSTED]
    result = check(tmp_path, *arguments)
    if case in {"hosted", "combined"}:
        assert result.returncode == 0, result.stderr
        assert f"Validated {35 if case == 'combined' else 21} wheels" in result.stdout
    else:
        assert result.returncode != 0
        assert "Wheel matrix mismatch" in result.stderr
