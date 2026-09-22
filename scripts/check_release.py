"""Check release versions and distribution contents before publishing (Python 3.11+)."""

import argparse
import os
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, help="directory of built wheels and sdist")
    parser.add_argument("--complete", action="store_true", help="require all 35 CPython wheels")
    args = parser.parse_args()
    if args.complete and not args.dist:
        parser.error("--complete requires --dist")
    with (ROOT / "Cargo.toml").open("rb") as handle:
        version = tomllib.load(handle)["package"]["version"]
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/") and ref != f"refs/tags/v{version}":
        raise SystemExit(f"Release tag {ref!r} does not match Cargo version {version}")
    if f"## [{version}] - " not in (ROOT / "CHANGELOG.md").read_text():
        raise SystemExit(f"Missing changelog entry for {version}")
    if args.dist:
        wheels = sorted(args.dist.glob("*.whl"))
        sdists = sorted(args.dist.glob("*.tar.gz"))
        if not wheels or len(sdists) != 1:
            raise SystemExit("Expected at least one wheel and exactly one sdist")
        actual = set()
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
                info = BytesParser().parsebytes(archive.read(metadata))
                if info["Version"] != version or info["Name"] != "rouge-rust":
                    raise SystemExit(f"Wrong package/version in {wheel.name}")
                if info["Requires-Python"] != ">=3.8":
                    raise SystemExit(f"Unexpected Python requirement in {wheel.name}")
                for required in ["fast_rouge/__init__.pyi", "fast_rouge/py.typed"]:
                    if required not in names:
                        raise SystemExit(f"Missing {required} in {wheel.name}")
            package, file_version, python, abi, platform = wheel.stem.split("-", 4)
            if package != "rouge_rust" or file_version != version or python != abi:
                raise SystemExit(f"Unexpected wheel version or ABI: {wheel.name}")
            if platform.startswith("manylinux") and platform.endswith("_aarch64"):
                target = "linux-aarch64"
            elif platform.startswith("manylinux") and platform.endswith("_x86_64"):
                target = "linux-x86_64"
            elif platform.startswith("macosx") and platform.endswith("_arm64"):
                target = "macos-arm64"
            elif platform.startswith("macosx") and platform.endswith("_x86_64"):
                target = "macos-x86_64"
            elif platform == "win_amd64":
                target = "windows-x64"
            else:
                raise SystemExit(f"Unexpected wheel platform: {wheel.name}")
            actual.add((python, target))
        with tarfile.open(sdists[0]) as archive:
            names = archive.getnames()
            prefix = f"rouge_rust-{version}/"
            for required in [
                "Cargo.toml",
                "Cargo.lock",
                "pyproject.toml",
                "LICENSE",
                "fast_rouge.pyi",
                "tests/test_parity.py",
                "docs/api.md",
            ]:
                if prefix + required not in names:
                    raise SystemExit(f"Missing {required} in sdist")
        if args.complete:
            expected = {
                (f"cp3{minor}", target)
                for minor in range(8, 15)
                for target in [
                    "linux-aarch64",
                    "linux-x86_64",
                    "macos-arm64",
                    "macos-x86_64",
                    "windows-x64",
                ]
            }
            if actual != expected or len(wheels) != len(expected):
                raise SystemExit(
                    f"Wheel matrix mismatch: missing={expected - actual}, extra={actual - expected}"
                )
        print(f"Validated {len(wheels)} wheels and one sdist for {version}")
    else:
        print(f"Version and changelog agree: {version}")


if __name__ == "__main__":
    main()
