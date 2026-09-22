# Releasing

The published product is the `rouge-rust` Python distribution. Deployment means
publishing wheels and an sdist to PyPI, followed by a GitHub Release with those
artifacts and `SHA256SUMS`. There is no separately deployed web service.

## Prepare a release

1. Start on up-to-date `main` with a clean working tree.
2. Update the version in `Cargo.toml`; refresh `Cargo.lock` with `cargo check`.
   Python metadata and `fast_rouge.__version__` both derive from the Cargo version.
3. Move changes from `Unreleased` into a dated version entry in `CHANGELOG.md`.
   Update the comparison links. Keep `Unreleased` for future changes.
4. Run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md) and `cargo audit`.
5. Build into a fresh version-specific output directory and check the artifacts:

```bash
# Example for 0.1.12; substitute the version being released.
uv run --extra dev maturin build --release --locked --sdist -o dist/0.1.12
uv run --extra dev twine check --strict dist/0.1.12/*
uv run --extra dev python scripts/check_release.py --dist dist/0.1.12
```

`scripts/check_release.py` requires Python 3.11+. Install the wheel in a fresh
virtual environment and run the Python tests there. Also build/install the sdist
in a separate environment. Inspect the wheel for its type stubs and `py.typed`;
the script checks both, along with essential source-distribution files.

6. Commit and push `main`. Wait for the full CI workflow to succeed.
7. Create and push **only the new tag** on that validated commit:

```bash
git tag -a v0.1.12 -m "Release 0.1.12"
git push origin v0.1.12
```

Never reuse a published version or move a released tag. The tag must equal `v`
plus the Cargo package version; CI rejects mismatches before publication.

## CI gates

The workflow in `.github/workflows/release.yml` performs:

- Rust formatting, Clippy, tests, documentation, and minimum-Rust-version checks.
- Python lint/format checks, parity/API tests, and installed-stub validation.
- Native build/install/test jobs for CPython 3.8–3.14 on five targets: Linux
  x86-64, Linux ARM64, macOS Intel, macOS ARM64, and Windows x64 (35 wheels).
- Source-distribution installation and Python tests.
- Strict distribution metadata validation and a complete wheel-matrix check.
- PyPI trusted publishing only for tags and only after every gate passes.
- A GitHub Release only after PyPI publishing succeeds, with changelog notes
  and SHA-256 checksums.

Actions are pinned to commit SHAs and tracked by Dependabot. Builds use the
committed Cargo lockfile. Release runs cannot be canceled by a newer push.
The checked build artifacts are passed to publication without rebuilding them.

## Trusted publishing configuration

The PyPI project must trust this GitHub publisher:

| Setting | Value |
| --- | --- |
| Owner | `kyle-mirich` |
| Repository | `rouge-rust` |
| Workflow | `release.yml` |
| Environment | `pypi` |

The publish job has OIDC permission; the release job has repository write
permission. Other jobs have read-only permissions. No long-lived PyPI token is
required. Any configured environment approval remains a maintainer action.

## Verify publication and recover from failures

After a successful tag workflow, verify the PyPI version and GitHub Release.
Install the exact version from PyPI with `--only-binary=:all:` in a fresh
environment, run the Python tests, and check `fast_rouge.__version__`.

If a build/test gate fails, nothing is published; fix `main` and validate it before
creating the next release tag. If publishing fails, inspect PyPI before retrying:
a network failure may occur after some files were accepted. Do not blindly retry
a partial upload or overwrite an existing version. Reconcile the existing files
and their checksums, or prepare a new patch version. If only GitHub Release
creation fails, rerun that failed job using the existing validated artifacts;
do not republish to PyPI.
