#!/usr/bin/env python3
"""Build clean, reproducible distribution archives (.tar.gz and .zip) of the project.

By default, builds archives strictly under the project's dist/ directory.
Optional export to another directory is supported via --export-dir, which refuses
to overwrite existing files unless --force is explicitly provided.

Excludes:
- .git / version control metadata
- .venv / virtual environments
- dist / generated archives
- __pycache__ / compiled bytecode (*.pyc, *.pyo)
- .pytest_cache / test artifacts
- .ruff_cache / linter caches
- .mypy_cache / type checker caches
- .DS_Store / OS metadata
- __MACOSX
"""

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__MACOSX",
}

EXCLUDED_FILES = {
    ".DS_Store",
}


def is_excluded(path: Path) -> bool:
    rel_parts = path.relative_to(PROJECT_ROOT).parts
    for part in rel_parts:
        if part in EXCLUDED_DIRS or part.endswith((".pyc", ".pyo")):
            return True
    return bool(path.name in EXCLUDED_FILES or path.name.endswith((".pyc", ".pyo")))


def get_files_to_pack() -> list[Path]:
    files = []
    for root, dirs, filenames in os.walk(PROJECT_ROOT):
        # Prune excluded directories in-place to avoid descending into them
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for f in filenames:
            file_path = Path(root) / f
            if not is_excluded(file_path):
                files.append(file_path)
    return sorted(files)


def build_tar_gz(output_path: Path, files: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        for file_path in files:
            arcname = os.path.relpath(file_path, PROJECT_ROOT)
            tar.add(file_path, arcname=f"Secure Smart Charging/{arcname}")


def build_zip(output_path: Path, files: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = os.path.relpath(file_path, PROJECT_ROOT)
            zf.write(file_path, arcname=f"Secure Smart Charging/{arcname}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean distribution archives.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional external destination directory to copy archives to.",
    )
    parser.add_argument(
        "--force",
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in --export-dir.",
    )
    args = parser.parse_args()

    files = get_files_to_pack()
    print(f"Collected {len(files)} files to package from {PROJECT_ROOT}")

    dist_dir = PROJECT_ROOT / "dist"
    tar_path = dist_dir / "secure_smart_charging.tar.gz"
    zip_path = dist_dir / "secure_smart_charging.zip"

    build_tar_gz(tar_path, files)
    print(f"Created {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)")

    build_zip(zip_path, files)
    print(f"Created {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    if args.export_dir is not None:
        export_dir = Path(args.export_dir).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)

        dest_tar = export_dir / tar_path.name
        dest_zip = export_dir / zip_path.name

        for dest, src in [(dest_tar, tar_path), (dest_zip, zip_path)]:
            if dest.exists() and not args.force:
                raise FileExistsError(
                    f"Destination file '{dest}' already exists. Use --force to overwrite."
                )
            shutil.copy2(src, dest)
            print(f"Exported to {dest} ({dest.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
