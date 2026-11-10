"""Moving recorded data from the lab computer to the storage of the university."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def transfer(source: str | Path, target: str | Path, delete_source: bool = False) -> dict:
    """Copy every file below `source` to `target` and verify each copy by its checksum.

    A local file is only deleted after its copy has been verified, and only when asked.
    """
    source, target = Path(source), Path(target)
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")
    if not target.is_dir():
        raise FileNotFoundError(f"Target folder not found or not reachable: {target}")
    if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents:
        raise ValueError("The target folder must not be the source folder or lie inside it.")
    copied, skipped, failed, deleted = [], [], [], []
    for file in sorted(p for p in source.rglob("*") if p.is_file()):
        dest = target / file.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = _digest(file)
        if dest.exists() and _digest(dest) == digest:
            skipped.append(str(file))
        else:
            shutil.copy2(file, dest)
            if _digest(dest) != digest:
                failed.append(str(file))
                continue
            copied.append(str(file))
        if delete_source:
            file.unlink()
            deleted.append(str(file))
    return {"copied": len(copied), "already_there": len(skipped), "failed": failed, "deleted": len(deleted)}
