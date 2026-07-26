#!/usr/bin/env python3
"""Remap absolute image paths inside frozen JSONL manifests.

Typical use (public clone on a new machine)::

    # Replace a previous absolute data prefix with your local data root
    python scripts/remap_manifest_paths.py \\
        --in-dir manifests_raw \\
        --out-dir manifests \\
        --old-prefix /PREVIOUS/DATASET/ROOT \\
        --new-prefix /LOCAL/DATASET/ROOT

    # Or set each dataset root explicitly
    python scripts/remap_manifest_paths.py \\
        --in-dir manifests_raw \\
        --out-dir manifests \\
        --map DiffusionForensics=/data/DF \\
        --map UniversalFakeDetect=/data/UFD \\
        --map GANGen-Detection=/data/GANGen \\
        --map flux_sd3_eval=/data/flux_eval

    # Emit paths relative to --new-prefix (portable manifests)
    python scripts/remap_manifest_paths.py \\
        --in-dir manifests_raw --out-dir manifests \\
        --old-prefix /PREVIOUS/DATASET/ROOT \\
        --new-prefix /LOCAL/DATASET/ROOT \\
        --relative

The script never modifies inputs in place unless --in-place is set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath


DEFAULT_MARKERS = (
    "DiffusionForensics",
    "UniversalFakeDetect",
    "GANGen-Detection",
    "flux_sd3_eval",
)


def parse_map(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--map expects NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip().replace("\\", "/")
        if not name or not path:
            raise SystemExit(f"invalid --map: {item}")
        out[name] = path.rstrip("/")
    return out


def join_root(root: str, *parts: str) -> str:
    """Join without Windows drive rewriting for Unix-style roots."""
    root = root.replace("\\", "/").rstrip("/")
    cleaned = [p.replace("\\", "/").strip("/") for p in parts if p]
    if root.startswith("/"):
        return str(PurePosixPath(root).joinpath(*cleaned))
    base = Path(root).expanduser()
    for p in cleaned:
        base = base / p
    return str(base)


def split_on_marker(path: str, markers: tuple[str, ...]) -> tuple[str, str] | None:
    """Return (marker, relative_suffix under marker) if path contains a known dataset marker."""
    norm = path.replace("\\", "/")
    for m in markers:
        token = f"/{m}/"
        idx = norm.find(token)
        if idx >= 0:
            return m, norm[idx + len(token) :]
        token2 = f"/{m}"
        idx2 = norm.find(token2)
        if idx2 >= 0 and (idx2 + len(token2) == len(norm) or norm[idx2 + len(token2)] == "/"):
            return m, norm[idx2 + len(token2) :].lstrip("/")
    return None


def remap_path(
    path: str,
    *,
    old_prefix: str | None,
    new_prefix: str | None,
    root_map: dict[str, str],
    markers: tuple[str, ...],
    relative: bool,
) -> str:
    norm = path.replace("\\", "/")

    hit = split_on_marker(norm, markers)
    if hit is not None:
        marker, suffix = hit
        if marker in root_map:
            if relative:
                return Path(suffix).as_posix()
            return join_root(root_map[marker], suffix)

    if old_prefix and new_prefix is not None:
        old = old_prefix.replace("\\", "/").rstrip("/")
        if norm.startswith(old + "/") or norm == old:
            tail = norm[len(old) :].lstrip("/")
            if relative:
                return Path(tail).as_posix()
            return join_root(new_prefix, tail)

    if hit is not None and new_prefix is not None and not root_map:
        marker, suffix = hit
        if relative:
            return f"{marker}/{suffix}" if suffix else marker
        return join_root(new_prefix, marker, suffix)

    return path


def process_file(
    src: Path,
    dst: Path,
    *,
    old_prefix: str | None,
    new_prefix: str | None,
    root_map: dict[str, str],
    markers: tuple[str, ...],
    relative: bool,
    check_exists: bool,
    dry_run: bool,
) -> dict:
    n = 0
    changed = 0
    missing = 0
    samples: list[tuple[str, str]] = []
    lines_out: list[str] = []

    with open(src, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            old = row.get("path", "")
            new = remap_path(
                old,
                old_prefix=old_prefix,
                new_prefix=new_prefix,
                root_map=root_map,
                markers=markers,
                relative=relative,
            )
            n += 1
            if new != old:
                changed += 1
                if len(samples) < 3:
                    samples.append((old, new))
            row["path"] = new
            if check_exists and not relative and not Path(new).exists():
                missing += 1
            lines_out.append(json.dumps(row, ensure_ascii=False))

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

    return {
        "file": str(src),
        "out": str(dst),
        "n": n,
        "changed": changed,
        "missing": missing,
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, help="Directory of *.jsonl manifests")
    ap.add_argument("--out-dir", type=Path, help="Output directory (required unless --in-place)")
    ap.add_argument("--files", nargs="*", help="Explicit jsonl files (overrides --in-dir glob)")
    ap.add_argument("--old-prefix", default="/PREVIOUS/DATASET/ROOT", help="Absolute prefix to strip/replace")
    ap.add_argument("--new-prefix", default=None, help="New absolute data root")
    ap.add_argument("--map", action="append", default=[], help="NAME=PATH dataset root map (repeatable)")
    ap.add_argument("--markers", default=",".join(DEFAULT_MARKERS), help="Comma-separated dataset folder names")
    ap.add_argument("--relative", action="store_true", help="Write paths relative to dataset root / new-prefix")
    ap.add_argument("--check-exists", action="store_true", help="Count missing files after remap (absolute paths)")
    ap.add_argument("--dry-run", action="store_true", help="Print stats only; do not write")
    ap.add_argument("--in-place", action="store_true", help="Overwrite input files (discouraged)")
    args = ap.parse_args()

    markers = tuple(m.strip() for m in args.markers.split(",") if m.strip())
    root_map = parse_map(args.map)

    if args.in_place and args.out_dir:
        raise SystemExit("use either --in-place or --out-dir, not both")
    if not args.in_place and args.out_dir is None and not args.dry_run:
        raise SystemExit("--out-dir is required unless --dry-run or --in-place")

    if args.files:
        files = [Path(p) for p in args.files]
    elif args.in_dir:
        files = sorted(args.in_dir.glob("*.jsonl"))
    else:
        raise SystemExit("provide --in-dir or --files")

    if not files:
        raise SystemExit("no jsonl files found")

    new_prefix = args.new_prefix.replace("\\", "/").rstrip("/") if args.new_prefix else None
    reports = []
    for src in files:
        dst = src if args.in_place else ((args.out_dir / src.name) if args.out_dir else src)
        rep = process_file(
            src,
            dst,
            old_prefix=args.old_prefix,
            new_prefix=new_prefix,
            root_map=root_map,
            markers=markers,
            relative=args.relative,
            check_exists=args.check_exists,
            dry_run=args.dry_run,
        )
        reports.append(rep)
        print(f"{src.name}: n={rep['n']} changed={rep['changed']} missing={rep['missing']}", flush=True)
        for old, new in rep["samples"]:
            print(f"  {old}\n  -> {new}", flush=True)

    total_missing = sum(r["missing"] for r in reports)
    if args.check_exists and total_missing:
        print(f"WARNING: {total_missing} paths do not exist after remap", file=sys.stderr)
        sys.exit(2)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
