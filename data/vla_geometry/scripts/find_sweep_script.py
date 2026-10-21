#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Locate the frozen entry script of controlled_vla_geometry_sweep_v1.

Run this ON THE MACHINE THAT PRODUCED THE SWEEP (the one whose D: drive held
D:\NN_FRI_SAVEX15_Rebuild).  It only reads files; nothing is modified.

    py find_sweep_script.py
    py find_sweep_script.py --root D:\ --root E:\
    py find_sweep_script.py --root D:\NN_FRI_SAVEX15_Rebuild

It reports, in this order:
  1. any file whose SHA-256 equals the hash recorded in PROTOCOL_LOCK.json
  2. any .py whose name suggests the sweep
  3. any .py that mentions the sweep's identifying strings
  4. hits found INSIDE zip archives as well as on disk
"""

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

# the value frozen in PROTOCOL_LOCK.json -> locked_payload.script_sha256
TARGET = "7e8b5064e16a7f1945a8a3aa6afbfb93f310e0f51c3a2fbae11200935f325158"
HELPER = "9640c7a536afd0ffb27c604dd1c40c355181ae2f3d84d4feb6cd47338617b9a6"

NAME_HINTS = ("geometry", "sweep", "controlled_vla", "receiver_mask", "mask")
CONTENT_HINTS = ("controlled_vla_geometry_sweep", "geometry_variants", "sparse_fullspan",
                 "same_full_array_Y_then_mask_rows", "regenerate_noise_after_masking",
                 "retune_any_algorithm_parameter_by_geometry")

PRUNE = {".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", "dist", "build",
         "site-packages", "CondaPkgs", "anaconda3", "AppData", "Windows", "$Recycle.Bin"}
MAXSIZE = 8_000_000


def digest(b):
    return hashlib.sha256(b).hexdigest()


def scan_file(p: Path, hits, helper_hits, name_hits, content_hits):
    try:
        if p.stat().st_size > MAXSIZE:
            return
        if p.suffix.lower() == ".py":
            data = p.read_bytes()
            h = digest(data)
            if h == TARGET:
                hits.append(p)
            elif h == HELPER:
                helper_hits.append(p)
            if any(k in p.name.lower() for k in NAME_HINTS):
                name_hits.append(p)
            try:
                t = data.decode("utf-8", "ignore")
            except Exception:
                t = ""
            if any(k in t for k in CONTENT_HINTS):
                content_hits.append(p)
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if not n.lower().endswith((".py", ".json", ".md", ".txt")):
                        continue
                    try:
                        data = z.read(n)
                    except Exception:
                        continue
                    h = digest(data)
                    tag = "%s :: %s" % (p, n)
                    if h == TARGET:
                        hits.append(tag)
                    elif h == HELPER:
                        helper_hits.append(tag)
                    if n.lower().endswith(".py") and any(k in Path(n).name.lower() for k in NAME_HINTS):
                        name_hits.append(tag)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None,
                    help="directory to scan; repeatable. Defaults to every drive letter.")
    args = ap.parse_args()

    roots = [Path(r) for r in args.root] if args.root else \
            [Path("%s:\\" % c) for c in "CDEFGHIJKLMNOPQRSTUVWXYZ" if Path("%s:\\" % c).exists()]

    hits, helper_hits, name_hits, content_hits = [], [], [], []
    scanned = 0
    for root in roots:
        print("scanning %s ..." % root, flush=True)
        for p in root.rglob("*"):
            if any(part in PRUNE for part in p.parts):
                continue
            if p.is_file():
                scanned += 1
                scan_file(p, hits, helper_hits, name_hits, content_hits)

    print()
    print("=" * 74)
    print("scanned %d files under %d root(s)" % (scanned, len(roots)))
    print("=" * 74)
    print()
    print("1. EXACT MATCH for script_sha256  %s" % TARGET[:16] + "...")
    print("   %s" % ("\n   ".join(map(str, hits)) if hits else "NOT FOUND"))
    print()
    print("2. exact match for dynamic_helper_sha256 %s..." % HELPER[:16])
    print("   %s" % ("\n   ".join(map(str, helper_hits)) if helper_hits else "not found"))
    print()
    print("3. .py files whose NAME suggests the sweep (%d)" % len(name_hits))
    for x in name_hits[:25]:
        print("   %s" % x)
    print()
    print("4. .py files whose CONTENT mentions the sweep (%d)" % len(content_hits))
    for x in content_hits[:25]:
        print("   %s" % x)
    print()
    if not hits:
        print("NOTE: the frozen script was not found.  If it is genuinely gone, a rebuilt")
        print("      implementation can reproduce the numbers but can never carry this hash.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
