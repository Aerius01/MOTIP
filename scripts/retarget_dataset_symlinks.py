"""Retarget OceanFish/all image symlinks from the original local path to the HPC path.

On the local machine, symlinks in datasets/OceanFish/all/*/img1/ point 7 levels up:
    ../../../../../../../16-marlin-dataset/...  →  /home/david-james/Desktop/16-marlin-dataset/

On the HPC the dataset root is one level shallower (/scratch/david-james/ not /home/.../Desktop/),
so the correct relative target is 6 levels up:
    ../../../../../../16-marlin-dataset/...  →  /scratch/david-james/16-marlin-dataset/
"""

import argparse
import os
import pathlib
import sys


OLD_PREFIX = "../../../../../../../16-marlin-dataset/"
NEW_PREFIX = "../../../../../../16-marlin-dataset/"


def retarget(root: pathlib.Path, *, dry_run: bool) -> None:
    changed = 0
    skipped = 0
    errors = 0

    for link in sorted(root.rglob("*")):
        if not link.is_symlink():
            continue
        target = os.readlink(link)
        if not target.startswith(OLD_PREFIX):
            skipped += 1
            continue
        new_target = NEW_PREFIX + target[len(OLD_PREFIX):]
        changed += 1
        if dry_run:
            print(f"[dry-run] {link} -> {new_target}")
        else:
            try:
                link.unlink()
                link.symlink_to(new_target)
            except OSError as e:
                changed -= 1
                print(f"[ERROR] {link}: {e}", file=sys.stderr)
                errors += 1

    if dry_run:
        print(f"\nDry run complete — {changed + skipped} symlinks found, {skipped} already correct, would retarget {changed}.")
    else:
        print(f"\nDone — retargeted {changed} symlinks, {skipped} skipped (already correct), {errors} errors.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("/scratch/david-james/01-MOTIP/datasets/OceanFish/all"),
        help="Root directory to search for symlinks (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without modifying anything",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        sys.exit(f"Root directory not found: {args.root}")

    print(f"Scanning {args.root} ...")
    print(f"  old prefix: {OLD_PREFIX!r}")
    print(f"  new prefix: {NEW_PREFIX!r}")
    if args.dry_run:
        print("  mode: dry-run\n")

    retarget(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
