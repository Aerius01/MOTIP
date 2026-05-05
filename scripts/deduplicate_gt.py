"""Remove duplicate (frame_id, obj_id) rows from MOT-format gt.txt files.

Keeps the first occurrence of each (frame, id) pair and discards subsequent
duplicates, which cause TrackEval to raise TrackEvalException.
"""

import argparse
import pathlib
import sys


def deduplicate_gt(root: pathlib.Path, *, dry_run: bool) -> None:
    gt_files = sorted(root.rglob("gt/gt.txt"))
    print(f"Scanning {len(gt_files)} gt.txt files under {root}...")
    total_removed = 0
    for gt_path in gt_files:
        lines = gt_path.read_text().splitlines()
        seen: set[tuple[str, str]] = set()
        deduped: list[str] = []
        removed = 0
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                key = (parts[0], parts[1])
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
            deduped.append(line)
        if removed:
            total_removed += removed
            if dry_run:
                print(f"  [dry-run] {gt_path.relative_to(root)}: would remove {removed} lines")
            else:
                gt_path.write_text("\n".join(deduped) + "\n")
                print(f"  {gt_path.relative_to(root)}: removed {removed} lines")
    if total_removed == 0:
        print("No duplicates found.")
    else:
        action = "would remove" if dry_run else "removed"
        print(f"Done — {action} {total_removed} total duplicate lines.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("/scratch/david-james/01-MOTIP/datasets/MFT_SeaDroneSee_merged"),
        help="Root directory to search for gt/gt.txt files (default: %(default)s)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying files")
    args = parser.parse_args()

    if not args.root.is_dir():
        sys.exit(f"Root directory not found: {args.root}")

    deduplicate_gt(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
