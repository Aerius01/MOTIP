"""
setup_kfold_splits.py

Creates the 5-fold cross-validation directory structure for the OceanFish dataset
under datasets/OceanFish/. Each fold gets three split sub-directories (train/, val/,
test/) populated with relative symlinks into datasets/OceanFish/all/, plus the
corresponding *_seqmap.txt files consumed by OceanFish dataset class and TrackEval.

Fold/group assignments (from EXPERIMENT_PLAN.md):

  Group 0: BB36_mob_m1, BB58_stat_s1, BB65_mob_m2
  Group 1: BB36_statmob_sm1, BB65_mob_m1, BB69_stat_s1
  Group 2: BB42_statmob_sm1, BB58_stat_s2, BB65_mob_m3
  Group 3: BB45_stat_s1, BB64_mob_m1, BB65_stat_s1
  Group 4: BB58_mob_m1, BB69_stat_s2

For fold k: test=group[k], val=group[(k+1)%5], train=remaining groups.

Pre-requisite:
  datasets/OceanFish/all/{seq_name}/ must exist for all 14 sequences, each
  containing seqinfo.ini, img1/, and gt/gt.txt in MOT format.

Usage:
  python tools/setup_kfold_splits.py [--data-root ./datasets]

After running, configure a fold in YAML with e.g.:
  DATASET_SPLITS: [fold0/train]
  INFERENCE_SPLIT: fold0/val       # or fold0/test for final eval
"""

import os
import argparse


FOLD_GROUPS = {
    0: ["BB36_mob_m1",      "BB58_stat_s1",  "BB65_mob_m2"],
    1: ["BB36_statmob_sm1", "BB65_mob_m1",   "BB69_stat_s1"],
    2: ["BB42_statmob_sm1", "BB58_stat_s2",  "BB65_mob_m3"],
    3: ["BB45_stat_s1",     "BB64_mob_m1",   "BB65_stat_s1"],
    4: ["BB58_mob_m1",      "BB69_stat_s2"],
}
K = 5


def get_fold_splits(fold_k: int) -> tuple[list, list, list]:
    """Return (train_seqs, val_seqs, test_seqs) for fold k."""
    test = FOLD_GROUPS[fold_k]
    val  = FOLD_GROUPS[(fold_k + 1) % K]
    train = []
    for g in range(K):
        if g != fold_k and g != (fold_k + 1) % K:
            train.extend(FOLD_GROUPS[g])
    return train, val, test


def write_seqmap(path: str, seq_names: list[str]) -> None:
    with open(path, "w") as f:
        f.write("name\n")
        for name in sorted(seq_names):
            f.write(f"{name}\n")


def make_symlink(link: str, target_abs: str) -> None:
    """Create (or replace) a relative symlink at `link` pointing to `target_abs`."""
    target_rel = os.path.relpath(target_abs, start=os.path.dirname(link))
    if os.path.islink(link):
        os.remove(link)
    elif os.path.exists(link):
        raise RuntimeError(f"{link} exists and is not a symlink — refusing to overwrite.")
    os.symlink(target_rel, link)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", default="./datasets",
        help="Path to the datasets root (default: ./datasets)",
    )
    args = parser.parse_args()

    oceanfish_dir = os.path.abspath(os.path.join(args.data_root, "OceanFish"))
    all_dir = os.path.join(oceanfish_dir, "all")

    # Validate that all expected sequences are present in all/
    all_seqs = [s for group in FOLD_GROUPS.values() for s in group]
    missing = [s for s in all_seqs if not os.path.isdir(os.path.join(all_dir, s))]
    if missing:
        print(f"ERROR: The following sequences are missing from {all_dir}:")
        for s in sorted(missing):
            print(f"  {s}")
        print("\nConvert the raw CVAT sequences to MOT format first, then re-run.")
        return 1

    print(f"Found all {len(all_seqs)} sequences in {all_dir}\n")

    for fold_k in range(K):
        train_seqs, val_seqs, test_seqs = get_fold_splits(fold_k)
        fold_dir = os.path.join(oceanfish_dir, f"fold{fold_k}")
        print(f"fold{fold_k}:")

        for split_name, seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
            split_dir = os.path.join(fold_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            for seq in seqs:
                make_symlink(
                    link=os.path.join(split_dir, seq),
                    target_abs=os.path.join(all_dir, seq),
                )

            seqmap_path = os.path.join(fold_dir, f"{split_name}_seqmap.txt")
            write_seqmap(seqmap_path, seqs)
            print(f"  {split_name:5s} ({len(seqs):2d} seqs): {sorted(seqs)}")

        print()

    print(f"Done. Fold directories created under {oceanfish_dir}/")
    print()
    print("To use in training, set in your YAML:")
    print("  DATASET_SPLITS: [fold0/train]")
    print("  INFERENCE_SPLIT: fold0/val    # for early-stopping val")
    print("  INFERENCE_DATASET: OceanFish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
