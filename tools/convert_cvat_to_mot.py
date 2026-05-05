"""
convert_cvat_to_mot.py

Converts the raw OceanFish CVAT dataset (CVAT XML 1.1 interpolation format) into
the MOT-challenge directory layout expected by MOTIP:

  datasets/OceanFish/all/{seq_name}/
    seqinfo.ini
    gt/gt.txt
    img1/{i:08d}.jpg  →  (symlink) raw PNG at 16-marlin-dataset/{seq}/images/{i-1}.png

Only gt/gt.txt and seqinfo.ini are generated — images are referenced via lightweight
relative symlinks, so no data is duplicated. PIL opens them correctly regardless of
the .jpg extension because it reads file magic bytes.

gt.txt format (MOT):
  frame_id, obj_id, x, y, w, h, conf, class_id, visibility
  - frame_id: 1-indexed
  - obj_id:   CVAT track id + 1 (1-indexed, unique per sequence)
  - x, y:     top-left corner in pixels
  - w, h:     width / height in pixels
  - conf:     1 (always)
  - class_id: 1-indexed (marlin=1 sealion=2 baitball=3 human=4 bird=5 mahi=6)
  - visibility: 1.0 (occluded boxes are written with visibility=0.5)

Usage:
  python tools/convert_cvat_to_mot.py \\
      --raw-dir /path/to/16-marlin-dataset \\
      --out-dir ./datasets/OceanFish/all

Run from the 01-MOTIP/ directory.
"""

import os
import argparse
import xml.etree.ElementTree as ET


LABEL_TO_CLASS_ID = {
    "marlin":   1,
    "sealion":  2,
    "baitball": 3,
    "human":    4,
    "bird":     5,
    "mahi":     6,
}

FRAME_RATE = 60  # all sequences are 60 fps drone footage


def convert_sequence(seq_name: str, raw_seq_dir: str, out_seq_dir: str) -> None:
    ann_path = os.path.join(raw_seq_dir, "annotations.xml")
    img_src_dir = os.path.join(raw_seq_dir, "images")

    tree = ET.parse(ann_path)
    root = tree.getroot()

    # Sequence metadata from the CVAT task header
    stop_frame = int(root.find(".//stop_frame").text)
    seq_length = stop_frame + 1  # frames are 0-indexed in CVAT

    # Image dimensions from the first image element (or fall back to parsing a PNG header)
    size_node = root.find(".//original_size") if root.find(".//original_size") is not None else root.find(".//image")
    if size_node is not None and size_node.find("width") is not None:
        im_w = int(size_node.find("width").text)
        im_h = int(size_node.find("height").text)
    else:
        # Read from the PNG header (bytes 16-24 of a PNG file are width/height)
        import struct
        sample = os.path.join(img_src_dir, "0.png")
        with open(sample, "rb") as f:
            f.read(16)
            im_w = struct.unpack(">I", f.read(4))[0]
            im_h = struct.unpack(">I", f.read(4))[0]

    # Build gt rows: one list entry per visible box in frame order
    gt_rows: list[tuple] = []  # (frame_id, obj_id, x, y, w, h, class_id, visibility)

    for track in root.findall("track"):
        label = track.get("label")
        if label not in LABEL_TO_CLASS_ID:
            continue
        class_id = LABEL_TO_CLASS_ID[label]
        obj_id = int(track.get("id")) + 1  # 1-indexed

        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame_idx = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            w = xbr - xtl
            h = ybr - ytl
            if w <= 0 or h <= 0:
                continue
            visibility = 0.5 if box.get("occluded") == "1" else 1.0
            gt_rows.append((frame_idx + 1, obj_id, xtl, ytl, w, h, class_id, visibility))

    # Sort by frame_id then obj_id for clean output
    gt_rows.sort(key=lambda r: (r[0], r[1]))

    # Write output files
    gt_dir = os.path.join(out_seq_dir, "gt")
    img1_dir = os.path.join(out_seq_dir, "img1")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(img1_dir, exist_ok=True)

    # seqinfo.ini
    with open(os.path.join(out_seq_dir, "seqinfo.ini"), "w") as f:
        f.write(f"[Sequence]\n"
                f"name={seq_name}\n"
                f"imDir=img1\n"
                f"frameRate={FRAME_RATE}\n"
                f"seqLength={seq_length}\n"
                f"imWidth={im_w}\n"
                f"imHeight={im_h}\n"
                f"imExt=.jpg\n")

    # gt/gt.txt
    with open(os.path.join(gt_dir, "gt.txt"), "w") as f:
        for row in gt_rows:
            frame_id, obj_id, x, y, w, h, class_id, vis = row
            f.write(f"{frame_id},{obj_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{class_id},{vis:.1f}\n")

    # img1/ symlinks: {i+1:08d}.jpg → relative path to images/{i}.png
    for i in range(seq_length):
        link = os.path.join(img1_dir, f"{i + 1:08d}.jpg")
        target_abs = os.path.abspath(os.path.join(img_src_dir, f"{i}.png"))
        target_rel = os.path.relpath(target_abs, start=img1_dir)
        if os.path.islink(link):
            os.remove(link)
        os.symlink(target_rel, link)

    print(f"  {seq_name}: {seq_length} frames, {len(gt_rows)} annotations, "
          f"{len(set(r[1] for r in gt_rows))} tracks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir",
                        default="/home/david-james/Desktop/16-marlin-dataset",
                        help="Root of the raw CVAT dataset (contains one sub-dir per sequence)")
    parser.add_argument("--out-dir",
                        default="./datasets/OceanFish/all",
                        help="Output root; each sequence gets a sub-directory here")
    parser.add_argument("--sequences", nargs="*",
                        help="Convert only these sequence names (default: all)")
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)

    sequences = args.sequences or [
        d for d in sorted(os.listdir(raw_dir))
        if os.path.isdir(os.path.join(raw_dir, d))
        and os.path.exists(os.path.join(raw_dir, d, "annotations.xml"))
    ]

    print(f"Converting {len(sequences)} sequence(s) → {out_dir}\n")
    for seq in sequences:
        raw_seq = os.path.join(raw_dir, seq)
        out_seq = os.path.join(out_dir, seq)
        os.makedirs(out_seq, exist_ok=True)
        convert_sequence(seq, raw_seq, out_seq)

    print(f"\nDone. Run setup_kfold_splits.py next to create fold symlink directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
