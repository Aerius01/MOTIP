"""
Convert SeaDronesSee_MOT (COCO-style JSON) train split to MOT format,
outputting into datasets/MFT_SeaDroneSee_merged/train/.

Class mapping (merged dataset):
  MFT25 fish          -> class_id 1  (already in place, untouched)
  SeaDronesSee:
    1 swimmer                -> class_id 2
    2 swimmer with lifejacket -> class_id 3
    3 boat                   -> class_id 4
    6 life jacket            -> class_id 5

Output structure per video:
  {out_root}/SDS-{folder_name}/
      seqinfo.ini
      img1/00000001.jpg ... (symlinks to original jpgs)
      gt/gt.txt
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SDS_ROOT = REPO_ROOT / "datasets" / "SeaDronesSee_MOT"
TRAIN_JSON = SDS_ROOT / "annotations" / "instances_train_objects_in_water.json"
TRAIN_JPG_DIR = SDS_ROOT / "SeaDronesSee_MOT_jpg" / "train"
OUT_ROOT = REPO_ROOT / "datasets" / "MFT_SeaDroneSee_merged" / "train"

# SeaDronesSee category_id -> merged dataset class_id (1-indexed)
CLASS_REMAP = {1: 2, 2: 3, 3: 4, 6: 5}

# ── Load JSON ────────────────────────────────────────────────────────────────
print("Loading train JSON...")
with open(TRAIN_JSON) as f:
    data = json.load(f)

# ── Build lookup structures ───────────────────────────────────────────────────
# video_id -> folder name (e.g. "DJI_0057")
vid_names: dict[int, str] = {}
for v in data["videos"]:
    raw_name = v.get("name") or v.get("name:") or str(v["id"])
    folder = raw_name.split("/")[-1].replace(".MP4", "").replace(".mp4", "")
    vid_names[v["id"]] = folder

# image_id -> image record
images_by_id: dict[int, dict] = {img["id"]: img for img in data["images"]}

# video_id -> list of image records, sorted by frame_index
vid_images: dict[int, list] = defaultdict(list)
for img in data["images"]:
    vid_images[img["video_id"]].append(img)
for vid_id in vid_images:
    vid_images[vid_id].sort(key=lambda x: x["frame_index"])

# image_id -> list of annotations
anns_by_image: dict[int, list] = defaultdict(list)
for ann in data["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)

# ── Process each video ────────────────────────────────────────────────────────
print(f"Converting {len(vid_images)} videos -> {OUT_ROOT}")

for vid_id, imgs in sorted(vid_images.items()):
    if vid_id in vid_names:
        folder_name = vid_names[vid_id]
    else:
        # video_id missing from videos list — fall back to source metadata in image
        folder_name = imgs[0].get("source", {}).get("folder_name", f"VID{vid_id}")
        print(f"  WARNING: video_id={vid_id} not in videos list, using folder_name={folder_name!r} from image source")
    seq_name = f"SDS-{folder_name}"
    seq_dir = OUT_ROOT / seq_name
    img1_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"

    img1_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    width = imgs[0]["width"]
    height = imgs[0]["height"]
    n_frames = len(imgs)

    # image_id -> dense 1-indexed frame number
    frame_map: dict[int, int] = {
        img["id"]: idx + 1 for idx, img in enumerate(imgs)
    }

    # ── Symlink images ────────────────────────────────────────────────────────
    for img in imgs:
        dense_frame = frame_map[img["id"]]
        src_stem = os.path.splitext(img["file_name"])[0]  # e.g. "0"
        src_path = TRAIN_JPG_DIR / (src_stem + ".jpg")
        dst_path = img1_dir / f"{dense_frame:08d}.jpg"

        if not src_path.exists():
            print(f"  WARNING: source image not found: {src_path}", file=sys.stderr)
            continue

        if dst_path.is_symlink() or dst_path.exists():
            dst_path.unlink()
        dst_path.symlink_to(src_path.resolve())

    # ── Write gt.txt ─────────────────────────────────────────────────────────
    gt_rows = []
    for img in imgs:
        dense_frame = frame_map[img["id"]]
        for ann in anns_by_image.get(img["id"], []):
            x, y, w, h = ann["bbox"]
            track_id = ann["track_id"] + 1   # 0-indexed in JSON -> 1-indexed in MOT
            mapped_class = CLASS_REMAP[ann["category_id"]]
            # MOT format: frame, id, x, y, w, h, conf, class, visibility
            gt_rows.append(
                f"{dense_frame},{track_id},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,{mapped_class},1.0"
            )

    gt_rows.sort(key=lambda r: (int(r.split(",")[0]), int(r.split(",")[1])))
    with open(gt_dir / "gt.txt", "w") as f:
        f.write("\n".join(gt_rows))
        if gt_rows:
            f.write("\n")

    # ── Write seqinfo.ini ─────────────────────────────────────────────────────
    seqinfo = (
        f"[Sequence]\n"
        f"name={seq_name}\n"
        f"imDir=img1\n"
        f"frameRate=25\n"
        f"seqLength={n_frames}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        f"imExt=.jpg\n"
    )
    with open(seq_dir / "seqinfo.ini", "w") as f:
        f.write(seqinfo)

    print(f"  {seq_name}: {n_frames} frames, {len(gt_rows)} annotations")

print("Done.")
