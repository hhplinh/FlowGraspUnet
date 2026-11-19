import pickle
import zipfile
from pathlib import Path

# --- CONFIGURATION ---
MAX_UNSEEN = None  # None = all 230 entries (200 scenes) in test/unseen.obj
DATA_DIR = Path("data/grasp-anything++")
UNSEEN_DIR = DATA_DIR / "unseen"

ZIP_MAP = {
    "grasp_instructions": DATA_DIR / "grasp_instructions.zip",
    "grasp_label_positive": DATA_DIR / "grasp_label_positive.zip",
    "part_mask": DATA_DIR / "part_mask.zip",
}

def get_scene_hash(entry):
    raw = entry[0] if isinstance(entry, (tuple, list)) else str(entry)
    return Path(raw).stem.split("_")[0]

# 1. Load ONLY unseen.obj (leaves data/grasp-anything++/seen untouched)
unseen_src = Path("split/grasp-anything++/test/unseen.obj")
with open(unseen_src, "rb") as f:
    entries = pickle.load(f)

subset = entries[:MAX_UNSEEN] if MAX_UNSEEN else entries
target_hashes = {get_scene_hash(e) for e in subset}
print(f"Loaded {unseen_src} -> total={len(entries)}, selected={len(subset)}, unique_scenes={len(target_hashes)}")

# 2. Write unseen split files
(UNSEEN_DIR / "test").mkdir(parents=True, exist_ok=True)
with open(UNSEEN_DIR / "unseen.obj", "wb") as f:
    pickle.dump(subset, f)
with open(UNSEEN_DIR / "test/unseen.obj", "wb") as f:
    pickle.dump(subset, f)

# 3. Extract only unseen scene hashes (skips files already extracted for seen)
for folder_name, zip_path in ZIP_MAP.items():
    if not zip_path.exists():
        print(f"[SKIP] {zip_path} not found")
        continue

    out_folder = DATA_DIR / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            scene_hash = Path(member).stem.split("_")[0]
            if scene_hash in target_hashes:
                dest = out_folder / Path(member).name
                if not dest.exists():
                    dest.write_bytes(zf.read(member))
                    extracted += 1

    print(f"[OK] {folder_name}: extracted {extracted} new unseen files into {out_folder}")