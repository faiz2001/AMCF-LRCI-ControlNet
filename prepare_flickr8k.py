import cv2
import json
import os
import numpy as np

# ── CONFIG ──────────────────────────────────────────
FLICKR_IMG_DIR  = '/data1/faiz/Flicker8k_Dataset'   # images folder
FLICKR_CAP_FILE = '/data1/faiz/Flickr8k.token.txt'  # captions file
OUT_DIR         = '/data1/faiz/training/fill50k'
NUM_IMAGES      = 1000   # start with 1000
IMG_SIZE        = 512
# ────────────────────────────────────────────────────

os.makedirs(f'{OUT_DIR}/source', exist_ok=True)
os.makedirs(f'{OUT_DIR}/target', exist_ok=True)

# Load captions
print("Loading captions...")
captions = {}
with open(FLICKR_CAP_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        img_name = parts[0].split('#')[0].strip()
        caption  = parts[1].strip()
        if img_name not in captions:
            captions[img_name] = caption

print(f"Loaded captions for {len(captions)} images")

# Get image list
all_images = sorted([
    f for f in os.listdir(FLICKR_IMG_DIR)
    if f.endswith('.jpg') and f in captions
])[:NUM_IMAGES]

print(f"Processing {len(all_images)} images...")

data   = []
failed = 0

for i, fname in enumerate(all_images):
    try:
        img_path = os.path.join(FLICKR_IMG_DIR, fname)
        img      = cv2.imread(img_path)

        if img is None:
            failed += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Canny edge map
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge     = cv2.Canny(gray, 100, 200)
        edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        tgt_name = f'{i:06d}.jpg'

        cv2.imwrite(f'{OUT_DIR}/target/{tgt_name}', img)
        cv2.imwrite(f'{OUT_DIR}/source/{tgt_name}', edge_rgb)

        data.append({
            "source": f"source/{tgt_name}",
            "target": f"target/{tgt_name}",
            "prompt": captions[fname]
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_images)} done")

    except Exception as e:
        print(f"  Error on {fname}: {e}")
        failed += 1

# Save prompt.json
with open(f'{OUT_DIR}/prompt.json', 'w') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')

print(f"\nDone! {len(data)} samples saved, {failed} failed.")
print(f"Dataset at: {OUT_DIR}")
print(f"prompt.json: {OUT_DIR}/prompt.json")
