import cv2, json, os
from pycocotools.coco import COCO

COCO_IMG_DIR  = '/data2/faiz/val2017'
COCO_ANN_FILE = '/data2/faiz/captions_val2017.json'
OUT_DIR       = '/data2/faiz/training/coco_val'
NUM_IMAGES    = 5000
IMG_SIZE      = 512

os.makedirs(f'{OUT_DIR}/source', exist_ok=True)
os.makedirs(f'{OUT_DIR}/target', exist_ok=True)

print("Loading COCO captions...")
coco = COCO(COCO_ANN_FILE)
img_ids = coco.getImgIds()[:NUM_IMAGES]

data, failed = [], 0
for i, img_id in enumerate(img_ids):
    try:
        info = coco.loadImgs(img_id)[0]
        img  = cv2.imread(f"{COCO_IMG_DIR}/{info['file_name']}")
        if img is None:
            failed += 1; continue
        img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 100, 200)
        edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'{OUT_DIR}/target/{i:06d}.jpg', img)
        cv2.imwrite(f'{OUT_DIR}/source/{i:06d}.jpg', edge_rgb)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        caption = anns[0]['caption'] if anns else "an image"
        data.append({"source": f"source/{i:06d}.jpg",
                     "target": f"target/{i:06d}.jpg",
                     "prompt": caption.strip()})
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{NUM_IMAGES} done")
    except Exception as e:
        print(f"  Error: {e}"); failed += 1

with open(f'{OUT_DIR}/prompt.json', 'w') as f:
    for d in data:
        f.write(json.dumps(d) + '\n')

print(f"\nDone. {len(data)} samples, {failed} failed.")
