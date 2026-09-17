#!/usr/bin/env python3
"""Train the two models and select the screening cutoffs exactly as described in the paper.

  python training.py signs      model 2: YOLO11s, 9 GRAZ classes, up to 45 epochs, patience 15
  python training.py fracture   model 1: YOLO11s, fracture + metal on the six-source pool, up to 40 epochs, patience 12
  python training.py cutoff     fracture cutoff = smallest value giving >= 92% image-level specificity on the
                                GRAZ validation partition with injury signs fixed at 0.30; written to weights/cutoffs.json

Both models start from COCO-pretrained YOLO11s weights at 1,024 px, batch 16, with horizontal flip,
translation, scale and mosaic augmentation (Ultralytics defaults, scale 0.7 for the fracture model).
Run `python datasets.py all` first.
"""
import argparse, glob, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA, WEIGHTS = HERE / 'data', HERE / 'weights'
SIGN_THRESHOLD, TARGET_SPECIFICITY = 0.30, 0.92

RECIPES = {
    'signs': dict(data=str(DATA / 'signs.yaml'), epochs=45, patience=15, scale=0.5, name='signs_model'),
    'fracture': dict(data=str(DATA / 'fracture.yaml'), epochs=40, patience=12, scale=0.7, name='fracture_model'),
}


def train(which, device=0):
    from ultralytics import YOLO
    r = RECIPES[which]
    YOLO('yolo11s.pt').train(data=r['data'], epochs=r['epochs'], patience=r['patience'], scale=r['scale'],
                             imgsz=1024, batch=16, device=device, workers=8, pretrained=True, mosaic=1.0,
                             fliplr=0.5, translate=0.1, project=str(HERE / 'runs'), name=r['name'], exist_ok=True)
    best = HERE / 'runs' / r['name'] / 'weights' / 'best.pt'
    WEIGHTS.mkdir(exist_ok=True)
    (WEIGHTS / f"{r['name']}.pt").write_bytes(best.read_bytes())
    print('saved', WEIGHTS / f"{r['name']}.pt")


def select_cutoff(device=0):
    """Score the GRAZ validation partition with both models and pick the fracture cutoff (paper rule)."""
    from inference import Screener
    s = Screener(WEIGHTS / 'fracture_model.pt', WEIGHTS / 'signs_model.pt', cutoffs=None, device=device)
    images = sorted(glob.glob(str(DATA / 'graz2' / 'images' / 'val' / '*.png')))
    scores = s.score_images(images)
    positive = {Path(p).stem: Path(DATA / 'graz2' / 'labels' / 'val' / f'{Path(p).stem}.txt').read_text().strip() != ''
                for p in images}
    negatives = [scores[Path(p).stem] for p in images if not positive[Path(p).stem]]
    for t in np.arange(0.01, 0.991, 0.005):
        spec = np.mean([not (v['fracture'] >= t or v['sign'] >= SIGN_THRESHOLD) for v in negatives])
        if spec >= TARGET_SPECIFICITY:
            break
    cut = dict(fracture_threshold=float(round(t, 3)), sign_threshold=SIGN_THRESHOLD, val_specificity=float(spec),
               val_sensitivity=float(np.mean([(scores[k]['fracture'] >= t or scores[k]['sign'] >= SIGN_THRESHOLD)
                                              for k, p in positive.items() if p])),
               rule=f'smallest fracture cutoff (0.005 grid) with image-level specificity >= {TARGET_SPECIFICITY} '
                    f'on GRAZ validation, injury signs fixed at {SIGN_THRESHOLD}')
    json.dump(cut, open(WEIGHTS / 'cutoffs.json', 'w'), indent=1)
    print(json.dumps(cut, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['signs', 'fracture', 'cutoff'])
    ap.add_argument('--device', default=0)
    a = ap.parse_args()
    select_cutoff(a.device) if a.step == 'cutoff' else train(a.step, a.device)
