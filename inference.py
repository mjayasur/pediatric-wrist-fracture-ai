#!/usr/bin/env python3
"""Run the two-model screening system, or reproduce the paper's evaluation.

  python inference.py run --images DIR [--group folder|graz] [--out results.csv] [--annotate DIR]
      Scores every radiograph; a radiograph is flagged when the fracture model reaches the fracture cutoff or
      any injury sign (periosteal reaction, pronator sign, soft-tissue swelling) reaches 0.30. Examinations
      are all views of one visit: grouped by sub-folder (--group folder) or by GRAZ file naming (--group graz).
  python inference.py evaluate-graz      examination-level metrics, per-type sensitivity, fracture AP50 (GRAZ test)
  python inference.py evaluate-pediurf   case-level sensitivity on the external PediURF cohort (distal, midshaft)

Weights and cutoffs are downloaded from the GitHub release into weights/ if absent.
"""
import argparse, collections, csv, json, random, re, urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA, WEIGHTS = HERE / 'data', HERE / 'weights'
RELEASE = 'https://github.com/mjayasur/pediatric-wrist-fracture-ai/releases/download/v1.0'
SIGN_CLASSES = {5: 'periosteal reaction', 6: 'pronator sign', 7: 'soft-tissue swelling'}
IMGSZ = 1024


def letterbox(path, size=IMGSZ):
    """Aspect-preserving square letterbox (grey 114), 16-bit PNG scaled to 8-bit; returns the geometry."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = (im // 256).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    h, w = im.shape[:2]
    sc = size / max(h, w); nw, nh = round(w * sc), round(h * sc)
    canvas = np.full((size, size, 3), 114, np.uint8); px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, sc, px, py, w, h


def ensure_weights():
    WEIGHTS.mkdir(exist_ok=True)
    for name in ('fracture_model.pt', 'signs_model.pt', 'cutoffs.json'):
        if not (WEIGHTS / name).exists():
            print('downloading', name)
            urllib.request.urlretrieve(f'{RELEASE}/{name}', WEIGHTS / name)


class Screener:
    def __init__(self, fracture_weights=None, signs_weights=None, cutoffs='auto', device=0):
        from ultralytics import YOLO
        if fracture_weights is None:
            ensure_weights()
        self.m1 = YOLO(str(fracture_weights or WEIGHTS / 'fracture_model.pt'))
        self.m2 = YOLO(str(signs_weights or WEIGHTS / 'signs_model.pt'))
        self.device = device
        c = json.load(open(WEIGHTS / 'cutoffs.json')) if cutoffs == 'auto' else (cutoffs or {})
        self.t_fracture, self.t_sign = c.get('fracture_threshold', 0.26), c.get('sign_threshold', 0.30)

    def score(self, path):
        """Per-radiograph scores and boxes in normalised xywh of the original image."""
        img, sc, px, py, w, h = letterbox(path)
        conv = lambda b: [((b[0] - px) / sc) / w, ((b[1] - py) / sc) / h, (b[2] / sc) / w, (b[3] / sc) / h]
        r1 = self.m1.predict(img, imgsz=IMGSZ, device=self.device, verbose=False, conf=0.001)[0]
        r2 = self.m2.predict(img, imgsz=IMGSZ, device=self.device, verbose=False, conf=0.001)[0]
        fractures = sorted([(float(b.conf), conv([float(v) for v in b.xywh[0]])) for b in r1.boxes if int(b.cls) == 0],
                           key=lambda t: -t[0])
        signs = sorted([(float(b.conf), conv([float(v) for v in b.xywh[0]]), SIGN_CLASSES[int(b.cls)])
                        for b in r2.boxes if int(b.cls) in SIGN_CLASSES], key=lambda t: -t[0])
        return dict(fracture=fractures[0][0] if fractures else 0.0, sign=signs[0][0] if signs else 0.0,
                    fracture_boxes=fractures, sign_boxes=signs)

    def flagged(self, s):
        return s['fracture'] >= self.t_fracture or s['sign'] >= self.t_sign

    def score_images(self, paths):
        return {Path(p).stem: self.score(p) for p in paths}


# ----------------------------------------------------------------------------- run on new radiographs
def run(a):
    s = Screener(device=a.device)
    paths = sorted(p for p in Path(a.images).rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    group = (lambda p: str(p.parent)) if a.group == 'folder' else \
            (lambda p: '_'.join(p.stem.split('_')[:1] + p.stem.split('_')[2:3]))   # GRAZ: patient_study
    rows, exams = [], collections.defaultdict(list)
    for p in paths:
        sc = s.score(p); exams[group(p)].append(sc)
        rows.append(dict(image=str(p), examination=group(p), fracture_conf=round(sc['fracture'], 4),
                         sign_conf=round(sc['sign'], 4), flagged=s.flagged(sc),
                         fracture_boxes=json.dumps([[round(c, 3)] + [round(v, 4) for v in b] for c, b in sc['fracture_boxes'][:5] if c >= s.t_fracture]),
                         sign_boxes=json.dumps([[round(c, 3), n] + [round(v, 4) for v in b] for c, b, n in sc['sign_boxes'][:3] if c >= s.t_sign])))
        if a.annotate:
            im = letterbox(p)[0].copy()
            for c, b in sc['fracture_boxes']:
                if c >= s.t_fracture:
                    draw(im, b, p, (255, 255, 0), f'fracture {c:.2f}')
            for c, b, n in sc['sign_boxes']:
                if c >= s.t_sign:
                    draw(im, b, p, (0, 140, 255), f'{n} {c:.2f}')
            Path(a.annotate).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.annotate) / f'{p.stem}.jpg'), im)
    for r in rows:
        r['examination_flagged'] = any(s.flagged(x) for x in exams[r['examination']])
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} radiographs, {len(exams)} examinations, {sum(any(s.flagged(x) for x in v) for v in exams.values())} flagged -> {a.out}')


def draw(im, b, path, color, text):
    _, sc, px, py, w, h = letterbox(path)
    x, y, bw, bh = b[0] * w * sc + px, b[1] * h * sc + py, b[2] * w * sc, b[3] * h * sc
    cv2.rectangle(im, (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2)), color, 2)
    cv2.putText(im, text, (int(x - bw / 2), int(y - bh / 2) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ----------------------------------------------------------------------------- paper evaluation
SUBTYPES = [('Torus or buckle, radius', ['23r-M/2.1']), ('Torus or buckle, ulna', ['23u-M/2.1']),
            ('Torus or buckle, both bones', ['23-M/2.1']), ('Complete metaphyseal, radius', ['23r-M/3.1']),
            ('Complete metaphyseal, ulna', ['23u-M/3.1']), ('Complete metaphyseal, both bones', ['23-M/3.1']),
            ('Salter-Harris I, radius', ['23r-E/1']), ('Salter-Harris II, radius', ['23r-E/2.1']),
            ('Salter-Harris III, radius', ['23r-E/3']), ('Salter-Harris IV, radius', ['23r-E/4.1', '23r-E/4.2']),
            ('Salter-Harris I, ulna', ['23u-E/1']), ('Salter-Harris III, ulna', ['23u-E/3']),
            ('Salter-Harris IV, ulna', ['23u-E/4']), ('Ulnar styloid', ['23u-E/7']), ('Styloid, both bones', ['23-E/7']),
            ('Scaphoid', ['72B(b)', '72B(c)']), ('Shaft bowing, radius', ['22r-D/1']), ('Shaft bowing, ulna', ['22u-D/1.1']),
            ('Shaft bowing, both bones', ['22-D/1.1']), ('Shaft greenstick, radius', ['22r-D/2.1']),
            ('Shaft transverse, radius', ['22r-D/4.1']), ('Shaft oblique or spiral, radius', ['22r-D/5.1']),
            ('Shaft oblique or spiral, both bones', ['22-D/5.1'])]


def metrics(groups):
    """groups: list of (positive, score_ratio) per examination; ratio >= 1 means flagged."""
    tp = sum(p and r >= 1 for p, r in groups); fn = sum(p and r < 1 for p, r in groups)
    fp = sum((not p) and r >= 1 for p, r in groups); tn = sum((not p) and r < 1 for p, r in groups)
    pos, neg = sorted([r for p, r in groups if p]), sorted([r for p, r in groups if not p])
    auc = float(np.mean([[x > y for y in neg] for x in pos])) if pos and neg else float('nan')
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1), accuracy=(tp + tn) / max(len(groups), 1), auroc=auc)


def evaluate_graz(a):
    s = Screener(device=a.device)
    meta = {r['filestem']: r for r in csv.DictReader(open(HERE / 'raw' / 'GRAZPEDWRI-DX' / 'dataset.csv', encoding='utf-8-sig'))}
    images = sorted((DATA / 'graz2' / 'images' / 'test').glob('*.png'))
    exams = collections.defaultdict(list)
    for p in images:
        sc = s.score(p); m = meta[p.stem]
        positive = (DATA / 'graz2' / 'labels' / 'test' / f'{p.stem}.txt').read_text().count('\n0 ') + \
                   (DATA / 'graz2' / 'labels' / 'test' / f'{p.stem}.txt').read_text().startswith('0 ') > 0
        exams[(m['patient_id'], m['study_number'])].append(dict(positive=positive, codes=[c.strip() for c in m['ao_classification'].split(';') if c.strip()],
                                                              ratio=max(sc['fracture'] / s.t_fracture, sc['sign'] / s.t_sign)))
    groups = [(any(v['positive'] for v in vs), max(v['ratio'] for v in vs)) for vs in exams.values()]
    m = metrics(groups)
    # patient-clustered bootstrap (1,000 resamples) for the overall metrics
    by_patient = collections.defaultdict(list)
    for (pt, st), vs in exams.items():
        by_patient[pt].append((any(v['positive'] for v in vs), max(v['ratio'] for v in vs)))
    pts, rng, boots = list(by_patient), random.Random(0), []
    for _ in range(1000):
        boots.append(metrics([g for pt in rng.choices(pts, k=len(pts)) for g in by_patient[pt]]))
    for k in ('sensitivity', 'specificity', 'ppv', 'accuracy', 'auroc'):
        v = sorted(b[k] for b in boots); m[f'{k}_ci95'] = [v[25], v[974]]
    print(f"GRAZ test: {len(exams)} examinations, {len(images)} radiographs")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=1))
    print('\nSensitivity by fracture type (examinations flagged / positive examinations with the code):')
    for name, codes in SUBTYPES:
        g = [vs for vs in exams.values() if any(v['positive'] and any(c in v['codes'] for c in codes) for v in vs)]
        if g:
            hit = sum(max(v['ratio'] for v in vs) >= 1 for vs in g)
            print(f'  {name:38s} {hit}/{len(g)}')


def evaluate_pediurf(a):
    s = Screener(device=a.device)
    rows = list(csv.DictReader(open(DATA / 'pediurf' / 'cases.csv')))
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        hit = any(s.flagged(s.score(v)) for v in r['views'].split(';'))
        out[r['category']][0] += hit; out[r['category']][1] += 1; out['all'][0] += hit; out['all'][1] += 1
    for k, (h, n) in out.items():
        print(f'PediURF {k}: {h}/{n} = {100 * h / n:.1f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['run', 'evaluate-graz', 'evaluate-pediurf'])
    ap.add_argument('--images'); ap.add_argument('--group', default='folder', choices=['folder', 'graz'])
    ap.add_argument('--out', default='results.csv'); ap.add_argument('--annotate')
    ap.add_argument('--device', default=0)
    a = ap.parse_args()
    {'run': run, 'evaluate-graz': evaluate_graz, 'evaluate-pediurf': evaluate_pediurf}[a.mode](a)
