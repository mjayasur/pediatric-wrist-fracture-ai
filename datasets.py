#!/usr/bin/env python3
"""Organise the public datasets into the exact training, validation and test sets used in the paper.

Download the sources listed in README.md into `raw/` (layout below), then run `python datasets.py all`.

  raw/GRAZPEDWRI-DX/images/*.png                 16-bit PNG radiographs
  raw/GRAZPEDWRI-DX/yolov5/labels/*.txt          YOLO labels, 9 classes (GRAZ_CLASSES)
  raw/GRAZPEDWRI-DX/dataset.csv                  per-image metadata (patient, study, AO code, projection)
  raw/FracAtlas/images/{Fractured,Non_fractured}/*.jpg  +  raw/FracAtlas/Annotations/YOLO/*.txt
  raw/MURA-v1.1/{train,valid}/XR_*/patientNNNNN/studyN_{negative,positive}/imageN.png
  raw/boneage-training-dataset/*.png             RSNA Pediatric Bone Age training images
  raw/PediURF/{train,test}/<category>/<case>/{front,side}.jpg  +  raw/PediURF/{train,test}.csv
  raw/scaphoid_scanner_images/*.jpg              fetched from the GitHub release automatically if absent

Outputs (symlinks, so nothing is duplicated):
  data/graz9/{images,labels}/{train,val,test}    9-class GRAZ partitions -> model 2 (injury signs)
  data/graz2/{images,labels}/{train,val,test}    2-class (fracture, metal) GRAZ partitions
  data/fracture_pool/{images,labels}/train       six-source training pool for model 1 (36,487 unique images)
  data/fracatlas_val/{images,labels}             FracAtlas validation images (added to model 1 validation)
  data/pediurf/cases.csv                         external test cohort: distal + midshaft cases, two views each
  data/signs.yaml, data/fracture.yaml            Ultralytics dataset files
"""
import argparse, csv, io, json, os, shutil, sys, urllib.request, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW, DATA = HERE / 'raw', HERE / 'data'
RELEASE = 'https://github.com/mjayasur/pediatric-wrist-fracture-ai/releases/download/v1.0'

GRAZ_CLASSES = ['boneanomaly', 'bonelesion', 'foreignbody', 'fracture', 'metal',
                'periostealreaction', 'pronatorsign', 'softtissue', 'text']
TWO_CLASS = {3: 0, 4: 1}            # GRAZ fracture -> 0, metal -> 1 for model 1
SCAPHOID_TO_FRACTURE = {0: 0, 1: 1, 2: 0}   # scaphoid-scanner / panel labels use class 2 = scaphoid fracture
GRAZ_TEST_SPLITS = {'test', 'hard_radius', 'hard_ulna'}   # hard_* = the reserved Salter-Harris III/IV patients
FOCUS_REPEATS = 20        # 27 reviewed GRAZ scaphoid images are sampled 20x
PANEL_REPEATS = {True: 10, False: 5}   # published panels: pediatric x10, other x5
PEDIURF_CATEGORIES = {'Distal ulna and radius fractures': 'distal',
                      'Midshaft ulna and radius fractures': 'midshaft'}   # proximal is outside the wrist FOV


def read_list(name):
    return [l.strip() for l in open(HERE / 'splits' / name) if l.strip()]


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(Path(src).resolve(), dst)


def write_label(dst, lines):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('\n'.join(lines) + ('\n' if lines else ''))


def remap(label_path, mapping):
    """Read a YOLO label file and remap/drop classes; returns the kept lines."""
    out = []
    if label_path and Path(label_path).exists():
        for line in open(label_path):
            t = line.split()
            if len(t) >= 5 and int(float(t[0])) in mapping:
                out.append(' '.join([str(mapping[int(float(t[0]))])] + t[1:5]))
    return out


def add(pool, name, image, label_lines):
    """Add one entry to a YOLO image/label directory pair."""
    link(image, pool / 'images' / 'train' / name)
    write_label(pool / 'labels' / 'train' / (Path(name).stem + '.txt'), label_lines)


# ----------------------------------------------------------------------------- GRAZPEDWRI-DX
def build_graz():
    """Patient-level partitions from splits/grazpedwri_split.csv (16,345 / 1,966 / 2,016 radiographs)."""
    img_dir, lab_dir = RAW / 'GRAZPEDWRI-DX' / 'images', RAW / 'GRAZPEDWRI-DX' / 'yolov5' / 'labels'
    counts = {}
    for r in csv.DictReader(open(HERE / 'splits' / 'grazpedwri_split.csv')):
        part = 'test' if r['split'] in GRAZ_TEST_SPLITS else r['split']
        stem, img = r['filestem'], img_dir / f"{r['filestem']}.png"
        if not img.exists():
            sys.exit(f'missing GRAZ image {img}')
        nine = [l.strip() for l in open(lab_dir / f'{stem}.txt')] if (lab_dir / f'{stem}.txt').exists() else []
        link(img, DATA / 'graz9' / 'images' / part / f'{stem}.png')
        write_label(DATA / 'graz9' / 'labels' / part / f'{stem}.txt', [l for l in nine if l])
        link(img, DATA / 'graz2' / 'images' / part / f'{stem}.png')
        write_label(DATA / 'graz2' / 'labels' / part / f'{stem}.txt', remap(lab_dir / f'{stem}.txt', TWO_CLASS))
        counts[part] = counts.get(part, 0) + 1
    print('GRAZ partitions:', counts)


# ----------------------------------------------------------------------------- model 1 training pool
def fetch_scaphoid_images():
    d = RAW / 'scaphoid_scanner_images'
    if d.exists() and any(d.glob('*.jpg')):
        return d
    print('downloading Scaphoid-scanner originals from the GitHub release ...')
    buf = urllib.request.urlopen(f'{RELEASE}/scaphoid_scanner_images.zip').read()
    zipfile.ZipFile(io.BytesIO(buf)).extractall(RAW)
    return d


def build_fracture_pool():
    """Six-source pool for model 1: GRAZ train, FracAtlas, MURA, RSNA bone age, Scaphoid-scanner, panels."""
    pool = DATA / 'fracture_pool'
    shutil.rmtree(pool, ignore_errors=True)
    n = {}
    # GRAZ training partition minus 32 scaphoid-coded images without a box; 27 reviewed scaphoid images x20
    excluded, focus = set(read_list('graz_train_excluded.txt')), set(read_list('graz_scaphoid_focus.txt'))
    for img in sorted((DATA / 'graz2' / 'images' / 'train').glob('*.png')):
        if img.stem in excluded:
            continue
        lines = [l.strip() for l in open(DATA / 'graz2' / 'labels' / 'train' / f'{img.stem}.txt') if l.strip()]
        add(pool, f'graz__{img.stem}.png', img, lines)
        for k in range(1, FOCUS_REPEATS if img.stem in focus else 1):
            add(pool, f'graz__{img.stem}__rep{k:02d}.png', img, lines)
        n['graz'] = n.get('graz', 0) + 1
    # FracAtlas (upper-limb subset used in the paper), labels are the dataset's own YOLO annotations
    fa = RAW / 'FracAtlas'
    for name in read_list('fracatlas_train.txt') + read_list('fracatlas_added.txt'):
        img = next(p for p in (fa / 'images' / 'Fractured' / name, fa / 'images' / 'Non_fractured' / name) if p.exists())
        add(pool, f'fracatlas__{name}', img, remap(fa / 'Annotations' / 'YOLO' / f'{Path(name).stem}.txt', {0: 0}))
        n['fracatlas'] = n.get('fracatlas', 0) + 1
    # MURA: normal studies as background; abnormal studies with model-generated boxes (labels/mura)
    for lst in ('mura_normal_wrist', 'mura_abnormal_wrist', 'mura_normal_forearm_hand',
                'mura_added_wrist', 'mura_added_forearm', 'mura_added_hand'):
        for rel in read_list(f'{lst}.txt'):
            key = rel.replace('/', '__').rsplit('.', 1)[0]
            add(pool, f'mura__{key}.png', RAW / 'MURA-v1.1' / rel,
                remap(HERE / 'labels' / 'mura' / f'{key}.txt', {0: 0, 1: 1}))
            n['mura'] = n.get('mura', 0) + 1
    # RSNA bone age: 4,000 non-trauma hand radiographs as background
    for name in read_list('bone_age_images.txt'):
        add(pool, f'boneage__{name}', RAW / 'boneage-training-dataset' / name, [])
        n['bone_age'] = n.get('bone_age', 0) + 1
    # Scaphoid-scanner: 501 adult scaphoid fractures, boxes from the source dataset
    for img in sorted(fetch_scaphoid_images().glob('*.jpg')):
        add(pool, f'scaphoid__{img.name}', img,
            remap(HERE / 'scaphoid_scanner' / 'labels' / f'{img.stem}.txt', SCAPHOID_TO_FRACTURE))
        n['scaphoid'] = n.get('scaphoid', 0) + 1
    # 24 reviewed panels from open-access articles (panels/manifest.json): pediatric x10, other x5
    for p in json.load(open(HERE / 'panels' / 'manifest.json')):
        img = HERE / 'panels' / p['image']
        lines = remap(HERE / 'panels' / p['label'], SCAPHOID_TO_FRACTURE)
        for k in range(PANEL_REPEATS[bool(p['pediatric_confirmed'] in (True, 'True'))]):
            add(pool, f'panel__{img.stem}__rep{k:02d}.png', img, lines)
        n['panels'] = n.get('panels', 0) + 1
    # FracAtlas validation images (added to the GRAZ validation partition for model 1 checkpoint selection)
    for name in read_list('fracatlas_val.txt'):
        img = next(p for p in (fa / 'images' / 'Fractured' / name, fa / 'images' / 'Non_fractured' / name) if p.exists())
        link(img, DATA / 'fracatlas_val' / 'images' / name)
        write_label(DATA / 'fracatlas_val' / 'labels' / f'{Path(name).stem}.txt',
                    remap(fa / 'Annotations' / 'YOLO' / f'{Path(name).stem}.txt', {0: 0}))
    total = len(list((pool / 'images' / 'train').iterdir()))
    print('fracture pool: unique images', n, 'sum', sum(n.values()), '| entries incl. repeats', total)
    (DATA / 'fracture.yaml').write_text(
        f"path: {DATA}\ntrain: [fracture_pool/images/train]\nval: [graz2/images/val, fracatlas_val/images]\n"
        "nc: 2\nnames: [fracture, metal]\n")
    (DATA / 'signs.yaml').write_text(
        f"path: {DATA}\ntrain: graz9/images/train\nval: graz9/images/val\nnc: 9\nnames: {GRAZ_CLASSES}\n")


# ----------------------------------------------------------------------------- MURA pseudo-labels (optional)
def pseudo_label_mura(weights):
    """Regenerate labels/mura with a trained injury-sign model: fracture >= 0.40, metal >= 0.50 (paper)."""
    from ultralytics import YOLO
    from inference import letterbox
    model, out = YOLO(weights), HERE / 'labels' / 'mura'
    for lst in ('mura_abnormal_wrist', 'mura_added_wrist', 'mura_added_forearm', 'mura_added_hand'):
        for rel in read_list(f'{lst}.txt'):
            if 'positive' not in rel:
                continue
            img, sc, px, py, w, h = letterbox(str(RAW / 'MURA-v1.1' / rel))
            res = model.predict(img, imgsz=1024, verbose=False, conf=0.001)[0]
            lines = []
            for b in res.boxes:
                c, conf = int(b.cls), float(b.conf)
                if (c == 3 and conf >= 0.40) or (c == 4 and conf >= 0.50):
                    x, y, bw, bh = [float(v) for v in b.xywh[0]]
                    lines.append(f"{TWO_CLASS[c]} {((x-px)/sc)/w:.6f} {((y-py)/sc)/h:.6f} {(bw/sc)/w:.6f} {(bh/sc)/h:.6f}")
            write_label(out / (rel.replace('/', '__').rsplit('.', 1)[0] + '.txt'), lines)
    print('MURA pseudo-labels written to', out)


# ----------------------------------------------------------------------------- PediURF external cohort
def build_pediurf():
    """All distal and midshaft cases from both PediURF partitions; none is used for training."""
    rows = []
    for split in ('train', 'test'):
        lines = open(RAW / 'PediURF' / f'{split}.csv', encoding='utf-8-sig').read().split('\n')[1:]
        for row in csv.reader(lines):
            if len(row) < 4 or row[0].strip() not in PEDIURF_CATEGORIES:
                continue
            case_dir = RAW / 'PediURF' / split / row[0].strip() / row[1].strip()
            views = sorted(str(p) for p in case_dir.glob('*.jpg'))
            if views:
                rows.append(dict(case=row[1].strip(), category=PEDIURF_CATEGORIES[row[0].strip()], split=split,
                                 sex=row[2].strip(), age=row[3].strip(), views=';'.join(views)))
    (DATA / 'pediurf').mkdir(parents=True, exist_ok=True)
    with open(DATA / 'pediurf' / 'cases.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('PediURF external cohort:', len(rows), 'cases',
          {c: sum(r['category'] == c for r in rows) for c in ('distal', 'midshaft')})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['all', 'graz', 'fracture', 'pediurf', 'pseudo-label'])
    ap.add_argument('--weights', default='weights/signs_model.pt', help='injury-sign model for pseudo-label')
    a = ap.parse_args()
    if a.step in ('all', 'graz'):
        build_graz()
    if a.step in ('all', 'fracture'):
        build_fracture_pool()
    if a.step in ('all', 'pediurf'):
        build_pediurf()
    if a.step == 'pseudo-label':
        pseudo_label_mura(a.weights)
