
import os
import sys
import json
import math
import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


FOLD_DIR = Path("/root/autodl-tmp/ultralytics-main/cv_results_1/fold_0").resolve()
OUT_BASE = FOLD_DIR / "final_fulltrain"      # will create final_train / final_val / final_dataset.yaml
VAL_RATIO = 0.10
SEED = 42
USE_SYMLINK = True
NAMES = ['Spalling', 'Squat', 'Wheel Burn', 'Corrugation']

BATCH_SIZE = 24
LR0 = 0.001
EPOCHS = 200

MODEL_CONFIGS = {
    # "both_innovations": "yolov8-SPPF-LSKA.yaml",
    "baseline_yolov8n": "ultralytics/cfg/models/v8/yolov8n.yaml",
}

PROJECT_TRAIN = FOLD_DIR.parent / "final_runs_bs24_lr1e-3"
PROJECT_EVAL  = FOLD_DIR.parent / "final_eval"



def safe_link_or_copy(src: Path, dst: Path, use_symlink: bool = True):
    """Create a symlink if possible, otherwise copy the file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        if use_symlink:
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def make_final_split_from_fold0(fold_dir: Path, out_base: Path, val_ratio: float, names, seed=42, use_symlink=True) -> Path:
    """
    Merge fold_0 train+val, randomly split a new final_val, keep fold_0/test as test.
    Return the generated final_dataset.yaml path.
    """
    rng = random.Random(seed)
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")

    train_img_dir = fold_dir / "train" / "images"
    val_img_dir   = fold_dir / "val" / "images"
    test_img_dir  = fold_dir / "test" / "images"

    train_lbl_dir = fold_dir / "train" / "labels"
    val_lbl_dir   = fold_dir / "val" / "labels"
    test_lbl_dir  = fold_dir / "test" / "labels"

    assert train_img_dir.exists() and val_img_dir.exists() and test_img_dir.exists(), "images directory missing in fold_0"
    assert train_lbl_dir.exists() and val_lbl_dir.exists() and test_lbl_dir.exists(),  "labels directory missing in fold_0"

    def collect(img_dir, lbl_dir):
        files = []
        for p in img_dir.iterdir():
            if p.suffix.lower() in img_exts:
                lbl = lbl_dir / (p.stem + ".txt")
                if lbl.exists():
                    files.append((p, lbl))
        return files

    pool = collect(train_img_dir, train_lbl_dir) + collect(val_img_dir, val_lbl_dir)
    if len(pool) == 0:
        raise RuntimeError("no eligible images found in fold_0 train/val")

    rng.shuffle(pool)
    n_total = len(pool)
    n_val   = max(1, int(math.floor(n_total * val_ratio)))
    val_set = set(pool[:n_val])
    train_set = set(pool[n_val:])

    print(f"[Split] total(non-test)={n_total} -> final_train={len(train_set)}, final_val={len(val_set)} (val_ratio={val_ratio})")

    # create final train/val structure
    final_train_img = out_base / "final_train" / "images"
    final_train_lbl = out_base / "final_train" / "labels"
    final_val_img   = out_base / "final_val" / "images"
    final_val_lbl   = out_base / "final_val" / "labels"
    for d in [final_train_img, final_train_lbl, final_val_img, final_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    for (img, lbl) in train_set:
        safe_link_or_copy(img, final_train_img / img.name, use_symlink)
        safe_link_or_copy(lbl, final_train_lbl / lbl.name, use_symlink)
    for (img, lbl) in val_set:
        safe_link_or_copy(img, final_val_img / img.name, use_symlink)
        safe_link_or_copy(lbl, final_val_lbl / lbl.name, use_symlink)

    # write final_dataset.yaml (absolute paths for robustness)
    yaml_path = out_base / "final_dataset.yaml"

    def norm(p: Path) -> str:
        return str(p.resolve()).replace("\\", "/")

    yaml_str = (
        f"train: {norm(final_train_img)}\n"
        f"val: {norm(final_val_img)}\n"
        f"test: {norm(test_img_dir)}\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(yaml_str, encoding="utf-8")
    print(f"[YAML] written: {yaml_path}")
    print(f"  train -> {norm(final_train_img)}")
    print(f"  val   -> {norm(final_val_img)}")
    print(f"  test  -> {norm(test_img_dir)}")
    return yaml_path


def _safe_get_speed_ms(results, key: str):
    """
    Robustly fetch speed in ms from Ultralytics results.
    Returns NaN if the key is missing or incompatible.
    """
    try:
        s = results.speed
        if isinstance(s, dict):
            v = s.get(key, None)
        else:
            v = getattr(s, key, None)
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def train_and_eval(model_name: str, model_cfg: str, dataset_yaml: Path):
    print(f"\n===== Train: {model_name} =====")
    print(f"cfg={model_cfg}")
    device = '0' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_cfg)

    run_name = f"{model_name}_full_bs{BATCH_SIZE}_lr{LR0:.0e}"
    results = model.train(
        data=str(dataset_yaml),
        imgsz=640,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        workers=2,          # more stable for your environment
        device=device,
        optimizer='AdamW',
        lr0=LR0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=4.0,
        dfl=1.5,
        amp=True,
        cache='disk',
        save=True,
        project=str(PROJECT_TRAIN),
        name=run_name,
        patience=30,
        # augmentations (kept as your original choices)
        mosaic=1.0,
        scale=0.5,
        fliplr=0.5,
        degrees=0.0,
        translate=0.2,
        mixup=0.1,
        copy_paste=0.1,
        auto_augment=True,
        erasing=0.4,
        verbose=True,
    )

    # pick best.pt (fallback to last.pt)
    weights_dir = Path(results.save_dir) / "weights"
    best_pt = weights_dir / "best.pt"
    if not best_pt.exists():
        print("best.pt not found, using last.pt")
        best_pt = weights_dir / "last.pt"

    print(f"\n===== Test on held-out TEST: {model_name} ({best_pt}) =====")
    model_best = YOLO(str(best_pt))
    test_res = model_best.val(
        data=str(dataset_yaml),
        split='test',
        save_json=True,
        project=str(PROJECT_EVAL),
        name=f"{model_name}_final_test_bs{BATCH_SIZE}_lr{LR0:.0e}",
    )

    # collect metrics (speed handled robustly)
    inf_ms  = _safe_get_speed_ms(test_res, "inference")
    metrics = {
        "model": model_name,
        "batch": BATCH_SIZE,
        "lr0": LR0,
        "mAP50": float(test_res.box.map50),
        "mAP50-95": float(test_res.box.map),
        "precision": float(test_res.box.precision.mean()),
        "recall": float(test_res.box.recall.mean()),
        "f1": float(
            2 * test_res.box.precision.mean() * test_res.box.recall.mean()
            / max(1e-9, (test_res.box.precision.mean() + test_res.box.recall.mean()))
        ),
        "speed_ms_per_image": float(inf_ms) if inf_ms == inf_ms else float("nan"),  # NaN-safe
    }

    # save JSON
    PROJECT_EVAL.mkdir(parents=True, exist_ok=True)
    out_json = PROJECT_EVAL / f"{model_name}_final_metrics_bs{BATCH_SIZE}_lr{LR0:.0e}.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"metrics saved: {out_json}")

    # free memory between models
    try:
        del model_best
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    print("Full retrain on fold_0 (train+val merged) with fixed test split")
    print(f"FOLD_DIR = {FOLD_DIR}")
    print(f"VAL_RATIO = {VAL_RATIO}")

    dataset_yaml = make_final_split_from_fold0(
        fold_dir=FOLD_DIR,
        out_base=OUT_BASE,
        val_ratio=VAL_RATIO,
        names=NAMES,
        seed=SEED,
        use_symlink=USE_SYMLINK
    )

    all_metrics = []
    for name, cfg in MODEL_CONFIGS.items():
        all_metrics.append(train_and_eval(name, cfg, dataset_yaml))

    print("\n==================== SUMMARY (Test set) ====================")
    for m in all_metrics:
        print(f"[{m['model']}] mAP50-95={m['mAP50-95']:.4f} | mAP50={m['mAP50']:.4f} "
              f"| P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f} "
              f"| {m['speed_ms_per_image']:.2f} ms/img")
    print("============================================================")


if __name__ == "__main__":
    main()
