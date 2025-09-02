import os
import shutil
import random
import glob
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")


class CrossValidationManager:
    """
    K-fold cross-validation manager.
    This version:
      - does NOT evaluate on a test set (train/val only for training/metrics)
      - avoids verbose metric printing to terminal
      - writes a cv_summary.json for downstream aggregation
    """

    def __init__(self, dataset_dir="ultralytics/datasets", output_dir="cv_results", k_folds=5, test_ratio=0.1):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.k_folds = k_folds
        self.test_ratio = test_ratio
        self.classes = ["Spalling", "Squat", "Wheel Burn", "Corrugation"]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "runs").mkdir(parents=True, exist_ok=True)

        print(f"[CV] dataset_dir: {self.dataset_dir}")
        print(f"[CV] output_dir : {self.output_dir}")

  
    def collect_all_data(self):
        """Collect all image/label pairs under train/ and val/."""
        all_images, all_labels = [], []
        for subset in ["train", "val"]:
            img_dir = self.dataset_dir / subset / "images"
            label_dir = self.dataset_dir / subset / "labels"
            for img_path in glob.glob(str(img_dir / "*.jpg")):
                stem = Path(img_path).stem
                label_path = label_dir / f"{stem}.txt"
                if label_path.exists():
                    all_images.append(img_path)
                    all_labels.append(str(label_path))
        print(f"[CV] collected {len(all_images)} samples")
        return all_images, all_labels

    def stratified_split(self, images, labels, seed: int = 42):
        """
        Create a small hold-out set (called 'test' here to keep folder structure),
        but we will NOT evaluate on it in this experiment. It is just kept to be
        compatible with the existing folder layout.
        """
        rng = random.Random(seed)
        class_counts = {i: [] for i in range(len(self.classes))}
        for idx, label_path in enumerate(labels):
            with open(label_path, "r") as f:
                lines = f.readlines()
                if lines:
                    class_id = int(lines[0].split()[0])
                    class_counts[class_id].append(idx)

        test_indices, cv_indices = [], []
        for _, indices in class_counts.items():
            rng.shuffle(indices)
            n_test = max(1, int(len(indices) * self.test_ratio))
            test_indices.extend(indices[:n_test])
            cv_indices.extend(indices[n_test:])

        rng.shuffle(test_indices)
        rng.shuffle(cv_indices)
        return test_indices, cv_indices

    def create_fold_datasets(self, images, labels, cv_indices, seed: int = 42):
        """Create K fold splits from cv_indices."""
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
        fold_splits = []
        for _, (train_idx, val_idx) in enumerate(kf.split(cv_indices)):
            train_indices = [cv_indices[i] for i in train_idx]
            val_indices = [cv_indices[i] for i in val_idx]
            fold_splits.append((train_indices, val_indices))
        return fold_splits

    def setup_fold_data(self, fold, train_indices, val_indices, test_indices, images, labels):
        """
        Materialize a fold directory with train/val/test (test kept for compatibility).
        If dataset.yaml already exists, reuse it.
        """
        fold_dir = self.output_dir / f"fold_{fold}"
        ds_yaml = fold_dir / "dataset.yaml"
        if ds_yaml.exists():
            print(f"[CV] reuse fold data: {ds_yaml}")
            return ds_yaml

        fold_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            for sub in ["images", "labels"]:
                (fold_dir / split / sub).mkdir(parents=True, exist_ok=True)

        splits = {"train": train_indices, "val": val_indices, "test": test_indices}
        for split, indices in splits.items():
            print(f"[CV] copying {split} files: {len(indices)}")
            for idx in indices:
                try:
                    src_img = images[idx]
                    dst_img = fold_dir / split / "images" / Path(src_img).name
                    shutil.copy2(src_img, dst_img)

                    src_lbl = labels[idx]
                    dst_lbl = fold_dir / split / "labels" / Path(src_lbl).name
                    shutil.copy2(src_lbl, dst_lbl)
                except Exception as e:
                    print(f"[CV] copy error idx={idx}: {e}")

        train_path = str((fold_dir / "train" / "images").resolve()).replace("\\", "/")
        val_path = str((fold_dir / "val" / "images").resolve()).replace("\\", "/")
        test_path = str((fold_dir / "test" / "images").resolve()).replace("\\", "/")

        yaml_content = (
            f"train: {train_path}\n"
            f"val: {val_path}\n"
            f"test: {test_path}\n\n"
            f"nc: 4\n"
            f"names: {self.classes}\n"
        )
        with open(ds_yaml, "w") as f:
            f.write(yaml_content)

        print(f"[CV] fold {fold}: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        return ds_yaml


    def train_model(self, model_config, dataset_yaml, fold, model_name, epochs=200):
        """
        Train a single fold. Keep training logs on disk only.
        No console metric printing; rely on Ultralytics logs and returned dict.
        """
        try:
            device = "0" if torch.cuda.is_available() else "cpu"
            if not Path(model_config).exists():
                print(f"[CV] missing model config: {model_config}")
                return None
            if not Path(dataset_yaml).exists():
                print(f"[CV] missing dataset yaml: {dataset_yaml}")
                return None

            model = YOLO(model_config)
            results = model.train(
                data=str(dataset_yaml),
                imgsz=640,
                epochs=epochs,
                batch=16,
                workers=8,
                device=device,
                optimizer="AdamW",
                lr0=0.001,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                box=7.5,
                cls=4.0,
                dfl=1.5,
                amp=True,
                cache=False,
                save=True,
                project=str(self.output_dir / "runs"),
                name=f"{model_name}_fold_{fold}",
                patience=30,
                mosaic=1.0,
                scale=0.5,
                fliplr=0.5,
                degrees=0.0,
                translate=0.2,
                mixup=0.1,
                copy_paste=0.1,
                auto_augment=True,
                erasing=0.4,
            )
            return results
        except Exception as e:
            print(f"[CV] training error: {e}")
            return None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run_cross_validation(self, model_configs, epochs=200):
        """Run K-fold CV for all provided model configs. No test evaluation."""
        # Try to reuse existing fold datasets
        fold_yamls = []
        reuse_ok = True
        for k in range(self.k_folds):
            p = self.output_dir / f"fold_{k}" / "dataset.yaml"
            if not p.exists():
                reuse_ok = False
            fold_yamls.append(p)

        if reuse_ok:
            print("[CV] existing fold datasets found, reusing.")
        else:
            print("[CV] creating fold datasets.")
            all_images, all_labels = self.collect_all_data()
            test_indices, cv_indices = self.stratified_split(all_images, all_labels)
            fold_splits = self.create_fold_datasets(all_images, all_labels, cv_indices)
            for fold, (train_idx, val_idx) in enumerate(fold_splits):
                ds_yaml = self.setup_fold_data(
                    fold, train_idx, val_idx, test_indices, all_images, all_labels
                )
                fold_yamls[fold] = ds_yaml

        all_results = {name: [] for name in model_configs.keys()}

        for fold in range(self.k_folds):
            dataset_yaml = fold_yamls[fold]
            print(f"[CV] fold {fold+1}/{self.k_folds} start")
            for model_name, cfg in model_configs.items():
                print(f"[CV] training model={model_name} fold={fold+1}")
                train_results = self.train_model(cfg, dataset_yaml, fold, model_name, epochs)
                if train_results:
                    all_results[model_name].append(
                        {"fold": fold, "train_results": train_results}
                    )

        self.save_cv_results(all_results)
        print("[CV] cross validation finished.")
        return all_results

    def save_cv_results(self, results):
        """
        Save per-model aggregated metrics to cv_summary.json.
        No console metric printing here.
        """
        summary = []
        metrics = ["precision", "recall", "mAP50", "mAP50-95"]

        for model_name, fold_results in results.items():
            if not fold_results:
                continue
            agg = {}
            for metric in metrics:
                vals = []
                for fr in fold_results:
                    try:
                        d = fr["train_results"].results_dict
                        key = f"metrics/val_{metric}(B)"
                        if key in d:
                            vals.append(d[key])
                    except Exception:
                        pass
                if vals:
                    agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
            summary.append({"model": model_name, "metrics": agg, "n_folds": len(fold_results)})

        import json
        with open(self.output_dir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        # intentionally no printing of metrics to terminal
