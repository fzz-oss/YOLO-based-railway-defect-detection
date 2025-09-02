# test_final_weights.py  
import json
from pathlib import Path
from ultralytics import YOLO
import torch

# paths (kept as your originals; edit if needed)
MODEL_WEIGHTS = {
    "both_innovations": "/root/autodl-tmp/ultralytics-main/cv_results_1/final_runs_bs24_lr1e-3/both_innovations_full_bs24_lr1e-03/weights/best.pt",
    "baseline_yolov8n": "/root/autodl-tmp/ultralytics-main/cv_results_1/final_runs_bs24_lr1e-3/baseline_yolov8n_full_bs24_lr1e-03/weights/best.pt",
}

DATASET_YAML = "/root/autodl-tmp/ultralytics-main/cv_results_1/fold_0/final_fulltrain/final_dataset.yaml"
PROJECT_EVAL = Path("/root/autodl-tmp/ultralytics-main/cv_results_1/final_test_eval")
PROJECT_EVAL.mkdir(parents=True, exist_ok=True)


def _safe_get_speed_ms(results, key: str):
    """Robustly read results.speed[key] across different Ultralytics versions."""
    try:
        s = results.speed
        if isinstance(s, dict):
            v = s.get(key, None)
        else:
            v = getattr(s, key, None)
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def _safe_fps_from_ms(ms):
    """Return FPS given milliseconds per image. NaN-safe and zero-safe."""
    try:
        ms = float(ms)
        if ms <= 0 or ms != ms:  # <=0 or NaN
            return float("nan")
        return 1000.0 / ms
    except Exception:
        return float("nan")


def run_test(model_name, weights, data_yaml):
    print(f"\nTesting {model_name} on TEST split ...")
    device = '0' if torch.cuda.is_available() else 'cpu'
    model = YOLO(weights)

    results = model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=1,
        device=device,
        save_json=True,
        project=str(PROJECT_EVAL),
        name=f"{model_name}_test_eval",
    )

    pre_ms  = _safe_get_speed_ms(results, "preprocess")
    inf_ms  = _safe_get_speed_ms(results, "inference")
    post_ms = _safe_get_speed_ms(results, "postprocess")
    total_ms = (pre_ms if pre_ms == pre_ms else 0.0) + (inf_ms if inf_ms == inf_ms else 0.0) + (post_ms if post_ms == post_ms else 0.0)

    metrics = {
        "model": model_name,
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.precision.mean()),
        "recall": float(results.box.recall.mean()),
        "f1": float(
            2 * results.box.precision.mean() * results.box.recall.mean() /
            max(1e-9, (results.box.precision.mean() + results.box.recall.mean()))
        ),
        "speed_ms_pre": float(pre_ms) if pre_ms == pre_ms else float("nan"),
        "speed_ms_inf": float(inf_ms) if inf_ms == inf_ms else float("nan"),
        "speed_ms_post": float(post_ms) if post_ms == post_ms else float("nan"),
        "fps_model": _safe_fps_from_ms(inf_ms),
        "fps_end2end": _safe_fps_from_ms(total_ms),
    }

    out_json = PROJECT_EVAL / f"{model_name}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"results saved: {out_json}")

    # release GPU memory between models
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    all_metrics = []
    for model_name, weights in MODEL_WEIGHTS.items():
        all_metrics.append(run_test(model_name, weights, DATASET_YAML))

    print("\n================== FINAL TEST RESULTS ==================")
    for m in all_metrics:
        print(f"[{m['model']}] mAP50-95={m['mAP50-95']:.4f}, "
              f"mAP50={m['mAP50']:.4f}, "
              f"P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, "
              f"FPS(model)={m['fps_model']:.1f}, FPS(end2end)={m['fps_end2end']:.1f}")
    print("========================================================")


if __name__ == "__main__":
    main()
