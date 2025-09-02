import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from cross_validation import CrossValidationManager


class CompleteEvaluationManager:
    """
    Post-processing for CV:
      - run K-fold CV (no test)
      - convert cv_summary.json to per-fold CSV and model-level mean/std CSV
    No metric tables are printed to terminal.
    """

    def __init__(self, dataset_dir="ultralytics/datasets", output_dir="cv_results", k_folds=5, test_ratio=0.1):
        self.cv_manager = CrossValidationManager(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            k_folds=k_folds,
            test_ratio=test_ratio,
        )
        self.output_dir = Path(output_dir)
        self.summary_dir = self.output_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def run_complete_evaluation(self, model_configs, epochs=200):
        """
        Run CV and export CSV summaries.
        No terminal metric printing; all artifacts are saved to disk.
        """
        _ = self.cv_manager.run_cross_validation(model_configs, epochs)
        df = self.export_per_fold_metrics()
        return {"per_fold_metrics": df}

    def export_per_fold_metrics(self):
        """
        Read cv_summary.json and write:
          - summary/per_fold_metrics.csv
          - summary/metrics_mean_std_by_model.csv
        """
        summary_path = self.cv_manager.output_dir / "cv_summary.json"
        if not summary_path.exists():
            print(f"[Eval] not found: {summary_path}")
            return None

        with open(summary_path, "r") as f:
            summary = json.load(f)

        rows = []
        for model_summary in summary:
            model_name = model_summary.get("model", "unknown")
            n_folds = int(model_summary.get("n_folds", 0))
            metrics = model_summary.get("metrics", {})

            vals_m95 = (metrics.get("mAP50-95") or metrics.get("mAP50_95") or {}).get("values", [])
            vals_m50 = (metrics.get("mAP50") or {}).get("values", [])
            vals_p = (metrics.get("precision") or {}).get("values", [])
            vals_r = (metrics.get("recall") or {}).get("values", [])

            max_len = max(len(vals_m95), len(vals_m50), len(vals_p), len(vals_r), n_folds)
            for i in range(max_len):
                rows.append(
                    {
                        "model": model_name,
                        "fold": i,
                        "mAP50-95": vals_m95[i] if i < len(vals_m95) else np.nan,
                        "mAP50": vals_m50[i] if i < len(vals_m50) else np.nan,
                        "precision": vals_p[i] if i < len(vals_p) else np.nan,
                        "recall": vals_r[i] if i < len(vals_r) else np.nan,
                    }
                )

        df = pd.DataFrame(rows)
        per_fold_csv = self.summary_dir / "per_fold_metrics.csv"
        df.to_csv(per_fold_csv, index=False)

        agg = (
            df.groupby("model")[["mAP50-95", "mAP50", "precision", "recall"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg_csv = self.summary_dir / "metrics_mean_std_by_model.csv"
        agg.to_csv(agg_csv, index=False)

        print(f"[Eval] per-fold metrics written: {per_fold_csv}")
        print(f"[Eval] mean/std summary written: {agg_csv}")
        return df


def main():
    # Minimal example usage; safe defaults.
    model_configs = {
        "both_innovations": "yolov8-CA-LSKA.yaml",
        "baseline_yolov8n": "ultralytics/cfg/models/v8/yolov8n.yaml",
        "only_CA":"yolov8-CA-only.yaml;"
        "only_LSKA":"yolov8-LSKA-only.yaml"
    }
    evaluator = CompleteEvaluationManager(
        dataset_dir="ultralytics/datasets",
        output_dir="cv_results_1",
        k_folds=5,
        test_ratio=0.1,
    )
    evaluator.run_complete_evaluation(model_configs, epochs=200)


if __name__ == "__main__":
    main()
