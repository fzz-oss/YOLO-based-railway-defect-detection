# YOLO-based Railway Defect Detection

This project focuses on railway defect detection using **YOLOv8** with two key improvements: **Coordinate Attention (CA)** and **Large Selective Kernel Attention (LSKA)**.  
The dataset is based on [Railway-defect-detection](https://github.com/hy199248/Railway-defect-detection).  
The baseline YOLOv8 implementation comes from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/ultralytics).

---

## Project Structure

The project is divided into two parts: **Experiment 1** and **Experiment 2**.  

- Both experiments used similar **5-fold cross-validation** code.  
- In this repository, redundant code has been integrated:
  - `complete_cv_evaluation.py` and `cross_validation.py`: unified cross-validation scripts.  
  - `final_train.py` and `run_final_test.py`: designed for Experiment 2, where the best hyperparameters were selected, followed by training and evaluation on the test set.  
- During the early phase, two servers were used to run both parts separately, but the scripts here consolidate those workflows.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Prepare the dataset

Download the dataset from Railway-defect-detection

and update the dataset path in configs/dataset.yaml.

3. Run cross-validation (Experiment 1 & 2)
python scripts/complete_cv_evaluation.py
# or
python scripts/cross_validation.py

4. Train the model with best hyperparameters (Experiment 2)
python scripts/final_train.py

5. Evaluate on the test set
python scripts/run_final_test.py


Training and evaluation results (metrics, logs, and visualizations) will be saved in the results/ folder.

Results
Test Results on the Dataset

The table below shows a comparison between the CA+LSKA model and the Baseline YOLOv8 on the test set after hyperparameter tuning.

Left: CA+LSKA model

Right: Baseline YOLOv8

CA+LSKA Model	Baseline YOLOv8

	

Metrics include precision (P), recall (R), mAP@50, and mAP@50-95 across defect categories: Spalling, Squat, Wheel Burn, Corrugation.

Requirements

To install dependencies, run:

pip install -r requirements.txt
