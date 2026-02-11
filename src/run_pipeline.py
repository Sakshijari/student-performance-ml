"""
Main pipeline: load data -> preprocess -> train all models -> evaluate -> save.
Run from project root: python -m src.run_pipeline
Or: python src/run_pipeline.py
"""

import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import load_data, preprocess_and_split
from src.train_and_evaluate import run_training_and_evaluation, print_results_summary


def main():
    data_path = os.path.join(PROJECT_ROOT, "data", "StudentsPerformance.csv")
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    print("Loading data...")
    df = load_data(data_path)
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

    print("Preprocessing and splitting (stratified)...")
    data_result = preprocess_and_split(df)
    print(f"Train: {data_result['X_train'].shape[0]}, Val: {data_result['X_val'].shape[0]}, Test: {data_result['X_test'].shape[0]}")

    output_dir = os.path.join(PROJECT_ROOT, "output")
    print("Training and evaluating models...")
    results = run_training_and_evaluation(data_result, output_dir=output_dir)

    print_results_summary(results)
    print(f"\nResults and models saved to: {output_dir}")


if __name__ == "__main__":
    main()
