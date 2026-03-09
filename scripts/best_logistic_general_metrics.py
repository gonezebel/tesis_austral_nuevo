from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

RANDOM_STATE = 42
MAX_ITER = 1000

BALANCE_METHODS = {
    "NONE": None,
    "SMOTE": SMOTE(random_state=RANDOM_STATE),
    "UNDER": RandomUnderSampler(random_state=RANDOM_STATE),
    "SMOTEENN": SMOTEENN(random_state=RANDOM_STATE),
}


def find_latest_logistic_dir(base_reports: Path) -> Path:
    candidates = sorted(base_reports.glob("*/Logistic_*"))
    if not candidates:
        raise FileNotFoundError(f"No se encontraron carpetas Logistic_* en: {base_reports}")
    return candidates[-1]


def build_model(balance_name: str) -> Pipeline:
    balancer = BALANCE_METHODS.get(str(balance_name).upper())
    class_weight = None if balancer is not None else "balanced"
    steps = []
    if balancer is not None:
        steps.append(("balancer", clone(balancer)))
    steps.append(
        (
            "model",
            LogisticRegression(max_iter=MAX_ITER, solver="lbfgs", class_weight=class_weight),
        )
    )
    return Pipeline(steps=steps)


def main():
    report_dir = find_latest_logistic_dir(Path("reports/06_modelo_logistic"))
    summary_path = report_dir / "resumen_modelos_logistic_regression.csv"
    data_dir = Path("data/intermediate/05_seleccion/04_2026_01_12")

    df = pd.read_csv(summary_path)
    req_cols = {"modelo", "balanceo", "cv_macro_f1", "cv_recall_target"}
    missing = req_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en resumen: {missing}")

    # Mejor modelo general: max cv_macro_f1; desempate por cv_recall_target.
    df_sorted = df.sort_values(["cv_macro_f1", "cv_recall_target"], ascending=[False, False]).reset_index(drop=True)
    best = df_sorted.iloc[0]

    model_name = str(best["modelo"])
    balance_name = str(best["balanceo"]).upper()

    x_train = pd.read_parquet(data_dir / f"X_train_{model_name}.parquet")
    x_test = pd.read_parquet(data_dir / f"X_test_{model_name}.parquet")
    y_train = pd.read_parquet(data_dir / f"y_train_{model_name}.parquet").squeeze()
    y_test = pd.read_parquet(data_dir / f"y_test_{model_name}.parquet").squeeze()

    pipe = build_model(balance_name)
    pipe.fit(x_train, y_train)

    y_pred_train = pipe.predict(x_train)
    y_pred_test = pipe.predict(x_test)

    report_train = pd.DataFrame(classification_report(y_train, y_pred_train, output_dict=True, zero_division=0)).T
    report_test = pd.DataFrame(classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)).T

    clases = sorted(set(pd.Series(y_train).astype(str)).union(set(pd.Series(y_test).astype(str))), key=lambda x: int(x) if x.isdigit() else x)

    global_metrics = {
        "modelo": model_name,
        "balanceo": balance_name,
        "criterio_general": "max cv_macro_f1 (desempate cv_recall_target)",
        "cv_macro_f1": float(best["cv_macro_f1"]),
        "cv_recall_target": float(best["cv_recall_target"]),
        "accuracy_test": float(accuracy_score(y_test, y_pred_test)),
        "macro_f1_test": float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
        "weighted_f1_test": float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0)),
        "accuracy_train": float(accuracy_score(y_train, y_pred_train)),
        "macro_f1_train": float(f1_score(y_train, y_pred_train, average="macro", zero_division=0)),
        "weighted_f1_train": float(f1_score(y_train, y_pred_train, average="weighted", zero_division=0)),
    }
    global_metrics["sobreajuste_macro_f1"] = global_metrics["macro_f1_train"] - global_metrics["macro_f1_test"]

    per_class_rows = []
    for c in clases:
        train_vals = report_train.loc[c] if c in report_train.index else None
        test_vals = report_test.loc[c] if c in report_test.index else None
        per_class_rows.append(
            {
                "clase": c,
                "f1_train": float(train_vals["f1-score"]) if train_vals is not None else np.nan,
                "recall_train": float(train_vals["recall"]) if train_vals is not None else np.nan,
                "precision_train": float(train_vals["precision"]) if train_vals is not None else np.nan,
                "support_train": float(train_vals["support"]) if train_vals is not None else np.nan,
                "f1_test": float(test_vals["f1-score"]) if test_vals is not None else np.nan,
                "recall_test": float(test_vals["recall"]) if test_vals is not None else np.nan,
                "precision_test": float(test_vals["precision"]) if test_vals is not None else np.nan,
                "support_test": float(test_vals["support"]) if test_vals is not None else np.nan,
            }
        )

    out_json = report_dir / "best_logistic_general_metrics.json"
    out_csv = report_dir / "best_logistic_general_metrics_per_class.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(global_metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(per_class_rows).to_csv(out_csv, index=False)

    print(f"BEST_GENERAL_MODELO={model_name}")
    print(f"BEST_GENERAL_BALANCEO={balance_name}")
    print(f"cv_macro_f1={global_metrics['cv_macro_f1']:.6f}")
    print(f"cv_recall_target={global_metrics['cv_recall_target']:.6f}")
    print(f"macro_f1_test={global_metrics['macro_f1_test']:.6f}")
    print(f"CSV_PER_CLASS={out_csv}")
    print(f"JSON_GLOBAL={out_json}")


if __name__ == "__main__":
    main()
