from __future__ import annotations

import argparse
import time
from pathlib import Path

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
            LogisticRegression(
                max_iter=MAX_ITER,
                solver="lbfgs",
                class_weight=class_weight,
            ),
        )
    )
    return Pipeline(steps=steps)


def load_variant(data_dir: Path, model_name: str):
    x_train = pd.read_parquet(data_dir / f"X_train_{model_name}.parquet")
    x_test = pd.read_parquet(data_dir / f"X_test_{model_name}.parquet")
    y_train = pd.read_parquet(data_dir / f"y_train_{model_name}.parquet").squeeze()
    y_test = pd.read_parquet(data_dir / f"y_test_{model_name}.parquet").squeeze()
    return x_train, y_train, x_test, y_test


def safe_report(y_true, y_pred):
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T


def fill_row_metrics(row: pd.Series, data_dir: Path, classes_sorted: list[str]) -> dict:
    model_name = str(row["modelo"])
    balance_name = str(row["balanceo"]).upper()

    t0 = time.time()

    x_train, y_train, x_test, y_test = load_variant(data_dir, model_name)
    pipe = build_model(balance_name)
    pipe.fit(x_train, y_train)

    y_pred_train = pipe.predict(x_train)
    y_pred_test = pipe.predict(x_test)

    report_train = safe_report(y_train, y_pred_train)
    report_test = safe_report(y_test, y_pred_test)

    out = {
        "nan_total_train": int(x_train.isna().sum().sum()),
        "nan_total_test": int(x_test.isna().sum().sum()),
        "accuracy_test": float(accuracy_score(y_test, y_pred_test)),
        "macro_f1_test": float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
        "weighted_f1_test": float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0)),
        "accuracy_train": float(accuracy_score(y_train, y_pred_train)),
        "macro_f1_train": float(f1_score(y_train, y_pred_train, average="macro", zero_division=0)),
        "weighted_f1_train": float(f1_score(y_train, y_pred_train, average="weighted", zero_division=0)),
        "tiempo_min": (time.time() - t0) / 60.0,
    }
    out["sobreajuste_macro_f1"] = out["macro_f1_train"] - out["macro_f1_test"]

    for cls in classes_sorted:
        if cls in report_test.index:
            out[f"f1_{cls}_test"] = float(report_test.loc[cls, "f1-score"])
            out[f"recall_{cls}_test"] = float(report_test.loc[cls, "recall"])
            out[f"precision_{cls}_test"] = float(report_test.loc[cls, "precision"])
        else:
            out[f"f1_{cls}_test"] = np.nan
            out[f"recall_{cls}_test"] = np.nan
            out[f"precision_{cls}_test"] = np.nan

        if cls in report_train.index:
            out[f"f1_{cls}_train"] = float(report_train.loc[cls, "f1-score"])
            out[f"recall_{cls}_train"] = float(report_train.loc[cls, "recall"])
            out[f"precision_{cls}_train"] = float(report_train.loc[cls, "precision"])
        else:
            out[f"f1_{cls}_train"] = np.nan
            out[f"recall_{cls}_train"] = np.nan
            out[f"precision_{cls}_train"] = np.nan

    return out


def infer_classes(data_dir: Path) -> list[str]:
    y_paths = sorted(data_dir.glob("y_train_*.parquet"))
    all_classes = set()
    for p in y_paths[:15]:
        y = pd.read_parquet(p).squeeze()
        all_classes.update(pd.Series(y).dropna().astype(str).unique().tolist())
    # Si por algun motivo no alcanza con 15, tomamos todo.
    if not all_classes:
        for p in y_paths:
            y = pd.read_parquet(p).squeeze()
            all_classes.update(pd.Series(y).dropna().astype(str).unique().tolist())

    def _sort_key(v: str):
        try:
            return (0, int(v))
        except Exception:
            return (1, v)

    return sorted(all_classes, key=_sort_key)


def main():
    parser = argparse.ArgumentParser(description="Completa metricas globales y por clase para todas las variantes logisticas.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Carpeta Logistic_* con resumen_modelos_logistic_regression.csv. Si no se indica, usa la mas reciente.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/intermediate/05_seleccion/04_2026_01_12"),
        help="Carpeta con X_train_*.parquet, y_train_*.parquet, etc.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Procesa solo N filas (prueba rapida).")
    parser.add_argument(
        "--output-name",
        type=str,
        default="resumen_modelos_logistic_regression_full_metrics.csv",
        help="Nombre del CSV de salida.",
    )
    parser.add_argument(
        "--overwrite-original",
        action="store_true",
        help="Sobrescribe tambien resumen_modelos_logistic_regression.csv con el resultado completo.",
    )
    args = parser.parse_args()

    reports_base = Path("reports/06_modelo_logistic")
    report_dir = args.report_dir if args.report_dir is not None else find_latest_logistic_dir(reports_base)

    summary_path = report_dir / "resumen_modelos_logistic_regression.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"No existe: {summary_path}")

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe data_dir: {data_dir}")

    df = pd.read_csv(summary_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    classes_sorted = infer_classes(data_dir)
    print(f"Clases detectadas: {classes_sorted}")
    print(f"Filas a procesar: {len(df)}")

    filled = []
    total = len(df)
    for i, row in df.iterrows():
        idx = i + 1
        model_name = row.get("modelo")
        balance_name = row.get("balanceo")
        print(f"[{idx}/{total}] {model_name} | {balance_name}")

        metricas = fill_row_metrics(row, data_dir=data_dir, classes_sorted=classes_sorted)
        row_out = row.to_dict()
        row_out.update(metricas)
        filled.append(row_out)

    out_df = pd.DataFrame(filled)

    output_path = report_dir / args.output_name
    out_df.to_csv(output_path, index=False)
    print(f"CSV completo guardado en: {output_path}")

    if args.overwrite_original:
        out_df.to_csv(summary_path, index=False)
        print(f"Original sobrescrito: {summary_path}")


if __name__ == "__main__":
    main()
