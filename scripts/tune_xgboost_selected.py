from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_BASE = REPO_ROOT / "data" / "intermediate" / "05_seleccion"
SUMMARY_GLOB = REPO_ROOT / "reports" / "08_modelo_xgboost"
OUTPUT_BASE = REPO_ROOT / "reports" / "13_tuning_xgboost"

BALANCE_METHODS = {
    "NONE": None,
    "SMOTE": SMOTE(random_state=42),
    "UNDER": RandomUnderSampler(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuning XGBoost solo sobre la base/balanceo ganadores del resumen previo."
    )
    parser.add_argument("--summary-csv", type=Path, default=None, help="Ruta al resumen_modelos_xgboost.csv")
    parser.add_argument("--base-name", type=str, default=None, help="Forzar base_name/modelo")
    parser.add_argument("--balance-name", type=str, default=None, help="Forzar balanceo (NONE/SMOTE/UNDER/SMOTEENN)")
    parser.add_argument("--target-class", type=int, default=None, help="Clase objetivo para recall (por defecto min(y_train))")
    parser.add_argument("--n-iter", type=int, default=24, help="Iteraciones de RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Folds de CV")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def find_latest_summary() -> Path:
    candidates = sorted(SUMMARY_GLOB.glob("**/resumen_modelos_xgboost.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No se encontro resumen_modelos_xgboost.csv en reports/08_modelo_xgboost")
    return candidates[-1]


def find_latest_input_path() -> Path:
    if not INPUT_BASE.exists():
        raise FileNotFoundError(f"No existe {INPUT_BASE}")
    candidates = sorted([p for p in INPUT_BASE.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No hay subdirectorios en data/intermediate/05_seleccion")
    return candidates[-1]


def choose_best_from_summary(summary_csv: Path) -> tuple[str, str]:
    df = pd.read_csv(summary_csv)
    required = {"modelo", "balanceo", "cv_recall_target", "cv_macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en resumen: {sorted(missing)}")

    df = df.dropna(subset=["cv_recall_target", "cv_macro_f1"]).copy()
    if df.empty:
        raise ValueError("El resumen no tiene filas validas para seleccionar mejor modelo.")

    best = df.sort_values(["cv_recall_target", "cv_macro_f1"], ascending=False).iloc[0]
    return str(best["modelo"]), str(best["balanceo"])


def resolve_target_class(y: pd.Series, target: int | None) -> int:
    classes = list(pd.Series(y).unique())
    target_val = classes[0] if target is None else target
    if target_val in classes:
        return int(target_val)
    if str(target_val) in [str(c) for c in classes]:
        for c in classes:
            if str(c) == str(target_val):
                return int(c)
    return int(classes[0])


def build_pipeline(balancer, random_state: int) -> Pipeline:
    model = XGBClassifier(
        objective="multi:softprob",
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    steps = []
    if balancer is not None:
        steps.append(("balance", balancer))
    steps.append(("model", model))
    return Pipeline(steps)


def main() -> None:
    args = parse_args()

    summary_csv = args.summary_csv if args.summary_csv else find_latest_summary()
    input_path = find_latest_input_path()

    if args.base_name and args.balance_name:
        base_name, balance_name = args.base_name, args.balance_name
    else:
        base_name, balance_name = choose_best_from_summary(summary_csv)

    if balance_name not in BALANCE_METHODS:
        raise ValueError(f"Balanceo no soportado: {balance_name}")

    x_train_path = input_path / f"X_train_{base_name}.parquet"
    x_test_path = input_path / f"X_test_{base_name}.parquet"
    y_train_path = input_path / f"y_train_{base_name}.parquet"
    y_test_path = input_path / f"y_test_{base_name}.parquet"
    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    X_train = pd.read_parquet(x_train_path)
    X_test = pd.read_parquet(x_test_path)
    y_train = pd.read_parquet(y_train_path).squeeze()
    y_test = pd.read_parquet(y_test_path).squeeze()

    target_class = resolve_target_class(y_train, args.target_class)
    y_min = int(y_train.min())
    y_train_adj = y_train - y_min
    y_test_adj = y_test - y_min
    target_class_adj = target_class - y_min

    balancer = BALANCE_METHODS[balance_name]
    pipeline = build_pipeline(balancer, args.random_state)

    recall_target = make_scorer(
        recall_score,
        labels=[target_class_adj],
        average="macro",
        zero_division=0,
    )
    scorers = {"recall_target": recall_target, "f1_macro": "f1_macro"}

    param_distributions = {
        "model__n_estimators": [120, 180, 240, 320],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0.0, 0.1, 0.3],
        "model__reg_alpha": [0.0, 0.1, 0.5],
        "model__reg_lambda": [0.5, 1.0, 2.0],
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=scorers,
        refit="recall_target",
        cv=cv,
        n_jobs=-1,
        random_state=args.random_state,
        verbose=1,
        return_train_score=False,
    )
    search.fit(X_train, y_train_adj)

    best_model = search.best_estimator_
    y_pred_test_adj = best_model.predict(X_test)
    y_pred_train_adj = best_model.predict(X_train)

    y_pred_test = y_pred_test_adj + y_min
    y_pred_train = y_pred_train_adj + y_min

    run_id = datetime.today().strftime("%Y%m%d")
    output_dir = OUTPUT_BASE / run_id / f"XGBoost_tuning_{datetime.today().strftime('%Y-%m-%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_recall_target")
    cv_results.to_csv(output_dir / "cv_results_xgboost_tuning.csv", index=False)

    summary = {
        "base_name": base_name,
        "balanceo": balance_name,
        "summary_csv_source": str(summary_csv),
        "input_path": str(input_path),
        "target_class_original": int(target_class),
        "cv_best_recall_target": float(search.best_score_),
        "cv_best_f1_macro": float(cv_results.iloc[0]["mean_test_f1_macro"]),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_f1_macro": float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
        "train_f1_macro": float(f1_score(y_train, y_pred_train, average="macro", zero_division=0)),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "resumen_tuning_xgboost.csv", index=False)

    with open(output_dir / "best_params_xgboost.json", "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, indent=2)

    report_test = pd.DataFrame(classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)).T
    report_test.to_csv(output_dir / "classification_report_test_xgboost.csv")

    print("Resumen tuning XGBoost")
    print(json.dumps(summary, indent=2))
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
