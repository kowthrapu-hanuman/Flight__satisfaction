import argparse, os, json
import joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

TARGET = "Satisfaction"

def infer_columns(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    feature_cols = [c for c in df.columns if c != target]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])

def try_roc_auc(y_true, proba, classes):
    try:
        labels = list(sorted(classes))
        if len(labels) != 2:
            return None
        pos = labels[-1]               # deterministic positive
        pos_idx = list(classes).index(pos)
        y_bin = (np.array(y_true) == pos).astype(int)
        return roc_auc_score(y_bin, proba[:, pos_idx])
    except Exception:
        return None

def train_and_select(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "dtree": DecisionTreeClassifier(random_state=42),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
    }
    results, best_name, best_score, best_pipe = {}, None, -1, None
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        auc = None
        if hasattr(model, "predict_proba"):
            try:
                auc = try_roc_auc(y_test, pipe.predict_proba(X_test), pipe.named_steps["model"].classes_)
            except Exception:
                pass
        score = auc if auc is not None else acc
        results[name] = {"accuracy": float(acc), "roc_auc": (float(auc) if auc is not None else None),
                         "selection_score": float(score)}
        if score > best_score:
            best_name, best_score, best_pipe = name, score, pipe
    return best_name, best_pipe, results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training CSV")
    ap.add_argument("--target", default=TARGET, help="Target column name")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default="models/model_best.joblib")
    ap.add_argument("--metrics-out", default="models/metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feat_cols, num_cols, cat_cols = infer_columns(df, args.target)
    X, y = df[feat_cols], df[args.target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    pre = build_preprocessor(num_cols, cat_cols)
    best_name, best_pipe, results = train_and_select(Xtr, Xte, ytr, yte, pre)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(best_pipe, args.out)
    meta = {"best_model": best_name, "results": results, "feature_columns": feat_cols,
            "numeric_columns": num_cols, "categorical_columns": cat_cols, "target": args.target}
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f: json.dump(meta, f, indent=2)

    print(f"Saved best pipeline → {args.out}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
