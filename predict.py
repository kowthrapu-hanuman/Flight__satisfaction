import argparse, joblib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV (no target column)")
    ap.add_argument("--model", default="models/model_best.joblib")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    clf = joblib.load(args.model)
    preds = clf.predict(df)

    out = df.copy()
    out["prediction"] = preds
    try:
        proba = clf.predict_proba(df)
        classes = list(clf.named_steps["model"].classes_)
        pos_idx = classes.index(sorted(classes)[-1])
        out["probability"] = proba[:, pos_idx]
    except Exception:
        pass

    out.to_csv(args.out, index=False)
    print(f"Saved predictions → {args.out}")

if __name__ == "__main__":
    main()
