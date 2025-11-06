import argparse, joblib, pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to labeled CSV")
    ap.add_argument("--target", default="Satisfaction")
    ap.add_argument("--model", default="models/model_best.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found.")
    X, y = df.drop(columns=[args.target]), df[args.target]

    clf = joblib.load(args.model)
    preds = clf.predict(X)

    print("Accuracy:", accuracy_score(y, preds))
    print("\nClassification report:\n", classification_report(y, preds))

if __name__ == "__main__":
    main()
