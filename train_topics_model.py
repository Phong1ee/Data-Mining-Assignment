"""
Train a label predictor (1–5) from OpenAlex topics/keywords + paper metadata.

Optimizes for Quadratic Weighted Kappa (QWK) via cross-validation.
Uses HistGradientBoostingClassifier (handles mixed numeric/categorical features well).

Requires: CSV/papers_topics_keywords.csv from extract_topics_keywords.py

Usage:
  python train_topics_model.py
  python train_topics_model.py --features-csv CSV/papers_topics_keywords.csv --predictions predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def clip_labels(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(y).astype(int), 1, 5)


def build_text_column(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ("title", "topic_1", "topic_2", "topic_3", "keyword_1", "keyword_2", "keyword_3"):
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out


def _to_csr(X) -> csr_matrix:
    if hasattr(X, "tocsr"):
        return X.tocsr()
    return csr_matrix(X)


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[np.ndarray, float, Pipeline]:
    y_train = train_df["Label"].astype(int).values
    text_train = build_text_column(train_df)
    text_test = build_text_column(test_df)

    numeric_cols = [
        c
        for c in [
            "year",
            "primary_topic_score",
            "topic_1_score",
            "topic_2_score",
            "topic_3_score",
            "keyword_1_score",
            "keyword_2_score",
            "keyword_3_score",
        ]
        if c in train_df.columns
    ]

    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)
    X_text_tr = tfidf.fit_transform(text_train)
    X_text_te = tfidf.transform(text_test)

    tab_pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            ("venue", OneHotEncoder(handle_unknown="ignore", max_categories=32), ["venue"]),
        ],
        remainder="drop",
    )
    X_tab_tr = tab_pre.fit_transform(train_df)
    X_tab_te = tab_pre.transform(test_df)

    X_tr = hstack([X_text_tr, _to_csr(X_tab_tr)])
    X_te = hstack([X_text_te, _to_csr(X_tab_te)])

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=12,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_cv = cross_val_predict(clf, X_tr, y_train, cv=cv)
    qwk = quadratic_weighted_kappa(y_train, y_cv)

    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te).astype(int)
    y_pred = clip_labels(y_pred)

    return y_pred, qwk, clf


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features-csv",
        type=Path,
        default=root / "CSV" / "papers_topics_keywords.csv",
    )
    ap.add_argument(
        "--train-csv",
        type=Path,
        default=root / "CSV" / "train.csv",
        help="Used only if features CSV has no Label column for some rows",
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        default=root / "predictions.csv",
    )
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    if not args.features_csv.is_file():
        print(f"Missing {args.features_csv}. Run: python extract_topics_keywords.py")
        raise SystemExit(1)

    df = pd.read_csv(args.features_csv)
    if "Label" not in df.columns and args.train_csv.is_file():
        labels = pd.read_csv(args.train_csv)[["id", "Label"]]
        df = df.merge(labels, on="id", how="left", suffixes=("", "_y"))

    train_df = df[df["Label"].notna()].copy()
    test_df = df[df["Label"].isna()].copy()

    if train_df.empty:
        print("No labeled training rows in features CSV.")
        raise SystemExit(1)

    # Rows with no topics: still usable via title + venue + year
    train_df["venue"] = train_df["venue"].fillna("unknown").astype(str)
    test_df["venue"] = test_df["venue"].fillna("unknown").astype(str)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    predictions, qwk_cv, _ = fit_predict(train_df, test_df, n_splits=args.splits)

    print(f"\n5-fold CV Quadratic Weighted Kappa: {qwk_cv:.4f}")
    print("\n(Train with topics/keywords + title + venue + year)")
    print("Model: HistGradientBoostingClassifier")

    out = pd.DataFrame({"id": test_df["id"].astype(int), "Label": predictions})
    out.to_csv(args.predictions, index=False)
    print(f"\nSaved {args.predictions} ({len(out)} rows)")
    print(out.head(10))


if __name__ == "__main__":
    main()
