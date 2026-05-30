from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Label" not in df.columns:
        raise ValueError(f"Features CSV {path} has no Label column")
    return df


def fit_model(train_df: pd.DataFrame):
    y_train = train_df["Label"].astype(int).values
    text_train = build_text_column(train_df)
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
    X_tr = hstack([X_text_tr, _to_csr(X_tab_tr)]).toarray().astype(np.float32)

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=12,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_train)
    return clf, tfidf, tab_pre, numeric_cols


def predict_dataset(df: pd.DataFrame, clf, tfidf, tab_pre, numeric_cols):
    text = build_text_column(df)
    X_text = tfidf.transform(text)
    X_tab = tab_pre.transform(df)
    X = hstack([X_text, _to_csr(X_tab)]).toarray().astype(np.float32)
    preds = clf.predict(X).astype(int)
    preds = np.clip(np.round(preds).astype(int), 1, 5)
    return preds


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    train_path = root / "CSV" / "train_papers_topics_keywords.csv"
    private_path = root / "CSV" / "private_test_topics_keywords.csv"
    public_path = root / "CSV" / "public_test_topics_keywords.csv"
    out_path = root / "prediction.csv"

    train_df = load_features(train_path)
    private_df = pd.read_csv(private_path)
    public_df = pd.read_csv(public_path)

    for df in (private_df, public_df):
        df["venue"] = df["venue"].fillna("unknown").astype(str)

    clf, tfidf, tab_pre, numeric_cols = fit_model(train_df)

    private_preds = predict_dataset(private_df, clf, tfidf, tab_pre, numeric_cols)
    public_preds = predict_dataset(public_df, clf, tfidf, tab_pre, numeric_cols)

    out_df = pd.concat(
        [
            pd.DataFrame({"id": private_df["id"].astype(int), "Label": private_preds}),
            pd.DataFrame({"id": public_df["id"].astype(int), "Label": public_preds}),
        ],
        ignore_index=True,
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(out_df)} rows")
    print(out_df.head(5).to_string(index=False))
