from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Shop Smart Dashboard",
    page_icon="SS",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(255, 213, 152, 0.45), transparent 32%),
            radial-gradient(circle at 85% 20%, rgba(147, 210, 255, 0.45), transparent 28%),
            linear-gradient(165deg, #fef6ea 0%, #f4f8fb 55%, #eef2f8 100%);
    }

    .hero {
        border: 1px solid rgba(17, 40, 54, 0.14);
        border-radius: 18px;
        padding: 20px;
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.92), rgba(255, 245, 224, 0.94));
        box-shadow: 0 12px 26px rgba(18, 40, 52, 0.14);
        margin-bottom: 0.8rem;
    }

    .hero h1 {
        font-family: 'Playfair Display', serif;
        color: #0f3f55;
        margin: 0 0 8px 0;
        font-size: 2rem;
    }

    .hero p {
        margin: 0;
        color: #25485c;
    }

    .panel {
        border: 1px solid rgba(17, 40, 54, 0.12);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.88);
        padding: 14px;
    }

    .good {
        border: 1px solid #85cc9a;
        background: #dbf5e3;
        color: #114a2a;
        border-radius: 12px;
        padding: 10px 12px;
        font-weight: 700;
    }

    .bad {
        border: 1px solid #efad9f;
        background: #ffe8e2;
        color: #7e1f1f;
        border-radius: 12px;
        padding: 10px 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_CANDIDATES = [
    Path("online_shoppers.csv"),
    Path("shop_smart_ecommerce"),
]
TARGET_COL = "Revenue"


def _find_data_file() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "No dataset found. Expected one of: online_shoppers.csv, shop_smart_ecommerce"
    )


def _normalize_target(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
    }
    return text.map(mapping).fillna(0).astype(int)


@st.cache_data
def load_data() -> pd.DataFrame:
    file_path = _find_data_file()
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_resource
def train_pipeline(df: pd.DataFrame) -> dict[str, object]:
    x = df.drop(columns=[TARGET_COL])
    y = _normalize_target(df[TARGET_COL])

    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    return {
        "pipeline": pipeline,
        "x": x,
        "metrics": metrics,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "cm": confusion_matrix(y_test, y_pred),
    }


def _make_user_input_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Session Input")

    numeric_cols = features_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = features_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    user_row: dict[str, object] = {}

    col_a, col_b = st.columns(2)

    for idx, col in enumerate(numeric_cols):
        col_data = pd.to_numeric(features_df[col], errors="coerce")
        safe_min = float(np.nanmin(col_data)) if not np.isnan(np.nanmin(col_data)) else 0.0
        safe_max = float(np.nanmax(col_data)) if not np.isnan(np.nanmax(col_data)) else safe_min + 1.0
        median = float(np.nanmedian(col_data)) if not np.isnan(np.nanmedian(col_data)) else safe_min

        target_col = col_a if idx % 2 == 0 else col_b
        with target_col:
            user_row[col] = st.slider(
                col,
                min_value=safe_min,
                max_value=safe_max if safe_max > safe_min else safe_min + 1.0,
                value=min(max(median, safe_min), safe_max if safe_max > safe_min else safe_min + 1.0),
            )

    for idx, col in enumerate(categorical_cols):
        options = (
            features_df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
            .unique()
            .tolist()
        )
        options = sorted(options) if options else ["Unknown"]

        target_col = col_a if idx % 2 == 0 else col_b
        with target_col:
            user_row[col] = st.selectbox(col, options=options, index=0)

    return pd.DataFrame([user_row])


def main() -> None:
    st.markdown(
        """
        <div class='hero'>
            <h1>Shop Smart Conversion Predictor</h1>
            <p>
                Decision-tree powered ecommerce dashboard based on your shop_smart notebook.
                Fill visitor behavior inputs and get conversion likelihood instantly.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    model_artifacts = train_pipeline(df)

    pipeline: Pipeline = model_artifacts["pipeline"]  # type: ignore[assignment]
    x: pd.DataFrame = model_artifacts["x"]  # type: ignore[assignment]
    metrics: dict[str, float] = model_artifacts["metrics"]  # type: ignore[assignment]
    report: dict[str, dict[str, float]] = model_artifacts["report"]  # type: ignore[assignment]
    cm: np.ndarray = model_artifacts["cm"]  # type: ignore[assignment]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Features", str(x.shape[1]))
    k3.metric("Conversion Rate", f"{(_normalize_target(df[TARGET_COL]).mean() * 100):.2f}%")
    k4.metric("F1 Score", f"{metrics['f1'] * 100:.2f}%")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        with st.form("prediction_form"):
            input_df = _make_user_input_frame(x)
            submit = st.form_submit_button("Predict Conversion")

    with right:
        st.subheader("Model Dashboard")
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        st.metric("F1", f"{metrics['f1'] * 100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Classification report"):
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)

        with st.expander("Confusion matrix"):
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: No Purchase", "Actual: Purchase"],
                columns=["Pred: No Purchase", "Pred: Purchase"],
            )
            st.dataframe(cm_df, use_container_width=True)

    if submit:
        pred = int(pipeline.predict(input_df)[0])
        probability = None

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            probability = float(pipeline.predict_proba(input_df)[0][1])

        st.markdown("### Prediction Output")

        if pred == 1:
            st.markdown(
                "<div class='good'>Likely buyer: this session has strong conversion intent.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='bad'>Low conversion intent: this visitor is less likely to purchase.</div>",
                unsafe_allow_html=True,
            )

        if probability is not None:
            st.progress(probability)
            st.write(f"Conversion probability: {probability * 100:.2f}%")

    with st.sidebar:
        st.header("Run Info")
        st.write(f"Dataset rows: {len(df):,}")
        st.write(f"Target column: {TARGET_COL}")
        st.caption("Data source auto-detected from local project files.")


if __name__ == "__main__":
    main()
