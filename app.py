from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "digits_dataset.npz"
MODEL_PATH = BASE_DIR / "models" / "digit_cnn.keras"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

st.set_page_config(page_title="CNN Digit Classifier", page_icon="🔢", layout="wide")
st.title("🔢 CNN Handwritten Digit Classifier")
st.write("Explore an included handwritten-digit dataset and use a trained Convolutional Neural Network (CNN) to classify 0–9 digits.")

@st.cache_data
def load_dataset(path: str):
    with np.load(path, allow_pickle=False) as z:
        images = z["images"].astype(np.float32)
        labels = z["labels"].astype(np.int64)
    return images, labels

@st.cache_resource
def load_cnn(path: str):
    return load_model(path)

if not DATA_PATH.exists():
    st.error("Dataset is missing. Make sure data/digits_dataset.npz is included in the project.")
    st.stop()

images, labels = load_dataset(str(DATA_PATH))

with st.sidebar:
    st.header("Dataset overview")
    st.metric("Images", f"{len(images):,}")
    st.metric("Classes", "10 (digits 0–9)")
    st.metric("Image shape", "8 × 8 pixels")
    st.caption("Dataset: scikit-learn handwritten digits. Pixel values range from 0 to 16.")

explore_tab, predict_tab, about_tab = st.tabs(["🔎 Explore data", "🧠 CNN prediction", "📘 About the project"])

with explore_tab:
    st.subheader("Explore a sample image")
    idx = st.slider("Choose a sample index", min_value=0, max_value=len(images)-1, value=0)
    left, right = st.columns([1, 2])
    with left:
        st.image(images[idx], clamp=True, width=240, caption=f"Sample #{idx}")
        st.markdown(f"**Actual label: `{labels[idx]}`**")
    with right:
        st.write("Each image is an 8×8 grayscale array. A CNN learns local visual patterns from these pixel values.")
        class_counts = pd.Series(labels).value_counts().sort_index()
        class_df = pd.DataFrame({"Digit": class_counts.index.astype(str), "Images": class_counts.values})
        st.bar_chart(class_df, x="Digit", y="Images")
    with st.expander("Preview dataset rows (flattened pixels + label)"):
        flat = images.reshape(len(images), -1).astype(int)
        df = pd.DataFrame(flat, columns=[f"pixel_{i}" for i in range(64)])
        df["label"] = labels
        st.dataframe(df.head(20), use_container_width=True)
        st.download_button("Download dataset CSV", data=(BASE_DIR / "data" / "digits_dataset.csv").read_bytes(), file_name="digits_dataset.csv", mime="text/csv")

with predict_tab:
    st.subheader("Predict the digit using a CNN")
    if not MODEL_PATH.exists():
        st.warning("The trained model file is not here yet. Run `python train_model.py` in VS Code, then refresh this page. Include `models/digit_cnn.keras` when pushing to GitHub for deployment.")
    else:
        try:
            model = load_cnn(str(MODEL_PATH))
            test_indices = np.arange(len(images))
            metrics = None
            if METRICS_PATH.exists():
                try:
                    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                    if metrics.get("test_indices"):
                        test_indices = np.array(metrics["test_indices"], dtype=int)
                except (ValueError, OSError, TypeError):
                    metrics = None
            if metrics and "test_accuracy" in metrics:
                st.metric("Held-out test accuracy", f"{metrics['test_accuracy']:.1%}")
            chosen = st.selectbox("Choose an image to classify", test_indices.tolist(), index=0, format_func=lambda i: f"Sample {i} — actual digit {labels[i]}")
            image = images[chosen]
            probabilities = model.predict((image / 16.0)[None, ..., None], verbose=0)[0]
            predicted = int(np.argmax(probabilities))
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(image, clamp=True, width=240, caption=f"Dataset sample #{chosen}")
                st.markdown(f"**Actual digit:** `{labels[chosen]}`")
                st.markdown(f"**Predicted digit:** `{predicted}`")
                if predicted == int(labels[chosen]):
                    st.success("Prediction matches the label.")
                else:
                    st.info("Prediction differs from the dataset label.")
            with c2:
                prob_df = pd.DataFrame({"Digit": [str(i) for i in range(10)], "Probability": probabilities})
                st.bar_chart(prob_df, x="Digit", y="Probability")
                st.caption("Probabilities are model confidence scores, not guarantees of correctness.")
        except Exception as exc:
            st.error(f"Could not load or use the CNN model: {exc}")

with about_tab:
    st.subheader("What is a CNN?")
    st.write("A Convolutional Neural Network (CNN) learns visual features such as edges, curves, and digit shapes using convolutional filters, then predicts one of ten classes.")
    st.markdown("**Project workflow**")
    st.markdown("1. Load the dataset\n2. Normalize pixel values\n3. Split into training and test sets\n4. Train a CNN\n5. Evaluate predictions\n6. Save the Keras model and deploy this Streamlit app")
    st.markdown("**Files created by training**")
    st.code("models/digit_cnn.keras\nmodels/metrics.json")
    st.caption("Educational demo: the included scikit-learn digits dataset has 1,797 samples of 8×8 images, not full-resolution photos.")
