
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "final_plant_disease_model.keras"
LABELS_PATH = PROJECT_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a plant-leaf image to obtain an AI-generated "
    "classification across 38 plant-health categories."
)

st.info(
    "This application provides image-classification estimates. "
    "It is not a substitute for professional agricultural diagnosis."
)

@st.cache_resource
def load_artifacts():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if model.output_shape[-1] != len(labels):
        raise ValueError(
            "The model output count does not match the class labels."
        )

    return model, labels

try:
    model, class_names = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

        if st.button("Analyze leaf", type="primary"):
            with st.spinner("Analyzing image..."):
                resized_image = image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.BILINEAR
                )

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )

                # Match MobileNetV2 training preprocessing:
                # pixel range [0, 255] -> [-1, 1]
                image_array = (
                    image_array / 127.5
                ) - 1.0

                batch = np.expand_dims(image_array, axis=0)

                probabilities = model.predict(
                    batch,
                    verbose=0
                )[0]

                top_indices = np.argsort(probabilities)[-3:][::-1]

                best_index = int(top_indices[0])
                best_class = class_names[best_index]
                confidence = float(probabilities[best_index])

            st.subheader("Prediction")
            st.write(f"**Predicted class:** {best_class}")
            st.metric(
                "Model confidence",
                f"{confidence:.2%}"
            )

            st.caption(
                "Confidence is the model's softmax probability, "
                "not a guarantee that the diagnosis is correct."
            )

            st.subheader("Top 3 predictions")

            for index in top_indices:
                label = class_names[int(index)]
                score = float(probabilities[index])

                st.write(f"**{label}** — {score:.2%}")
                st.progress(score)

    except Exception as exc:
        st.error(f"Could not process the uploaded image: {exc}")

st.divider()

st.caption(
    "Model: MobileNetV2 | Classes: 38 | "
    "Research prototype for educational use"
)
