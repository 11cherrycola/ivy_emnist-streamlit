import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="EMNIST Huruf A-Z", layout="centered")
st.title("✍️ Gambar Huruf Tangan - Model EMNIST")

# Load model
model = tf.keras.models.load_model("huruf_model_terbaik.h5")

# Canvas area
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="pink",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Tombol prediksi
if st.button("Prediksi Huruf"):
    if canvas_result.image_data is not None:
        # Ambil gambar dari canvas
        img = canvas_result.image_data
        img = Image.fromarray((255 - img[:, :, 0]).astype(np.uint8))  # ambil channel merah & invert

        # Auto-crop area berisi tinta
        img_np = np.array(img)
        coords = np.column_stack(np.where(img_np < 255))
        if coords.any():
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            img_np = img_np[y0:y1+1, x0:x1+1]  # crop ke area berisi tulisan
        else:
            st.warning("Gambar terlalu kosong untuk diproses.")
            st.stop()

        # Resize ke 28x28 + normalize
        img = Image.fromarray(img_np).resize((28, 28)).convert('L')
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 28, 28, 1)

        # Prediksi
        prediction = model.predict(img_array)
        label = chr(np.argmax(prediction[0]) + 65)  # convert index ke huruf A-Z

        st.success(f"Model memprediksi: **{label}**")
    else:
        st.warning("Gambar dulu ya!")

# Tombol clear
if st.button("Clear Canvas"):
    st.rerun()

st.caption("Model dilatih dengan dataset EMNIST Letters")
