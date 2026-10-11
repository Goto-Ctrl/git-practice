import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


def calculate_channel_averages(image):
    image_array = np.asarray(image)
    return image_array.shape, image_array.mean(axis=(0, 1))


st.title("画像解析アプリ")

uploaded_image = st.file_uploader(
    "JPEGまたはPNG画像を1枚アップロードしてください",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_image is not None:
    try:
        with Image.open(uploaded_image) as image:
            width, height = image.size
            rgb_image = image.convert("RGB")
            array_shape, channel_averages = calculate_channel_averages(rgb_image)
            st.write(f"幅: {width}px、高さ: {height}px")
            st.write(f"array shape: {array_shape}")
            st.write(
                "RGB channel average: "
                f"R={channel_averages[0]:.2f}, "
                f"G={channel_averages[1]:.2f}, "
                f"B={channel_averages[2]:.2f}"
            )
            st.image(rgb_image, caption=uploaded_image.name)
    except (UnidentifiedImageError, OSError):
        st.error("有効なJPEGまたはPNG画像を選択してください。")
