import streamlit as st
from PIL import Image, UnidentifiedImageError


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
            st.write(f"幅: {width}px、高さ: {height}px")
            st.image(image, caption=uploaded_image.name)
    except (UnidentifiedImageError, OSError):
        st.error("有効なJPEGまたはPNG画像を選択してください。")
