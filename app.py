import streamlit as st
from PIL import Image, UnidentifiedImageError

from analysis import calculate_channel_averages, calculate_channel_histograms


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
        st.subheader("RGB histogram")
        st.line_chart(
            calculate_channel_histograms(rgb_image),
            color=["#FF0000", "#00AA00", "#0000FF"],
            x_label="画素値",
            y_label="画素数",
        )
        st.image(rgb_image, caption=uploaded_image.name)
    except (UnidentifiedImageError, OSError):
        st.error("有効なJPEGまたはPNG画像を選択してください。")
