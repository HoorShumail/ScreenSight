"""
Streamlit demo: upload a screenshot, type an instruction, see where
ScreenSight predicts you should click.

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image, ImageDraw

from src.model import ScreenSightModel

st.set_page_config(page_title="ScreenSight", layout="wide")
st.title("ScreenSight — GUI Click-Target Grounding")
st.caption(
    "Upload a screenshot and describe what to click. Qwen2.5-VL-3B predicts "
    "the click coordinate, shown as a red marker."
)


@st.cache_resource
def load_model():
    return ScreenSightModel()


uploaded = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"])
instruction = st.text_input("Instruction", placeholder='e.g. "click the submit button"')
run = st.button("Predict click location", disabled=not (uploaded and instruction))

if run:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Loading model and running inference..."):
        model = load_model()
        point = model.predict_click(image, instruction)

    if point is None:
        st.error(
            "Couldn't parse a coordinate from the model's output. "
            "Try rephrasing the instruction, or check the console for the raw response."
        )
    else:
        x, y = point
        marked = image.copy()
        draw = ImageDraw.Draw(marked)
        r = max(6, min(image.size) // 100)
        draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=4)
        draw.line([x - r * 2, y, x + r * 2, y], fill="red", width=2)
        draw.line([x, y - r * 2, x, y + r * 2], fill="red", width=2)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(marked, use_container_width=True)
        with col2:
            st.metric("Predicted x", f"{x:.0f}px")
            st.metric("Predicted y", f"{y:.0f}px")
else:
    st.info("Upload a screenshot and enter an instruction to get started.")
