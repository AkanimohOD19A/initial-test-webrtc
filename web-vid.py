import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.threshold1 = 100
        self.threshold2 = 200

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  ## Convert Image to Arrays
        img = cv2.cvtColor(cv2.Canny(img, self.threshold1, self.threshold2),
                           cv2.COLOR_GRAY2BGR)  ## Use the cv2 Library to convert image to **scale
        return img


col1, col2 = st.columns(2)

with col1:
    webrtc_streamer(key="example1")

with col2:
    ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
    if ctx.video_transformer:
        ctx.video_transformer.threshold1 = st.sidebar.slider("Threshold1", 0, 1000, 100)
        ctx.video_transformer.threshold2 = st.sidebar.slider("Threshold2", 0, 1000, 100)

# print(transformer_widget_1, transformer_widget_2)
