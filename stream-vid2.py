import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def video_frame_callback(frame):
    success, image = video_frame_callback(frame)
    img = cv2.resize(image, (720, int(720 * (9 / 16))))

    results = model.predict(img)
    result_plot = results[0].plot()
    output_img = np.squeeze(result_plot.render())

    return av.VideoFrame.from_ndarray(output_img, format="bgr24"), results


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)