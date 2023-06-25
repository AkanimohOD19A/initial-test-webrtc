import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.threshold1 = 100
        self.threshold2 = 200

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  ## Convert Image to Arrays
        img = cv2.cvtColor(cv2.Canny(img, self.threshold1, self.threshold2),
                           cv2.COLOR_GRAY2BGR)  ## Use the cv2 Library to convert image to **scale
        return img


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

# col1, col2 = st.columns(2)
#
# with col1:
#     ctv = webrtc_streamer(key="example1")
#     cap = cv2.VideoCapture(ctv)
#
#     success, frame = cap.read()
#
#     results = model(frame)
#     # Visualize the results on the frame
#     annotated_frame = results[0].plot()
#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Inference", annotated_frame)
#
#
# with col2:
#     # ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
#     # if ctx.video_transformer:
#     #     ctx.video_transformer.threshold1 = st.sidebar.slider("Threshold1", 0, 1000, 100)
#     #     ctx.video_transformer.threshold2 = st.sidebar.slider("Threshold2", 0, 1000, 100)
#     st.write("PL")
# # print(transformer_widget_1, transformer_widget_2)
#
#

# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="bgr24")
#
#     results = model(img)
#     output_img = np.squeeze(results.render())
#
#     # # Loop through the video frames
#     # while cap.isOpened():
#     #     # Read a frame from the video
#     #     success, frame = cap.read()
#     #
#     #     if success:
#     #         # Run YOLOv8 inference on the frame
#     #         results = model(frame)
#     #
#     #         # Visualize the results on the frame
#     #         annotated_frame = results[0].plot()
#     #
#     #         # Display the annotated frame
#     #         cv2.imshow("YOLOv8 Inference", annotated_frame)
#
#     return av.VideoFrame.from_ndarray(output_img, format="bgr24")
#
#
# webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

# vid_cap = cv2.VideoCapture(0)
# st_frame = st.empty()
# while (vid_cap.isOpened()):
#     success, image = vid_cap.read()
#     if success:
#         _display_detected_frames(0.5,
#                                  model,
#                                  st_frame,
#                                  image,
#                                  is_display_tracker,
#                                  tracker,
#                                  )
#     else:
#         vid_cap.release()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model(img)
    output_img = np.squeeze(results.render())

    return av.VideoFrame.from_ndarray(output_img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)