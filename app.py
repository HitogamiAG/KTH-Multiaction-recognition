import streamlit as st

import tempfile

from inference_scripts.video_processing import process_video
from inference_scripts.inference import predict
import onnxruntime as ort
from local_utils import load_model

# Create session for model inference
ort_session = ort.InferenceSession("checkpoints/lstm_8_15_30.onnx")
seq_length = 15

# Upload video
st.title('Multiaction recognition algorithm')

video_data = st.file_uploader(
    "**Upload video from KTH Action dataset:**", type='AVI')

col1, col2 = st.columns(2)

# Set parameters for mediapipe
with col1:
    st.header('Mediapipe params')
    min_detection_confidence = st.slider(
        '**Min detection confidence:**', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    min_tracking_confidence = st.slider(
        '**Min tracking confidence:**', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    model_complexity = st.slider(
        '**Model complexity**', min_value=0, max_value=2, value=1, step=1)

# Set parameters for LSTM Model
with col2:
    st.header('LSTM Model params')
    step = st.slider('**Step**', min_value=1, max_value=10, value=5, step=1)
    frames_diff = st.slider('**Frame step**', min_value=1,
                            max_value=10, value=5, step=1)

# Recognize action and visualize result and processed video
if st.button('Predict'):
    if video_data is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_data.read())
        video, dataframe = process_video(tfile.name, min_detection_confidence,
                                         min_tracking_confidence, model_complexity, frames_diff)

        predicted_class = predict(ort_session, dataframe, seq_length, step)

        st.subheader(f'Predicted class: :red[**{predicted_class}**]')

        width = 50
        width = max(width, 0.01)
        side = max((100 - width) / 2, 0.01)

        _, container, _ = st.columns([side, width, side])
        container.video(data=video)

    else:
        st.subheader(':red[**Upload video!**]')
