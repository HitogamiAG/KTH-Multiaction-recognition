import streamlit as st

import tempfile

from inference_scripts.video_processing import process_video
from inference_scripts.inference import predict
from models.model import LSTMModel
from local_utils import load_model

model = LSTMModel(6, 30, 1, None)
model = load_model(model, 'checkpoints/lstm_model_100e_92acc.pt')

st.title('Multiaction recognition algorithm')

video_data = st.file_uploader(
    "**Upload video from KTH Action dataset:**", type='AVI')

col1, col2 = st.columns(2)

with col1:
    st.header('Mediapipe params')
    min_detection_confidence = st.slider(
        '**Min detection confidence:**', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    min_tracking_confidence = st.slider(
        '**Min tracking confidence:**', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    model_complexity = st.slider(
        '**Model complexity**', min_value=0, max_value=2, value=1, step=1)

with col2:
    st.header('LSTM Model params')
    seq_length = st.slider('**Sequence length**',
                           min_value=5, max_value=30, value=15, step=1)
    step = st.slider('**Step**', min_value=1, max_value=10, value=5, step=1)
    frames_diff = st.slider('**Frame step**', min_value=1,
                            max_value=10, value=5, step=1)

if st.button('Predict'):
    if video_data is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_data.read())
        video, dataframe = process_video(tfile.name, min_detection_confidence,
                                         min_tracking_confidence, model_complexity, frames_diff)

        predicted_class = predict(model, dataframe, seq_length, step)

        st.subheader(f'Predicted class: :red[**{predicted_class}**]')

        width = 50
        width = max(width, 0.01)
        side = max((100 - width) / 2, 0.01)

        _, container, _ = st.columns([side, width, side])
        container.video(data=video)

    else:
        st.subheader(':red[**Upload video!**]')
