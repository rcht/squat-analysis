import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
import os
import get_angles
import save_metrics
import json_analyzer
import seperate_reps
import json
with open('squat_feedback.json', 'r') as f:
    feedback = json.load(f)

st.set_page_config(layout="wide")
# Streamlit UI
st.title("Squat Form Analysis")

uploaded_file = st.file_uploader("Upload your squat video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read() 
    
# Save video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_bytes)
        temp_file.close()

        # Create two columns: left for video, right for analysis
        col1, col2 = st.columns([1, 2])  # 1:2 width ratio

        with col1:
            st.video(temp_file.name, format="video/mp4", start_time=0)

        with col2:
            angles = get_angles.main_logic(temp_file.name)
            rep_range = seperate_reps.rep_sep(
                angles["knee_angles"],
                angles["torso_angles"],
                angles["head_angles"],
                angles["symmetry_scores"]
            )

            st.markdown("### Rep Count")
            st.write(len(rep_range))

            save_metrics.save_rep_metrics(
                angles["knee_angles"],
                angles["torso_angles"],
                angles["head_angles"],
                angles["symmetry_scores"],
                angles["alignment_scores"],
                angles["hip_angles"],
                angles["toe_distances"],
                angles["heel_angles"],
                angles["back_angles"],
                rep_range,
                os.path.splitext(os.path.basename(temp_file.name))[0],
            )

            json_path = f'rep_metrics_{os.path.splitext(os.path.basename(temp_file.name))[0]}.json'
            results = json_analyzer.analyze_json(json_path)

            st.markdown("### Analysis Results")
            for rep in results:
                with st.container():
                    st.markdown(f"### Rep {rep['rep_number']}")
                    st.write("**Detected Labels:**")
                    for label in rep["labels"]:
                        with st.expander(f"🔍 {feedback[label]['text']}"):
                            explanation = feedback.get(label, "No explanation available.")
                            if label == "good":
                                st.success(explanation['description'])
                            else:
                                st.error(explanation['description'])
        
