import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
import os

# Initialize mediapipe and necessary utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def minvisibility(objs):
    return min([i.visibility for i in objs])

def obj_to_coord(obj):
    return [obj.x, obj.y, obj.z]

def vector_minus(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return np.arccos(np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)) * (180 / np.pi)

class EMADictSmoothing:
    def __init__(self, window_size=5, alpha=0.3):
        self.window_size = window_size
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.buffer = []

    def __call__(self, data_dict):
        self.buffer.append(data_dict.copy())
        
        while len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        smoothed = {}
        for key in data_dict.keys():
            smoothed[key] = 0.0
            for i, entry in enumerate(reversed(self.buffer)):
                weight = self.alpha * (1 - self.alpha)**i
                smoothed[key] += entry.get(key, 0.0) * weight
        
        return smoothed

def analyze_video(video_file):
    cap = cv2.VideoCapture(video_file)
    
    # Video writer to save output with overlays
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_file.close()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(temp_output_file.name, fourcc, 30, (640, 960))

    ema_smoother = EMADictSmoothing(window_size=5, alpha=0.3)

    knee_angles = []
    torso_angles = []
    hip_angles = []
    symmetry_scores = []
    alignment_scores = []
    rep_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 960))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            results = pose.process(image)
        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Calculate angles and perform form analysis (same as before)
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                lh_pos = obj_to_coord(left_hip)
                rh_pos = obj_to_coord(right_hip)
                lk_pos = obj_to_coord(left_knee)
                rk_pos = obj_to_coord(right_knee)
                la_pos = obj_to_coord(left_ankle)
                ra_pos = obj_to_coord(right_ankle)
                ls_pos = obj_to_coord(left_shoulder)
                rs_pos = obj_to_coord(right_shoulder)

                l_thigh_vec = vector_minus(lk_pos, lh_pos)
                r_thigh_vec = vector_minus(rk_pos, rh_pos)
                l_shin_vec = vector_minus(la_pos, lk_pos)
                r_shin_vec = vector_minus(ra_pos, rk_pos)

                l_torso_vec = vector_minus(lh_pos, ls_pos)
                r_torso_vec = vector_minus(rh_pos, rs_pos)

                l_hip_vec = vector_minus(lh_pos, ls_pos)
                r_hip_vec = vector_minus(rh_pos, rs_pos)

                l_knee_angle = get_angle(l_thigh_vec, l_shin_vec)
                r_knee_angle = get_angle(r_thigh_vec, r_shin_vec)
                l_torso_angle = get_angle(l_torso_vec, [0, 1, 0])  
                r_torso_angle = get_angle(r_torso_vec, [0, 1, 0])
                l_hip_angle = get_angle(l_thigh_vec, l_hip_vec)
                r_hip_angle = get_angle(r_thigh_vec, r_hip_vec)

                avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
                avg_torso_angle = (l_torso_angle + r_torso_angle) / 2

                symmetry_score = abs(l_knee_angle - r_knee_angle) + abs(l_torso_angle - r_torso_angle)

                l_alignment = abs(la_pos[0] - lk_pos[0]) < 0.1  
                r_alignment = abs(ra_pos[0] - rk_pos[0]) < 0.1
                alignment_score = int(l_alignment and r_alignment)

                if minvisibility([left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle]) > 0.5:
                    knee_angles.append(avg_knee_angle)
                    torso_angles.append(avg_torso_angle)
                    hip_angles.append((l_hip_angle + r_hip_angle) / 2)
                    symmetry_scores.append(symmetry_score)
                    alignment_scores.append(alignment_score)

                    is_standing = 1 if (avg_knee_angle > 160 and avg_torso_angle < 30) else 0

                    current_pose = {
                        "squat_down": 1 if (np.mean(knee_angles[-5:]) < 100) else 0,
                        "good_form": 1 if (
                            np.mean(torso_angles[-5:]) < 50 and 
                            np.mean(knee_angles[-5:]) < 100 and  
                            alignment_score == 1 and             
                            symmetry_score < 50                 
                        ) else 0,
                        "standing": is_standing 
                    }

                    smoothed_pose = ema_smoother(current_pose)
                    if smoothed_pose["squat_down"] >= 0.5 and rep_count % 2 == 0:
                        rep_count += 1
                    elif smoothed_pose["squat_down"] < 0.5 and rep_count % 2 == 1:
                        rep_count += 1

            except Exception as e:
                print(f"Error: {e}")
                pass

            # Write the frame to the output video file
            out.write(image)

    cap.release()
    out.release()
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    axs[0].plot(knee_angles, label='Knee Angle (°)')
    axs[0].axhline(y=100, color='r', linestyle='--', label='Depth Threshold')
    axs[0].set_title('Knee Flexion Angle')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Angle (°)')
    axs[0].legend()

    axs[1].plot(torso_angles, label='Torso Angle (°)')
    axs[1].axhline(y=50, color='r', linestyle='--', label='Upright Threshold')
    axs[1].set_title('Torso Inclination')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Angle (°)')
    axs[1].legend()

    axs[2].plot(symmetry_scores, label='Symmetry Score', color='green')
    axs[2].axhline(y=10, color='g', linestyle='--', label='Symmetry Tolerance')
    axs[2].plot(alignment_scores, label='Alignment Score', color='blue')
    axs[2].axhline(y=0.5, color='b', linestyle='--', label='Alignment Tolerance')
    axs[2].set_title('Form Metrics')
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Score')
    axs[2].legend()

    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    return rep_count // 2,img_buf, temp_output_file.name

def calculate_metrics(predicted_reps, actual_reps):
    if predicted_reps == 0 and actual_reps == 0:
        return 1.0, 1.0
    elif predicted_reps == 0:
        return 0.0, 0.0

    tp = min(predicted_reps, actual_reps)

    fp = max(0, predicted_reps - actual_reps)

    fn = max(0, actual_reps - predicted_reps)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

# Streamlit UI
st.title("Squat Form Analysis")
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read()

    # Save video to temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_file.close()

        st.video(temp_file.name)
        rep_count, plot_buf, output_video_file = analyze_video(temp_file.name)
        precision,recall=calculate_metrics(rep_count, 5)
        # st.video(output_video_file)
        st.write(f"Predicted reps: {rep_count}")
        st.write(f"Actual reps: {5}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}\n") 
        st.write(f"Reps Count: {rep_count}")
        st.image(plot_buf)
        # os.remove(output_video_file)
