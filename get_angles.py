import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def minvisibility(objs):
    return min([i.visibility for i in objs])

def obj_to_coord(obj):
    return np.array([obj.x, obj.y, obj.z])

def vector_minus(v1, v2):
    return v1 - v2

def get_angle(v1, v2):
    dot    = np.dot(v1, v2)
    norm1  = np.linalg.norm(v1)
    norm2  = np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0)))

class EMADictSmoothing:
    def __init__(self, window_size=5, alpha=0.3):
        self.window_size = window_size
        self.alpha       = alpha
        self.reset()

    def reset(self):
        self.buffer = []

    def __call__(self, data_dict):
        self.buffer.append(data_dict.copy())
        while len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        smoothed = {}
        for key in data_dict:
            smoothed[key] = 0.0
            for i, entry in enumerate(reversed(self.buffer)):
                weight = self.alpha * (1 - self.alpha)**i
                smoothed[key] += entry[key] * weight
        return smoothed

def main_logic(file):
    cap = cv2.VideoCapture(file)
    ema_smoother = EMADictSmoothing(window_size=5, alpha=0.3)

    # existing metrics
    knee_angles      = []
    torso_angles     = []
    hip_angles       = []
    symmetry_scores  = []
    alignment_scores = []
    rep_count        = 0

    # NEW: head angle & toe‑distance trackers
    head_angles     = []
    toe_distances   = []
    heel_angles    = []
    back_angles     = []

    with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                lm = results.pose_landmarks.landmark

                left_knee     = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_knee    = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                left_hip      = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip     = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_ankle    = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle   = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder= lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                nose_pt       = lm[mp_pose.PoseLandmark.NOSE.value]          # NEW
                left_heel       = lm[mp_pose.PoseLandmark.LEFT_HEEL.value]
                right_heel      = lm[mp_pose.PoseLandmark.RIGHT_HEEL.value]
                left_foot_index = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                right_foot_index= lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

                lhe = obj_to_coord(left_heel)
                rhe = obj_to_coord(right_heel)
                lfi = obj_to_coord(left_foot_index)
                rfi = obj_to_coord(right_foot_index)
                lk = obj_to_coord(left_knee)
                rk = obj_to_coord(right_knee)
                lh = obj_to_coord(left_hip)
                rh = obj_to_coord(right_hip)
                la = obj_to_coord(left_ankle)
                ra = obj_to_coord(right_ankle)
                ls = obj_to_coord(left_shoulder)
                rs = obj_to_coord(right_shoulder)
                no = obj_to_coord(nose_pt)                                  # NEW

                l_thigh    = vector_minus(lk, lh)
                r_thigh    = vector_minus(rk, rh)
                l_shin     = vector_minus(la, lk)
                r_shin     = vector_minus(ra, rk)
                l_torso_v  = vector_minus(lh, ls)
                r_torso_v  = vector_minus(rh, rs)
                l_hip_v    = vector_minus(lh, ls)
                r_hip_v    = vector_minus(rh, rs)
                l_heel_vec = vector_minus(lfi, lhe)
                r_heel_vec = vector_minus(rfi,rhe)
                l_back_vec   = vector_minus(ls, lh)
                r_back_vec   = vector_minus(rs, rh)
                back_vec_avg = (l_back_vec + r_back_vec) / 2
                thigh_vec_avg = (l_thigh + r_thigh) / 2
                # existing angles & scores
                l_knee_ang  = get_angle(l_thigh, l_shin)
                r_knee_ang  = get_angle(r_thigh, r_shin)
                l_torso_ang = get_angle(l_torso_v, [0,1,0])
                r_torso_ang = get_angle(r_torso_v, [0,1,0])
                l_hip_ang   = get_angle(l_thigh, l_hip_v)
                r_hip_ang   = get_angle(r_thigh, r_hip_v)
                r_heel_ang = get_angle(r_heel_vec, [0,1,0])
                l_heel_ang = get_angle(l_heel_vec, [0,1,0])
                back_angle = get_angle(back_vec_avg, thigh_vec_avg)

                avg_knee    = (l_knee_ang + r_knee_ang)/2
                avg_torso   = (l_torso_ang + r_torso_ang)/2
                sym_score   = abs(l_knee_ang - r_knee_ang) + abs(l_torso_ang - r_torso_ang)
                l_align     = abs(la[0] - lk[0]) < 0.1
                r_align     = abs(ra[0] - rk[0]) < 0.1
                align_score = int(l_align and r_align)
                heel_angle= max(l_heel_ang, r_heel_ang)
                # NEW: head angle (to vertical)
                mid_sho     = (ls + rs) / 2
                head_vec    = no - mid_sho
                head_ang    = get_angle(head_vec, [0,1,0])

                # NEW: toe‑distance = avg horizontal ankle–knee separation
                toe_dist    = (abs(la[0]-lk[0]) + abs(ra[0]-rk[0])) / 2

                if minvisibility([left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle]) > 0.7:
                    # append your old lists
                    knee_angles.append(avg_knee)
                    torso_angles.append(avg_torso)
                    hip_angles.append((l_hip_ang + r_hip_ang)/2)
                    symmetry_scores.append(sym_score)
                    alignment_scores.append(align_score)
                    back_angles.append(back_angle)
                    # append the new trackers
                    head_angles.append(head_ang)
                    toe_distances.append(toe_dist)
                    heel_angles.append(heel_angle)
                    is_standing = 1 if (avg_knee>160 and avg_torso<30) else 0

                    current_pose = {
                        "squat_down": 1 if (np.mean(knee_angles[-5:]) < 100) else 0,
                        "good_form": 1 if (
                            np.mean(torso_angles[-5:]) < 50 and 
                            np.mean(knee_angles[-5:]) < 100 and  
                            align_score == 1 and             
                            sym_score < 50                 
                        ) else 0,
                        "standing": is_standing 
                    }

                    smoothed = ema_smoother(current_pose)
                    if smoothed["squat_down"] >= 0.5 and rep_count % 2 == 0:
                        rep_count += 1
                    elif smoothed["squat_down"] < 0.5 and rep_count % 2 == 1:
                        rep_count += 1

                    cv2.putText(image, f"Reps: {rep_count//2}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if smoothed["squat_down"] >= 0.5:
                        if smoothed["good_form"] >= 0.5:
                            cv2.putText(image, "GOOD SQUAT!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            cv2.putText(image, "BAD FORM!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,   0, 255), 2)
                    else:
                        cv2.putText(image, "Standing", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            except Exception as e:
                print(f"Error: {e}")

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    # your original plots (unchanged) will still work on knee_angles, torso_angles, etc.
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(knee_angles, label='Knee Angle (°)')
    plt.axhline(y=100, color='r', linestyle='--', label='Depth Threshold')
    plt.title('Knee Flexion Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (°)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(torso_angles, label='Torso Angle (°)')
    plt.axhline(y=50, color='r', linestyle='--', label='Upright Threshold')
    plt.title('Torso Inclination')
    plt.xlabel('Frame')
    plt.ylabel('Angle (°)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(symmetry_scores, label='Symmetry Score')
    plt.plot(alignment_scores, label='Alignment Score')
    plt.title('Form Metrics')
    plt.xlabel('Frame')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    return {
        "knee_angles": knee_angles,
        "torso_angles": torso_angles,
        "hip_angles": hip_angles,
        "symmetry_scores": symmetry_scores,
        "alignment_scores": alignment_scores,
        "rep_count": rep_count//2,
        "head_angles": head_angles,  
        "toe_distances": toe_distances,  
        "heel_angles": heel_angles , 
        "back_angles": back_angles  
    }
