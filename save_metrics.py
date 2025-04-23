import json

def save_rep_metrics(knee_angles, torso_angles, head_angles, symmetry_scores, alignment_scores, hip_angles, toe_distances,heel_angles,inter_thigh_angles,
                     back_angles, rep_ranges, file_name):
    TEMP_OUTPUT = f'rep_metrics_{file_name}.json'
    

    rep_data = []

    for idx, (start, end) in enumerate((rep_ranges)):
        rep_metrics = {
            "rep_number": int(idx + 1),
            "start_frame": int(start),
            "end_frame": int(end),
            "knee_angle":       [float(val) for val in knee_angles[start:end+1]],
            "torso_angle":      [float(val) for val in torso_angles[start:end+1]],
            "hip_angle":        [float(val) for val in hip_angles[start:end+1]],
            "symmetry_score":   [float(val) for val in symmetry_scores[start:end+1]],
            "alignment_score":  [float(val) for val in alignment_scores[start:end+1]],
            "head_angle":       [float(val) for val in head_angles[start:end+1]],
            "toe_distance":     [float(val) for val in toe_distances[start:end+1]],
            "heel_angle":       [float(val) for val in heel_angles[start:end+1]],
            "back_angle":       [float(val) for val in back_angles[start:end+1]],
            "inter_thigh_angle": [float(val) for val in inter_thigh_angles[start:end+1]],
        }
        rep_data.append(rep_metrics)

    with open(TEMP_OUTPUT, "w") as f:
        json.dump(rep_data, f, indent=2)