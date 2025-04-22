import numpy as np
import matplotlib.pyplot as plt

def rep_sep(knee_angles, torso_angles, head_angles, symmetry_scores):

    ka = np.array(knee_angles)
    THRESH = 100.0

    down_idxs = np.where((ka[:-1] >= THRESH) & (ka[1:] < THRESH))[0] + 1
    up_idxs   = np.where((ka[:-1] <= THRESH) & (ka[1:] > THRESH))[0] + 1

    rep_ranges = []
    j = 0
    for down in down_idxs:
        while j < len(up_idxs) and up_idxs[j] <= down:
            j += 1
        if j < len(up_idxs):
            rep_ranges.append([down, up_idxs[j]])
            j += 1

    for i in range(1, len(rep_ranges)):
        rep_ranges[i][0] = rep_ranges[i-1][1]
    if len(rep_ranges) ==0:
        return rep_ranges
    rep_ranges.append([rep_ranges[-1][1], len(ka)])
    rep_ranges = [r for r in rep_ranges if r[1]- r[0]> 10]
    
    plt.figure(figsize=(14, 6))
    plt.plot(ka, label='Knee Angle (°)', color='blue')
    plt.axhline(y=THRESH, color='r', linestyle='--', label='Depth Threshold (100°)')

    for i, (start, end) in enumerate(rep_ranges):
        plt.axvspan(start, end, color='green', alpha=0.2)
        plt.text((start + end) // 2, 60, f'Rep {i+1}', color='green', ha='center', fontsize=9)

    plt.title('Detected Squat Reps from Knee Angle')
    plt.xlabel('Frame')
    plt.ylabel('Knee Angle (°)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    metrics = {
        "Torso Angle (°)": np.array(torso_angles),
        "Head Angle (°)":  np.array(head_angles),
        "Symmetry Score":  np.array(symmetry_scores)
    }

    plt.figure(figsize=(14, 10))

    for i, (label, data) in enumerate(metrics.items()):
        plt.subplot(len(metrics), 1, i+1)
        plt.plot(data, label=label, color='blue')

        for idx, (start, end) in enumerate(rep_ranges):
            plt.axvspan(start, end, color='green', alpha=0.2)
            plt.text((start + end) // 2, np.min(data), f'Rep {idx+1}', color='green',
                    ha='center', fontsize=8, va='bottom')

        plt.title(label)
        plt.xlabel('Frame')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()
    print("Rep ranges:", rep_ranges)
    return rep_ranges