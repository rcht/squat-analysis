# Squat Technique Analysis with Keypoint Detection Models

A Computer Vision project by Samridh Gupta, Saksham Kapoor and Rachit Arora

## Dependencies

```
streamlit
mediapipe
pytorch
cv2
matplotlib
numpy
```

## Methodology

- First, we extract body keypoints using Mediapipe
- After that, we calculate relevant angles from all the keypoints
- We pass these angles into a 1D convolution classifier (multilabel classification) and it predicts the mistakes and gives feedback to the user

## Running

```
streamlit run squat_form_app.py
```

Tested on Linux (requires `\tmp` directory)

