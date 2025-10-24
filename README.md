# Football Skills Analysis

It is a web-based application for analyzing football skills (Receiving, Shooting, Passing) using computer vision. It processes side and optional front/back view videos to detect biomechanical metrics like head direction, torso angle, pelvis angle, and ankle-ball distance, providing insights for performance improvement.

### Goal

Tactiq aims to assist football players and coaches in evaluating and refining skills by analyzing video footage. It uses computer vision to detect key biomechanical metrics during receiving, shooting, or passing actions, enabling data-driven performance optimization.

### Technologies Used

* **Python** : Core programming language for backend logic and video processing.
* **Streamlit** : Web framework for building the user interface and displaying results.
* **OpenCV** : For video frame processing and drawing annotations.
* **MediaPipe** : For pose estimation to track body landmarks (e.g., ankles, hips, shoulders).
* **YOLOv8** : For object detection (e.g., person, ball) in video frames.
* **Pandas** : For organizing and displaying biomechanical metrics in tables.
* **PIL (Pillow)** : For image handling and conversion.
* **NumPy** : For numerical computations, such as distance and angle calculations.
