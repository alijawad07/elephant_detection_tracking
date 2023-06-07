# Elephant Detection & Tracking using Yolov8 and DeepSort

Elephant Detection & Tracking with YOLOv8 and DeepSort is a computer vision project that aims to detect and track elephants using the YOLOv8 object detection model and DeepSort tracker. This project provides a real-time detection and tracking solution by analyzing video streams.

## Features

- Utilizes the YOLOv8 object detection model for accurate elephant detection
- Real-time detection and tracking using DeepSort
- Suitable for various scenarios including Protecting Elephants, Stopping Poaching, Enhancing Wildlife Tourism, Managing Elephant Habitats, Planning Wildlife-Friendly Areas
- Built with efficiency and ease-of-use in mind

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- deep-sort-realtime

## Getting Started

1. Clone the repository:

```
https://github.com/alijawad07/elephant_detection_tracking
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Update the configuration file with the appropriate paths and parameters.

4. Run the elephant_tracker script:
```
python3 elephant_tracker.py --source --output --weights
```
- --source => Path to directory containing video

- --output => Path to save the detection results

- --weights => Path to yolov8 weights file


## Acknowledgments

- Thanks to Roboflow for providing the comprehensive fall detection dataset used in training the YOLOv8 model.
- Special appreciation to Ultralytics for developing the YOLOv8 model and its integration with the project.

## References

- [YOLOv8](https://github.com/ultralytics/yolov5)
- [Roboflow](https://roboflow.com/)
