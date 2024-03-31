# Surveillance YOLO

This Python project utilizes the YOLOv8 object detection model to detect people entering and leaving a camera's field of view. It logs the events and determines whether a person is on side A or side B of the frame.

## Prerequisites
Ensure the following are installed on your system:
- Python 3.x
- PyTorch
- NumPy
- OpenCV
- Ultralytics YOLO

## Installation
Clone the repository
```
git clone https://github.com/Jyublee/Surveillance_Yolo.git
```

Install the Prerequisites
```
pip install -r requirements.txt
```

## Usage 
The program will start capturing video from the default camera (index 0) and perform person detection and logging.
The program will display the video feed with bounding boxes around detected people and their assigned IDs.
The program will log events in the `logs.txt` file whenever a person enters or leaves the frame, along with the timestamp and the side (A or B) they entered or left from.
Press 'Esc' to stop the program.

## Configuration
- `capture_index`: The index of the camera to capture video from (default is 0).
- `log_delay`: The delay in seconds between consecutive log entries for the same person (default is 2.0 seconds).
- `presence_threshold`: The number of frames a person needs to be present to be considered as entered (default is 10 frames).
- `absence_threshold`: The number of frames a person needs to be absent to be considered as left (default is 5 frames).

## TODO
- [ ] Modify the Delay to give accurate logs
- [x] Add Function to Save the footage as photos or videos when detection occurs
- [ ] Add potential Frontend and UI

## License
This project is licensed under the MIT License.
