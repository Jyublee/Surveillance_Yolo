import torch
import numpy as np
import cv2
from time import time
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = YOLO("yolov8n.pt")
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.person_detected = False
        self.recording = False
        self.clip_count = 0
        self.out = None
        self.no_person_frames = 0
        self.max_no_person_frames = 30  # Adjust this value as needed
        
        # Create directory to save clips
        self.clip_dir = "detected_clips"
        os.makedirs(self.clip_dir, exist_ok=True)

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def add_timestamp(self, frame):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, self.frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)
            
            if 0 in class_ids:
                self.person_detected = True
                self.no_person_frames = 0
            else:
                self.person_detected = False
                self.no_person_frames += 1
            
            if self.person_detected:
                if not self.recording:
                    self.clip_count += 1
                    filename = os.path.join(self.clip_dir, f"clip_{self.clip_count}.mp4")
                    self.out = cv2.VideoWriter(filename, fourcc, 30.0, (self.frame_width, self.frame_height))
                    self.recording = True
            else:
                if self.recording and self.no_person_frames > self.max_no_person_frames:
                    self.out.release()
                    self.recording = False
            
            if self.recording:
                im0 = self.add_timestamp(im0)
                self.out.write(im0)
            
            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()