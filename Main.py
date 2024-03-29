import torch
import numpy as np
import cv2
from time import time
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = YOLO("yolov8n.pt")
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.person_ids = {}
        self.next_person_id = 1
        self.log_delay = 5.0  # Increased delay to 5 seconds between consecutive logs
        self.last_log_time = {}  # Dictionary to store last log time for each person

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

    def assign_person_id(self, box):
        for person_id, prev_box in self.person_ids.items():
            iou = self.calculate_iou(box, prev_box)
            if iou > 0.5:
                self.person_ids[person_id] = box
                return person_id
        
        person_id = self.next_person_id
        self.person_ids[person_id] = box
        self.next_person_id += 1
        return person_id

    def calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def log_event(self, person_id, side):
        current_time = time()
        if person_id not in self.last_log_time or current_time - self.last_log_time[person_id] >= self.log_delay:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - Person {person_id} detected at {side}\n"
            with open("logs.txt", "a") as log_file:
                log_file.write(log_entry)
            self.last_log_time[person_id] = current_time

    def determine_side(self, box):
        x1, _, x2, _ = box
        frame_center = self.frame_width // 2
        person_center = (x1 + x2) // 2
        return "side A" if person_center < frame_center else "side B"

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)
            
            for box, cls in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.cls.cpu().tolist()):
                if cls == 0:  # Check if person detected
                    person_id = self.assign_person_id(box.tolist())
                    side = self.determine_side(box.tolist())
                    self.log_event(person_id, side)

            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()
