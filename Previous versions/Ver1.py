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
        self.person_positions = {}
        self.person_frames = {}
        self.next_person_id = 1
        self.log_delay = 2.0  # Delay in seconds between consecutive logs
        self.presence_threshold = 10  # Number of frames to consider a person present
        self.absence_threshold = 5  # Number of frames to consider a person absent
        self.max_distance = 50  # Maximum distance between bounding boxes to consider the same person

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
        min_distance = float('inf')
        assigned_id = None

        for person_id, prev_box in self.person_positions.items():
            distance = self.calculate_distance(box, prev_box)
            if distance < min_distance and distance < self.max_distance:
                min_distance = distance
                assigned_id = person_id

        if assigned_id is None:
            assigned_id = self.next_person_id
            self.next_person_id += 1

        self.person_positions[assigned_id] = box
        return assigned_id

    def calculate_distance(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
        center2 = ((x3 + x4) // 2, (y3 + y4) // 2)

        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance

    def log_event(self, person_id, event_type, side):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - Person {person_id} {event_type} {side}\n"
        with open("logs.txt", "a") as log_file:
            log_file.write(log_entry)

    def determine_side(self, box):
        x1, _, x2, _ = box
        frame_center = self.frame_width // 2
        person_center = (x1 + x2) // 2
        return "side A" if person_center < frame_center else "side B"

    def update_person_frames(self, person_id):
        if person_id not in self.person_frames:
            self.person_frames[person_id] = 1
        else:
            self.person_frames[person_id] += 1

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        last_log_time = time()
        
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            current_person_ids = set()
            
            for box, cls in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.cls.cpu().tolist()):
                if cls == 0:  # Check if person detected
                    person_id = self.assign_person_id(box.tolist())
                    current_person_ids.add(person_id)
                    self.update_person_frames(person_id)

                    if person_id not in self.person_ids:
                        self.person_ids[person_id] = box.tolist()
                        side = self.determine_side(box.tolist())
                        if time() - last_log_time >= self.log_delay:
                            self.log_event(person_id, "came in from", side)
                            last_log_time = time()
                    else:
                        prev_box = self.person_ids[person_id]
                        distance = self.calculate_distance(box.tolist(), prev_box)
                        if distance > self.max_distance:
                            side = self.determine_side(box.tolist())
                            if time() - last_log_time >= self.log_delay:
                                self.log_event(person_id, "left to", side)
                                last_log_time = time()
                            self.person_ids[person_id] = box.tolist()

            for person_id in set(self.person_ids.keys()) - current_person_ids:
                if self.person_frames[person_id] >= self.presence_threshold:
                    if person_id not in self.person_positions:
                        continue
                    side = self.determine_side(self.person_positions[person_id])
                    if time() - last_log_time >= self.log_delay:
                        self.log_event(person_id, "left to", side)
                        last_log_time = time()
                    del self.person_positions[person_id]
                del self.person_ids[person_id]
                del self.person_frames[person_id]

            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()
