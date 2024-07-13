import os
import cv2
import time
import torch
from torchvision import transforms
from PIL import Image
from model import YOLOv1
import json

class VehicleDetectionPipeline:
    def __init__(self, weights_path, split_size, num_boxes, num_classes, threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLOv1(split_size, num_boxes, num_classes).to(self.device)
        self.threshold = threshold
        self.num_boxes = num_boxes  
        self.load_model_weights(weights_path)
        self.transform = transforms.Compose([
            transforms.Resize((448, 448), Image.NEAREST),
            transforms.ToTensor(),
        ])

    def load_model_weights(self, weights_path):
        try:
            weights = torch.load(weights_path)
            self.model.load_state_dict(weights["state_dict"])
            self.model.eval()
            print("Model weights loaded successfully.")
        except FileNotFoundError:
            raise RuntimeError(f"Error: Weights file {weights_path} not found.")

    def process_video(self, input_video, output_video):
        category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                         "truck", "train", "other person", "bus", "car", "rider", 
                         "motorcycle", "bicycle", "trailer"]
        category_color = [(255,255,0),(255,0,0),(255,128,0),(0,255,255),(255,0,255),
                          (128,255,0),(0,255,128),(255,0,127),(0,255,0),(0,0,255),
                          (127,0,255),(0,128,255),(128,128,128)]
        
        print("Loading input video file...")
        vs = cv2.VideoCapture(input_video)
        
        if not vs.isOpened():
            raise RuntimeError(f"Error: Could not open video file {input_video}.")

        frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))
        
        ratio_x = frame_width / 448
        ratio_y = frame_height / 448
        
        idx = 1
        sum_fps = 0
        amount_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            grabbed, frame = vs.read()
            if not grabbed:
                break
            
            print(f"Processing frame {idx} out of {amount_frames}...")
            idx += 1
            
            img = Image.fromarray(frame)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                start_time = time.time()
                output = self.model(img_tensor)
                curr_fps = int(1.0 / (time.time() - start_time))
                sum_fps += curr_fps
                print(f"FPS for YOLO prediction: {curr_fps}")

            corr_class = torch.argmax(output[0, :, :, 10:23], dim=2)

            for cell_h in range(output.shape[1]):
                for cell_w in range(output.shape[2]):                
                    best_box = 0
                    max_conf = 0
                    for box in range(int(self.num_boxes)):
                        if output[0, cell_h, cell_w, box * 5] > max_conf:
                            best_box = box
                            max_conf = output[0, cell_h, cell_w, box * 5]
                    
                    if output[0, cell_h, cell_w, best_box * 5] >= self.threshold:
                        confidence_score = output[0, cell_h, cell_w, best_box * 5]
                        center_box = output[0, cell_h, cell_w, best_box * 5 + 1: best_box * 5 + 5]
                        best_class = corr_class[cell_h, cell_w]
                        
                        centre_x = center_box[0] * 32 + 32 * cell_w
                        centre_y = center_box[1] * 32 + 32 * cell_h
                        width = center_box[2] * 448
                        height = center_box[3] * 448
                        
                        x1 = int((centre_x - width / 2) * ratio_x)
                        y1 = int((centre_y - height / 2) * ratio_y)
                        x2 = int((centre_x + width / 2) * ratio_x)
                        y2 = int((centre_y + height / 2) * ratio_y)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), category_color[best_class], 2)
                        labelsize = cv2.getTextSize(category_list[best_class], cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - 20), (x1 + labelsize[0][0] + 45, y1), category_color[best_class], -1)
                        cv2.putText(frame, f"{category_list[best_class]} {int(confidence_score.item() * 100)}%", 
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            out.write(frame)

        print("Average FPS:", int(sum_fps / idx))
        vs.release()
        out.release()

if __name__ == "__main__":
    pipeline = VehicleDetectionPipeline(
        weights_path="./model/YOLO_bdd100k.pt",
        split_size=14,
        num_boxes=2,
        num_classes=13,
        threshold=0.5
    )
    pipeline.process_video(
        input_video="./data/traffic.mp4",
        output_video="./data/traffic_pipe.mp4"
    )
