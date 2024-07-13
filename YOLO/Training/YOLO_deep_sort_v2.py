from torchvision import transforms
from model import YOLOv1
from PIL import Image
import argparse
import time
import os
import cv2
import torch
from deep_sort import DeepSort
import numpy as np

category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                 "truck", "train", "other person", "bus", "car", "rider", 
                 "motorcycle", "bicycle", "trailer"]
category_color = [(255,255,0),(255,0,0),(255,128,0),(0,255,255),(255,0,255),
                  (128,255,0),(0,255,128),(255,0,127),(0,255,0),(0,0,255),
                  (127,0,255),(0,128,255),(128,128,128)]

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", default="./model/YOLO_bdd100k.pt", help="path to the model weights")
ap.add_argument("-t", "--threshold", default=0.5, help="threshold for the confidence score of the bounding box prediction")
ap.add_argument("-ss", "--split_size", default=14, help="split size of the grid which is applied to the image")
ap.add_argument("-nb", "--num_boxes", default=2, help="number of bounding boxes which are being predicted")
ap.add_argument("-nc", "--num_classes", default=13, help="number of classes which are being predicted")
ap.add_argument("-i", "--input", default="./data/traffic.mp4", help="path to your input video")
ap.add_argument("-o", "--output", default="./data/outv6.mp4", help="path to your output video")
args = ap.parse_args()

def main(): 
    print("##### YOLO OBJECT DETECTION FOR VIDEOS #####\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = YOLOv1(split_size=args.split_size, num_boxes=args.num_boxes, num_classes=args.num_classes).to(device)
    try:
        weights = torch.load(args.weights, map_location=device)
        model.load_state_dict(weights["state_dict"])
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((448, 448), Image.NEAREST),
        transforms.ToTensor(),
    ])
    
    vs = cv2.VideoCapture(args.input)
    if not vs.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

    ratio_x = frame_width / 448
    ratio_y = frame_height / 448

    idx = 1
    sum_fps = 0
    amount_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    deepsort_checkpoint = "./Training/deep_sort/deep/checkpoint/ckpt.t7"

    deepsort = DeepSort(model_path=deepsort_checkpoint, max_age=70)
    car_class_id = category_list.index("car")
    tracked_car_ids = set()  

    while True:              
        grabbed, frame = vs.read()
        if not grabbed:
            break 
        
        print(f"Processing frame {idx}/{amount_frames} ({(idx / amount_frames) * 100:.2f}%)")
        
        idx += 1
        if frame is None:
            print("Frame not read correctly. Exiting...")
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            start_time = time.time()
            output = model(img_tensor) 
            curr_fps = int(1.0 / (time.time() - start_time))
            sum_fps += curr_fps
            print(f"FPS for YOLO prediction: {curr_fps}")

        corr_class = torch.argmax(output[0, :, :, 10:23], dim=2)
        confidences = [], []

        for cell_h in range(output.shape[1]):
            for cell_w in range(output.shape[2]):                
                best_box = 0
                max_conf = 0
                for box in range(args.num_boxes):
                    if output[0, cell_h, cell_w, box * 5] > max_conf:
                        best_box = box
                        max_conf = output[0, cell_h, cell_w, box * 5]
                
                if output[0, cell_h, cell_w, best_box * 5] >= args.threshold:
                    confidence_score = output[0, cell_h, cell_w, best_box * 5]
                    center_box = output[0, cell_h, cell_w, best_box * 5 + 1:best_box * 5 + 5]
                    best_class = corr_class[cell_h, cell_w]
                    
                    centre_x = center_box[0] * 32 + 32 * cell_w
                    centre_y = center_box[1] * 32 + 32 * cell_h
                    width = center_box[2] * 448
                    height = center_box[3] * 448
                    
                    x1 = int((centre_x - width / 2) * ratio_x)
                    y1 = int((centre_y - height / 2) * ratio_y)
                    x2 = int((centre_x + width / 2) * ratio_x)
                    y2 = int((centre_y + height / 2) * ratio_y)
                    conf = float(confidence_score.item())

                    if frame is not None and isinstance(frame, np.ndarray):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), category_color[best_class], 1)
                        labelsize = cv2.getTextSize(category_list[best_class], cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - 20), (x1 + labelsize[0][0] + 45, y1), category_color[best_class], -1)
                        cv2.putText(frame, f"{category_list[best_class]} {int(confidence_score.item() * 100)}%", 
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        detections = np.array([x1, y1, x2, y2, conf])
                        confidences.append(conf)

        if len(detections) > 0:
            result_tracker = deepsort.update(np.array(detections), confidences, [], frame)
            print(result_tracker)
            for res in result_tracker:
                x1, y1, x2, y2, id = map(int, res)
                car_idx = int(id)
                if car_idx == car_class_id:
                    tracked_car_ids.add(id) 

        cv2.putText(frame, f"Unique Car Count: {len(tracked_car_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)

    print(f"Total number of unique cars detected: {len(tracked_car_ids)}")
    print(f"Average FPS: {int(sum_fps / amount_frames)}")

if __name__ == '__main__':
    main()
