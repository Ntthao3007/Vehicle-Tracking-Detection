import cv2
import numpy as np
import torch
from deep_sort import DeepSort 
from torchvision import transforms
from PIL import Image
from model import YOLOv1  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_yolo_model(model_path):
    model = YOLOv1(split_size=14, num_boxes=2, num_classes=13).to(device)
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights["state_dict"])
    model.eval()
    return model

def process_video(video_path, yolo_model, deepsort, output_path, threshold=0.5):
    category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                     "truck", "train", "other person", "bus", "car", "rider", 
                     "motorcycle", "bicycle", "trailer"]
    
    car_class_id = category_list.index("car")  
    tracked_car_ids = set()  
    
    transform = transforms.Compose([
        transforms.Resize((448, 448), Image.NEAREST),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, img = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = yolo_model(img_tensor)

        detections = np.empty((0, 5))
        confidences = []
        for prediction in predictions:
            for i in range(yolo_model.split_size):
                for j in range(yolo_model.split_size):
                    cell = prediction[i][j]
                    for b in range(yolo_model.num_boxes):
                        conf = cell[4 + b*5]
                        if conf > threshold:
                            x_center, y_center, width, height = cell[b*5:b*5+4]
                            x_center = (x_center + j) / yolo_model.split_size
                            y_center = (y_center + i) / yolo_model.split_size
                            width = width / yolo_model.split_size
                            height = height / yolo_model.split_size

                            x1 = int((x_center - width / 2) * img.shape[1])
                            y1 = int((y_center - height / 2) * img.shape[0])
                            x2 = int((x_center + width / 2) * img.shape[1])
                            y2 = int((y_center + height / 2) * img.shape[0])
                            cls = torch.argmax(cell[10:])
                            conf = float(conf.item())
                            
                            cv2.putText(img, f"{category_list[cls]} {conf:.2f}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            current_array = np.array([x1, y1, x2, y2, conf])
                            detections = np.vstack((detections, current_array))
                            confidences.append(conf)

        detections = detections[:, :4]
        
        if detections.size > 0:
            result_tracker = deepsort.update(detections, confidences, [], img)
            for res in result_tracker:
                x1, y1, x2, y2, id = res
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                w, h = x2 - x1, y2 - y1
                cls = int(res[5])
                
                if cls == car_class_id:
                    tracked_car_ids.add(id)

                cv2.putText(img, f"ID: {id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        cv2.putText(img, f"Car Count: {tracked_car_ids}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(img)

    cap.release()
    out.release()

    print(f"Total unique cars detected: {len(tracked_car_ids)}")

def main():
    video_path = "./data/traffic.mp4"
    yolo_model_path = "./model/YOLO_bdd100k.pt"
    deepsort_checkpoint = "./Training/deep_sort/deep/checkpoint/ckpt.t7"
    output_path = "./data/traffic_out.mp4"
    yolo_model = load_yolo_model(yolo_model_path)
    deepsort = DeepSort(model_path=deepsort_checkpoint, max_age=70)
    process_video(video_path, yolo_model, deepsort, output_path)
    print("Successfully processed the video.")

if __name__ == "__main__":
    main()
