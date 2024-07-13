import torch
from torchvision import transforms
from PIL import Image
import time
import cv2
import gradio as gr
from model import YOLOv1

category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", "truck", "train", 
                 "other person", "bus", "car", "rider", "motorcycle", "bicycle", "trailer"]
category_color = [(255,255,0),(255,0,0),(255,128,0),(0,255,255),(255,0,255),(128,255,0),(0,255,128),
                  (255,0,127),(0,255,0),(0,0,255),(127,0,255),(0,128,255),(128,128,128)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv1(split_size=14, num_boxes=2, num_classes=13).to(device)
weights = torch.load("./model.YOLO_bdd100k.pt", map_location=device)
model.load_state_dict(weights["state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((448, 448), Image.NEAREST),
    transforms.ToTensor(),
])

def yolo_inference(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_width, img_height = img.shape[1], img.shape[0]
    
    ratio_x, ratio_y = img_width / 448, img_height / 448
    PIL_img = Image.fromarray(img)
    img_tensor = transform(PIL_img).unsqueeze(0).to(device)

    with torch.no_grad():
        start_time = time.time()
        output = model(img_tensor) #
        curr_fps = int(1.0 / (time.time() - start_time)) 
        
    corr_class = torch.argmax(output[0, :, :, 10:23], dim=2)
        
    for cell_h in range(output.shape[1]):
        for cell_w in range(output.shape[2]):                
            best_box = 0
            max_conf = 0
            for box in range(2):
                if output[0, cell_h, cell_w, box*5] > max_conf:
                    best_box = box
                    max_conf = output[0, cell_h, cell_w, box*5]
                
            if output[0, cell_h, cell_w, best_box*5] >= 0.5:
                confidence_score = output[0, cell_h, cell_w, best_box*5]
                center_box = output[0, cell_h, cell_w, best_box*5+1:best_box*5+5]
                best_class = corr_class[cell_h, cell_w]
                    
                centre_x = center_box[0] * 32 + 32 * cell_w
                centre_y = center_box[1] * 32 + 32 * cell_h
                width = center_box[2] * 448
                height = center_box[3] * 448
                    
                x1 = int((centre_x - width / 2) * ratio_x)
                y1 = int((centre_y - height / 2) * ratio_y)
                x2 = int((centre_x + width / 2) * ratio_x)
                y2 = int((centre_y + height / 2) * ratio_y)                    
                            
                cv2.rectangle(img, (x1, y1), (x2, y2), category_color[best_class], 1)
                labelsize = cv2.getTextSize(category_list[best_class], cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1-20), (x1 + labelsize[0][0] + 45, y1), category_color[best_class], -1)
                cv2.putText(img, f"{category_list[best_class]} {round(confidence_score.item(), 2)}", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, f"{curr_fps}FPS", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

iface = gr.Interface(
    fn=yolo_inference,
    inputs=gr.Image(type="numpy", label="Input Image"),
    outputs="image",
    title="Autonomous Vehicle Detection (Image)",
    description="Upload an image for object detection."
)

iface.launch()
