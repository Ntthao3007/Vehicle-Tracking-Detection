import gradio as gr
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import tempfile
from model import YOLOv1

category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                 "truck", "train", "other person", "bus", "car", "rider", 
                 "motorcycle", "bicycle", "trailer"]
category_color = [(255,255,0),(255,0,0),(255,128,0),(0,255,255),(255,0,255),
                  (128,255,0),(0,255,128),(255,0,127),(0,255,0),(0,0,255),
                  (127,0,255),(0,128,255),(128,128,128)]
                  
def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv1(14, 2, 13).to(device)
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights["state_dict"])
    model.eval()
    return model, device

model, device = load_model("./model.YOLO_bdd100k.pt")

def process_video(video_path, threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((448,448), Image.NEAREST),
        transforms.ToTensor(),
    ])
    
    vs = cv2.VideoCapture(video_path)
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vs.get(cv2.CAP_PROP_FPS)
    
    ratio_x = frame_width / 448
    ratio_y = frame_height / 448
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, 
                          (frame_width, frame_height))
    
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
        
        corr_class = torch.argmax(output[0,:,:,10:23], dim=2)
        
        for cell_h in range(output.shape[1]):
            for cell_w in range(output.shape[2]):                
                best_box = 0
                max_conf = 0
                for box in range(2):
                    if output[0, cell_h, cell_w, box*5] > max_conf:
                        best_box = box
                        max_conf = output[0, cell_h, cell_w, box*5]
                
                if output[0, cell_h, cell_w, best_box*5] >= threshold:
                    confidence_score = output[0, cell_h, cell_w, best_box*5]
                    center_box = output[0, cell_h, cell_w, best_box*5+1:best_box*5+5]
                    best_class = corr_class[cell_h, cell_w]
                    
                    centre_x = center_box[0]*32 + 32*cell_w
                    centre_y = center_box[1]*32 + 32*cell_h
                    width = center_box[2] * 448
                    height = center_box[3] * 448
                    
                    x1 = int((centre_x - width/2) * ratio_x)
                    y1 = int((centre_y - height/2) * ratio_y)
                    x2 = int((centre_x + width/2) * ratio_x)
                    y2 = int((centre_y + height/2) * ratio_y)
                    
                    cv2.rectangle(frame, (x1,y1), (x2,y2), category_color[best_class], 1)
                    label = f"{category_list[best_class]} {int(confidence_score.item()*100)}%"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1-20), (x1+label_size[0][0]+45, y1), 
                                  category_color[best_class], -1)
                    cv2.putText(frame, label, (x1, y1-5), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        out.write(frame)
    
    out.release()
    vs.release()
    
    return temp_output.name

def yolo_inference(video):
    output_video = process_video(video)
    return output_video

gr_interface = gr.Interface(
    fn=yolo_inference, 
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Video(label="Output Video"),
    title="Autonomous Vehicle Detection (Video)",
    description="Upload a video for object detection."
)

if __name__ == "__main__":
    gr_interface.launch()
