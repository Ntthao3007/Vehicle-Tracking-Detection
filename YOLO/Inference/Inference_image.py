from torchvision import transforms
from model import YOLOv1
from PIL import Image
import argparse
import time
import os
import cv2
import torch

category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                 "truck", "train", "other person", "bus", "car", "rider", 
                 "motorcycle", "bicycle", "trailer"]
category_color = [(255,255,0),(255,0,0),(255,128,0),(0,255,255),(255,0,255),
                  (128,255,0),(0,255,128),(255,0,127),(0,255,0),(0,0,255),
                  (127,0,255),(0,128,255),(128,128,128)]

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to the modell weights")
ap.add_argument("-t", "--threshold", default=0.5, 
                help="threshold for the confidence score of the bouding box prediction")
ap.add_argument("-ss", "--split_size", default=14, 
                help="split size of the grid which is applied to the image")
ap.add_argument("-nb", "--num_boxes", default=2, 
                help="number of bounding boxes which are being predicted")
ap.add_argument("-nc", "--num_classes", default=13, 
                help="number of classes which are being predicted")
ap.add_argument("-i", "--input", required=True, help="path to your input image")
ap.add_argument("-o", "--output", required=True, help="path to your output image")
args = ap.parse_args()


def main(): 
    print("")
    print("##### YOLO OBJECT DETECTION FOR IMAGES #####")
    print("")   
    print("Loading the model")
    print("...")
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv1(int(args.split_size), int(args.num_boxes), int(args.num_classes)).to(device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Amount of YOLO parameters: " + str(num_param))
    print("...")
    print("Loading model weights")
    print("...")
   
    weights = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights["state_dict"])
    model.eval()
    
    transform = transforms.Compose([
    transforms.Resize((448,448), Image.NEAREST),
    transforms.ToTensor(),
    ])  
    
    print("Loading input image file")
    print("...")
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    ratio_x = img_width/448
    ratio_y = img_height/448

    PIL_img = Image.fromarray(img)
    img_tensor = transform(PIL_img).unsqueeze(0).to(device)

    with torch.no_grad():
        start_time = time.time()
        output = model(img_tensor) 
        curr_fps = int(1.0 / (time.time() - start_time)) 
        print("FPS for YOLO prediction: " + str(curr_fps))
        print("")
        
    corr_class = torch.argmax(output[0,:,:,10:23], dim=2)
        
    for cell_h in range(output.shape[1]):
        for cell_w in range(output.shape[2]):                
            best_box = 0
            max_conf = 0
            for box in range(int(args.num_boxes)):
                if output[0, cell_h, cell_w, box*5] > max_conf:
                    best_box = box
                    max_conf = output[0, cell_h, cell_w, box*5]
                
            if output[0, cell_h, cell_w, best_box*5] >= float(args.threshold):
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
                            
                cv2.rectangle(img, (x1,y1), (x2,y2), category_color[best_class], 1)
                labelsize = cv2.getTextSize(category_list[best_class], 
                                            cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1-20), (x1+labelsize[0][0]+45,y1), 
                              category_color[best_class], -1)
                cv2.putText(img, category_list[best_class] + " " + 
                            str(round(confidence_score.item(), 2)), (x1,y1-5), 
                            cv2.FONT_HERSHEY_DUPLEX , 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(img, str(curr_fps) + "FPS", (25, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imwrite(args.output, img) 
    
if __name__ == '__main__':
    main()

'''
example usage:
python3 ./Inference/Inference_image.py -w ./model/YOLO_bdd100k.pt -i ./data/bdd100k/images/100k/test/cabc30fc-e7726578.jpg -o ./Inference/output.png
'''