import torch
import os
import argparse
from dataset import DataLoader
from utils import MidtoCorner, IoU
from collections import Counter
from model import YOLOv1


ap = argparse.ArgumentParser()
ap.add_argument("-tip", "--test_img_files_path", default="./data/bdd100k/images/100k/val/", 
                help="path to the test image folder")
ap.add_argument("-ttp", "--test_target_files_path", 
                default="./data/bdd100k/labels/det_20/det_val.json", 
                help="path to json file containing the test labels")
ap.add_argument("-bs", "--batch_size", default=10, help="batch size")
ap.add_argument("-ls", "--load_size", default=1000, 
                help="amount of batches which are being loaded in one take")
ap.add_argument("-nb", "--number_boxes", default=2, 
                help="amount of bounding boxes which should be predicted")
ap.add_argument("-lc", "--lambda_coord", default=5, 
                help="hyperparameter penalizeing predicted bounding boxes in the loss function")
ap.add_argument("-ln", "--lambda_noobj", default=0.5, 
                help="hyperparameter penalizeing prediction confidence scores in the loss function")
ap.add_argument("-lmf", "--load_model_file", default="./model/YOLO_bdd100k.pt", 
                help="name of the file containing the model weights")
ap.add_argument("-ioutn", "--iou_threshold_nms", default=0.9, 
                help="threshold for the IoU between the predicted boxes and the ground-truth boxes for NMS")
ap.add_argument("-ioutm", "--iou_threshold_map", default=0.7, 
                help="threshold for the IoU between the predicted boxes and the ground-truth boxes for mAP")
ap.add_argument("-t", "--threshold", default=0.5, 
                help="threshold for the confidence score of predicted bounding boxes")
ap.add_argument("-unms", "--use_nms", default=1, 
                help="1 if non max suppression should be used, else 0")
args = ap.parse_args()

test_img_files_path = args.test_img_files_path 
test_target_files_path = args.test_target_files_path 
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign", 
                "truck", "train", "other person", "bus", "car", "rider", "motorcycle", 
                "bicycle", "trailer"] 

batch_size = int(args.batch_size) 
load_size = int(args.load_size)
split_size = 14 
num_boxes = int(args.number_boxes)
lambda_coord = float(args.lambda_coord)
lambda_noobj = float(args.lambda_noobj)
iou_threshold_nms = float(args.iou_threshold_nms)
iou_threshold_map = float(args.iou_threshold_map)
threshold = float(args.threshold)
use_nms = int(args.use_nms)
cell_dim = int(448/split_size) 
num_classes = len(category_list) 
load_model_file = args.load_model_file 


def validate(test_img_files_path, test_target_files_path, category_list, split_size, 
             batch_size, load_size, model, cell_dim, num_boxes, num_classes, device, 
             iou_threshold_nms, iou_threshold_map, threshold, use_nms):
    """
    Parameters:
        test_img_files_path (str): System path to the image directory containing 
        the test dataset images.
        test_target_files_path (str): System path to the json file containing the 
        ground-truth labels for the test dataset.
        category_list (list): A list containing all classes which should be detected.
        split_size (int): Size of the grid which is applied to the image.
        batch_size (int): Batch size.
        load_size (int): Amount of batches which are loaded in one function call.
        model (): The YOLOv1-model.
        cell_dim (int): The dimension of a single cell.
        num_boxes (int): Amount of bounding boxes which are being predicted by 
        the model.
        num_classes (int): Amount of classes which are being predicted.
        device (): Device which is used for training and testing the model.
        iou_threshold_nms (float): Threshold for the IoU between the predicted boxes 
        and the ground-truth boxes for non maximum suppression.
        iou_threshold_map (float): Threshold for the IoU between the predicted boxes 
        and the ground-truth boxes for mean average precision.
        threshold (float): Threshold for the confidence score of predicted 
        bounding boxes.
        use_nms (bool): Specifies if non max suppression should be applied to the
        bounding box predictions.
    """
    
    model.eval()
       
    print("DATA IS BEING LOADED FOR VALIDATION")
    print("")
    data = DataLoader(test_img_files_path, test_target_files_path, category_list, 
                      split_size, batch_size, load_size)
    data.LoadFiles()
    all_pred_boxes = []
    all_target_boxes = []
    
    train_idx = 0 
    while len(data.img_files) > 0:
        print("LOADING NEW VALIDATION BATCHES")            
        print("Remaining validation files:" + str(len(data.img_files)))
        print("")
        data.LoadData()
            
        for batch_idx, (img_data, target_data) in enumerate(data.data):
            img_data = img_data.to(device)
            target_data = target_data.to(device)
    
            with torch.no_grad():
                predictions = model(img_data)
            
            print('Extracting bounding boxes')
            print('Batch: {}/{} ({:.0f}%)'.format(batch_idx+1, len(data.data), 
                                                  (batch_idx+1) / len(data.data) * 100.))
            print('')
            pred_boxes = extract_boxes(predictions, num_classes, num_boxes, 
                                       cell_dim, threshold)
            target_boxes = extract_boxes(target_data, num_classes, 1, cell_dim, 
                                         threshold)
            
            for sample_idx in range(len(pred_boxes)):
                if use_nms:
                    nms_boxes = non_max_suppression(pred_boxes[sample_idx], 
                                                    iou_threshold_nms) 
                else:
                    nms_boxes = pred_boxes[sample_idx]
                
                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)
                
                for box in target_boxes[sample_idx]:
                    all_target_boxes.append([train_idx] + box)

                train_idx += 1
 
    pred = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0}
    for prediction in all_pred_boxes:
        cls_idx = prediction[1]
        pred[cls_idx] += 1
    print(pred)
    pred = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0}
    for prediction in all_target_boxes:
        cls_idx = prediction[1]
        pred[cls_idx] += 1
    print(pred)


def extract_boxes(yolo_tensor, num_classes, num_boxes, cell_dim, threshold):
    """
    Parameters:
        yolo_tensor (tensor): The tensor from which the bounding boxes need to 
        be extracted.
        num_classes (int): Amount of classes which are being predicted.
        num_boxes (int): Amount of bounding boxes which are being predicted.
        cell_dim (int): Dimension of a single cell.
        threshold (float): Threshold for the confidence score of predicted 
        bounding boxes.
        
    Returns:
        all_bboxes (list): A list where each element is a list representing one 
        image from the batch. This inner list contains other lists which represent 
        the bounding boxes within this image.
        The box lists are specified as [class_pred, conf_score, x1, y1, x2, y2]
    """
    
    all_bboxes = [] 
            
    for sample_idx in range(yolo_tensor.shape[0]):
        bboxes = [] 
        for cell_h in range(yolo_tensor.shape[1]):
            for cell_w in range(yolo_tensor.shape[2]):             
                best_box = 0
                max_conf = 0.               
                for box_idx in range(num_boxes):
                    if yolo_tensor[sample_idx, cell_h, cell_w, box_idx*5] > max_conf:
                        max_conf = yolo_tensor[sample_idx, cell_h, cell_w, box_idx*5]
                        best_box = box_idx
                conf = yolo_tensor[sample_idx, cell_h, cell_w, best_box*5]
                if conf < threshold:
                    continue
                
                best_class = 0
                max_conf = 0.
                for class_idx in range(num_classes):
                    if yolo_tensor[sample_idx, cell_h, cell_w, num_boxes*5+class_idx] > max_conf:
                        max_conf = yolo_tensor[sample_idx, cell_h, cell_w, num_boxes*5+class_idx]
                        best_class = class_idx
                        
                cords = MidtoCorner(yolo_tensor[sample_idx, cell_h, cell_w, 
                                                best_box*5+1:best_box*5+5], cell_h, cell_w, cell_dim)
                x1 = cords[0]
                y1 = cords[1]
                x2 = cords[2]
                y2 = cords[3]
                
                bboxes.append([best_class, conf, x1, y1, x2, y2])               
        all_bboxes.append(bboxes)       
    return all_bboxes


def non_max_suppression(bboxes, iou_threshold):
    """
    Parameters:
        bboxes (list): List of lists containing all bboxes with each bboxes
        specified as [class_pred, conf_score, x1, y1, x2, y2].
        iou_threshold (float): Threshold for the IOU with the ground truth bbox.
        
    Returns:
        bboxes_after_nms (list): bboxes after performing NMS given a specific 
        IoU threshold.
    """

    assert type(bboxes) == list

    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or IoU(
                chosen_box[2:],
                box[2:]
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold, box_format, 
                           num_classes):
    """
    Parameters:
        pred_boxes (list): List of lists containing all predicted bboxes with 
        each bbox specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2].
        true_boxes (list): List of lists containing all ground truth bboxes with 
        each bbox specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2].
        iou_threshold (float): Threshold for counting a predicted bbox as true positive.
        box_format (str): "midpoint" or "corners" used to specify the bbox format.
        num_classes (int): Number of classes.
        
    Returns:
        float: mAP value across all classes given a specific IoU threshold .
    """
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        print('Calculating average precision for class: ' + str(category_list[c]))
        detections = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths) 
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IoU(
                    gt[3:],detection[3:]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    print(average_precisions)
    return sum(average_precisions) / len(average_precisions)


def main():
    device = torch.device('cuda')
    model = YOLOv1(split_size, num_boxes, num_classes).to(device)
    print("###################### LOADING YOLO MODEL ######################")
    print("")
    model_weights = torch.load(load_model_file)
    model.load_state_dict(model_weights["state_dict"])
    print("###################### STARTING VALIDATION ######################")
    print("")
    validate(test_img_files_path, test_target_files_path, category_list, split_size,
             batch_size, load_size, model, cell_dim, num_boxes, num_classes, device,
             iou_threshold_nms, iou_threshold_map, threshold, use_nms)


if __name__ == "__main__":
    main()