import os
import pandas as pd
import shutil
import torch
from PIL import Image
from ultralytics import YOLO
import PIL
import numpy as np


def find_value(value_lst, my_lst):
    for item in my_lst:
        if item in value_lst:
            return True
    return False

def get_indexes(path):
    class_indexes = []
    with open(path, 'r') as file:
        for line in file:
            class_index = int(line.strip().split()[0])
            class_indexes.append(class_index)
    return class_indexes

def create_class_lst(file_path):
    classes = []
    with open(file_path, 'r') as file:
        for line in file:
            # Append the entire line as a single element in the list
            classes.append(line.strip())
    return classes


def inference(model_path, iou=0.7, conf=0.5, inference_dir=None, output_dir=None, save=True, labels_dir=None, class_txt=None):
    classes = []
    if (labels_dir is not None):
        classes = create_class_lst(class_txt)
    model = YOLO(model_path)
    model.iou = iou
    model.conf = conf

    main_folder = inference_dir
    output_summary_file = os.path.join(output_dir, "summary.csv")
    output_predictions_file =  os.path.join(output_dir, "predictions.csv")

    
    img_paths = []
    summary = []
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    total_count = 0
    predictions = []
    for image_name in os.listdir(inference_dir):
        image_path = os.path.join(inference_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
            results = model.predict(source=image)
            result = results[0]
            master_list = []
            classes_list = []
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                cords.append(conf)
                cords.append(class_id)
                master_list.append(cords)
                classes_list.append(class_id)
            results_list = pd.DataFrame.from_records(master_list, columns=['xmin','ymin','xmax','ymax','confidence','name'])
            results_list = results_list[results_list['confidence'] >= conf]
            results_list.reset_index(drop=True, inplace=True)
            if(save):
                print("Saving results")
                img = Image.fromarray(result.plot()[:,:,::-1])
                os.makedirs(os.path.join(output_dir, "preds"), exist_ok=True)
                save_path = os.path.join(output_dir, "preds", image_name)
                img.save(save_path)
            ground_truths = []
            total_count += 1
            if (labels_dir is not None):
                label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
                if (os.path.exists(label_path)):
                    idx = get_indexes(label_path)
                    ground_truths = [classes[index] for index in idx]
                else:
                    ground_truths = []
                if results_list.empty and len(ground_truths) == 0:
                    total_tn += 1
                elif results_list.empty and len(ground_truths) != 0:
                    total_fn += 1
                elif (len(ground_truths) != 0 and (set(ground_truths) == set(results_list['name'].tolist()))):
                    total_tp += 1
                else:
                    total_fp += 1
                predictions.append([image_path, ground_truths, results_list['name'].tolist(), results_list['confidence'].tolist()])
            else:
                predictions.append([image_path, results_list['name'].tolist(), results_list['confidence'].tolist()])
        except (OSError, PIL.UnidentifiedImageError):
            continue
    try:
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        print(f"Accuracy: {accuracy}")
    except ZeroDivisionError:
        accuracy = np.nan
    try:
        precision = total_tp / (total_tp + total_fp)
        print(f"Precision: {precision}")
    except ZeroDivisionError:
        precision = np.nan
    try:
        recall = total_tp / (total_tp + total_fn)
        print(f"Recall: {recall}")
    except ZeroDivisionError:
        recall = np.nan
    try:
        f_one = 2 * ((precision * recall) / (precision + recall)) 
        print(f"F1: {f_one}")
    except ZeroDivisionError:
        f_one = np.nan
    summary.append(["Overall", total_count, total_tp, total_fp, total_tn, total_fn, accuracy, precision, recall, f_one])
    summary_df = pd.DataFrame(summary, columns=["Name", "Images Count", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"])
    summary_df.to_csv(output_summary_file, index=False)
    if (labels_dir is not None):
        prediction_df = pd.DataFrame(predictions, columns=["Image Path", "Ground Truths", "Predictions", "Confidence"])
        prediction_df.to_csv(output_predictions_file, index=False)
    else:
        prediction_df = pd.DataFrame(predictions, columns=["Image Path", "Predictions", "Confidence"])
        prediction_df.to_csv(output_predictions_file, index=False)

if __name__ == "__main__":
    main()