import os
import pandas as pd
import shutil
import torch
from PIL import Image
from ultralytics import YOLO
import PIL
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt


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


## Function from : https://github.com/satojkovic/DeepLogo2/blob/main/Train_DeepLogo2_by_detr.ipynb

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

## Function from : https://github.com/satojkovic/DeepLogo2/blob/main/Train_DeepLogo2_by_detr.ipynb

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

## Function from : https://github.com/satojkovic/DeepLogo2/blob/main/Train_DeepLogo2_by_detr.ipynb

def plot_finetuned_results(pil_img, prob=None, boxes=None, save_dir=None, image_name=None, classes=None):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    preds = []
    probs = []
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          text = f'{classes[cl]}: {p[cl]:0.2f}'
          preds.append(classes[cl])
          probs.append(p[cl])
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    if save_dir is not None and image_name is not None:
      plt.savefig(os.path.join(save_dir, image_name))
      return True, preds, probs
    return False, None, None


def yolo_inference(model_path, iou=0.7, conf=0.5, inference_dir=None, output_dir=None, save=True, labels_dir=None, class_txt=None):
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

def detr_inference(model_path, iou=0.7, conf=0.5, inference_dir=None, output_dir=None, save=True, labels_dir=None, class_txt=None):
    classes = []
    if class_txt is None:
        raise ValueError("Please provide a classes.txt file with each class in a new line")
    classes = create_class_lst(class_txt)

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    main_folder = inference_dir
    output_summary_file = os.path.join(output_dir, "summary.csv")
    output_predictions_file =  os.path.join(output_dir, "predictions.csv")
    model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=len(classes))
    checkpoint = torch.load(model_path, map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                      strict=False)

    model.eval();

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
            img = transform(image).unsqueeze(0)
            outputs = model(img)
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > conf

            probas_to_keep = probas[keep]

            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
            ret, preds, probs = plot_finetuned_results(image, probas_to_keep, bboxes_scaled, output_dir, image_name, classes)
            ground_truths = []
            total_count += 1
            if (labels_dir is not None):
                label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
                if (os.path.exists(label_path)):
                    idx = get_indexes(label_path)
                    ground_truths = [classes[index] for index in idx]
                else:
                    ground_truths = []
                if ret == False and len(ground_truths) == 0:
                    total_tn += 1
                elif ret == False and len(ground_truths) != 0:
                    total_fn += 1
                elif (len(ground_truths) != 0 and (set(ground_truths) == set(preds))):
                    total_tp += 1
                else:
                    total_fp += 1
                predictions.append([image_path, ground_truths, preds, probs])
            else:
                predictions.append([image_path, preds, probs])
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