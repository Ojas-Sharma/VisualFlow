import cv2
import numpy as np
import math
import random
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import colorsys
import os
import shutil
from .utils import *
from tqdm import tqdm

def read_yolo_txt_file(file_path):
    data_list = []

    with open(file_path, 'r') as file:
        for line in file:
            numbers = line.strip().split()
            if len(numbers) != 5:
                raise ValueError(f"Invalid data format in line: {line.strip()}")
            data_list.append([float(num) for num in numbers])

    return data_list

def save_yolo_txt_file(yolo_data, output_file):
    with open(output_file, 'w') as file:
        for data in yolo_data:
            if len(data) != 5:
                raise ValueError("Each inner list should contain 5 numbers.")
            
            class_index, x_center, y_center, width, height = data
            line = f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            file.write(line)


def cutout(image_dir=None, labels_dir=None, output_dir=None, max_num_cutouts=3):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Applying Cutouts...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, ext = os.path.splitext(image_name)
                # Load the image
                image = Image.open(image_path)
                image = image.convert("RGB")
                image_width, image_height = image.size
                # Create a copy of the original image
                augmented_image = image.copy()
                cutout_size=(40, 40)

                # Apply cut-out augmentation
                draw = ImageDraw.Draw(augmented_image)
                augmented_bbox_list = []
                for _ in range(max_num_cutouts):
                    cutout_xmin = random.randint(0, image_width - int(cutout_size[0]))
                    cutout_ymin = random.randint(0, image_height - int(cutout_size[1]))
                    cutout_xmax = cutout_xmin + int(cutout_size[0])
                    cutout_ymax = cutout_ymin + int(cutout_size[1])
                    cutout_bbox = (cutout_xmin, cutout_ymin, cutout_xmax, cutout_ymax)
                    augmented_bbox_list.append(cutout_bbox)

                    # Fill the selected bounding box with black pixels
                    draw.rectangle(cutout_bbox, fill=(0, 0, 0))
                new_image_name = f"{image_name}_cutout_{max_num_cutouts}{ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                augmented_image.save(new_image_path)
            except Exception as e:
                print(f"{image_name}{ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)
    if (labels_dir):
        labels = os.listdir(labels_dir)
        output_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_labels_dir, exist_ok=True)
        for label in labels:
            label_path = os.path.join(labels_dir, label)
            label_name, ext = os.path.splitext(label)
            new_label_name = f"{label_name}_cutout_{max_num_cutouts}{ext}"
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            shutil.copy(label_path, new_label_path)

def grayscale(image_dir=None, labels_dir=None, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Grayscaling...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, ext = os.path.splitext(image_name)
                image = Image.open(image_path)
                image = image.convert("RGB")
                grayscale_image = image.convert('L')
                new_image_name = f"{image_name}_grayscale_{ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                grayscale_image.save(new_image_path)
            except Exception as e:
                print(f"{image_name}{ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)

    if (labels_dir):
        labels = os.listdir(labels_dir)
        output_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_labels_dir, exist_ok=True)
        for label in labels:
            label_path = os.path.join(labels_dir, label)
            label_name, ext = os.path.splitext(label)
            new_label_name = f"{label_name}_grayscale_{ext}"
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            shutil.copy(label_path, new_label_path)


def brightness(image_dir=None, labels_dir=None, output_dir=None, factor=1.5):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Applying Brightness...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, ext = os.path.splitext(image_name)
                image = Image.open(image_path)
                image = image.convert("RGB")
                enhancer = ImageEnhance.Brightness(image)
                brightened_image = enhancer.enhance(factor)
                new_image_name = f"{image_name}_brightened_{factor}{ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                brightened_image.save(new_image_path)
            except Exception as e:
                print(f"{image_name}{ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)

    if (labels_dir):
        labels = os.listdir(labels_dir)
        output_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_labels_dir, exist_ok=True)
        for label in labels:
            label_path = os.path.join(labels_dir, label)
            label_name, ext = os.path.splitext(label)
            new_label_name = f"{label_name}_brightened_{factor}{ext}"
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            shutil.copy(label_path, new_label_path)

def noise(image_dir=None, labels_dir=None, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Adding Noise...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, ext = os.path.splitext(image_name)
                # Load the image
                image = cv2.imread(image_path)
                # Generate random Gaussian noise
                mean = 0
                stddev = 180
                noise = np.zeros(image.shape, np.uint8)
                cv2.randn(noise, mean, stddev)
                # Add noise to image
                noisy_img = cv2.add(image, noise)
                noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
                noisy_img = Image.fromarray(noisy_img)
                new_image_name = f"{image_name}_noise{ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                noisy_img.save(new_image_path)
            except Exception as e:
                print(f"{image_name}{ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)

    if (labels_dir):
        labels = os.listdir(labels_dir)
        output_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_labels_dir, exist_ok=True)
        for label in labels:
            label_path = os.path.join(labels_dir, label)
            label_name, ext = os.path.splitext(label)
            new_label_name = f"{label_name}_noise{ext}"
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            shutil.copy(label_path, new_label_path)


def blur(image_dir=None, labels_dir=None, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Blurring...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, ext = os.path.splitext(image_name)
                image = Image.open(image_path)
                image = image.convert("RGB")
                # Apply blur
                blurred_image = image.filter(ImageFilter.BLUR)
                new_image_name = f"{image_name}_blurred{ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                blurred_image.save(new_image_path)
            except Exception as e:
                print(f"{image_name}{ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)

    if (labels_dir):
        labels = os.listdir(labels_dir)
        output_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_labels_dir, exist_ok=True)
        for label in labels:
            label_path = os.path.join(labels_dir, label)
            label_name, ext = os.path.splitext(label)
            new_label_name = f"{label_name}_blurred{ext}"
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            shutil.copy(label_path, new_label_path)

def hue(image_dir=None, labels_dir=None, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Applying Hue...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, ext = os.path.splitext(image_name)
                image = Image.open(image_path)
                image = image.convert("RGB")
                # Convert the image to the HSV color space
                hsv_image = image.convert('HSV')
                # Split the HSV image into separate channels
                h, s, v = hsv_image.split()
                # Adjust the hue channel by a certain value
                hue_shift = 60  # Adjust the hue by 30 degrees (you can change this value as desired)
                shifted_hue = h.point(lambda x: (x + hue_shift) % 256)
                # Merge the modified hue channel with the original saturation and value channels
                modified_hsv_image = Image.merge('HSV', (shifted_hue, s, v))
                # Convert the modified HSV image back to the RGB color space
                modified_rgb_image = modified_hsv_image.convert('RGB')
                new_image_name = f"{image_name}_hue{ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                modified_rgb_image.save(new_image_path)
            except Exception as e:
                print(f"{image_name}{ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)

    if (labels_dir):
        labels = os.listdir(labels_dir)
        output_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_labels_dir, exist_ok=True)
        for label in labels:
            label_path = os.path.join(labels_dir, label)
            label_name, ext = os.path.splitext(label)
            new_label_name = f"{label_name}_hue{ext}"
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            shutil.copy(label_path, new_label_path)

def exposure(image_dir=None, labels_dir=None, output_dir=None, factor=2.0):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Applying Exposure...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, ext = os.path.splitext(image_name)
                image = Image.open(image_path)
                image = image.convert("RGB")
                # Adjust the exposure
                enhancer = ImageEnhance.Contrast(image)
                exposure_adjusted_image = enhancer.enhance(factor)
                new_image_name = f"{image_name}_exposure_{factor}{ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                exposure_adjusted_image.save(new_image_path)
            except Exception as e:
                print(f"{image_name}{ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)

    if (labels_dir):
        labels = os.listdir(labels_dir)
        output_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_labels_dir, exist_ok=True)
        for label in labels:
            label_path = os.path.join(labels_dir, label)
            label_name, ext = os.path.splitext(label)
            new_label_name = f"{label_name}_exposure_{factor}{ext}"
            new_label_path = os.path.join(output_labels_dir, new_label_name)
            shutil.copy(label_path, new_label_path)

def flip90(image_dir=None, labels_dir=None, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Flipping...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, image_ext = os.path.splitext(image_name)
                label_name, label_ext = image_name, ".txt"
                label_path = os.path.join(labels_dir, label_name + label_ext)
                new_image_name = f"{image_name}_flip90{image_ext}"
                new_label_name = f"{label_name}_flip90{label_ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                new_label_path = os.path.join(output_labels_dir, new_label_name)
                image = Image.open(image_path)
                image = image.convert("RGB")
                image_width, image_height = image.size
                # Rotate the image by 90 degrees clockwise
                rotated_image = image.rotate(90, expand=True)
                rotated_image.save(new_image_path)
                bboxes = read_yolo_txt_file(label_path)
                new_bboxes = []
                for bbox in bboxes:
                    converted_bbox = yolo2pascalvoc(bbox[1], bbox[2], bbox[3], bbox[4], image_width, image_height)
                    x_min,y_min,x_max,y_max = converted_bbox
                    new_xmin = y_min
                    new_ymin = image_width-x_max
                    new_xmax = y_max
                    new_ymax = image_width-x_min
                    new_yolo_bbox = pascalvoc2yolo(new_xmin,new_ymin,new_xmax,new_ymax, image_height, image_width)
                    new_bboxes.append([int(bbox[0]), new_yolo_bbox[0], new_yolo_bbox[1], new_yolo_bbox[2], new_yolo_bbox[3]])
                try:
                    save_yolo_txt_file(new_bboxes, new_label_path)
                except ValueError as e:
                    print(e)
            except Exception as e:
                print(f"{image_name}{image_ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)

def shear(image_dir=None, labels_dir=None, output_dir=None, shear_factor = 0.2):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Shearing...") as pbar:
        for image_name in image_paths:
            try:
                shear_factor = (-1 * shear_factor, shear_factor)
                shear_factor = random.uniform(*shear_factor)
                image_path = os.path.join(image_dir, image_name)
                image_name, image_ext = os.path.splitext(image_name)
                label_name, label_ext = image_name, ".txt"
                label_path = os.path.join(labels_dir, label_name + label_ext)
                new_image_name = f"{image_name}_sheared{image_ext}"
                new_label_name = f"{label_name}_sheared{label_ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                new_label_path = os.path.join(output_labels_dir, new_label_name)
                img = cv2.imread(image_path)
                w,h = img.shape[1], img.shape[0]
                M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
                nW =  img.shape[1] + abs(shear_factor*img.shape[0])
                img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
                img = cv2.resize(img, (w,h))
                scale_factor_x = nW / w
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.save(new_image_path)
                # Rotate the image by 90 degrees clockwise
                bboxes = read_yolo_txt_file(label_path)
                converted_bboxes = []
                for bbox in bboxes:
                    converted_bbox = yolo2pascalvoc(bbox[1], bbox[2], bbox[3], bbox[4], w, h)
                    converted_bboxes.append(converted_bbox)
                converted_bboxes = np.array(converted_bboxes)
                converted_bboxes[:,[0,2]] += ((converted_bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int)
                converted_bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1]
                new_bboxes = []
                for idx, converted_bbox in enumerate(converted_bboxes):
                    new_yolo_bbox = pascalvoc2yolo(converted_bbox[0], converted_bbox[1], converted_bbox[2], converted_bbox[3], w, h)
                    new_bboxes.append([int(bboxes[idx][0]), new_yolo_bbox[0], new_yolo_bbox[1], new_yolo_bbox[2], new_yolo_bbox[3]])
                try:
                    save_yolo_txt_file(new_bboxes, new_label_path)
                except ValueError as e:
                    print(e)
            except Exception as e:
                print(f"{image_name}{image_ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)

def rotate(image_dir=None, labels_dir=None, output_dir=None, angle=30):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(image_dir)
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    with tqdm(total=len(image_paths), desc="Rotating...") as pbar:
        for image_name in image_paths:
            try:
                image_path = os.path.join(image_dir, image_name)
                image_name, image_ext = os.path.splitext(image_name)
                label_name, label_ext = image_name, ".txt"
                label_path = os.path.join(labels_dir, label_name + label_ext)
                new_image_name = f"{image_name}_rotate_{angle}.png"
                new_label_name = f"{label_name}_rotate_{angle}{label_ext}"
                new_image_path = os.path.join(output_images_dir, new_image_name)
                new_label_path = os.path.join(output_labels_dir, new_label_name)
                image = Image.open(image_path)
                image = image.convert("RGBA")
                image_width, image_height = image.size
                new_image = image.copy()
                draw = ImageDraw.Draw(new_image)
                bboxes = read_yolo_txt_file(label_path)
                converted_bboxes = []
                for bbox in bboxes:
                    converted_bbox = yolo2pascalvoc(bbox[1], bbox[2], bbox[3], bbox[4], image_width, image_height)
                    converted_bboxes.append(converted_bbox)
                new_bboxes = []
                for idx, bbox in enumerate(converted_bboxes):
                    xmin, ymin, xmax, ymax = bbox
                    roi = image.crop((xmin, ymin, xmax, ymax))
                    rotated_roi = roi.rotate(angle, resample=Image.BILINEAR, expand=True)
                    width, height = rotated_roi.size
                    cx, cy = width / 2, height / 2
                    radian_angle = math.radians(angle)
                    cos_theta = math.cos(radian_angle)
                    sin_theta = math.sin(radian_angle)
                    # new_width = width*cos_theta + height*sin_theta
                    # new_height = height*cos_theta + width*sin_theta
                    new_xmin = int(cx + (xmin - cx) * cos_theta - (ymin - cy) * sin_theta)
                    new_ymin = int(cy + (xmin - cx) * sin_theta + (ymin - cy) * cos_theta)
                    new_xmax = new_xmin + width
                    new_ymax = new_ymin + height
                    new_yolo_bbox = pascalvoc2yolo(new_xmin, new_ymin, new_xmax, new_ymax, image_width, image_height)
                    new_bboxes.append([int(bboxes[idx][0]), new_yolo_bbox[0], new_yolo_bbox[1], new_yolo_bbox[2], new_yolo_bbox[3]])
                    # Resize the rotated ROI to match the adjusted bounding box size
                    # rotated_roi = rotated_roi.resize((new_width, new_height))
                    draw.rectangle([(xmin, ymin), (xmax, ymax)], fill='black')
                    # Paste the rotated ROI back into the new image at the appropriate location
                    new_image.paste(rotated_roi, (new_xmin, new_ymin), mask=rotated_roi)
                new_image.save(new_image_path)
                try:
                    save_yolo_txt_file(new_bboxes, new_label_path)
                except ValueError as e:
                    print(e)
            except Exception as e:
                print(f"{image_name}{image_ext}: Skipped due to OSError")
                print(e)
            pbar.update(1)



# Example usage
# cutout(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output", max_num_cutouts=3)
# grayscale(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output")
# brightness(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output")
# noise(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output")
# blur(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output")
# hue(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output")
# exposure(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output")
# flip90(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output")
# shear(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output")
# rotate(image_dir="/home/ubuntu/test/train/images", labels_dir="/home/ubuntu/test/train/labels", output_dir="/home/ubuntu/test/output", angle=30)