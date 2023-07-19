from pascal_voc_writer import Writer
from tqdm import tqdm
import time
import os
from PIL import Image
import xml.etree.ElementTree as ET
import json

def to_voc(in_format=None, images=None, annotations=None, class_file=None, out_dir=None, json_file=None):
    if in_format is None:
        raise ValueError("Missing input argument: Please provide input type ('yolo' or 'coco')")
    if out_dir is None:
        raise ValueError("Missing argument: out_dir")
    if images is None:
        raise ValueError("Missing argument: images")
    if not os.path.isdir(os.path.join(out_dir, "xmls")):
        os.mkdir(os.path.join(out_dir, "xmls"))
    if in_format == "yolo":
        if class_file is None:
            raise ValueError("Please provide path to classes.txt")
        if annotations is None:
            raise ValueError("Missing argument: annotations")
        else:
            class_dict = {}
            with open(class_file, 'r') as file:
                for line_num, name in enumerate(file):
                    name = name.strip()
                    class_dict[line_num] = name

            image_info = []
            image_files = [filename for filename in os.listdir(images) if filename.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            with tqdm(total=len(image_files), desc="Converting...") as pbar:
                for filename in os.listdir(images):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_path = os.path.join(images, filename)
                        label_path = os.path.join(annotations, filename.rsplit(".", 1)[0] + ".txt")
                        try:
                            image_width, image_height = 0, 0
                            with Image.open(image_path) as img:
                                image_width, image_height = img.size
                            with open(label_path, 'r') as label:
                                bboxes = []
                                for line in label:
                                    line = line.strip()
                                    numbers = line.split()
                                    class_name = class_dict[int(numbers[0])]
                                    x_center = float(numbers[1])
                                    y_center = float(numbers[2])
                                    w = float(numbers[3])
                                    h = float(numbers[4])
                                    bboxes.append([x_center, y_center, w, h, class_name])
                                writer = Writer(image_path, image_width, image_height)
                                for bbox in bboxes:
                                    voc_bbox = yolo2pascalvoc(bbox[0], bbox[1], bbox[2], bbox[3], image_width, image_height)
                                    writer.addObject(bbox[4], voc_bbox[0], voc_bbox[1], voc_bbox[2], voc_bbox[3])
                                writer.save(os.path.join(output, "xmls", filename.rsplit(".", 1)[0] + ".xml"))
                        except OSError:
                            print(image_path + " skipped")
                            pass
                        pbar.update(1)
            print("Completed!")

    if in_format == "coco":
        if json_file is None:
            raise ValueError("Missing argument: json_file. Please provide path to json file")
        else:
            print("starting")
            bbox_mapping = {}
            class_mapping = {}
            coco = json.load(open(json_file, 'r'))
            image_ids = []
            for c in coco['categories']:
                class_mapping[c["id"] - 1] = c["name"]
            for ann in coco['annotations']:
                image_id = ann['image_id']
                if image_id not in bbox_mapping:
                    bbox_mapping[image_id] = []
                xmin = ann['bbox'][0]
                ymin = ann['bbox'][1]
                w = ann['bbox'][2]
                h = ann['bbox'][3]
                bbox_mapping[image_id].append([ann["category_id"] - 1, xmin, ymin, w, h])
            with tqdm(total=len(coco['images']), desc="Converting...") as pbar:
                for im in coco['images']:
                    result = []
                    bboxes = bbox_mapping[im["id"]]
                    image_path = os.path.join(images, os.path.basename(im["file_name"]))
                    writer = Writer(image_path, im["width"], im["height"])
                    for bbox in bboxes:
                        voc_bbox = coco2pascalvoc(bbox[1], bbox[2], bbox[3], bbox[4])
                        writer.addObject(class_mapping[bbox[0]], voc_bbox[0], voc_bbox[1], voc_bbox[2], voc_bbox[3])
                    xml_filename = os.path.basename(im["file_name"]).rsplit(".", 1)[0]
                    writer.save(os.path.join(out_dir, "xmls", xml_filename + ".xml"))
                    pbar.update(1)
            print("Completed!")


def to_yolo(in_format=None, images=None, annotations=None, out_dir=None, json_file=None):
    if in_format is None:
        raise ValueError("Missing input argument: Please provide input type ('voc' or 'coco')")
    if out_dir is None:
        raise ValueError("Missing argument: out_dir")
    if images is None:
        raise ValueError("Missing argument: images")
    if not os.path.isdir(os.path.join(out_dir, "labels")):
        os.mkdir(os.path.join(out_dir, "labels"))
    if in_format == "voc":
        if annotations is None:
            raise ValueError("Missing argument: annotations")
        else:
            class_lst = []
            class_path = os.path.join(out_dir, "classes.txt")
            image_files = [filename for filename in os.listdir(images) if filename.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            with tqdm(total=len(image_files), desc="Converting...") as pbar:
                for filename in os.listdir(images):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_path = os.path.join(images, filename)
                        label_path = os.path.join(annotations, filename.rsplit(".", 1)[0] + ".xml")
                        try:
                            image_width, image_height = 0, 0
                            with Image.open(image_path) as img:
                                image_width, image_height = img.size
                            result = []
                            tree = ET.parse(label_path)
                            root = tree.getroot()
                            height = int(root.find("size").find("height").text)
                            width = int(root.find("size").find("width").text)
                            for obj in root.findall('object'):
                                label = obj.find("name").text
                                if label not in class_lst:
                                    class_lst.append(label)
                                index = class_lst.index(label)
                                bbox = [float(x.text) for x in obj.find("bndbox")]
                                yolo_bbox = pascalvoc2yolo(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
                                bbox_string = " ".join([str(x) for x in yolo_bbox])
                                result.append(f"{index} {bbox_string}")

                            if result:
                                label_name = filename.rsplit(".", 1)[0] + ".txt"
                                with open(os.path.join(out_dir, "labels", label_name), "w", encoding="utf-8") as f:
                                    f.write("\n".join(result))
                        except OSError:
                            print(image_path + " skipped")
                            pass
                        pbar.update(1)
                with open(class_path, 'w', encoding='utf8') as f:
                    for item in class_lst:
                        f.write(item + '\n')
            print("Completed!")

    if in_format == "coco":
        if json_file is None:
            raise ValueError("Missing argument: json_file. Please provide path to json file")
        else:
            print("starting")
            bbox_mapping = {}
            class_mapping = {}
            coco = json.load(open(json_file, 'r'))
            image_ids = []
            for c in coco['categories']:
                class_mapping[c["id"] - 1] = c["name"]
            for ann in coco['annotations']:
                image_id = ann['image_id']
                if image_id not in bbox_mapping:
                    bbox_mapping[image_id] = []
                xmin = ann['bbox'][0]
                ymin = ann['bbox'][1]
                w = ann['bbox'][2]
                h = ann['bbox'][3]
                bbox_mapping[image_id].append([ann["category_id"] - 1, xmin, ymin, w, h])
            with tqdm(total=len(coco['images']), desc="Converting...") as pbar:
                for im in coco['images']:
                    result = []
                    bboxes = bbox_mapping[im["id"]]
                    for bbox in bboxes:
                        yolo_bbox = coco2yolo(bbox[1], bbox[2], bbox[3], bbox[4], im["width"], im["height"])
                        bbox_string = " ".join([str(x) for x in yolo_bbox])
                        result.append(f"{bbox[0]} {bbox_string}")
                    if result:
                        image_filename = os.path.basename(im["file_name"]).rsplit(".", 1)[0]

                        with open(os.path.join(out_dir, "labels", f"{image_filename}.txt"), "w", encoding="utf-8") as f:
                            f.write("\n".join(result))
                    pbar.update(1)
            print("Completed!")



def to_coco(in_format=None, images=None, annotations=None, class_file=None, output_file_path=None):
    if in_format is None:
        raise ValueError("Missing input argument: Please provide input type ('yolo' or 'voc')")
    if output_file_path is None:
        raise ValueError("Missing argument: output_file_path")
    if images is None:
        raise ValueError("Missing argument: images")
    if annotations is None:
        raise ValueError("Missing argument: annotations")
    if in_format == "yolo":
        if class_file is None:
            raise ValueError("Please provide path to classes.txt which has a list of all your classes (one class on each line)")
        coco = {"images": [{}], "categories": [], "annotations": [{}]}
        image_id = 0
        ann_id = 1
        image_info = []
        ann_info = []
        image_files = [filename for filename in os.listdir(images) if filename.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        with tqdm(total=len(image_files), desc="Converting...") as pbar:
            for filename in os.listdir(images):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_path = os.path.join(images, filename)
                    label_path = os.path.join(annotations, filename.rsplit(".", 1)[0] + ".txt")
                    try:
                        image_width, image_height = 0, 0
                        with Image.open(image_path) as img:
                            image_width, image_height = img.size
                            image_stats = {
                                            "file_name": image_path,
                                            "height": image_height,
                                            "width": image_width,
                                            "id": image_id,
                                        }
                            image_info.append(image_stats)
                        with open(label_path, 'r') as label:
                            bboxes = []
                            for line in label:
                                line = line.strip()
                                numbers = line.split()
                                class_id = int(numbers[0]) + 1
                                x_center = float(numbers[1])
                                y_center = float(numbers[2])
                                w = float(numbers[3])
                                h = float(numbers[4])
                                coco_bbox = yolo2coco(x_center, y_center, w, h,  image_width, image_height)
                                area = coco_bbox[2]*coco_bbox[3]
                                bbox_stats = {
                                            "id": ann_id,
                                            "image_id": image_id,
                                            "bbox": (coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3]),
                                            "area": area,
                                            "iscrowd": 0,
                                            "category_id": class_id,
                                        }
                                ann_info.append(bbox_stats)
                                ann_id = ann_id + 1
                        image_id = image_id + 1
                    except OSError:
                        print(image_path + " skipped")
                        pass
                    pbar.update(1)
            coco["images"] = image_info
            coco["annotations"] = ann_info
            with open(class_file, 'r') as file:
                for line_num, name in enumerate(file):
                    name = name.strip()
                    categories = {
                        "supercategory": "Null",
                        "id": line_num + 1,
                        "name": name,
                    }
                    coco["categories"].append(categories)
            with open(output_file_path, "w") as outfile:
                json.dump(coco, outfile, indent=4)
        print("Completed!")
    
    if in_format == "voc":
        if class_file is None:
            raise ValueError("Please provide path to classes.txt which has a list of all your classes (one class on each line)")
        coco = {"images": [{}], "categories": [], "annotations": [{}]}
        image_id = 0
        ann_id = 1
        image_info = []
        ann_info = []
        class_dict = {}
        with open(class_file, 'r') as file:
            for line_num, name in enumerate(file):
                name = name.strip()
                class_dict[name] = line_num + 1
        image_files = [filename for filename in os.listdir(images) if filename.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        with tqdm(total=len(image_files), desc="Converting...") as pbar:
            for filename in os.listdir(images):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    image_path = os.path.join(images, filename)
                    label_path = os.path.join(annotations, filename.rsplit(".", 1)[0] + ".xml")
                    tree = ET.parse(label_path)
                    root = tree.getroot()
                    image_height = int(root.find("size")[0].text)
                    image_width = int(root.find("size")[1].text)
                    image_channels = int(root.find("size")[2].text)
                    try:
                        image_width, image_height = 0, 0
                        with Image.open(image_path) as img:
                            image_width, image_height = img.size
                            image_stats = {
                                            "file_name": image_path,
                                            "height": image_height,
                                            "width": image_width,
                                            "id": image_id,
                                        }
                            image_info.append(image_stats)
                        for member in root.findall('object'):
                            class_name = member[0].text
                            xmin = float(member[4][0].text)
                            ymin = float(member[4][1].text)
                            xmax = float(member[4][2].text)
                            ymax = float(member[4][3].text)
                            coco_bbox = pascalvoc2coco(xmin,ymin,xmax,ymax)
                            area = coco_bbox[2]*coco_bbox[3]
                            bbox_stats = {
                                        "id": ann_id,
                                        "image_id": image_id,
                                        "bbox": (coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3]),
                                        "area": area,
                                        "iscrowd": 0,
                                        "category_id": class_dict[class_name],
                                    }
                            ann_info.append(bbox_stats)
                            ann_id = ann_id + 1
                        image_id = image_id + 1
                    except OSError:
                        print(image_path + " skipped")
                        pass
                    pbar.update(1)
            coco["images"] = image_info
            coco["annotations"] = ann_info
            with open(class_file, 'r') as file:
                for line_num, name in enumerate(file):
                    name = name.strip()
                    categories = {
                        "supercategory": "Null",
                        "id": line_num + 1,
                        "name": name,
                    }
                    coco["categories"].append(categories)
            with open(output_file_path, "w") as outfile:
                json.dump(coco, outfile, indent=4)
        print("Completed!")