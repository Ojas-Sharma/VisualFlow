# Convert Yolo bb to Pascal_voc bb
def yolo2pascalvoc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    xmin = ((2 * x_center * image_w) - w)/2
    ymin = ((2 * y_center * image_h) - h)/2
    xmax = xmin + w
    ymax = ymin + h
    return [xmin, ymin, xmax, ymax]

def coco2pascalvoc(xmin, ymin, w, h):
    return [xmin,ymin, xmin + w, ymin + h]


def pascalvoc2yolo(xmin, ymin, xmax, ymax, image_w, image_h):
    x_center = ((xmax + xmin)/(2*image_w))
    y_center = ((ymax + ymin)/(2*image_h))
    w = (xmax - xmin)/image_w
    h = (ymax - ymin)/image_h
    return [x_center, y_center, w, h]

def yolo2coco(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    xmin = ((2 * x_center * image_w) - w)/2
    ymin = ((2 * y_center * image_h) - h)/2
    return [xmin, ymin, w, h]

def pascalvoc2coco(xmin, ymin, xmax, ymax):
    w = xmax-xmin
    h = ymax - ymin
    return [xmin,ymin, w, h]

def coco2yolo(xmin, ymin, w, h, image_w, image_h):
    x_center = ((2*xmin + w)/(2*image_w))
    y_center = ((2*ymin + h)/(2*image_h))
    return [x_center , y_center, w/image_w, h/image_h]


# image_path = os.path.join(image_dir, image_name)
#             image_name, image_ext = os.path.splitext(image_name)
#             label_name, label_ext = image_name, ".txt"
#             label_path = os.path.join(labels_dir, label_name + label_ext)
#             new_image_name = f"{image_name}_rotate_{angle}{image_ext}"
#             new_label_name = f"{label_name}_rotate_{angle}{label_ext}"
#             new_image_path = os.path.join(output_images_dir, new_image_name)
#             new_label_path = os.path.join(output_labels_dir, new_label_name)
#             image = Image.open(image_path)
#             image = image.convert("RGBA")
#             image_width, image_height = image.size
#             new_image = image.copy()
#             draw = ImageDraw.Draw(new_image)
#             bboxes = read_yolo_txt_file(label_path)
#             converted_bboxes = []
#             new_bboxes = []
#             for bbox in bboxes:
#                 radian_angle = math.radians(angle)
#                 cos_theta = math.cos(radian_angle)
#                 sin_theta = math.sin(radian_angle)
#                 new_xmin = int(cx + (xmin - cx) * c
#                 unnormalized_bbox = [bbox[0], bbox[1]*image_width, bbox[2]*image_height, bbox[3]*image_width, bbox[4]*image_height]
#                 new_width = unnormalized_bbox[3]*cos_theta + unnormalized_bbox[4]*sin_theta
#                 new_height = unnormalized_bbox[4]*cos_theta + unnormalized_bbox[3]*sin_theta
#                 new_xcenter = new_width / 2
#                 new_ycenter = new_height / 2
#                 new_bbox = [bbox[0], new_xcenter / image_width, new_ycenter / image_height, new_width / image_width, new_height / image_height]