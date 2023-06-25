import cvtoolkit as cvt

## YOLO
## -- images folder
## -- labels folder (.txt)
## -- classes.txt

## PASCAL VOC
## -- images folder
## -- xmls folder (.xml)

## COCO
## -- Images folder
## -- 1 JSON file

# to_coco(input_type="voc", image_folder="/home/ubuntu/test/images", ann_folder="/home/ubuntu/test/xmls/", output_file_path="/home/ubuntu/test/testing_voc.json", class_file="/home/ubuntu/test/classes.txt")
# to_yolo(input_type="voc", image_folder="/home/ubuntu/3k_dump/images", ann_folder="/home/ubuntu/3k_dump/xmls/", output_folder="/home/ubuntu/3k_dump/")
# to_coco(input_type="yolo", image_folder="/home/ubuntu/test/images", ann_folder="/home/ubuntu/test/labels/", output_file_path="/home/ubuntu/test/testing.json", class_file="/home/ubuntu/test/classes.txt")
# to_yolo(input_type="coco", image_folder="/home/ubuntu/test/images", output_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")


# Empty image folder

## TESTING
# to_voc

# No input type -- tested
try:
    cvt.to_voc(image_folder="/home/ubuntu/test/images", output_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# no Images folder -- tested
try:
    cvt.to_voc(input_type="coco", output_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))
# No output folder -- tested

try:
    cvt.to_voc(input_type="coco", image_folder="/home/ubuntu/test/images", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))
# No json file given -- tested

try:
    cvt.to_voc(input_type="coco", image_folder="/home/ubuntu/test/images", output_folder="/home/ubuntu/test/")
except ValueError as e:
    print("Test passed: {}".format(e))

# input type something else -- tested
try:
    cvt.to_voc(input_type="something", image_folder="/home/ubuntu/test/images", output_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# json file given when going from yolo -- tested
try:
    cvt.to_voc(input_type="yolo", image_folder="/home/ubuntu/test/images", output_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# incorrect parameter name -- tested
try:
    cvt.to_voc(input="coco", image_foldr="/home/ubuntu/test/images", outut_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except TypeError as e:
    print("Test passed: {}".format(e))

# relative paths given -- tested
try:
    cvt.to_voc(input_type="coco", image_folder="../../test/images", output_folder="../../test/", json_file="../../test/testing.json")
    print("Test passed")
except ValueError as e:
    print("Test failed: {}".format(e))



## TESTING
# to_coco
# No input type
try:
    cvt.to_coco(image_folder="/home/ubuntu/test/images", output_file_path="/home/ubuntu/test/testing_coco.json", class_file="/home/ubuntu/test/classes.txt")
except ValueError as e:
    print("Test passed: {}".format(e))

# no Images folder -- tested
try:
    cvt.to_coco(input_type="voc", output_file_path="/home/ubuntu/test/testing_coco.json", class_file="/home/ubuntu/test/classes.txt")
except ValueError as e:
    print("Test passed: {}".format(e))

# No output file path -- tested
try:
    cvt.to_coco(input_type="yolo", image_folder="/home/ubuntu/test/images", ann_folder = "../../test/labels_test/", class_file="/home/ubuntu/test/classes.txt")
except ValueError as e:
    print("Test passed: {}".format(e))

# No class file with yolo input type -- tested
try:
    cvt.to_coco(input_type="yolo", image_folder="/home/ubuntu/test/images", ann_folder = "../../test/labels_test/", output_file_path="/home/ubuntu/test/testing_coco.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# input type something else -- tested
try:
    cvt.to_coco(input_type="something", image_folder="/home/ubuntu/test/images", ann_folder = "../../test/labels_test/", output_file_path="/home/ubuntu/test/testing_coco.json", class_file="/home/ubuntu/test/classes.txt")
except ValueError as e:
    print("Test passed: {}".format(e))

# relative paths given -- tested
try:
    cvt.to_coco(input_type="voc", image_folder="../../test/images", ann_folder = "../../test/xmls/", output_file_path="../../test/testing_coco.json", class_file="../../test/classes.txt")
    print("Test passed")
except ValueError as e:
    print("Test failed: {}".format(e))

# classes.txt path given but does not exist -- tested
try:
    cvt.to_coco(input_type="yolo", image_folder="/home/ubuntu/test/images", ann_folder = "../../test/labels_test/", output_file_path="/home/ubuntu/test/testing_coco.json", class_file="../../test/classes.txt")
except FileNotFoundError as e:
    print("Test passed: {}".format(e))


## TESTING

# to_yolo

# No input type
try:
    cvt.to_yolo(image_folder="/home/ubuntu/test/images", output_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# no Images folder
try:
    cvt.to_yolo(input_type="coco", output_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# No output folder
try:
    cvt.to_yolo(input_type="coco", image_folder="/home/ubuntu/test/images", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# No json file given
try:
    cvt.to_yolo(input_type="coco", image_folder="/home/ubuntu/test/images", output_folder="/home/ubuntu/test/")
except ValueError as e:
    print("Test passed: {}".format(e))

# input type something else
try:
    cvt.to_yolo(input_type="something", image_folder="/home/ubuntu/test/images", output_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except TypeError as e:
    print("Test passed: {}".format(e))

# relative paths given
try:
    cvt.to_yolo(input_type="coco", image_folder="../../test/images", output_folder="../../test/", json_file="../../test/testing.json")
    print("Test passed")
except ValueError as e:
    print("Test failed: {}".format(e))

