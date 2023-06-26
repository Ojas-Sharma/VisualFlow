import VisualFlow as vf
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

vf.to_coco(in_format="voc", images="/home/ubuntu/test/images", annotations="/home/ubuntu/test/xmls/", output_file_path="/home/ubuntu/test/testing_voc.json", class_file="/home/ubuntu/test/classes.txt")
vf.to_yolo(in_format="voc", images="/home/ubuntu/3k_dump/images", annotations="/home/ubuntu/3k_dump/xmls/", out_dir="/home/ubuntu/3k_dump/")
vf.to_coco(in_format="yolo", images="/home/ubuntu/test/images", annotations="/home/ubuntu/test/labels/", output_file_path="/home/ubuntu/test/testing.json", class_file="/home/ubuntu/test/classes.txt")
vf.to_yolo(in_format="coco", images="/home/ubuntu/test/images", out_dir="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")


# Empty image folder

## TESTING
# to_voc

# No input type -- tested
try:
    vf.to_voc(images="/home/ubuntu/test/images", out_dir="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# no Images folder -- tested
try:
    vf.to_voc(in_format="coco", out_dir="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))
# No output folder -- tested

try:
    vf.to_voc(in_format="coco", images="/home/ubuntu/test/images", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))
# No json file given -- tested

try:
    vf.to_voc(in_format="coco", images="/home/ubuntu/test/images", out_dir="/home/ubuntu/test/")
except ValueError as e:
    print("Test passed: {}".format(e))

# input type something else -- tested
try:
    vf.to_voc(in_format="something", images="/home/ubuntu/test/images", out_dir="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# json file given when going from yolo -- tested
try:
    vf.to_voc(in_format="yolo", images="/home/ubuntu/test/images", out_dir="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# incorrect parameter name -- tested
try:
    vf.to_voc(input="coco", image_foldr="/home/ubuntu/test/images", outut_folder="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except TypeError as e:
    print("Test passed: {}".format(e))

# relative paths given -- tested
try:
    vf.to_voc(in_format="coco", images="../../test/images", out_dir="../../test/", json_file="../../test/testing.json")
    print("Test passed")
except ValueError as e:
    print("Test failed: {}".format(e))



## TESTING
# to_coco
# No input type
try:
    vf.to_coco(images="/home/ubuntu/test/images", output_file_path="/home/ubuntu/test/testing_coco.json", class_file="/home/ubuntu/test/classes.txt")
except ValueError as e:
    print("Test passed: {}".format(e))

# no Images folder -- tested
try:
    vf.to_coco(in_format="voc", output_file_path="/home/ubuntu/test/testing_coco.json", class_file="/home/ubuntu/test/classes.txt")
except ValueError as e:
    print("Test passed: {}".format(e))

# No output file path -- tested
try:
    vf.to_coco(in_format="yolo", images="/home/ubuntu/test/images", annotations = "../../test/labels_test/", class_file="/home/ubuntu/test/classes.txt")
except ValueError as e:
    print("Test passed: {}".format(e))

# No class file with yolo input type -- tested
try:
    vf.to_coco(in_format="yolo", images="/home/ubuntu/test/images", annotations = "../../test/labels_test/", output_file_path="/home/ubuntu/test/testing_coco.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# input type something else -- tested
try:
    vf.to_coco(in_format="something", images="/home/ubuntu/test/images", annotations = "../../test/labels_test/", output_file_path="/home/ubuntu/test/testing_coco.json", class_file="/home/ubuntu/test/classes.txt")
except ValueError as e:
    print("Test passed: {}".format(e))

# relative paths given -- tested
try:
    vf.to_coco(in_format="voc", images="../../test/images", annotations = "../../test/xmls/", output_file_path="../../test/testing_coco.json", class_file="../../test/classes.txt")
    print("Test passed")
except ValueError as e:
    print("Test failed: {}".format(e))

# classes.txt path given but does not exist -- tested
try:
    vf.to_coco(in_format="yolo", images="/home/ubuntu/test/images", annotations = "../../test/labels_test/", output_file_path="/home/ubuntu/test/testing_coco.json", class_file="../../test/classes.txt")
except FileNotFoundError as e:
    print("Test passed: {}".format(e))


## TESTING

# to_yolo

# No input type
try:
    vf.to_yolo(images="/home/ubuntu/test/images", out_dir="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# no Images folder
try:
    vf.to_yolo(in_format="coco", out_dir="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# No output folder
try:
    vf.to_yolo(in_format="coco", images="/home/ubuntu/test/images", json_file="/home/ubuntu/test/testing.json")
except ValueError as e:
    print("Test passed: {}".format(e))

# No json file given
try:
    vf.to_yolo(in_format="coco", images="/home/ubuntu/test/images", out_dir="/home/ubuntu/test/")
except ValueError as e:
    print("Test passed: {}".format(e))

# input type something else
try:
    vf.to_yolo(in_format="something", images="/home/ubuntu/test/images", out_dir="/home/ubuntu/test/", json_file="/home/ubuntu/test/testing.json")
except TypeError as e:
    print("Test passed: {}".format(e))

# relative paths given
try:
    vf.to_yolo(in_format="coco", images="../../test/images", out_dir="../../test/", json_file="../../test/testing.json")
    print("Test passed")
except ValueError as e:
    print("Test failed: {}".format(e))

