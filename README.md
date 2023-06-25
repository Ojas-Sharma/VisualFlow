# VisualFlow

VisualFlow is a Python library for object detection that provides conversion functions between Pascal VOC, YOLO, and COCO formats. It aims to simplify the process of converting annotated datasets between these popular object detection formats.

## Installation

You can install VisualFlow using pip:

```bash
pip install visualflow
```
## Usage

VisualFlow provides three main conversion functions: to_voc(), to_yolo(), and to_coco(). Here's how you can use them:

### Conversion to YOLO Format
To convert from PASCAL VOC or COCO format to YOLO format, use the to_yolo() function.

For VOC to YOLO:
```python
import visualflow as vf

vf.to_yolo(input_type='voc',
       image_folder='path/to/images',
       ann_folder='path/to/annotations',
       output_folder='path/to/output')
```
For COCO to YOLO:
```python
import visualflow as vf

vf.to_yolo(input_type='coco',
       image_folder='path/to/images',
       output_folder='path/to/output',
       json_file='path/to/annotations.json')
```

### Conversion to Pascal VOC Format
To convert from COCO or YOLO format to Pascal VOC format, use the to_voc() function.

For COCO to VOC:
```python
import visualflow as vf

vf.to_voc(input_type='coco',
       image_folder='path/to/images',
       output_folder='path/to/output',
       json_file='path/to/annotations.json')
```
For YOLO to VOC:
```python
import visualflow as vf

vf.to_voc(input_type='yolo',
       image_folder='path/to/images',
       ann_folder='path/to/annotations',
       class_file='path/to/classes.txt',
       output_folder='path/to/output')
```

### Conversion to COCO Format
To convert from PASCAL VOC or YOLO format to COCO format, use the to_coco() function.

For VOC to COCO:
```python
import visualflow as vf

vf.to_coco(input_type='voc',
       image_folder='path/to/images',
       ann_folder='path/to/annotations',
       class_file='path/to/classes.txt',
       output_file_path='path/to/output.json')
```
For YOLO to COCO:
```python
import visualflow as vf

vf.to_coco(input_type='yolo',
       image_folder='path/to/images',
       ann_folder='path/to/annotations',
       class_file='path/to/classes.txt',
       output_file_path='path/to/output.json')
```

Make sure to replace 'path/to/images', 'path/to/annotations', 'path/to/classes.txt', and 'path/to/output' with the actual paths to your dataset files and folders.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on [GitHub](https://github.com/Ojas-Sharma/VisualFlow).

## License

[MIT](https://choosealicense.com/licenses/mit/)