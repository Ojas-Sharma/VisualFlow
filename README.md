# VisualFlow

VisualFlow is a Python library for object detection that provides conversion functions between Pascal VOC, YOLO, and COCO formats. It aims to simplify the process of converting annotated datasets between these popular object detection formats.

We have started this library with the vision of providing end to end object detection, from formatting all the way to inferencing multiple types of object detection models.

Our initial version of VisualFlow allows format conversions between PASCAL VOC, COCO and YOLO. Stay tuned for future updates!

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
import VisualFlow as vf

vf.to_yolo(in_format='voc',
       images='path/to/images',
       annotations='path/to/annotations',
       out_dir='path/to/output')
```
For COCO to YOLO:
```python
import VisualFlow as vf

vf.to_yolo(in_format='coco',
       images='path/to/images',
       out_dir='path/to/output',
       json_file='path/to/annotations.json')
```

### Conversion to Pascal VOC Format
To convert from COCO or YOLO format to Pascal VOC format, use the to_voc() function.

For COCO to VOC:
```python
import VisualFlow as vf

vf.to_voc(in_format='coco',
       images='path/to/images',
       out_dir='path/to/output',
       json_file='path/to/annotations.json')
```
For YOLO to VOC:
```python
import VisualFlow as vf

vf.to_voc(in_format='yolo',
       images='path/to/images',
       annotations='path/to/annotations',
       class_file='path/to/classes.txt',
       out_dir='path/to/output')
```

### Conversion to COCO Format
To convert from PASCAL VOC or YOLO format to COCO format, use the to_coco() function.

For VOC to COCO:
```python
import VisualFlow as vf

vf.to_coco(in_format='voc',
       images='path/to/images',
       annotations='path/to/annotations',
       class_file='path/to/classes.txt',
       output_file_path='path/to/output.json')
```
For YOLO to COCO:
```python
import VisualFlow as vf

vf.to_coco(in_format='yolo',
       images='path/to/images',
       annotations='path/to/annotations',
       class_file='path/to/classes.txt',
       output_file_path='path/to/output.json')
```

Make sure to replace 'path/to/images', 'path/to/annotations', 'path/to/classes.txt', and 'path/to/output' with the actual paths to your dataset files and folders.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on [GitHub](https://github.com/Ojas-Sharma/VisualFlow).

## License

[MIT](https://choosealicense.com/licenses/mit/)