# VisualFlow

![VisualFlow Logo](images/vf_logo.webp)

[![PyPI version](https://badge.fury.io/py/visualflow.svg)](https://badge.fury.io/py/visualflow)
[![Downloads](https://static.pepy.tech/badge/visualflow)](https://pepy.tech/project/visualflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VisualFlow is a Python library for object detection that provides conversion functions between Pascal VOC, YOLO, and COCO formats. It aims to simplify the process of converting annotated datasets between these popular object detection formats.

We have started this library with the vision of providing end to end object detection, from formatting all the way to inferencing multiple types of object detection models.

Our initial version of VisualFlow allows format conversions between PASCAL VOC, COCO and YOLO. Stay tuned for future updates!

- [Installation](#installation)
- [Usage](#usage)
  - [Conversions](#conversions)
  - [Augmentations](#augmentations)
  - [Inferences](#inferences)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install VisualFlow using pip:

```bash
pip install visualflow
```
## Usage

### Conversions

VisualFlow provides three main conversion functions: to_voc(), to_yolo(), and to_coco(). Here's how you can use them:

#### Conversion to YOLO Format
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

#### Conversion to Pascal VOC Format
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

#### Conversion to COCO Format
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

### Augmentations

VisualFlow's powerful data augmentations can enhance your object detection training data. Easily apply these transformations to your dataset with just a few lines of code:

- **Cutout**: Create up to three random cutouts to encourage robustness and generalization in your models.
- **Grayscale**: Convert images to grayscale, adding diversity to your training data.
- **Brightness**: Adjust the brightness of your images, ensuring your models can handle varying lighting conditions.
- **Noise**: Introduce noise to diversify your dataset and improve model resilience.
- **Blur**: Apply blurring to images, simulating real-world scenarios and enhancing model adaptability.
- **Hue**: Adjust the hue of images, enriching color variations and augmenting the dataset.
- **Exposure**: Manipulate exposure levels to help models cope with different lighting environments.
- **Flip90**: Perform 90-degree flips for data variation and better model generalization.
- **Shear**: Apply shear transformations on bounding boxes to augment your dataset and improve model robustness.
- **Rotate**: Rotate bounding boxes by a specified angle to create diverse training examples.

Some examples are available below
```python
import VisualFlow as vf

vf.cutout(image_dir='path/to/images', 
          labels_dir='path/to/labels', # optional
          output_dir='path/to/output', 
          max_num_cutouts=3) # optional, set by default

vf.grayscale(image_dir='path/to/images', 
             labels_dir='path/to/labels', # optional
             output_dir='path/to/output')

vf.brightness(image_dir='path/to/images', 
              labels_dir='path/to/labels', # optional
              output_dir='path/to/output', 
              factor=1.5) # optional, set by default

vf.noise(image_dir='path/to/images', 
         labels_dir='path/to/labels', #optional
         output_dir='path/to/output')

vf.blur(image_dir='path/to/images', 
        labels_dir='path/to/labels', # optional
        output_dir='path/to/output')

vf.hue(image_dir='path/to/images', 
       labels_dir='path/to/labels', # optional
       output_dir='path/to/output')

vf.exposure(image_dir='path/to/images', 
            labels_dir='path/to/labels', # optional
            output_dir='path/to/output', 
            factor=2.0) # optional, set by default

vf.flip90(image_dir='path/to/images', 
          labels_dir='path/to/labels', 
          output_dir='path/to/output')

vf.shear(image_dir='path/to/images', 
         labels_dir='path/to/labels', 
         output_dir='path/to/output', 
         shear_factor= 0.2) # optional, set by default

vf.rotate(image_dir='path/to/images', 
          labels_dir='path/to/labels', 
          output_dir='path/to/output', angle=30) # optional, set by default
```

### Inferences

VisualFlow now empowers you to harness the full potential of your YOLO models, making object detection inferencing a seamless part of your workflow. With this new feature, you can confidently evaluate your trained models on your trained models.

Inference with VisualFlow is a breeze. Here's a simple example of how you can perform inferencing on your YOLO models:
```python
import VisualFlow as vf

model_path = "/home/ubuntu/walmart_corrected/runs/detect/train/weights/best.pt"
inference_dir = "/home/ubuntu/test/Walmart/images"
labels_dir = "/home/ubuntu/test/Walmart/labels"
class_txt = "/home/ubuntu/test/Walmart/classes.txt"
output_dir = "/home/ubuntu/test/output"

vf.inference(model_path=model_path,
             inference_dir=inference_dir,
             labels_dir=labels_dir,
             class_txt=class_txt,
             output_dir=output_dir)
# additional arguments: iou, conf
```
We understand that each object detection project may require different configurations. Therefore, VisualFlow's inference() function now supports two additional parameters:

- **iou**: Set to 0.7 by default, this parameter controls the minimum threshold for bounding box overlap.
- **conf**: Set to 0.5 by default, this parameter determines the minimum confidence level required for an object detection prediction.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on [GitHub](https://github.com/Ojas-Sharma/VisualFlow).

## License

[MIT](https://choosealicense.com/licenses/mit/)
