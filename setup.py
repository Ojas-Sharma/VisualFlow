from setuptools import setup, find_packages

setup(
    name='VisualFlow',
    version='0.1.3',
    author='Ojas Sharma',
    author_email='ojassharma1607@gmail.com',
    description='A Python library for object detection format conversion',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Ojas-Sharma/VisualFlow',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='object-detection cvtoolkit pascal-voc yolo coco computer-vision detr image-classification detection format conversion',
    install_requires=[
        'numpy',
        'opencv-python',
        'pascal-voc-writer',
        'tqdm',
        'Pillow',
    ],
)