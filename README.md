
# FlowersTaskTransfer 🌷

Implementing a task transfer learning while using a pre-trained neural network (ResNet50V2).

- ### Part A - Task Transfer 🚍
     ResNet50V2 was originally used for identifying between 1000 different classes. I changed the output layer to a single neuron, so it can be used for a binary classification - decide whether the picture is flower or not. The training set is 472 images, each is labeled as 1 (flower) or 0 (not flower). After loading the net, I used an average pooling to get a 2048 neurons global layer. I activated the sigmoid function on this layer to calculate the single neuron in the output level, by which we can classify the image.

- ### Part B - Model Improvement 👌
    I used to methods to try and improve the model's accuracy. The first method is Data Augmentation - making the training set 3 times bigger than the original. The second method is changing the net architecture.


## Installation 🔗

Use the package manager pip to install the following:

```bash 
 pip install Pillow
 pip install h5py==2.10.0
```

**Requirments**
```bash
import tensorflow as tf
import cv2
import scipy.io
import numpy as np
import random
import time
```

This project uses the following Python libraries

PyPDF2 : For extracting text from PDF files.
spaCy : For passing the extracted text into an NLP pipeline.
NumPy : For fast matrix operations.
pandas : For analysing and getting insights from datasets.
matplotlib : For creating graphs and plots.
seaborn : For enhancing the style of matplotlib plots.
geopandas : For plotting maps.

Go to FlowerData-20210623.zip and click "view raw" to download the flowers' images.
