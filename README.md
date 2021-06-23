
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
<br>
## Importing the Data 📚
To import the data, first download it to your computer.
Choose arbitrary test indexes (171), and train indexes (300).

     data_path = "" # The path where your data is located
     test_images_indices = list(range(301, 473))
     train_images_indices = list(range(1, 301))

Now you can use the import_data function:

     def import_data(folder_path_of_data, train_images_index, test_images_index):
         """
         description:
         This function returns numpy arrays (compatible for keras fit function and preprocessed for resnet),
         of train and test data and labels given a data folder path, train and test indices.
         parameters:
         folder_path_of_data - the directory where the data files are stored.
         train_images_index - indexes of train images.
         test_images_index - indexes of test images
         returns:
         train_images, train_labels, test_images, test_labels - lists of images and labels for train and test
         """
         from keras.applications.resnet_v2 import preprocess_input
         IMG_SIZE = (224, 224, 3)  # Hard coded
         data = scipy.io.loadmat(folder_path_of_data + "\\" + "FlowerDataLabels.mat")
         resized_images = []
         labels = []
         from keras.applications.resnet_v2 import preprocess_input
         for index in range(len(data['Data'][0])):
             #                              image                 dim 1        dim2
             resized_image = cv2.resize(data['Data'][0][index], (IMG_SIZE[0], IMG_SIZE[1]))
             resized_image = resized_image.astype(np.float32)
             resized_images.append(resized_image)
             labels.append(data['Labels'][0][index])
         resized_images = np.array(resized_images)
         labels = np.array(labels)
         train_images = []
         train_labels = []
         test_images = []
         test_labels = []
         for index in train_images_index:
             train_images.append(resized_images[index - 1])
             train_labels.append(labels[index - 1])
         for index in test_images_index:
             test_images.append(resized_images[index - 1])
             test_labels.append(labels[index - 1])
         train_images = np.array(train_images)
         train_labels = np.array(train_labels)
         test_images = np.array(test_images)
         test_labels = np.array(test_labels)
         train_images = preprocess_input(train_images)
         test_images = preprocess_input(test_images)
         return train_images, train_labels, test_images, test_labels
