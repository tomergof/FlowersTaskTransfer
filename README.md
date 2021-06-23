
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

Go to FlowerData-20210623.zip and click "view raw" to download the flowers' images.
<br>
## Usage 🤔
To import the data, first download it to your computer.
Choose arbitrary test indexes (171), and train indexes (300).

     data_path = "" # The path where your data is located
     test_images_indices = list(range(301, 473))
     train_images_indices = list(range(1, 301))
     
The code above enables you to call main_no_imporovements AND main_final, which will activate part A or part B pipeline:

*For Part A:*

     main_no_improvements(data_path, train_images_indices, test_images_indices)
*For Part B:*

     main_final(data_path, train_images_indices, test_images_indices)
<br>
## Results & Conclusion

**Before** implementing the improvements, the basic pipeline's results where 72.67% on the test.
Below is the confusion matrix (threshold=0.5) for the basic pipeline:

![image](https://user-images.githubusercontent.com/61631269/123073263-f0439380-d41e-11eb-9661-71ed99d94ffa.png)
<br>
**After** using data augmentations technichs, and incresasing the training set to 3 times bigger, the results imporved to 80.23% accuracy on the same test set.
Below is the confusion matrix (threshold=0.5) after the data augmentation:

![image](https://user-images.githubusercontent.com/61631269/123086259-dc525e80-d42b-11eb-9b68-e9c295fe77f3.png)

It is worth mentioning that changing the architectre of the net, by adding another global layer (after the pooling layer), with 512 neurons, did not improve the results so it is not reported here.

     

