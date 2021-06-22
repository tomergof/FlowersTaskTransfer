# FlowersTaskTransfer 🌷
Applying task transfer learning while using a pre-trained neural network (ResNet50V2).

## Part A - Task Transfer 🚍
ResNet50V2 was originally used for identifying between 1000 different classes. I changed the output layer to a single neuron, so it can be used for a binary classification - decide whether the picture is flower or not.
The training set is 472 images, each is labeled as 1 (flower) or 0 (not flower).
After loading the net, I used an average pooling to get a 2048 neurons global layer. I activated the sigmoid function on this layer to calculate the single neuron in the output level, by which we can classify the image.
<br>
## Part B - Model Improvement 👌
I used to methods to try and improve the model's accuracy.
The first method is Data Augmentation - making the training set 3 times bigger than the original.
The second method is changing the net architecture.
<br>
## Files in the Repository 📚
*Hyperparameter Tuning*
The full procces of chossing all the relevant hyperparameters in the algorithms.
*Function & Packages*
All the packages that needs to be installed.
All the function for activating the project's pipe.
*Main*
Calling the main function of the project.
*Packages and Libraries*
All required packages and libraries.

