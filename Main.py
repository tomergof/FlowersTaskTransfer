# Important! Change the data path to the images' folder on your computer
data_path = ""
test_images_indices = list(range(301, 473))
train_images_indices = list(range(1, 301))


# might be needed:
# pip install Pillow
# pip install h5py==2.10.0

import tensorflow as tf
import cv2
import scipy.io
import numpy as np
import random
import time


def set_seed(seed):
    """
    Description:
    This function sets random seed for numpy and tensorflow.
    parameters:
    seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return


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


def load_model():
    """
    This function return a ResNet50V2 CNN with imagenet weights, input_shape of (224, 224, 3)
    and all the weights frozen.
    returns:
    model - ResNet50 pre-trained model
    """
    model = tf.keras.applications.ResNet50V2(
        include_top=False,  # No top level needed
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),  # HardCoded
        pooling=None,
    )
    model.trainable = False  # Freeze lower layers
    return model


def add_pooling_and_prediction_layers(model, global_average_layer, prediction_layer):
    """
    This function gets a basic model, global average layer and dense layer and returns a combined model.
    parameters:
    model - model before architecture change
    global_average_layer - adding average pooling layer to the ResNet50
    prediction_layer - 1 neuron prediction layer
    returns:
    model -  model after architecture change, for binary flower classification.
    """
    inputs = tf.keras.Input(shape=(224, 224, 3))  # hard coded
    x = model(inputs, training=False)
    x = global_average_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def change_architecture(model):
    """
    This function gets a basic model and returns this model with additional
    global average pooling layer, global 512 dense layer and a prediction layer with 1 neuron.
    parameters:
    model - model before architecture change.
    returns:
    model - model after architecture change, and additional global layer
    """
    global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
    additional_global_512 = tf.keras.layers.Dense(512, activation='sigmoid')
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    inputs = tf.keras.Input(shape=(224, 224, 3))  # hard coded
    x = model(inputs, training=False)
    x = global_avg_layer(x)
    x = additional_global_512(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def data_augmentation(data, labels, multiply=5):
    """
    Description:
    This function used for creating augmented data from original images.
    :param data: list of original images
    :param labels: list of original images labels
    :param multiply: determines number of augmented data. multiply = 2 creates data larger 3 times than original one
    :return:
    lists (numpy arrays) of augmented images and labels.
    """
    from keras.preprocessing.image import ImageDataGenerator
    from numpy import expand_dims

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 rotation_range=20,
                                 zoom_range=0.2,  # instead of cropping
                                 )
    augmented_data = []
    augmented_labels = []
    for image, label in zip(data, labels):
        augmented_data.append(image)
        augmented_labels.append(label)
        image = expand_dims(image, 0)
        for _ in range(multiply):
            it = datagen.flow(image, batch_size=1)
            batch = it.next()
            augmented_image = batch
            augmented_image = np.squeeze(augmented_image, axis=0)
            augmented_data.append(augmented_image)
            augmented_labels.append(label)
    return np.array(augmented_data), np.array(augmented_labels)


def create_roc(test_labels, predicted_labels):
    """
    description:
    This function creates a Recall precision curve and calculates area under the curve.
    :param test_labels: True data labels
    :param predicted_labels: model predicted labels
    :return:
    plt - the ROC plot
    auc_keras - the area under the curve.
    """
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, predicted_labels)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(1)
    plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    return plt, auc_keras


def main_no_improvements(path, train_idxs, test_idxs):
    """
    Description:
    Main function used for building the basic pipe model
    :param path: the directory where the data files are stored.
    :param train_idxs: indexes of train images.
    :param test_idxs: indexes of test images.
    :return:
    model - basic pipe model
    train_images, train_labels, test_images, test_labels -  lists of images and labels for train and test
    """
    set_seed(10)
    train_images, train_labels, test_images, test_labels = import_data(path, train_idxs, test_idxs)
    model = load_model()
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    model = add_pooling_and_prediction_layers(model, global_average_layer, prediction_layer)
    return model, train_images, train_labels, test_images, test_labels


def test_before_improvements():
    """
    description:
    This function trains basic pipe model and reports results. (Hard coded hyperparameters).
    """
    model, train_images, train_labels, test_images, test_labels = main_no_improvements(data_path, train_images_indices, test_images_indices)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.fit(x=train_images, y=train_labels,
              epochs=2,
              batch_size=32
              )
    pred = model.predict(test_images)
    accuracy = model.evaluate(test_images, test_labels)[1]
    print("Accuracy is " + str(accuracy))
    print("Confusion Matrix:" + str(accuracy))
    print_confusion_matrix(pred, test_labels)
    return


def print_confusion_matrix(y_pred, y_labels):
    """
    This function prints confusion matrix.
    :param y_pred: labels predicted by model.
    :param y_labels: true test labels.
    """
    from sklearn.metrics import confusion_matrix
    pred = [1 if x > 0.5 else 0 for x in y_pred]
    print(confusion_matrix(pred, y_labels, labels=(0, 1)))


def main_final(path, train_idxs, test_idxs):
    """
    Description:
    This function runs full final selected pipe, reports data details, results, confusion matrix, ROC, model structure and total run time.
    :param path: the directory where the data files are stored.
    :param train_idxs: indexes of train images.
    :param test_idxs: indexes of test images.
    """
    start = time.time()
    set_seed(10)
    print("Importing data...")
    train_images, train_labels, test_images, test_labels = import_data(path, train_idxs, test_idxs)
    num_train_imgs = len(train_images)
    num_train_imgs_true = sum(train_labels)
    print("Creating data augmentation...")
    augmented_data, augmented_labels = data_augmentation(train_images, train_labels, multiply=2)
    num_agmntd_imgs = len(augmented_data)
    num_agmntd_imgs_true = sum(augmented_labels)
    print("Loading model...")
    model = load_model()
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    model = add_pooling_and_prediction_layers(model, global_average_layer, prediction_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    print("Training network...")
    model.fit(x=augmented_data, y=augmented_labels,
              epochs=2,
              batch_size=32
              )
    print("Testing...")
    pred = model.predict(test_images)
    accuracy = model.evaluate(test_images, test_labels,)[1]
    print("------------- REPORT: -------------")
    print(" ")
    print("The number of original training images: " + str(num_train_imgs) + "\n")
    print("Amount of ???True??? flower images out of them: " + str(num_train_imgs_true) + "\n")
    print("The number of augmented training images: " + str(num_agmntd_imgs - num_train_imgs) + "\n")
    print("Amount of ???True??? flower images out of them: " + str(num_agmntd_imgs_true - num_train_imgs_true) + "\n")
    plt, auc_keras = create_roc(test_labels, pred)
    print("Precision-recall curve AUC score: " + str(auc_keras) + "\n")
    print("Test error rate: " + str(1-accuracy) + "\n")
    print("Test accuracy rate: " + str(accuracy) + "\n")
    print("Confusion Matrix:")
    print_confusion_matrix(pred, test_labels)
    print("Threshold = 0.5 \n X-axis is True labels \n Y-axis is Predicted labels \n")
    print("Final model structure:")
    model.summary()
    end = time.time()
    total_time = end - start
    print("Total run time: " + str(round(total_time / 60)) + " minutes.")
    plt.show()
    return

#Runing Part A:
main_no_improvements(data_path, train_images_indices, test_images_indices)

#Runing Part B - Data Augmnetation:
main_final(data_path, train_images_indices, test_images_indices)

