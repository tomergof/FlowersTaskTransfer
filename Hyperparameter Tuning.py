# Hyperparameter Tuning - Full Procces

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=False)

# Learning rate tuning (Learning rate = 0.1 was selected)

model, train_images, train_labels, test_images, test_labels = main_no_improvements(data_path, train_images_indices,
                                                                                   test_images_indices)
learning_rate_grid = [0.1, 0.01, 0.001, 0.0001]
results = []
initial_epochs = 5

for learning_rate in learning_rate_grid:
    learning_rate_CV_res = []
    for train_index, val_index in kf.split(X=train_images):
        model, train_images, train_labels, test_images, test_labels = main_no_improvements(data_path,
                                                                                           train_images_indices,
                                                                                           test_images_indices)

        training_data = train_images[train_index]
        training_y = train_labels[train_index]
        validation_data = train_images[val_index]
        validation_y = train_labels[val_index]

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        model.fit(x=training_data, y=training_y,
                  epochs=initial_epochs,
                  batch_size=32,
                  )

        result = model.evaluate(validation_data, validation_y)
        learning_rate_CV_res.append(result[1])
    results.append(sum(learning_rate_CV_res) / len(learning_rate_CV_res))

# Batch size tuning (Batch size = 32 was selected)

model, train_images, train_labels, test_images, test_labels = main_no_improvements(data_path, train_images_indices,
                                                                                   test_images_indices)
batch_size_grid = [4, 8, 16, 32, 64]
results = []
initial_epochs = 5

for batch_size in batch_size_grid:
    batch_size_CV_res = []
    for train_index, val_index in kf.split(X=train_images):
        model, train_images, train_labels, test_images, test_labels = main_no_improvements(data_path,
                                                                                           train_images_indices,
                                                                                           test_images_indices)

        training_data = train_images[train_index]
        training_y = train_labels[train_index]
        validation_data = train_images[val_index]
        validation_y = train_labels[val_index]

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        model.fit(x=training_data, y=training_y,
                  epochs=initial_epochs,
                  batch_size=batch_size,
                  )

        result = model.evaluate(validation_data, validation_y)
        batch_size_CV_res.append(result[1])
    results.append(sum(batch_size_CV_res) / len(batch_size_CV_res))

# Number of epochs tuning (2 epochs were selected)

model, train_images, train_labels, test_images, test_labels = main_no_improvements(data_path, train_images_indices,
                                                                                   test_images_indices)
results = []

for train_index, val_index in kf.split(X=train_images):
    model, train_images, train_labels, test_images, test_labels = main_no_improvements(data_path, train_images_indices,
                                                                                       test_images_indices)

    training_data = train_images[train_index]
    training_y = train_labels[train_index]
    validation_data = train_images[val_index]
    validation_y = train_labels[val_index]

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    result = model.fit(x=training_data, y=training_y,
                       epochs=8,
                       batch_size=32,
                       validation_data=(validation_data, validation_y)
                       )

    results.append(result)