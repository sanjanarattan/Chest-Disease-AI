# Chest Disease AI
 **Project Report:**

The following code is a machine learning project that involves the classification of chest diseases using image recognition. It includes Python code for a ResNet50 model, as well as code for evaluating the accuracy of the model using test data and generating a confusion matrix. The report also includes several images, including example chest x-rays and visualizations of the model's predictions.

**Transfer learning** is a technique used in machine learning where a model trained on one task is re-purposed for a different task. In the case of this code, transfer learning was used to classify chest diseases using image recognition. The ResNet50 model, which was pre-trained on a large dataset of images, was imported and used as a starting point for the model. The weights of the pre-trained model were then fine-tuned on the chest disease dataset to improve performance. This approach can significantly reduce the amount of data and time required for training, while still achieving high accuracy.

**Here's an explanation of each of the imports in the Python code:**

- `os`: This module provides a way of using operating system dependent functionality like reading or writing to the file system.
- `tensorflow`: This is a popular open-source platform for building and training machine learning models.
- `numpy`: This is a Python library used for working with arrays.
- `sklearn`: This is a machine learning library for Python. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN.
- `cv2`: This is a library used for real-time computer vision. It is mainly used to do image processing.
- `matplotlib`: This is a plotting library
- `seaborn`: This is a Python data visualization library.
- `pandas`: This is a data analysis and manipulation tool.

**Explanation of the Code:**

- `train_generator=image_generator.flow_from_directory(batch_size=40,directory=XRay_Directory,shuffle=True,target_size=(256,256),class_mode='categorical',subset='training')`: This line generates batches of image data from the train directory, resizes the images to 256x256 pixels, and applies categorical labels.
- `validation_generator=image_generator.flow_from_directory(batch_size=40,directory=XRay_Directory,shuffle=True,target_size=(256,256),class_mode='categorical',subset='validation')`: This line generates batches of image data from the validation directory, resizes the images to 256x256 pixels, and applies categorical labels.
- `basemodel=ResNet50(weights='imagenet',include_top=False,input_tensor=Input(shape=(256,256,3)))`: This line imports the ResNet50 model pre-trained on a large dataset of images, sets the weights to 'imagenet', and uses an input shape of 256x256 pixels with 3 channels.
- `for layer in basemodel.layers[:-10]: layers.trainable=False`: This line freezes the weights of the first few layers of the imported model so that they are not retrained during the fine-tuning process.
- `headmodel=Dense(4, activation='softmax')(headmodel)`: This line adds a fully connected layer with a softmax activation function to the model, with 4 outputs corresponding to the 4 classes of chest diseases.
- `model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-4), metrics=["accuracy"])`: This line compiles the model with a categorical cross-entropy loss function, an RMSprop optimizer with a learning rate of 0.0001, and an accuracy metric.
- `history = model.fit(train_generator, steps_per_epoch= train_generator.n // 4, epochs = 1, validation_data= val_generator, validation_steps=val_generator.n // 4, callbacks=[checkpointer, earlystopping])`: This line trains the model on the training data for one epoch, with a batch size of 4, and validates the model on the validation data.
- `test_gen = ImageDataGenerator(rescale=1./255)`: This line generates a test data generator that rescales the images to 256x256 pixels.
- `test_generator = test_gen.flow_from_directory(batch_size=40, directory=test_directory, shuffle=True, target_size=(256, 256), class_mode='categorical')`: This line generates batches of image data from the test directory, resizes the images to 256x256 pixels, and applies categorical labels.
- `evaluation = model.evaluate(test_generator, steps=test_generator.n // 4, verbose=1)`: This line evaluates the model's performance on the test data and prints the results, including the accuracy.
- `predict = model.predict(img)`: This line generates a prediction for a single image using the trained model.
- `cm = confusion_matrix(np.asarray(original), np.asarray(prediction))`: This line generates a confusion matrix for the model's predictions on the test data, comparing the predicted labels to the true labels.
- `sns.heatmap(cm, annot = True, ax = ax)`: This line creates a heatmap of the confusion matrix using the seaborn library.
