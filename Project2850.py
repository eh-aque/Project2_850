import numpy as np
import pandas as pd
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator


# Step 1: Data Processing
# -----------------------
# Define constants
IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS = 100, 100, 3
BATCH_SIZE = 32

# Define the train and validation data directories
train_data_dir = './Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 2/Project 2 Data/Train'
validation_data_dir = './Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 2/Project 2 Data/Validation'

# Create an ImageDataGenerator for training data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Create an ImageDataGenerator for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create the train generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Create the validation generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Step 2: Neural Network Architecture Design
# ------------------------------------------
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# # Define the model
# model1 = Sequential()
# model1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
# model1.add(MaxPooling2D(pool_size=(2, 2)))
# model1.add(Conv2D(64, (3, 3), activation='relu'))
# model1.add(MaxPooling2D(pool_size=(2, 2)))
# model1.add(Flatten())
# model1.add(Dense(128, activation='relu'))
# model1.add(Dropout(0.5))
# model1.add(Dense(4, activation='softmax'))  # 4 neurons for 4 classes

# # Define the model 2
# model2 = Sequential()
# model2.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))  # increased filters and kernel size, added stride
# model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))  # increased filters, added stride
# model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))  # increased filters, added stride
# model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Flatten())
# model2.add(Dense(512, activation='relu'))  # increased neurons
# model2.add(Dropout(0.5))
# model2.add(Dense(4, activation='softmax'))  # 4 neurons for 4 classes

# # Compile models before training
# model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Step 3: Hyperparameter Analysis
# -------------------------------

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))  # try 'LeakyRelu' or 'elu'
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))  # try 'LeakyRelu' or 'elu'
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))  # try 'LeakyRelu' or 'elu'
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # try 'elu'
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # 4 neurons for 4 classes

# Compile model before training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # try different optimizers like 'sgd', 'rmsprop'

# Define the model 2
model2 = Sequential()
model2.add(Conv2D(128, (3, 3), strides=(1, 1), activation='LeakyReLU', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))  # increased filters, changed activation
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(256, (3, 3), strides=(1, 1), activation='LeakyReLU'))  # increased filters, changed activation
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(512, (3, 3), strides=(1, 1), activation='LeakyReLU'))  # increased filters, changed activation
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(1024, activation='LeakyReLU'))  # increased neurons, changed activation
model2.add(Dropout(0.5))
model2.add(Dense(4, activation='softmax'))  # 4 neurons for 4 classes


# Define X_train and y_train
X_train = np.array(...)  # Replace ... with your actual training data
y_train = np.array(...)  # Replace ... with your actual training labels

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  # changed optimizer
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  # changed optimizer

# Step 4: Model Evaluation

# history and fit the model
history1 = model.fit(train_generator, validation_data=validation_generator, epochs=10)
history2 = model2.fit(train_generator, validation_data=validation_generator, epochs=10)
# history1 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
# history2 = model2.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Plot the training and validation accuracy and loss
import matplotlib.pyplot as plt

# Plot training & validation accuracy values for model 1
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Model 1 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values for model 1
plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model 1 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy values for model 2
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model 2 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values for model 2
plt.subplot(1, 2, 2)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model 2 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()