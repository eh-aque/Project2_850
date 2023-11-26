# Step 5: Model Testing

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model2 = load_model('model2.h5')

# Create a data generator for the test images
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the test images from a directory
test_generator = test_datagen.flow_from_directory(
    './Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 2/Project 2 Data/Test',
    target_size=(100, 100),  # Adjusted to match the model's input shape
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Create a mapping from indices to class names
index_to_class = {v: k for k, v in test_generator.class_indices.items()}

# List of image paths
image_paths = [
    './Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 2/Project 2 Data/Test/Medium/Crack__20180419_06_19_09,915.bmp', 
    './Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 2/Project 2 Data/Test/Large/Crack__20180419_13_29_14,846.bmp'
]

for image_path in image_paths:
    # Load the image
    img = load_img(image_path, target_size=(100, 100))  # Adjust the target size to match your model's input shape

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Normalize the image
    img_array /= 255.

    # Add an extra dimension for the batch size
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction for the image
    pred = model2.predict(img_array)

    # Find the class with the highest probability
    class_index = np.argmax(pred, axis=1)[0]

    # Get the class name for the prediction
    predicted_class = index_to_class[class_index]

    # Create a figure
    plt.figure()

    # Display the image
    plt.imshow(img)

    # Add a title with the predicted class
    plt.title(f'Predicted class: {predicted_class}')

    # Convert the predicted probabilities to percentages
    pred_percentages = pred[0] * 100

    # Round the percentages to 2 decimal places
    pred_percentages = np.round(pred_percentages, 2)

    # Convert the predicted probabilities to a string, with each element separated by a comma
    pred_percentages_str = ', '.join(map(str, pred_percentages))

    # Add a caption with the predicted probabilities
    plt.figtext(0.5, 0.01, f'Predicted probabilities (L, M, N, S): ({pred_percentages_str})%', wrap=True, horizontalalignment='center', fontsize=12)

    # Show the figure
    plt.show()


























# ---------------------------------------------------------------
# # Classifying ALL images in the test set - unfinshed, does not display seperate images, but lists all and its classifications
# # Create a data generator for the test images
# test_datagen = ImageDataGenerator(rescale=1./255)

# # Load the test images from a directory
# test_generator = test_datagen.flow_from_directory(
#     './Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 2/Project 2 Data/Test',
#     target_size=(100, 100),  # Adjusted to match the model's input shape
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False
# )

# # Make predictions on the test images
# preds = model2.predict(test_generator)

# # Find the class with the highest probability for each test image
# class_indices = np.argmax(preds, axis=1)

# # Create a mapping from indices to class names
# index_to_class = {v: k for k, v in test_generator.class_indices.items()}

# # Get the class names for the predictions
# predicted_classes = [index_to_class[i] for i in class_indices]

# print(f'The predicted classes are: {predicted_classes}')
# ---------------------------------------------------------------