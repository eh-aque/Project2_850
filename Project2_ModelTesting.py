from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

# Step 5: Model Testing

# Load the model
model2 = load_model('model2.h5')

# Full path to the image
image_path = './Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 2/Project 2 Data/Test/Medium/Crack__20180419_06_16_35,563.bmp'

# Load the image
img = load_img(image_path, target_size=(64, 64))

# Convert the image to an array and normalize it
img = img_to_array(img) / 255.

# Add an extra dimension
img = np.expand_dims(img, axis=0)

# Make a prediction
preds = model2.predict(img)

# Find the class with the highest probability
class_index = np.argmax(preds[0])

print(f'The predicted class is: {class_index}')


# Test multiple images using code below:
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import load_model
# import numpy as np
# import os

# # Load the model
# model2 = load_model('model2.h5')

# # Directory where your test images are located
# test_dir = 'path/test/test2/tesea/blah/blah/blah'

# # Loop over each file in the test directory
# for filename in os.listdir(test_dir):
#     # Only process .jpg files
#     if filename.endswith('.jpg'):
#         # Load the image
#         img = load_img(os.path.join(test_dir, filename), target_size=(64, 64))

#         # Convert the image to an array and normalize it
#         img = img_to_array(img) / 255.

#         # Add an extra dimension
#         img = np.expand_dims(img, axis=0)

#         # Make a prediction
#         preds = model2.predict(img)

#         # Find the class with the highest probability
#         class_index = np.argmax(preds[0])

#         print(f'The predicted class for {filename} is: {class_index}')