import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = tf.keras.models.load_model("image_classifier_model.h5")

# Classes (same order as training)
classes = ['cars', 'cats', 'dogs']

# 👉 Custom image path (change this as per your image)
# Example: agar image yahan rakhi hai → C:\image_classifier\dataset\test\car1.jpg
img_path = r"C:\image_classifier\dataset\test\000001.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = classes[np.argmax(predictions)]

print(f"\n🧩 Prediction: {predicted_class.upper()} ✅")