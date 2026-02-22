import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os

train_dir = r"C:\image_classifier\dataset\train"
val_dir = r"C:\image_classifier\dataset\val"

# Load dataset
train_ds = image_dataset_from_directory(train_dir, image_size=(160, 160), batch_size=32)
val_ds = image_dataset_from_directory(val_dir, image_size=(160, 160), batch_size=32)

# Normalize images
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False  # Freeze base model

# Add custom classifier on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation='softmax')  # 3 classes: cats, dogs, cars
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Save model
model.save("mobilenet_classifier.h5")

# Plot accuracy & loss
os.makedirs("graphs", exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='lime')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='cyan')
plt.title('📈 MobileNetV2 Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("graphs/mobilenet_accuracy.png")
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss', color='orange')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('📉 MobileNetV2 Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("graphs/mobilenet_loss.png")
plt.close()

print("\n✅ MobileNetV2 training completed and graphs saved.")