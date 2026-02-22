import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Validation data path
val_dir = r"C:\image_classifier\dataset\val"

# Load trained model
model = tf.keras.models.load_model("image_classifier_model.h5")

# Get model input size
input_shape = model.input_shape[1:3]
print(f"📏 Model expects image size: {input_shape}")

# Load validation dataset (categorical labels for 3 classes)
val_ds = image_dataset_from_directory(
    val_dir,
    image_size=input_shape,
    batch_size=32,
    label_mode='categorical'  # 👈 important fix
)

# Evaluate model
loss, accuracy = model.evaluate(val_ds)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Loss: {loss:.4f}")