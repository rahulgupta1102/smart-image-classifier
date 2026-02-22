from tensorflow.keras.preprocessing import image_dataset_from_directory

# training folder ka path
ds = image_dataset_from_directory("dataset/train", batch_size=1)
print("📂 Classes found in training folder:")
print(ds.class_names)