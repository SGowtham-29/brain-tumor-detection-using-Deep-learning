import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset paths (Modify these if needed)
TRAIN_DIR = r"D:\tumordatas\Training"
TEST_DIR = r"D:\tumordatas\Testing"

# Define model save path (Change this to your desired location)
SAVE_PATH = r"D:\tumordatas"
os.makedirs(SAVE_PATH, exist_ok=True)
MODEL_NAME = "brain_tumor_model.h5"

# Image Parameters
IMG_SIZE = (224, 224)  # MobileNetV2 requires 224x224
BATCH_SIZE = 32

# Data Augmentation for better generalization
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

# Training and Validation Data Loaders
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Load Pretrained MobileNetV2 Model (Efficient & Accurate)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pretrained layers

# Define Model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")  # 4 classes (Glioma, Meningioma, No Tumor, Pituitary)
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
EPOCHS = 10
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save Model
model.save(os.path.join(SAVE_PATH, MODEL_NAME))
print(f"Model saved at: {os.path.join(SAVE_PATH, MODEL_NAME)}")
