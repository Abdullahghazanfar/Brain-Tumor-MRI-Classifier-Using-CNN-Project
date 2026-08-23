# Brain-Tumor-MRI-Classifier-Using-CNN-Project
# ==============================================================================
# Project: CNN-Based Brain Tumor MRI Classifier
# Author: Abdullah Ghazanfar | Python Developer & AI/ML Researcher
# Description: End-to-end deep learning pipeline for MRI tumor classification.
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow Version:", tf.__version__)
print("All libraries loaded successfully!")
# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("Loading training dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    './Training', # Make sure your folder path is correct
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

print("Loading testing dataset...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    './Testing',
    shuffle=False, # Important: keep False for accurate confusion matrix mapping
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Extract class names automatically from folders
class_names = train_dataset.class_names
print("Classes detected:", class_names)

# Optimize pipeline for memory and speed
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
# ==============================================================================
# BUILD CONVOLUTIONAL NEURAL NETWORK (CNN)
# ==============================================================================
model = models.Sequential([
    # Input & Rescaling Layer (Normalize pixels to 0-1)
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    
    # Conv Block 1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Conv Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Regularization to prevent overfitting
    
    # Output Layer (4 classes)
    layers.Dense(4, activation='softmax')
])

model.summary()
# ==============================================================================
# COMPILE AND TRAIN THE MODEL
# ==============================================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS = 10
print(f"Starting training for {EPOCHS} epochs...")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)
# ==============================================================================
# EVALUATION GRAPHS
# ==============================================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))

# Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
# ==============================================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT (Precision, Recall, F1)
# ==============================================================================
print("Generating predictions on the test set...")
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted Diagnoses')
plt.ylabel('Actual Diagnoses')
plt.title('Brain Tumor MRI Confusion Matrix')
plt.show()

# Print detailed text report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
# ==============================================================================
# SINGLE IMAGE PREDICTION TEST
# ==============================================================================
# Replace this string with the path to any MRI image you want to test
test_image_path = './Testing/glioma/Te-gl_10.jpg' 

def predict_mri(image_path):
    # Load and preprocess
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Add batch dimension

    # Predict
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds[0])]
    confidence = np.max(preds[0]) * 100

    # Display
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"AI Prediction: {pred_class}\nConfidence: {confidence:.2f}%")
    plt.axis("off")
    plt.show()

# Run the function
predict_mri(test_image_path)
