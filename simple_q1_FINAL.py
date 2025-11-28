import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
import os

TRAIN_DIR = 'C:\\Users\\khwah\\Downloads\\Assignment2\\Assignment2\\Q1\\train'
TEST_DIR = 'C:\\Users\\khwah\\Downloads\\Assignment2\\Assignment2\\Q1\\test'

np.random.seed(42)
tf.random.set_seed(42)
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

print("="*60)
print("Q1: CAT VS DOG CLASSIFICATION")
print("="*60)

# Check if folders exist
if not os.path.exists(TRAIN_DIR):
    print(f"\nERROR: Cannot find training folder: {TRAIN_DIR}")
    print("Please update TRAIN_DIR path in the code!")
    exit()

if not os.path.exists(TEST_DIR):
    print(f"\nWARNING: Cannot find test folder: {TEST_DIR}")
    print("Will skip testing on test images")

# Load data
print("\n1. Loading images...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='training'
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation'
)

print(f"Training images: {train_gen.samples}")
print(f"Validation images: {val_gen.samples}")

# Build Simple CNN
print("\n2. Building Simple CNN Model...")
cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Simple CNN
print("\n3. Training Simple CNN (this takes ~15-20 minutes)...")
print("   Please be patient, grab a coffee ☕")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_cnn = cnn_model.fit(
    train_gen, epochs=20, validation_data=val_gen,
    callbacks=[early_stop], verbose=1
)

cnn_acc = cnn_model.evaluate(val_gen, verbose=0)[1]
print(f"\n✓ Simple CNN Accuracy: {cnn_acc:.2%}")

# Build VGG16
print("\n4. Building VGG16 Transfer Learning Model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

vgg_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train VGG16
print("\n5. Training VGG16 Model (this takes ~10-15 minutes)...")
print("   Downloading pre-trained weights first time...")
history_vgg = vgg_model.fit(
    train_gen, epochs=15, validation_data=val_gen,
    callbacks=[early_stop], verbose=1
)

vgg_acc = vgg_model.evaluate(val_gen, verbose=0)[1]
print(f"\n✓ VGG16 Accuracy: {vgg_acc:.2%}")

# Compare and save best
print("\n6. Comparing models...")
print(f"Simple CNN: {cnn_acc:.2%}")
print(f"VGG16:      {vgg_acc:.2%}")

if vgg_acc > cnn_acc:
    best_model = vgg_model
    best_name = "VGG16"
    best_acc = vgg_acc
    best_history = history_vgg
else:
    best_model = cnn_model
    best_name = "Simple CNN"
    best_acc = cnn_acc
    best_history = history_cnn

print(f"\n✓ Best Model: {best_name} with {best_acc:.2%} accuracy")

# Save model
best_model.save('cat_dog_model.h5')
print(f"Model saved: cat_dog_model.h5")

# Create plots
print("\n7. Creating visualizations...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(best_history.history['accuracy'], label='Training')
ax1.plot(best_history.history['val_accuracy'], label='Validation')
ax1.set_title(f'{best_name} - Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(best_history.history['loss'], label='Training')
ax2.plot(best_history.history['val_loss'], label='Validation')
ax2.set_title(f'{best_name} - Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('q1_training_results.png', dpi=300, bbox_inches='tight')
print("Training plot saved: q1_training_results.png")

# Test on samples
if os.path.exists(TEST_DIR):
    print("\n8. Testing on sample images...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=10,
        class_mode='binary', shuffle=True
    )
    
    x_batch, y_batch = next(test_gen)
    predictions = best_model.predict(x_batch, verbose=0)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_batch[i])
        pred_class = 'Dog' if predictions[i] > 0.5 else 'Cat'
        confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]
        ax.set_title(f'Predicted: {pred_class}\n{confidence:.1%}', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('q1_test_predictions.png', dpi=300, bbox_inches='tight')
    print("Test predictions saved: q1_test_predictions.png")

print("\n" + "="*60)
print("Q1 COMPLETE!")
print("="*60)
print(f"Final Accuracy: {best_acc:.2%}")
print("\nFiles created:")
print("  - cat_dog_model.h5")
print("  - q1_training_results.png")
if os.path.exists(TEST_DIR):
    print("  - q1_test_predictions.png")
