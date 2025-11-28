import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os

TRAIN_PATH = 'C:\\Users\\khwah\\Downloads\\Assignment2\\Assignment2\\Q2\\mnist_train.csv'
TEST_PATH = 'C:\\Users\\khwah\\Downloads\\Assignment2\\Assignment2\\Q2\\mnist_test.csv'

np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("Q2: MNIST DIGIT CLASSIFICATION")
print("="*60)

# Check if files exist
if not os.path.exists(TRAIN_PATH):
    print(f"\nERROR: Cannot find training file: {TRAIN_PATH}")
    print("Please update TRAIN_PATH in the code!")
    exit()

if not os.path.exists(TEST_PATH):
    print(f"\nERROR: Cannot find test file: {TEST_PATH}")
    print("Please update TEST_PATH in the code!")
    exit()

# Load data
print("\n1. Loading MNIST data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values / 255.0

y_test = test_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values / 255.0

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Visualize samples
print("\n2. Creating sample visualization...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.suptitle('Sample MNIST Digits', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_samples.png', dpi=300, bbox_inches='tight')
print("✓ Sample images saved: q2_samples.png")

# Create validation split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

results = []

# Method 1: Random Forest
print("\n3. Training Random Forest (takes ~3 minutes)...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train_split, y_train_split)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"✓ Random Forest Accuracy: {rf_acc:.2%}")
results.append(('Random Forest', rf_acc))

# Method 2: Neural Network
print("\n4. Building and training Neural Network (takes ~5 minutes)...")
nn_model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_nn = nn_model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val, y_val),
    epochs=30, batch_size=128,
    callbacks=[early_stop], verbose=1
)

nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\n✓ Neural Network Accuracy: {nn_acc:.2%}")
results.append(('Neural Network', nn_acc))

# Method 3: CNN
print("\n5. Building and training CNN (takes ~7 minutes)...")
X_train_cnn = X_train_split.reshape(-1, 28, 28, 1)
X_val_cnn = X_val.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_cnn = cnn_model.fit(
    X_train_cnn, y_train_split,
    validation_data=(X_val_cnn, y_val),
    epochs=30, batch_size=128,
    callbacks=[early_stop], verbose=1
)

cnn_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)[1]
print(f"\n✓ CNN Accuracy: {cnn_acc:.2%}")
results.append(('CNN', cnn_acc))

# Compare results
print("\n6. Comparing all methods...")
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for name, acc in results:
    status = "✓ PASSED (>90%)" if acc >= 0.90 else "✗ FAILED"
    print(f"{name:20s}: {acc:.2%} {status}")

best_name, best_acc = max(results, key=lambda x: x[1])
print(f"\n✓ Best Model: {best_name} with {best_acc:.2%} accuracy")

# Create comparison plot
print("\n7. Creating comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
names = [r[0] for r in results]
accs = [r[1] for r in results]
colors = ['green' if a >= 0.90 else 'orange' for a in accs]

bars = ax.bar(names, accs, color=colors, alpha=0.7)
ax.axhline(y=0.90, color='r', linestyle='--', label='90% Target')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Comparison - MNIST Classification', fontsize=14, fontweight='bold')
ax.set_ylim([0.85, 1.0])
ax.legend()

for bar, acc in zip(bars, accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('q2_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved: q2_comparison.png")

# Create confusion matrix for best model
print("\n8. Creating confusion matrix for best model...")
if best_name == 'CNN':
    y_pred = np.argmax(cnn_model.predict(X_test_cnn, verbose=0), axis=1)
elif best_name == 'Neural Network':
    y_pred = np.argmax(nn_model.predict(X_test, verbose=0), axis=1)
else:
    y_pred = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('q2_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved: q2_confusion_matrix.png")

# Save best model
print("\n9. Saving best model...")
if best_name == 'CNN':
    cnn_model.save('mnist_model.h5')
elif best_name == 'Neural Network':
    nn_model.save('mnist_model.h5')
else:
    import joblib
    joblib.dump(rf_model, 'mnist_model.pkl')
print("Best model saved")

print("\n" + "="*60)
print("Q2 COMPLETE!")
print("="*60)
print(f"Best Accuracy: {best_acc:.2%}")
if best_acc >= 0.90:
    print("TARGET ACHIEVED (>90%)")
else:
    print("Target not achieved")
print("\nFiles created:")
print("  - q2_samples.png")
print("  - q2_comparison.png")
print("  - q2_confusion_matrix.png")
print("  - mnist_model.h5 (or .pkl)")
