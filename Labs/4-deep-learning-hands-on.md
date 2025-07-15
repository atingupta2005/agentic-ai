# 📝 Hands-On Worksheet: MNIST Classification + Model Introspection  

## 📦 Part 1: Training a Neural Network on MNIST

### 🔹 Step 1: Import Required Libraries

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
```

---

### 🔹 Step 2: Load and Prepare the MNIST Dataset

```python
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode output labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

---

### 🔹 Step 3: Visualize Sample Images

```python
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i].argmax()}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

🗣️ *Prompt for Learners:* What patterns do you notice in how digits are written? Can you guess which ones might be harder to classify?

---

### 🔹 Step 4: Build the Neural Network

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

### 🔹 Step 5: Compile and Train

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

---

### 🔹 Step 6: Evaluate Model Performance

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
```

---

## 🔍 Part 2: Model Introspection — Understanding Decisions  

---

### 🔹 Step 1: Visualizing Weights from First Dense Layer

```python
weights = model.layers[1].get_weights()[0]  # Weight matrix of first Dense layer

# Display 10 learned filters
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(weights[:, i].reshape(28, 28), cmap='viridis')
    plt.title(f"Neuron {i}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

💬 *Ask Learners:* What features do these neurons seem to focus on? Edges? Center strokes?

---

### 🔹 Step 2: Predict + Confidence Visualizer

```python
import numpy as np

# Choose a test image
img_idx = 42
image = x_test[img_idx].reshape(1, 28, 28)

# Predict
prediction = model.predict(image)[0]

# Confidence bar chart
plt.bar(range(10), prediction)
plt.title("Prediction Confidence")
plt.xlabel("Digit")
plt.ylabel("Probability")
plt.show()

# Actual image
plt.imshow(x_test[img_idx], cmap='gray')
plt.title(f"True Label: {y_test[img_idx].argmax()}")
plt.axis('off')
plt.show()
```

---

### 🔹 Step 3: Analyze Misclassified Images

```python
preds = model.predict(x_test)
pred_labels = preds.argmax(axis=1)
true_labels = y_test.argmax(axis=1)

# Find where model got it wrong
errors = np.where(pred_labels != true_labels)[0]

# Show a few
plt.figure(figsize=(10, 4))
for i in range(5):
    idx = errors[i]
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {pred_labels[idx]}, Truth: {true_labels[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

---
