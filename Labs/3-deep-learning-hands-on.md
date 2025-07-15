## 🧠 Module 1: MNIST Image Classification with a Neural Network

### 📦 Objective:
Train a basic neural network to recognize handwritten digits (0–9) from grayscale images in the MNIST dataset.

---

### 🔧 Step-by-Step Breakdown

#### 🔹 Step 1: Import Libraries

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
```

---

#### 🔹 Step 2: Load and Preprocess Data

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

---

#### 🔹 Step 3: Visualize Sample Images

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

---

#### 🔹 Step 4: Build the Neural Network Model

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

#### 🔹 Step 5: Compile the Model

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

---

#### 🔹 Step 6: Train the Model

```python
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

---

#### 🔹 Step 7: Evaluate Performance

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

---

## 🔍 Module 2: Visualizing Model Decisions

Let’s interpret how the network sees and processes digit images.

---

### 🔹 Step 1: Visualizing Learned Weights

```python
weights = model.layers[1].get_weights()[0]  # First Dense layer weights

# Display weights for first 10 neurons
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(weights[:, i].reshape(28, 28), cmap='viridis')
    plt.title(f"Neuron {i}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

> 💬 **What This Shows:** Each neuron learns to respond to certain pixel patterns—almost like a filter or shape detector.

---

### 🔹 Step 2: Predict and Visualize Model Confidence

```python
import numpy as np

# Select a test image
test_idx = 42
image = x_test[test_idx].reshape(1, 28, 28)
prediction = model.predict(image)[0]

# Display prediction scores
plt.bar(range(10), prediction)
plt.title("Model Confidence Scores")
plt.xlabel("Digit")
plt.ylabel("Probability")
plt.show()

# Display the actual image
plt.imshow(x_test[test_idx], cmap='gray')
plt.title(f"True Label: {y_test[test_idx].argmax()}")
plt.axis('off')
plt.show()
```

> 💬 **Takeaway:** This shows not only what the model predicted, but how confident it is about each digit.

---

### 🔹 Step 3: Visualize Misclassifications

```python
preds = model.predict(x_test)
pred_labels = preds.argmax(axis=1)
true_labels = y_test.argmax(axis=1)

# Find incorrect predictions
errors = np.where(pred_labels != true_labels)[0]

# Visualize a few
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

> 💬 **Insight:** Error analysis helps uncover biases or weak spots in model learning—great for classroom discussion.

---

