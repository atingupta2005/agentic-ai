# 🔍 Model Introspection — Understanding Decisions  

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
