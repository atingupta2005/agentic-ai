## Fine-Tuning BERT for Sentiment Analysis

### Prerequisites

- Python 3.7+  
- PyTorch or TensorFlow backend  
- Install required libraries:  
  ```bash
  pip install transformers datasets scikit-learn torch
  ```

---

### 1. Choose and Load a Real-World Dataset

We’ll use the IMDb movie-review dataset via the 🤗 Datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
train_ds, test_ds = dataset["train"], dataset["test"]
```

---

### 2. Tokenize the Text

Use BERT’s tokenizer with a maximum sequence length of 128:

```python
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

train_ds = train_ds.map(tokenize_batch, batched=True)
test_ds  = test_ds.map(tokenize_batch, batched=True)

# Set the format for PyTorch
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch",  columns=["input_ids", "attention_mask", "label"])
```

---

### 3. Define the Model and Training Arguments

We’ll use `BertForSequenceClassification` and the Hugging Face `Trainer`:

```python
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="bert-finetuned-imdb",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True
)
```

---

### 4. Define Metrics and Trainer

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)
```

---

### 5. Train and Evaluate

```python
trainer.train()
metrics = trainer.evaluate()
print(metrics)
```

You’ll see epoch-wise metrics: accuracy, precision, recall, F1.

---

### 6. Inference on New Sentences

Save and reload the best model, then run predictions:

```python
model_path = "bert-finetuned-imdb/checkpoint-best"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs   = torch.softmax(outputs.logits, dim=1)
    label   = torch.argmax(probs).item()
    return ("negative", "positive")[label], probs[0][label].item()

for sample in [
    "I absolutely loved this movie—wonderful performances!",
    "It was a total waste of time, so boring..."
]:
    sentiment, confidence = predict_sentiment(sample)
    print(f"Text: {sample}\nSentiment: {sentiment} (confidence {confidence:.2f})\n")
```

---
