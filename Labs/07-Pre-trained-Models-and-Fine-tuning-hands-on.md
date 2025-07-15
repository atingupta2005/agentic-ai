# 🛠 Practical Hands-On Guide: Pre-trained Models and Fine-Tuning

This guide walks you through two hands-on labs:  
1. Using OpenAI’s GPT API for text generation  
2. Fine-tuning a BERT model for sentiment analysis on a real-world dataset  

You can run both labs in a local Python environment or in Google Colab.

---

## Part 1: Text Generation with OpenAI’s GPT API

### Prerequisites

- Python 3.7+  
- An OpenAI API key (set as environment variable `OPENAI_API_KEY`)  
- Install the `openai` library:  
  ```bash
  pip install openai
  ```

---

### 1. Configure Your Environment

1. Export your API key in the terminal (Linux/macOS):  
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```  
   Or on Windows PowerShell:  
   ```powershell
   setx OPENAI_API_KEY "your_api_key_here"
   ```

2. Verify your key is available in Python:  
   ```python
   import os
   assert "OPENAI_API_KEY" in os.environ
   ```

---

### 2. Basic Text Generation Script

Create a file `generate_text.py` with the following code:

```python
import os
import openai

def generate_text(prompt: str, 
                  model: str = "text-davinci-003", 
                  max_tokens: int = 100, 
                  temperature: float = 0.7,
                  top_p: float = 1.0,
                  n_samples: int = 1):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n_samples
    )
    return [choice.text.strip() for choice in response.choices]

if __name__ == "__main__":
    prompt = "Write a short poem about a sunrise over the mountains."
    outputs = generate_text(prompt, max_tokens=50, temperature=0.8, n_samples=2)
    for i, text in enumerate(outputs, 1):
        print(f"\n=== Sample {i} ===\n{text}")
```

Run it:
```bash
python generate_text.py
```

---

### 3. Experimenting with Parameters

- **`temperature`**: Higher values (0.8–1.0) → more creative outputs; lower (0.2–0.5) → more focused.  
- **`max_tokens`**: Limits response length.  
- **`top_p`** (nucleus sampling): Restricts vocabulary to cumulative probability _p_.  
- **`n`**: Number of samples to generate per prompt.

> 💡 Try generating three variations of a product description by setting `n_samples=3` and compare diversity.

---

### 4. Advanced: Chat Completions (GPT-3.5 / GPT-4)

OpenAI’s chat endpoint uses a list of messages:

```python
from openai import ChatCompletion

chat = ChatCompletion(api_key=os.getenv("OPENAI_API_KEY"))
response = chat.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the benefits of electric vehicles."}
    ],
    max_tokens=150,
    temperature=0.6
)
print(response.choices[0].message.content)
```

> 🔍 Notice how you can steer tone and style with the “system” message.

---

## Part 2: Fine-Tuning BERT for Sentiment Analysis

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
