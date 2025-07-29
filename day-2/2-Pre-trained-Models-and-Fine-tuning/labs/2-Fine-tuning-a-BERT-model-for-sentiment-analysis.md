**Fine-Tuning BERT for Sentiment Analysis**

---

### **1. Enhanced Prerequisites**

#### **1.1 System Requirements**
- Python 3.8+ (recommended for latest transformer support)
- CUDA 11.7+ (if using GPU acceleration)
- 16GB+ RAM (for efficient batch processing)

#### **1.2 Installation**
```bash
# Core packages
pip install transformers datasets scikit-learn torch

# Optional but recommended
pip install wandb tensorboard accelerate
```

#### **1.3 Hardware Configuration**
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

### **2. Dataset Preparation (Enhanced)**

#### **2.1 Dataset Loading with Validation Split**
```python
from datasets import load_dataset, DatasetDict

# Load with 10% validation split
dataset = load_dataset("imdb")
split_dataset = dataset["train"].train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],  # Our validation set
    "test": dataset["test"]              # Original test set
})
```

#### **2.2 Advanced Tokenization**
```python
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,  # Increased from 128 for better context
        add_special_tokens=True
    )

# Apply to all splits
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=256,
    remove_columns=["text"]
)

# Convert format
tokenized_datasets.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)
```

---

### **3. Model Configuration (Enhanced)**

#### **3.1 Model Initialization with Advanced Settings**
```python
from transformers import BertForSequenceClassification, TrainingArguments

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=True,  # For potential analysis
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Freeze early layers (optional)
for param in model.bert.encoder.layer[:6].parameters():
    param.requires_grad = False
```

#### **3.2 Comprehensive Training Arguments**
```python
training_args = TrainingArguments(
    output_dir="./bert-finetuned-imdb",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="wandb"  # Optional for experiment tracking
)
```

---

### **4. Enhanced Training Process**

#### **4.1 Custom Metrics with Class Balance**
```python
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    # Get per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )
    
    # Get macro-averaged metrics
    macro_f1 = np.mean(f1)
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "f1_positive": f1[1],
        "f1_negative": f1[0],
        "macro_f1": macro_f1,
        "precision": precision[1],
        "recall": recall[1]
    }
```

#### **4.2 Trainer with Early Stopping**
```python
from transformers import Trainer
from transformers.trainer_callback import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Start training with progress tracking
trainer.train()
```

---

### **5. Advanced Evaluation**

#### **5.1 Comprehensive Model Evaluation**
```python
# Evaluate on test set
test_results = trainer.evaluate(tokenized_datasets["test"])

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
```

#### **5.2 Model Interpretation (Integrated Gradients)
```python
# Optional: Install interpretability tools
# pip install captum

from captum.attr import IntegratedGradients

def interpret_sentence(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    ig = IntegratedGradients(model)
    
    # Compute attributions
    attributions = ig.attribute(
        inputs["input_ids"],
        target=1  # Positive class
    )
    
    # Visualize
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    show_text_heatmap(tokens, attributions[0])
```

---

### **6. Production-Ready Inference**

#### **6.1 Optimized Prediction Pipeline**
```python
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text, return_confidences=False):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        
        if return_confidences:
            return probs.cpu().numpy()[0]
        return "positive" if torch.argmax(probs).item() == 1 else "negative"

# Usage
analyzer = SentimentAnalyzer("bert-finetuned-imdb/checkpoint-best")
print(analyzer.predict("This film was breathtaking!"))
```

#### **6.2 ONNX Export (Optional)**
```python
torch.onnx.export(
    model,
    (torch.randint(0, 10000, (1, 128)),  # Example input
    "bert_sentiment.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)
```
