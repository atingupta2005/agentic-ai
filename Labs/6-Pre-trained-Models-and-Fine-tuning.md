# 🌟 Pre-trained Models and Fine-Tuning  

## Introduction  
Pre-training and fine-tuning have revolutionized natural language processing by enabling models to learn from massive unlabeled text and adapt quickly to specific tasks. Instead of training from scratch, developers can leverage off-the-shelf models via APIs, dramatically cutting development time and improving performance. This document walks you through:

- The landscape of pre-trained transformer models  
- Using Hugging Face and OpenAI APIs  
- Text generation and classification workflows  
- Transfer learning and domain adaptation methods  

---

## What Are Pre-trained Models?  
Pre-trained models are neural networks trained on large generic datasets (e.g., Wikipedia, Common Crawl) to learn rich representations of language.

- **Why pre-train?**  
  - Capture grammar, semantics, and world knowledge  
  - Avoid massive labeled datasets per task  
  - Provide strong starting points for downstream tasks  

- **Common pre-trained architectures:**  
  | Model Family | Role                  | Pretraining Objective       | Typical Use Cases          |
  |--------------|-----------------------|-----------------------------|----------------------------|
  | BERT         | Encoding text         | Masked language modeling    | Classification, NER        |
  | GPT          | Generating text       | Autoregressive language     | Text completion, dialogue  |
  | T5           | Text-to-text tasks    | “Fill in the blank” tasks   | Translation, summarization |
  | RoBERTa      | Robust BERT variant   | Improved MLM                | Same as BERT               |
  | DistilBERT   | Smaller, faster BERT  | Knowledge distillation      | On-device inference        |

---

## APIs for Pre-trained Models  

### Hugging Face Transformers Library  
Hugging Face provides an open-source Python library called **transformers**, which exposes:

- **Model Hub**: 10,000+ community-contributed models  
- **`pipeline` API**: High-level tasks (e.g., generation, classification)  
- **Trainer**: Utilities for custom fine-tuning  
- **Tokenizer classes**: Convert raw text to token IDs  

Installation:
```bash
pip install transformers datasets
```

Basic usage with pipeline:
```python
from transformers import pipeline

# Load a sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Run inference
print(classifier("I love using pretrained models!"))
```

---

### OpenAI API  
OpenAI offers hosted access to GPT-series models (e.g., GPT-3.5, GPT-4) via REST endpoints.

- **Authentication**: Obtain an API key from OpenAI  
- **Endpoints**: `/v1/completions`, `/v1/chat/completions`, etc.  
- **Pricing**: Pay-as-you-go per token  

Installation:
```bash
pip install openai
```

Example in Python:
```python
import openai

openai.api_key = "YOUR_API_KEY"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Write a short poem about the sunrise.",
  max_tokens=50,
  temperature=0.7
)

print(response.choices[0].text.strip())
```

---

## Using Transformers for Text Generation  
Text generation models predict next tokens given a prompt. Two main approaches:

1. **Autoregressive (GPT-style)**  
   - Predict token _t_ from _1…t−1_  
   - Great for free-form text completion  

2. **Encoder–Decoder (T5, BART)**  
   - Encode input text → decode generated text  
   - Supports text-to-text tasks, e.g., translation  

### Key Parameters for Generation  
- **`prompt`**: Starting text  
- **`max_length`**: Number of tokens to generate  
- **`temperature`**: Controls randomness (0.0–1.0)  
- **`top_k` / `top_p`**: Nucleus sampling  

#### Example with Hugging Face
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
out = generator("Once upon a time", max_length=30, num_return_sequences=3)
for idx, res in enumerate(out):
    print(f"=== Sample {idx+1} ===\n{res['generated_text']}\n")
```

#### Example with OpenAI
```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Summarize the key benefits of AI in one paragraph.",
  max_tokens=100
)
print(response.choices[0].text.strip())
```

---

## Using Transformers for Text Classification  
Sequence classification assigns labels (e.g., sentiment, topic) to whole text sequences.

### Hugging Face Pipeline
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("The movie was fantastic!"))
```

### Custom Classification with Trainer  
1. **Load dataset** (e.g., IMDb, custom CSV)  
2. **Tokenize** inputs with appropriate `Tokenizer`  
3. **Define** `Trainer` with `TrainingArguments`  
4. **Train** and **evaluate**  

```python
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import datasets

# Load dataset
ds = datasets.load_dataset("imdb")

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
ds = ds.map(preprocess, batched=True)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Training
args = TrainingArguments(
    output_dir="out",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch"
)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"])
trainer.train()
```

---

## Transfer Learning and Domain Adaptation  

### Why Transfer Learning?  
- **Reuses knowledge** from general corpora  
- **Speeds convergence** on small datasets  
- **Improves performance** in low-data regimes  

### Fine-Tuning Strategies  
| Strategy               | Description                                                       | When to Use               |
|------------------------|-------------------------------------------------------------------|---------------------------|
| Full Fine-Tuning       | Update all model weights                                          | Ample labeled data        |
| Feature-Based (Freeze) | Freeze backbone, train only task-specific head                    | Very small dataset        |
| Adapter Modules        | Insert small trainable layers inside transformer                  | Multi-task/adaptive use   |
| Prompt-Tuning          | Learn soft prompt embeddings                                       | Minimal compute budget    |

#### Simple Fine-Tuning Workflow  
1. Load pre-trained model & tokenizer.  
2. Replace head with task-specific layer.  
3. Compile and train on your dataset.  

```python
from transformers import BertTokenizer, TFBertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Tokenize inputs
inputs = tokenizer(["Hello world", "How are you?"], padding=True, truncation=True, return_tensors="tf")
labels = tf.constant([0,1])

# Train step
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(inputs.data, labels, epochs=3, batch_size=2)
```

### Domain Adaptation Techniques  
1. **Continued Pre-Training** on domain texts (e.g., legal, medical).  
2. **Multi-Task Learning** with in-domain auxiliary tasks.  
3. **Adapters**: Plug-and-play small modules that learn domain shifts.  

#### Example: Continued Pre-Training  
```python
from transformers import Trainer, TrainingArguments, BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# Prepare domain-specific corpus as MaskedLM dataset…
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="dom_pretrain", num_train_epochs=1, per_device_train_batch_size=16),
    train_dataset=domain_dataset
)
trainer.train()
# Save and fine-tune on task afterwards
```

---

## Hands-On Examples  

### Example: Text Generation with Hugging Face  
1. Create a pipeline with GPT-2.  
2. Experiment with `temperature`, `top_p`.  
3. Compare outputs for different prompts.  

### Example: Text Classification with Hugging Face  
1. Use the SST-2 fine-tuned DistilBERT model.  
2. Run inference on tweets or reviews.  
3. Analyze confidence scores.  

### Example: Fine-Tuning BERT on a Custom Dataset  
1. Prepare CSV with `text,label` columns.  
2. Load with `datasets`.  
3. Tokenize and fine-tune with `Trainer`.  
4. Evaluate and inspect misclassifications.

---

## Best Practices and Tips  
- Always **match tokenizer & model** (e.g., BERT-uncased tokenizer with BERT-uncased).  
- Use **learning rate warm-up** and **weight decay** to stabilize fine-tuning.  
- Monitor **validation loss** to avoid overfitting.  
- Leverage **mixed precision** (`fp16`) for faster training on GPUs.  
- For domain adaptation, start with **continued pre-training** before fine-tuning.

---

## Conclusion  
Pre-trained transformers and their fine-tuning workflows empower developers to build high-quality NLP applications quickly. By combining APIs (Hugging Face, OpenAI) with transfer learning strategies, you can tackle generation, classification, and specialized domain tasks with minimal data and compute.

---