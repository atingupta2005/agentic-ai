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
