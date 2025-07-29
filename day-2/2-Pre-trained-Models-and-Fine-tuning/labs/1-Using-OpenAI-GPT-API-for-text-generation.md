**OpenAI GPT API Text Generation Guide**  

---

### **1. Introduction**  
This document provides a complete implementation guide for using OpenAI's GPT API in Jupyter Notebook, featuring:  
- Interactive parameter controls  
- Support for both chat and completion models  
- Advanced generation techniques  
- Error handling and best practices  

---

### **2. Environment Setup**  

#### **2.1 Package Installation**  
Run this cell first to install required packages:  
```python
!pip install openai ipywidgets
```

#### **2.2 API Configuration**  
```python
import os
import openai
from IPython.display import display, Markdown

# Set your API key
api_key = "sk-your-key-here"  # Replace with your actual key
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key
```

---

### **3. Interactive Generation System**  

#### **3.1 Model Parameters Widget**  
Creates interactive controls for generation settings:  
```python
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown

model_choices = ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"]

@interact(
    model=Dropdown(options=model_choices, value="gpt-3.5-turbo", description="Model:"),
    temperature=FloatSlider(min=0, max=2, step=0.1, value=0.7, description="Creativity:"),
    max_tokens=IntSlider(min=50, max=1000, value=256, description="Max Length:"),
    top_p=FloatSlider(min=0.1, max=1, value=1, description="Diversity:")
)
def set_parameters(model, temperature, max_tokens, top_p):
    global generation_params
    generation_params = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }
    print("Parameters updated ✓")
```

#### **3.2 Enhanced Generation Function**  
```python
def generate_text(prompt, system_message=None, n=1):
    try:
        if generation_params["model"] in ["gpt-3.5-turbo", "gpt-4"]:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = openai.ChatCompletion.create(
                messages=messages,
                **generation_params
            )
            return [choice.message.content for choice in response.choices]
        else:
            response = openai.Completion.create(
                prompt=prompt,
                **generation_params
            )
            return [choice.text for choice in response.choices]
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
```

---

### **4. Practical Implementation**  

#### **4.1 Basic Text Generation**  
```python
prompt = "Explain quantum computing to a 10-year-old"
result = generate_text(prompt)
display(Markdown(f"**Response:**\n\n{result[0]}"))
```

#### **4.2 Chat Conversation Simulation**  
```python
conversation = [
    {"role": "system", "content": "You are a helpful science tutor."},
    {"role": "user", "content": "How do black holes form?"}
]

response = generate_text(
    prompt=conversation[-1]["content"],
    system_message=conversation[0]["content"]
)
display(Markdown(f"**Tutor:** {response[0]}"))
```

#### **4.3 Creative Generation with Parameters**  
```python
creative_params = {
    "model": "gpt-4",
    "temperature": 0.9,
    "max_tokens": 500
}

story = generate_text(
    "Write a short sci-fi story about AI discovering emotions",
    **creative_params
)
display(Markdown(f"**Story:**\n\n{story[0]}"))
```

---

### **5. Advanced Features**  

#### **5.1 Multiple Variations**  
```python
product_descriptions = generate_text(
    "Create marketing copy for a smartwatch",
    n=3  # Generate 3 variants
)

for i, desc in enumerate(product_descriptions, 1):
    display(Markdown(f"**Version {i}:**\n{desc}\n---"))
```

#### **5.2 JSON Mode (GPT-4 Only)**  
```python
structured_response = generate_text(
    prompt="List 5 European capitals with their countries in JSON format",
    system_message="Always respond with valid JSON"
)
print(structured_response[0])
```

---

### **6. Best Practices**  

1. **Rate Limiting**:  
   ```python
   import time
   def safe_generate(prompt, retries=3):
       for i in range(retries):
           try:
               return generate_text(prompt)
           except openai.error.RateLimitError:
               wait = 2 ** i  # Exponential backoff
               print(f"Rate limited. Waiting {wait} seconds...")
               time.sleep(wait)
       return None
   ```

2. **Cost Monitoring**:  
   ```python
   def estimate_cost(response):
       if isinstance(response, dict):
           model = response.get("model", "")
           usage = response.get("usage", {})
           tokens = usage.get("total_tokens", 0)
           
           # Sample pricing (check current rates)
           rates = {
               "gpt-3.5-turbo": 0.002/1000,
               "gpt-4": 0.06/1000
           }
           rate = rates.get(model, 0)
           return tokens * rate
       return 0
   ```

---

### **7. Complete Workflow Example**  

```python
# 1. Set parameters via widget UI
# 2. Generate content
business_plan = generate_text(
    "Create a 1-page business plan for an AI tutoring startup",
    system_message="You are a business consultant specializing in edtech"
)

# 3. Display formatted output
display(Markdown(f"## Business Plan\n\n{business_plan[0]}"))

# 4. Estimate cost
response = generate_text("Sample", return_full_response=True)
print(f"Estimated cost: ${estimate_cost(response):.4f}")
```
