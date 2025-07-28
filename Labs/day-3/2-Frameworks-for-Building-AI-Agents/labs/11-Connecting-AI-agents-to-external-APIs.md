# 🌐 Connecting AI Agents to External APIs  
_A hands‐on guide to endowing your AI agents with real‐world knowledge via Google Search and Wolfram Alpha integrations._

## 1. Introduction  

Most AI agents excel at language understanding but lack up-to-the-minute world knowledge or precise computational abilities. By hooking into external APIs—like Google Search for web data and Wolfram Alpha for math, scientific facts, and structured knowledge—agents become far more capable.  

In this lab, you will:  
- Acquire API credentials  
- Wrap each API as a **LangChain Tool**  
- Assemble an agent that chooses between search and computation  
- Handle errors, timeouts, and caching  

---

## 2. Prerequisites  

- Python 3.8+ installed  
- Familiarity with virtual environments  
- Basic knowledge of LangChain (or similar agent framework)  
- `pip` access to install packages  
- Accounts and API keys for:  
  - SerpAPI (Google Search)  
  - Wolfram Alpha  

---

## 3. Why External APIs?  

- **Google Search** gives you live web results, news, definitions, and obscure facts.  
- **Wolfram Alpha** excels at math, unit conversions, scientific constants, and data queries.  
- **Agents** can automatically decide which tool to use:  
  - “When was the Eiffel Tower built?” → Google Search  
  - “Compute 12 inches in centimeters.” → Wolfram Alpha  

---

## 4. Google Search API Integration  

### 4.1. Obtaining a SerpAPI Key  

1. Sign up at https://serpapi.com/  
2. Copy your **API Key** from the dashboard.  
3. Store it in your environment:

   ```bash
   export SERPAPI_API_KEY="your_serpapi_key"
   ```

### 4.2. Installing Dependencies  

```bash
pip install langchain serpapi python-dotenv
```

### 4.3. Building a Google Search Tool  

In `agent_search.py`:

```python
import os
from langchain import Tool
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def google_search(query: str) -> str:
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": "5"
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    if not results:
        return "No results found."
    # Concatenate top 3 titles + snippets
    summary = []
    for res in results[:3]:
        title = res.get("title")
        snippet = res.get("snippet")
        summary.append(f"{title}: {snippet}")
    return "\n\n".join(summary)

google_tool = Tool(
    name="google_search",
    func=google_search,
    description="Fetch top web results for a query via Google Search API."
)
```

### 4.4. Testing the Search Tool

```python
if __name__ == "__main__":
    print(google_search("latest Mars rover discoveries"))
```

Expect 3–5 summarized entries from Google’s organic results.

### 4.5. LangChain Example: Web-Enabled QA Agent

```python
from langchain.llms import OpenAI
from langchain import initialize_agent, AgentType

# Initialize LLM
llm = OpenAI(temperature=0)

# Create agent with only the Google tool
agent = initialize_agent(
    tools=[google_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Sample run
print(agent.run("What are the three most recent NASA missions to the Moon?"))
```

Agent will detect the need for web data and call `google_search`.

---

## 5. Wolfram Alpha API Integration  

### 5.1. Getting a Wolfram Alpha AppID  

1. Sign up at https://developer.wolframalpha.com/portal/myapps/  
2. Create a new **Simple App** and obtain the **AppID**.  
3. Store it securely:

   ```bash
   export WOLFRAM_APP_ID="your_app_id"
   ```

### 5.2. Installing Dependencies  

```bash
pip install wolframalpha langchain python-dotenv
```

### 5.3. Building a Wolfram Alpha Tool  

In `agent_compute.py`:

```python
import os
from langchain import Tool
import wolframalpha
from dotenv import load_dotenv

load_dotenv()
APP_ID = os.getenv("WOLFRAM_APP_ID")
client = wolframalpha.Client(APP_ID)

def wolfram_query(question: str) -> str:
    res = client.query(question)
    # Extract pod titles and plaintext answers
    answers = []
    for pod in res.pods:
        title = pod.title
        text = "".join([sub.text for sub in pod.subpods if sub.text])
        if text:
            answers.append(f"{title}: {text}")
    return "\n\n".join(answers) or "No answer available."

wa_tool = Tool(
    name="wolfram_alpha",
    func=wolfram_query,
    description="Compute answers using Wolfram Alpha API."
)
```

### 5.4. Testing the Wolfram Alpha Tool

```python
if __name__ == "__main__":
    print(wolfram_query("integrate x^2 sin(x) dx"))
```

Should return symbolic integration result with step titles.

### 5.5. LangChain Example: Math & Facts Agent

```python
from langchain.llms import OpenAI
from langchain import initialize_agent, AgentType

llm = OpenAI(temperature=0)

agent = initialize_agent(
    tools=[wa_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print(agent.run("How far is Mercury from the Sun in astronomical units?"))
```

Agent chooses the Wolfram tool for precise scientific data.

---

## 6. Combining Tools in One Agent

Bring both `google_tool` and `wa_tool` together:

```python
agent = initialize_agent(
    tools=[google_tool, wa_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Mixed query example
print(agent.run("What is 15% of 350? Also, summarize the dictator Octavian in two sentences."))
```

- For “15% of 350?” → calls Wolfram Alpha  
- For “Octavian” → calls Google Search

---

## 7. Error Handling & Rate-Limiting

1. **API Errors:** Wrap calls in `try/except` and return user-friendly messages.  
2. **Rate Limits:** Implement simple caching (in-memory or Redis) for repeated queries.  
3. **Timeouts:** Use `requests` timeouts:

   ```python
   resp = requests.get(url, timeout=5)
   ```

4. **Retries:** Use `tenacity` for robust retry logic:

   ```python
   from tenacity import retry, wait_exponential, stop_after_attempt

   @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
   def google_search(...):
       # call API
   ```

---

## 8. Best Practices

- **Keep API Keys Secret:** Never commit `.env` to source control.  
- **Use Dedicated Service Accounts:** Limit permissions and monitor usage.  
- **Cache Results:** Web searches often repeat. Cache to reduce cost and latency.  
- **Limit Output Size:** Summarize or truncate long responses.  
- **Monitor Costs:** APIs charge per request or token—track your usage.

---

## 9. Next Steps & Extensions

- Swap SerpAPI with **Google Programmable Search** or **Bing Search**.  
- Extend Wolfram tool to parse images or unit conversions.  
- Build a **hybrid agent** that also uses a local knowledge base (vector store).  
- Integrate **OpenAI Function Calling** for more structured API usage.  
- Deploy your agent as a web service with **FastAPI** or **Flask**.

---

## 10. Appendix: Full Example Code

See `agent_search.py` and `agent_compute.py` for full implementations, plus a combined `main.py` that ties everything together.

```python
# main.py
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain import initialize_agent, AgentType
from agent_search import google_tool
from agent_compute import wa_tool

load_dotenv()
llm = OpenAI(temperature=0.1)

agent = initialize_agent(
    tools=[google_tool, wa_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def chat():
    print("Agent ready. Type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        print("Agent:", agent.run(q), "\n")

if __name__ == "__main__":
    chat()
```

---
