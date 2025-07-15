# 🏭 Lab: Building a Simple AI Agent Using LangChain

## Lab Objectives
By the end of this lab, you will:
1. Set up a Python environment with LangChain and an LLM provider (OpenAI).
2. Implement a custom “Wikipedia Search” tool.
3. Instantiate a LangChain agent that dynamically chooses tools.
4. Interact with your agent in a REPL loop.
5. Extend the agent with memory to preserve context.

---

## Prerequisites
- Python 3.8+  
- An OpenAI API key (exported as `OPENAI_API_KEY`)  
- Basic familiarity with Python and REST APIs  
- (Optional) Git installed for cloning examples

---

## Environment Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows PowerShell
   ```

2. Install dependencies:
   ```bash
   pip install langchain openai requests python-dotenv
   ```

3. In your project folder, create a file `.env` containing:
   ```
   OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXX
   ```

4. Create a Python file `agent_lab.py` and add:
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
   ```

---

## Task 1: Implement the Wikipedia Search Tool

1. Add imports at top of `agent_lab.py`:
   ```python
   import requests
   from langchain import Tool
   ```

2. Define a function to fetch a Wikipedia summary:
   ```python
   def wiki_search(topic: str) -> str:
       url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
       resp = requests.get(url)
       if resp.status_code != 200:
           return "No article found."
       data = resp.json()
       return data.get("extract", "No summary available.")
   ```

3. Wrap it as a LangChain Tool:
   ```python
   wiki_tool = Tool(
       name="wiki_search",
       func=wiki_search,
       description="Use this tool to get a summary of any Wikipedia topic."
   )
   ```

4. Test the tool:
   ```python
   if __name__ == "__main__":
       print(wiki_search("LangChain"))
   ```

---

## Task 2: Initialize the LLM

1. Import and instantiate an OpenAI LLM wrapper:
   ```python
   from langchain.llms import OpenAI

   llm = OpenAI(
       temperature=0.2,
       openai_api_key=OPENAI_API_KEY
   )
   ```

2. Test a basic prompt:
   ```python
   if __name__ == "__main__":
       print(llm("Explain LangChain in one sentence."))
   ```

---

## Task 3: Create and Run the Agent

1. Import the agent initializer:
   ```python
   from langchain import initialize_agent, AgentType
   ```

2. Instantiate a zero-shot agent with your tool:
   ```python
   agent = initialize_agent(
       tools=[wiki_tool],
       llm=llm,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       verbose=True
   )
   ```

3. Add an interactive REPL loop at the bottom:
   ```python
   if __name__ == "__main__":
       while True:
           query = input("\nEnter your question (or 'exit'): ")
           if query.lower() == "exit":
               break
           print(agent.run(query))
   ```

4. Run the agent:
   ```bash
   python agent_lab.py
   ```
   - **Try queries**:  
     - “What is the capital of France?”  
     - “Tell me about the Mars Rover.”

---

## Task 4: Add Conversation Memory

1. Install a memory module:
   ```bash
   pip install langchain[embeddings]
   ```

2. Add imports and memory setup:
   ```python
   from langchain.memory import ConversationBufferMemory

   memory = ConversationBufferMemory(memory_key="chat_history")
   ```

3. Re-initialize the agent with memory:
   ```python
   agent = initialize_agent(
       tools=[wiki_tool],
       llm=llm,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       memory=memory,
       verbose=True
   )
   ```

4. Relaunch and observe that follow-up questions maintain context:
   ```text
   You: Who is Ada Lovelace?
   Agent: Ada Lovelace is…
   You: When did she live?
   Agent: She lived from 1815 to 1852.
   ```

---

## Task 5: Extend with a Calculator Tool

1. Define a simple calculator:
   ```python
   def calc(expression: str) -> str:
       try:
           return str(eval(expression, {}, {}))
       except Exception:
           return "Invalid calculation."
   ```

2. Wrap it as a tool:
   ```python
   calc_tool = Tool(
       name="calculator",
       func=calc,
       description="Evaluate arithmetic expressions."
   )
   ```

3. Reinitialize the agent with both tools:
   ```python
   agent = initialize_agent(
       tools=[wiki_tool, calc_tool],
       llm=llm,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       memory=memory,
       verbose=True
   )
   ```

4. Test mixed queries:
   ```text
   You: What is 128 * 7?
   Agent: 896
   You: And who is Alan Turing?
   Agent: …
   ```

---

## Lab Deliverables

- `agent_lab.py` containing:
  - Wikipedia search tool  
  - Calculator tool  
  - OpenAI LLM initializer  
  - Agent with memory  
  - REPL loop  
- Screenshots or logs demonstrating:
  - Tool calls (wiki and calculator)  
  - Memory in follow-up queries  

---

## Reflection Questions

1. How does the agent decide which tool to invoke?  
2. What are the benefits of adding memory to the agent?  
3. How might you add error handling or retries for failed API calls?  
4. Propose an additional tool (e.g., weather, stock prices) and sketch its implementation.

---

## Next Steps & Extensions

- Swap OpenAI for a local Hugging Face model backend.  
- Persist memory in a Redis or SQL store.  
- Build a multi-agent system: one agent scrapes data, another analyzes it.  
- Integrate function-calling APIs (e.g., OpenAI Functions) for richer tool definitions.
