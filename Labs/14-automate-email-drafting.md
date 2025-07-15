# 📝 Lab: Automating Email Drafting with an AI Agent

In this lab, you’ll build an autonomous AI agent that:

- Reads a client’s natural‐language query  
- Drafts a professional email response using an LLM  
- (Optionally) “sends” the email via a stubbed API  
- Maintains conversation context across queries  

You’ll use **LangChain** for orchestration, **OpenAI** as the LLM backend, and a simple in‐memory memory module.

---

## Lab Objectives

1. Set up your Python environment and dependencies  
2. Create a `draft_email` tool using an LLM  
3. (Optional) Create a `send_email` stub tool  
4. Assemble a LangChain agent that chooses between drafting and sending  
5. Add conversation memory so follow‐up drafts refer to earlier context  
6. Test your agent in an interactive REPL  

---

## Prerequisites

- Python 3.8 or newer  
- An OpenAI API key (export as `OPENAI_API_KEY`)  
- Basic familiarity with Python and REST APIs  
- (Optional) A SendGrid or SMTP account for real email sending  

---

## 1. Environment Setup

1. **Create & activate** a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

2. **Install** required packages:

   ```bash
   pip install langchain openai python-dotenv
   ```

3. **Create** a file `.env` in your project root:

   ```bash
   echo "OPENAI_API_KEY=sk-XXXXXXXXXX" > .env
   ```

4. **Create** `email_agent_lab.py` and add the following boilerplate:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
   ```

---

## 2. Implement the `draft_email` Tool

This tool will prompt the LLM to generate a polished email based on a client request.

1. **Add** imports at the top of `email_agent_lab.py`:

   ```python
   from langchain import Tool
   from langchain.llms import OpenAI
   ```

2. **Instantiate** the LLM:

   ```python
   llm = OpenAI(
       temperature=0.2,
       openai_api_key=OPENAI_API_KEY
   )
   ```

3. **Define** the drafting function:

   ```python
   def draft_email(query: str) -> str:
       """
       query: natural‐language description of what the email should cover,
       including recipient name, tone, and key points.
       """
       prompt = f"""
       You are an expert business communicator.
       Draft a professional email in response to the client request below.
       
       Client Request:
       {query}
       
       Requirements:
       - Use a polite, engaging tone.
       - Include a subject line.
       - Sign off with your name (e.g., “Best regards, [Your Name]”).
       """
       response = llm(prompt)
       return response.strip()
   ```

4. **Wrap** it as a LangChain `Tool`:

   ```python
   draft_email_tool = Tool(
       name="draft_email",
       func=draft_email,
       description="Draft a professional email based on a client query."
   )
   ```

5. **Test** the tool:

   ```python
   if __name__ == "__main__":
       sample = "I’d like to introduce our new AI-driven analytics platform to Acme Corp’s CTO, Maria Gonzalez. Highlight features and request a demo."
       print(draft_email(sample))
   ```

---

## 3. (Optional) Implement the `send_email` Tool

To simulate sending, we'll stub out an API call and print the result.

1. **Add** this function:

   ```python
   def send_email(payload: str) -> str:
       """
       payload: A JSON-like string containing "to", "subject", and "body".
       In a real setup, replace this with an API call to SendGrid, SMTP, etc.
       """
       # Here, just print to console
       print("=== Sending Email ===")
       print(payload)
       return "Email sent successfully (stub)."
   ```

2. **Wrap** as a `Tool`:

   ```python
   send_email_tool = Tool(
       name="send_email",
       func=send_email,
       description="Send an email given a structured payload."
   )
   ```

---

## 4. Assemble the AI Agent

Use LangChain’s agent initializer to pick between `draft_email` and (optionally) `send_email`.

1. **Import** the initializer:

   ```python
   from langchain import initialize_agent, AgentType
   ```

2. **List** your tools:

   ```python
   tools = [draft_email_tool]
   # If you implemented sending:
   # tools = [draft_email_tool, send_email_tool]
   ```

3. **Initialize** the agent:

   ```python
   agent = initialize_agent(
       tools=tools,
       llm=llm,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       verbose=True
   )
   ```

4. **Add** a REPL loop at the bottom of `email_agent_lab.py`:

   ```python
   if __name__ == "__main__":
       print("AI Email Agent ready. Type 'exit' to quit.")
       while True:
           query = input("\nClient Request: ")
           if query.lower() == "exit":
               break
           # Agent decides which tool(s) to call
           result = agent.run(query)
           print("\nAgent Response:\n")
           print(result)
   ```

5. **Run** the agent:

   ```bash
   python email_agent_lab.py
   ```

6. **Sample Interaction**:

   ```text
   Client Request: Please propose a project timeline for our website redesign, addressing budget and milestones.
   Agent Response:
   Subject: Proposal: Website Redesign Timeline & Budget

   Hi [Client Name],

   Thank you for considering our services for your website redesign. Below is a proposed timeline and budget outline:
   ...
   Best regards,
   [Your Name]
   ```

---

## 5. Add Conversation Memory

To allow your agent to remember previous drafts or context (e.g., client name), add a memory module.

1. **Install** embeddings support:

   ```bash
   pip install langchain[embeddings]
   ```

2. **Import** and instantiate memory:

   ```python
   from langchain.memory import ConversationBufferMemory

   memory = ConversationBufferMemory(memory_key="chat_history")
   ```

3. **Reinitialize** your agent with memory:

   ```python
   agent = initialize_agent(
       tools=tools,
       llm=llm,
       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       memory=memory,
       verbose=True
   )
   ```

4. **Test** context carryover:

   ```text
   Client Request: Draft an email to John from Acme introducing our AI service.
   Agent: [drafts email]

   Client Request: Ask him for his availability next week.
   Agent: [drafts follow-up referencing John and Acme from memory]
   ```

---

## 6. Lab Deliverables

- Source file `email_agent_lab.py` with:
  - LLM initialization  
  - `draft_email` (and optional `send_email`) tools  
  - Agent setup with memory  
  - REPL loop  

- **Demonstration** logs or screenshots of:
  - Drafting multiple emails in a session  
  - Follow-up drafts leveraging memory  

---

## 7. Reflection Questions

1. How does the agent choose which tool to invoke for a given query?  
2. What prompts or prompt‐template changes improve draft quality?  
3. How does memory influence follow‐up drafts?  
4. What real‐world security or compliance considerations arise when sending emails automatically?  

---

## 8. Next Steps & Extensions

- Integrate a real email service (SendGrid, Mailgun, SMTP).  
- Add an **attachment** feature: upload files via agent tool.  
- Implement **error handling** for API failures or bad prompts.  
- Extend to a **multi-agent** system: one agent drafts, another proofreads.  
- Deploy your agent as a web service using **FastAPI** or **Streamlit**.

---