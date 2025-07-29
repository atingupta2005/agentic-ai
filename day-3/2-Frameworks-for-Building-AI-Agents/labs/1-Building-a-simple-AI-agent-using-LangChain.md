# %% [markdown]
# # 🚀 Advanced AI Agent with LangChain
# *Fully self-contained Jupyter Notebook*
# 
# **Features:**
# - Wikipedia search tool
# - Safe calculator
# - Conversation memory
# - Rich interactive interface
# - Error handling
# 
# *No external files needed - runs entirely in this notebook*

# %% [markdown]
# ## 1️⃣ Environment Setup

# %%
# Install required packages (run this first)
!pip install -q langchain==0.1.14 openai==1.12.0 python-dotenv==1.0.0 
!pip install -q requests==2.31.0 wikipedia==1.4.0 langchain-community==0.0.27
!pip install -q rich==13.7.0 tiktoken==0.6.0

# %%
# @title 🔑 API Configuration
import os
from getpass import getpass
from rich.console import Console
from rich.panel import Panel

console = Console()

# Get API key securely
api_key = getpass("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key

console.print(Panel.fit("✅ Environment Ready!", style="bold green"))

# %% [markdown]
# ## 2️⃣ Tool Implementations

# %%
# @title 🌐 Wikipedia Search Tool
import requests
from langchain.tools import Tool
from typing import Optional

def wiki_search(topic: str, lang: str = "en") -> Optional[str]:
    """Enhanced Wikipedia search with error handling"""
    headers = {"User-Agent": "MyAIAgent/1.0 (educational-use)"}
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{topic}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("extract", "No summary available.")
    except Exception as e:
        print(f"⚠️ Wikipedia Error: {str(e)}")
        return None

wiki_tool = Tool(
    name="wikipedia",
    func=wiki_search,
    description="Access Wikipedia summaries. Input format: 'search_term lang_code'",
    handle_tool_error=True
)

# Test
wiki_search("LangChain")

# %%
# @title ➗ Safe Calculator Tool
import re
from langchain.tools import BaseTool

class SafeCalculator(BaseTool):
    name = "calculator"
    description = """Evaluates math expressions. 
    Only numbers and +-*/.() allowed. Example: '(3 + 5) * 2'"""
    
    def _run(self, expression: str) -> str:
        if not re.fullmatch(r'^[\d\s+\-*\/\.\(\)]+$', expression):
            return "❌ Invalid input - only basic math operations allowed"
        try:
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"⚠️ Calculation error: {str(e)}"

calc_tool = SafeCalculator()

# Test
calc_tool.run("(5 + 3) * 2")

# %% [markdown]
# ## 3️⃣ Agent Initialization

# %%
# @title 🧠 Initialize LLM & Memory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    streaming=True
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# %%
# @title 🤖 Build the Agent
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[wiki_tool, calc_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

# %% [markdown]
# ## 4️⃣ Interactive Chat Interface

# %%
# @title 💬 Run the Agent (Interactive Mode)
from rich.markdown import Markdown
from IPython.display import display, clear_output
import ipywidgets as widgets

output = widgets.Output()
text_input = widgets.Text(
    placeholder='Type your question here...',
    layout=widgets.Layout(width='80%')
submit_button = widgets.Button(description="Ask")

def on_submit(_):
    with output:
        clear_output()
        query = text_input.value
        if query.lower() in ('exit', 'quit'):
            print("👋 Ending conversation...")
            return
            
        print(f"🧑 You: {query}")
        try:
            response = agent({"input": query})
            display(Markdown(f"🤖 **Agent**: {response['output']}"))
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
        text_input.value = ''  # Clear input

submit_button.on_click(on_submit)
display(widgets.VBox([text_input, submit_button, output]))

# %% [markdown]
# ## 5️⃣ Sample Queries to Try
# ```
# What is LangChain?
# Tell me about Paris in French (use "Paris fr")
# Calculate (25 * 4) + (18 / 3)
# Who is Marie Curie? Then ask: What was her most famous discovery?
# ```

# %% [markdown]
# ## 🛠️ Troubleshooting
# - If you get API errors, verify your key is correct
# - Restart the kernel if tools stop responding
# - For timeouts, wait 1 minute and retry