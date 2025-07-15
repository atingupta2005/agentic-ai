# 🏗️ Lab: Deploying an AI Agent for Financial Report Insights

This hands-on guide walks you through building, containerizing, and deploying an AI agent that:

1. Ingests PDF financial reports  
2. Extracts key metrics (Revenue, EBITDA, Net Income, etc.)  
3. Generates narrative insights using an LLM  
4. Exposes a REST API for integration  

---

## Lab Objectives

- Set up a Python environment with necessary libraries  
- Implement tools for PDF parsing and metric extraction  
- Create an LLM-driven “Insight Generator” tool  
- Assemble a LangChain agent that orchestrates parsing, extraction, and insight generation  
- Wrap the agent behind a FastAPI service  
- Containerize with Docker and test deployment  

---

## Prerequisites

- Python 3.8+  
- `pip` package manager  
- Docker (for containerization)  
- An OpenAI API key (export as `OPENAI_API_KEY`)  
- Sample financial report PDFs (place in `./reports/`)  

---

## 1. Environment Setup

1. Clone or create your project folder:
   ```bash
   mkdir finance_agent && cd finance_agent
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows
   ```

3. Install required packages:
   ```bash
   pip install langchain openai fastapi uvicorn pdfplumber pandas python-dotenv
   ```

4. Create a `.env` file with your OpenAI key:
   ```
   OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXX
   ```

---

## 2. PDF Parsing & Metric Extraction Tool

We’ll use pdfplumber to extract tables and text, and pandas to process tables.

### 2.1. Create `tools/pdf_parser.py`

```python
import pdfplumber
import pandas as pd

def parse_financial_pdf(path: str) -> dict:
    """
    Extracts numerical tables and key text passages from a PDF financial report.
    Returns a dict with table DataFrames and raw text.
    """
    tables, text = [], ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            for table in page.extract_tables():
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    return {"text": text, "tables": tables}
```

### 2.2. Create `tools/metric_extractor.py`

```python
import pandas as pd
from typing import List, Dict

def extract_metrics(tables: List[pd.DataFrame]) -> Dict[str, float]:
    """
    Scans DataFrames for common financial metrics (Revenue, EBITDA, Net Income).
    Returns a dict of metric names and values.
    """
    metrics = {}
    keywords = {
        "Revenue": ["Revenue", "Total Revenue", "Net Sales"],
        "EBITDA": ["EBITDA", "Earnings Before Interest"],
        "Net Income": ["Net Income", "Profit After Tax"]
    }
    for df in tables:
        for metric, keys in keywords.items():
            for key in keys:
                mask = df.iloc[:, 0].str.contains(key, case=False, na=False)
                if mask.any():
                    # Assume last column holds the most recent figure
                    val = df.loc[mask, df.columns[-1]].values[0]
                    try:
                        metrics[metric] = float(val.replace(",", ""))
                    except:
                        pass
    return metrics
```

---

## 3. Insight Generation Tool

Wrap the LLM call to generate a CFO-style narrative from the extracted metrics.

### 3.1. Create `tools/insight_generator.py`

```python
import os
from langchain.llms import OpenAI

llm = OpenAI(
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def generate_insights(metrics: dict) -> str:
    """
    Given a dict of financial metrics, returns a narrative summary.
    """
    prompt = f"""
    You are a corporate finance expert. Analyze the following key metrics and write a concise executive summary:
    {metrics}

    Focus on trends, significant changes, and possible drivers.
    """
    response = llm(prompt)
    return response.strip()
```

---

## 4. Assemble the LangChain Agent

Create an agent that chains PDF parsing, metric extraction, and insight generation.

### 4.1. Create `agent.py`

```python
import os
from langchain import Tool, LLMChain
from langchain.agents import initialize_agent, AgentType
from tools.pdf_parser import parse_financial_pdf
from tools.metric_extractor import extract_metrics
from tools.insight_generator import generate_insights

# Wrap each function as a Tool
pdf_tool = Tool(
    name="parse_pdf",
    func=parse_financial_pdf,
    description="Parse a financial PDF and return raw text and tables."
)
metric_tool = Tool(
    name="extract_metrics",
    func=extract_metrics,
    description="Extract Revenue, EBITDA, and Net Income from tables."
)
insight_tool = Tool(
    name="generate_insights",
    func=generate_insights,
    description="Generate narrative insights from extracted metrics."
)

tools = [pdf_tool, metric_tool, insight_tool]

# Initialize a zero-shot agent
agent = initialize_agent(
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def analyze_report(path: str) -> str:
    """
    Runs the agent on the given PDF path to return insights.
    """
    # Step 1: parse PDF
    parsed = pdf_tool.run(path)
    # Step 2: extract metrics
    metrics = metric_tool.run(parsed["tables"])
    # Step 3: generate narrative
    insights = insight_tool.run(metrics)
    return insights
```

### 4.2. Test Locally

Add a `__main__` block to `agent.py`:

```python
if __name__ == "__main__":
    report_path = "./reports/Q1_2025_Financials.pdf"
    summary = analyze_report(report_path)
    print("\n=== Executive Summary ===\n", summary)
```

Run:
```bash
python agent.py
```

---

## 5. Expose as a FastAPI Service

Wrap the agent in a REST API to accept PDF uploads or file paths.

### 5.1. Create `app.py`

```python
import uvicorn
from fastapi import FastAPI, UploadFile, File
from agent import analyze_report
import shutil

app = FastAPI(title="Financial Insights Agent")

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    # Save uploaded PDF
    path = f"./temp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Run analysis
    summary = analyze_report(path)
    return {"filename": file.filename, "summary": summary}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 5.2. Test the API

From another terminal:
```bash
curl -X POST "http://localhost:8000/analyze/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./reports/Q1_2025_Financials.pdf"
```

You should receive a JSON response with your narrative summary.

---

## 6. Containerize with Docker

### 6.1. Create `Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2. Build & Run

```bash
# Freeze dependencies
pip freeze > requirements.txt

# Build Docker image
docker build -t finance-insights-agent .

# Run container
docker run -d -p 8000:8000 --name fin-agent finance-insights-agent
```

Test with the same `curl` command against `localhost:8000`.

---

## 7. Deployment Strategies

- **Cloud Run / AWS Fargate:** Deploy the Docker image as a serverless container.  
- **Kubernetes:** Create a Deployment and Service; use an Ingress for external access.  
- **Monitoring:** Integrate Prometheus/Grafana to track request latency, error rates.  
- **Logging:** Ensure FastAPI logs are shipped to ELK or Splunk for audit trails.  

---