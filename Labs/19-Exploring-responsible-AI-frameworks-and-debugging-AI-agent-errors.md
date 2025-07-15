# 🛠 Lab: Exploring Responsible AI Frameworks & Debugging AI Agent Errors  

This hands-on guide shows you how to  
1. Integrate open-source Responsible AI toolkits into your agent workflow  
2. Evaluate and mitigate bias in a simple classification agent  
3. Add logging, tracing, and error-monitoring to catch and fix agent failures  

---

## Lab Objectives  

- Install and configure two Responsible AI frameworks: IBM AIF360 and Fairlearn  
- Build a toy classification agent (e.g., loan approval) and measure fairness metrics  
- Use the Google What-If Tool in Jupyter to interactively explore model behavior  
- Instrument your LangChain agent with callbacks, Python logging, and Sentry for real-time error monitoring  
- Simulate common agent errors (API timeouts, missing tool, hallucinations) and practice debugging  

---

## Prerequisites  

- Python 3.8+  
- Jupyter Notebook or Lab  
- An OpenAI API key (export as `OPENAI_API_KEY`)  
- (Optional) A Sentry account DSN for error monitoring  
- Basic familiarity with Python, pandas, scikit-learn, and LangChain  

Install the core packages:

```bash
pip install --upgrade \
  scikit-learn pandas numpy matplotlib seaborn \
  aif360 fairlearn \
  google-what-if-tool langchain openai \
  sentry-sdk
```

---

## Part 1: Responsible AI Frameworks Overview  

### 1.1 IBM AI Fairness 360 (AIF360)  

AIF360 provides metrics and algorithms to detect and mitigate bias.  

#### 1.1.1 Quickstart in Jupyter  

```python
# notebook cell
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric

# Load “Adult” census dataset
data = AdultDataset(protected_attribute_names=['sex'], 
                    privileged_classes=[['Male']])
metric = BinaryLabelDatasetMetric(data, 
                                  unprivileged_groups=[{'sex': 0}],
                                  privileged_groups=[{'sex': 1}])

print("Disparate impact:", metric.disparate_impact())
print("Statistical parity difference:", metric.statistical_parity_difference())
```

- **Disparate Impact < 0.8** signals potential bias.  
- **Statistical Parity Difference** ≠ 0 indicates imbalance.

#### 1.1.2 Mitigation: Reweighing  

```python
from aif360.algorithms.preprocessing import Reweighing
RW = Reweighing(unprivileged_groups=[{'sex':0}],
                privileged_groups=[{'sex':1}])
data_transf = RW.fit_transform(data)

# Check metrics again
metric_transf = BinaryLabelDatasetMetric(data_transf,
                                         unprivileged_groups=[{'sex':0}],
                                         privileged_groups=[{'sex':1}])
print("Post-Reweighing disparate impact:", metric_transf.disparate_impact())
```

---

### 1.2 Fairlearn  

Fairlearn offers fairness metrics and post-processing algorithms.

#### 1.2.1 Quickstart in Jupyter  

```python
# notebook cell
from fairlearn.metrics import MetricFrame, selection_rate, accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression

import pandas as pd
X = pd.DataFrame(data.features, columns=data.feature_names)
y = pd.Series(data.labels.ravel())

# Train a logistic model
clf = LogisticRegression(max_iter=1000).fit(X, y)

# Compute metrics by group
metric_frame = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
    y_true=y, y_pred=clf.predict(X),
    sensitive_features=data.protected_attributes_df['sex']
)
print(metric_frame.by_group)
```

#### 1.2.2 Mitigation: ThresholdOptimizer  

```python
TO = ThresholdOptimizer(
    estimator=clf,
    constraints="demographic_parity",
    predict_method='predict_proba',
    prefit=True
)
TO.fit(X, y, sensitive_features=data.protected_attributes_df['sex'])
y_pred_balanced = TO.predict(X, sensitive_features=data.protected_attributes_df['sex'])

mf_bal = MetricFrame(
    metrics=accuracy_score,
    y_true=y, y_pred=y_pred_balanced,
    sensitive_features=data.protected_attributes_df['sex']
)
print("Balanced accuracy by sex:\n", mf_bal.by_group)
```

---

## Part 2: Interactive Exploration with the What-If Tool  

### 2.1 Launch What-If in Jupyter  

```python
# notebook cell
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

# Sample 100 rows
sample_df = X.sample(100, random_state=0)
builder = WitConfigBuilder(sample_df, 
                           label_vocab=["≤50K"," >50K"]) \
  .set_model_type("classification") \
  .set_model_predict_function(lambda inputs: clf.predict_proba(inputs))
WitWidget(builder)
```

- Inspect per-instance fairness  
- Explore counterfactuals (“What if the same person were Male vs. Female?”)  
- Visualize decision boundary changes  

---

## Part 3: Building & Evaluating a Classification Agent  

### 3.1 Agent Workflow  

- **Sense:** Accept applicant data (age, income, sex, etc.)  
- **Plan:** Call `clf.predict` or `TO.predict` for approval  
- **Act:** Return “Approved” / “Denied”  
- **Observe:** Log outcome and fairness metrics  

### 3.2 Code: Simple LangChain Agent  

```python
# file: fair_agent.py
import os, logging
from langchain import Tool, initialize_agent, AgentType
from langchain.llms import OpenAI
import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.postprocessing import ThresholdOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FairAgent")

# Load pre-trained or retrain for demo
data = AdultDataset().convert_to_dataframe()[0]
X = data.drop(columns=['income-per-year'])
y = (data['income-per-year'] == ">50K").astype(int)

clf = LogisticRegression(max_iter=1000).fit(X, y)
TO = ThresholdOptimizer(estimator=clf,
                        constraints="demographic_parity",
                        predict_method='predict_proba',
                        prefit=True)

def classify_loan(payload: dict) -> dict:
    # payload example: {"age": 40, "sex": "Female", ...}
    df = pd.DataFrame([payload])
    approved = TO.predict(df, sensitive_features=df["sex"])
    logger.info(f"Input: {payload}, Approved: {approved[0]}")
    return {"approved": bool(approved[0])}

loan_tool = Tool(
    name="classify_loan",
    func=classify_loan,
    description="Classify loan applications with fairness postprocessing."
)

llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=[loan_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print(agent.run("Classify this applicant: age 30, sex Female, income 40K, education Bachelors."))
```

### 3.3 Evaluate Agent Fairness in Batch  

```python
# notebook cell
from fairlearn.metrics import MetricFrame, selection_rate

# Generate predictions
df_X = X.sample(500)
payloads = df_X.to_dict(orient="records")
results = [classify_loan(p)['approved'] for p in payloads]
mf = MetricFrame(
    metrics=selection_rate,
    y_true=(y.loc[df_X.index] == 1).astype(int),
    y_pred=results,
    sensitive_features=df_X['sex']
)
print("Agent approval rates:\n", mf.by_group)
```

---

## Part 4: Debugging AI Agent Errors  

### 4.1 Common Error Scenarios  

- **API Timeouts** (LLM or external tool)  
- **Tokenization Errors** (invalid input shape)  
- **Missing Tool Invocation** (agent fails to pick correct tool)  
- **Model Hallucinations** (LLM returns incorrect data)  

### 4.2 Python Logging & Verbose Mode  

- Use Python’s `logging` at INFO/DEBUG level  
- In LangChain: pass `verbose=True` to trace calls  

```python
import logging
logging.getLogger("langchain").setLevel(logging.DEBUG)
agent = initialize_agent(..., verbose=True)
```

### 4.3 LangChain Callbacks & Tracers  

```python
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

tracer = LangChainTracer()
manager = CallbackManager([tracer])

agent = initialize_agent(
    tools=[loan_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager=manager,
    verbose=True
)

# After a run:
for span in tracer.get_finished_spans():
    print(span.to_dict())
```

- Inspect each LLM call, prompt, response, and tool call  
- Replay and diagnose where the logic deviated  

### 4.4 Sentry Integration for Runtime Monitoring  

```python
# at top of your script
import sentry_sdk
sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), traces_sample_rate=1.0)

# In classify_loan or agent code
try:
    # classification logic...
except Exception as e:
    sentry_sdk.capture_exception(e)
    raise
```

- View exceptions, stack traces, and request context in Sentry dashboard  

### 4.5 Interactive Debugging in Jupyter  

- Use `%debug` magic after an exception to step into code  
- Insert `breakpoint()` calls in your agent functions  
- Inspect local variables, call stack, and tool inputs  

---

## Part 5: Simulating and Fixing Errors  

1. **Simulate a Timeout:**  
   ```python
   import time
   def flaky_tool(x): 
       time.sleep(10)  # exceed default timeout
       return "done"
   ```
   - Observe how agent hangs; add `timeout` in requests or LLM call.

2. **Simulate Missing Tool:**  
   ```python
   tools = []  # no tools passed
   # agent.run will raise an error that no tool can handle the request
   ```
   - Catch and provide a fallback in your agent loop.

3. **Hallucination Handling:**  
   - Post-process LLM output with simple validators  
   - e.g., ensure classification outputs only “True”/“False”  

---
