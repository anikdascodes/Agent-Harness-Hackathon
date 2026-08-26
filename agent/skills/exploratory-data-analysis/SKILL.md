---
name: exploratory-data-analysis
description: Best practices for writing clean, sandboxed Python scripts to profile distributions, correlation matrices, and generate publication-quality executive charts.
---

# Exploratory Data Analysis (EDA) Protocol

## Principles:
1. **Always execute code inside the sandbox** via `run_sandbox_python`.
2. **Design for Executives:** Charts must have clear titles, labelled axes, readable fonts, and uncluttered palettes (`sns.set_theme(style="whitegrid")`).
3. **Connect Visuals to Business Meaning:** Never output a chart without 1-2 bullet points explaining what it means for revenue, retention, or growth.

## Required Python Patterns:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({'font.size': 11})

# Example: Churn rate by contract type
df = pd.read_csv("datasets/saas_customer_churn.csv")

plt.figure(figsize=(8, 5))
churn_by_contract = df.groupby("contract_type")["churned"].mean().reset_index()
churn_by_contract["churn_rate_pct"] = churn_by_contract["churned"] * 100

ax = sns.barplot(data=churn_by_contract, x="contract_type", y="churn_rate_pct", palette="Blues_r")
plt.title("Customer Churn Rate by Contract Type", fontsize=14, pad=15)
plt.xlabel("Contract Type", fontweight="bold")
plt.ylabel("Churn Rate (%)", fontweight="bold")

# Annotate percentages on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 7), 
                textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.show() # Automatically captured by DataForge sandbox
```
