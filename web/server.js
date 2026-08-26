import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import multer from "multer";

import { inspectDataset } from "../mcp-servers/dataforge-tools/dist/tools/dataset-inspector.js";
import { runPythonInSandbox } from "../mcp-servers/dataforge-tools/dist/tools/sandbox-runner.js";
import { benchmarkMLModels } from "../mcp-servers/dataforge-tools/dist/tools/model-evaluator.js";
import { generateExecutiveBrief } from "../mcp-servers/dataforge-tools/dist/tools/consultant-brief.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");
const DATASETS_DIR = path.resolve(REPO_ROOT, "datasets");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));
app.use("/outputs", express.static(path.join(REPO_ROOT, "outputs")));
app.use("/datasets", express.static(DATASETS_DIR));

const upload = multer({ dest: path.join(DATASETS_DIR, "uploads") });

/**
 * Validates and resolves a dataset path strictly within the allowed datasets directory.
 * Prevents arbitrary file reads and directory traversal attacks.
 */
function resolveSafeDatasetPath(clientPath) {
  if (!clientPath || typeof clientPath !== "string") {
    throw new Error("Invalid or missing datasetPath");
  }

  // If path starts with "datasets/", remove prefix to resolve against DATASETS_DIR
  const normalizedRel = clientPath.replace(/^datasets\//, "");
  const resolved = path.resolve(DATASETS_DIR, normalizedRel);

  // Security boundary check: must be strictly inside DATASETS_DIR
  if (!resolved.startsWith(DATASETS_DIR)) {
    throw new Error("Access Denied: Path traversal outside datasets directory is restricted.");
  }

  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`Dataset file not found: ${path.basename(resolved)}`);
  }

  return resolved;
}

// 1. List Available Datasets
app.get("/api/datasets", (req, res) => {
  try {
    const files = fs.readdirSync(DATASETS_DIR).filter(f => f.endsWith(".csv"));
    const list = files.map(f => ({
      filename: f,
      relativePath: `datasets/${f}`,
      displayName: f === "saas_customer_churn.csv" ? "SaaS Customer Churn (1,200 accounts)" :
                   f === "retail_sales_forecast.csv" ? "Retail Sales & Promotions (11,680 rows)" : f
    }));
    res.json({ success: true, datasets: list });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 2. Inspect Dataset
app.post("/api/inspect", async (req, res) => {
  try {
    const safePath = resolveSafeDatasetPath(req.body.datasetPath);
    const meta = await inspectDataset(safePath, REPO_ROOT);
    res.json({ success: true, data: meta });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

// 3. Run Sandboxed EDA with dynamically calculated real statistics
app.post("/api/run-eda", async (req, res) => {
  try {
    const safePath = resolveSafeDatasetPath(req.body.datasetPath);
    const safeLiteral = JSON.stringify(safePath);

    const pyScript = `
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")
df = pd.read_csv(${safeLiteral})

insights = []

# Dynamic EDA analysis based on actual columns present
if "contract_type" in df.columns and "churned" in df.columns:
    plt.figure(figsize=(7, 4.5))
    churn_by_contract = df.groupby("contract_type")["churned"].mean().reset_index()
    churn_by_contract["churn_rate_pct"] = churn_by_contract["churned"] * 100
    ax = sns.barplot(data=churn_by_contract, x="contract_type", y="churn_rate_pct", palette="Blues_r")
    plt.title("Customer Churn Rate by Contract Type", fontsize=13, fontweight="bold")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate (%)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.show()

    m2m_rate = churn_by_contract.loc[churn_by_contract['contract_type'] == 'Month-to-Month', 'churn_rate_pct'].values
    ann_rate = churn_by_contract.loc[churn_by_contract['contract_type'] == 'One Year', 'churn_rate_pct'].values
    if len(m2m_rate) > 0 and len(ann_rate) > 0:
        insights.append(f"Contract Disparity: Month-to-Month accounts exhibit a {m2m_rate[0]:.1f}% churn rate, vs {ann_rate[0]:.1f}% for Annual accounts.")

if "support_tickets_count" in df.columns and "churned" in df.columns:
    plt.figure(figsize=(7, 4.5))
    churn_by_tickets = df.groupby("support_tickets_count")["churned"].mean().reset_index()
    churn_by_tickets["churn_rate_pct"] = churn_by_tickets["churned"] * 100
    sns.barplot(data=churn_by_tickets, x="support_tickets_count", y="churn_rate_pct", palette="Reds_d")
    plt.title("Churn Rate vs Support Ticket Frequency", fontsize=13, fontweight="bold")
    plt.xlabel("Support Tickets in First 90 Days")
    plt.ylabel("Churn Rate (%)")
    plt.tight_layout()
    plt.show()

    low_ticket_churn = df[df['support_tickets_count'] <= 1]['churned'].mean() * 100
    high_ticket_churn = df[df['support_tickets_count'] >= 3]['churned'].mean() * 100
    if not np.isnan(low_ticket_churn) and not np.isnan(high_ticket_churn):
        ratio = (high_ticket_churn / low_ticket_churn) if low_ticket_churn > 0 else 1.0
        insights.append(f"Support Escalation Risk: Accounts with >= 3 tickets churn at {high_ticket_churn:.1f}% ({ratio:.1f}x higher than accounts with <= 1 ticket at {low_ticket_churn:.1f}%).")

if "category" in df.columns and "revenue_usd" in df.columns:
    plt.figure(figsize=(7, 4.5))
    rev_by_cat = df.groupby("category")["revenue_usd"].sum().reset_index().sort_values(by="revenue_usd", ascending=False)
    sns.barplot(data=rev_by_cat, x="category", y="revenue_usd", palette="Blues_r")
    plt.title("Total Revenue by Product Category", fontsize=13, fontweight="bold")
    plt.xlabel("Product Category")
    plt.ylabel("Total Revenue (USD)")
    plt.tight_layout()
    plt.show()

    top_cat = rev_by_cat.iloc[0]
    total_rev = rev_by_cat['revenue_usd'].sum()
    top_share = (top_cat['revenue_usd'] / total_rev * 100) if total_rev > 0 else 0
    insights.append(f"Top Category Driver: '{top_cat['category']}' generated ${top_cat['revenue_usd']:,.0f} ({top_share:.1f}% of total recorded revenue).")

if "is_promotion" in df.columns and "revenue_usd" in df.columns:
    plt.figure(figsize=(7, 4.5))
    promo_rev = df.groupby("is_promotion")["revenue_usd"].mean().reset_index()
    promo_rev["Promo"] = promo_rev["is_promotion"].map({0: "Regular Days", 1: "Promotional Days"})
    sns.barplot(data=promo_rev, x="Promo", y="revenue_usd", palette="Greens_d")
    plt.title("Average Daily Revenue: Promo vs Regular", fontsize=13, fontweight="bold")
    plt.xlabel("Day Type")
    plt.ylabel("Average Daily Revenue (USD)")
    plt.tight_layout()
    plt.show()

    regular_avg = df[df['is_promotion'] == 0]['revenue_usd'].mean()
    promo_avg = df[df['is_promotion'] == 1]['revenue_usd'].mean()
    if not np.isnan(regular_avg) and not np.isnan(promo_avg) and regular_avg > 0:
        lift = ((promo_avg - regular_avg) / regular_avg) * 100
        insights.append(f"Promotion Impact: Active promotional campaigns generate ${promo_avg:,.2f} average daily revenue (+{lift:.1f}% lift over regular baseline of ${regular_avg:,.2f}).")

# Fallback generic insight if custom columns not matched
if not insights:
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        plt.figure(figsize=(7, 4.5))
        df[num_cols[:3]].hist(figsize=(7, 4.5))
        plt.tight_layout()
        plt.show()
        insights.append(f"Exploratory profiling completed across {len(df)} records and {len(df.columns)} features.")

print("__EDA_JSON_START__" + json.dumps(insights) + "__EDA_JSON_END__")
`;
    const result = await runPythonInSandbox(pyScript, REPO_ROOT);
    let dynamicInsights = [];
    const match = result.stdout.match(/__EDA_JSON_START__(.*?)__EDA_JSON_END__/s);
    if (match) {
      dynamicInsights = JSON.parse(match[1]);
    }
    res.json({ success: true, data: { ...result, insights: dynamicInsights } });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

// 4. Benchmark ML Models
app.post("/api/benchmark-ml", async (req, res) => {
  try {
    const { datasetPath, targetColumn, taskType, missingStrategy } = req.body;
    const safePath = resolveSafeDatasetPath(datasetPath);
    const result = await benchmarkMLModels(safePath, targetColumn, taskType, REPO_ROOT, missingStrategy || "impute");
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

// 5. Generate Executive Brief
app.post("/api/generate-brief", (req, res) => {
  try {
    const briefInput = req.body;
    const result = generateExecutiveBrief(briefInput, REPO_ROOT);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`\n🚀 DataForge AI Executive Web Dashboard running at: http://localhost:${PORT}\n`);
});
