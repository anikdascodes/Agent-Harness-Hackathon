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

// 3. Run Sandboxed EDA
app.post("/api/run-eda", async (req, res) => {
  try {
    const safePath = resolveSafeDatasetPath(req.body.datasetPath);
    const safeLiteral = JSON.stringify(safePath);

    const pyScript = `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")
df = pd.read_csv(${safeLiteral})

# SaaS Churn Plots
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

# Retail Sales Plots
if "category" in df.columns and "revenue_usd" in df.columns:
    plt.figure(figsize=(7, 4.5))
    rev_by_cat = df.groupby("category")["revenue_usd"].sum().reset_index()
    sns.barplot(data=rev_by_cat, x="category", y="revenue_usd", palette="Blues_r")
    plt.title("Total Revenue by Product Category", fontsize=13, fontweight="bold")
    plt.xlabel("Product Category")
    plt.ylabel("Total Revenue (USD)")
    plt.tight_layout()
    plt.show()

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
`;
    const result = await runPythonInSandbox(pyScript, REPO_ROOT);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
});

// 4. Benchmark ML Models
app.post("/api/benchmark-ml", async (req, res) => {
  try {
    const { datasetPath, targetColumn, taskType } = req.body;
    const safePath = resolveSafeDatasetPath(datasetPath);
    const result = await benchmarkMLModels(safePath, targetColumn, taskType, REPO_ROOT);
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
