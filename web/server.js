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

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));
app.use("/outputs", express.static(path.join(REPO_ROOT, "outputs")));
app.use("/datasets", express.static(path.join(REPO_ROOT, "datasets")));

const upload = multer({ dest: path.join(REPO_ROOT, "datasets", "uploads") });

// 1. List Available Datasets
app.get("/api/datasets", (req, res) => {
  try {
    const datasetsDir = path.join(REPO_ROOT, "datasets");
    const files = fs.readdirSync(datasetsDir).filter(f => f.endsWith(".csv"));
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
    const { datasetPath } = req.body;
    const absPath = path.isAbsolute(datasetPath) ? datasetPath : path.join(REPO_ROOT, datasetPath);
    const meta = await inspectDataset(absPath, REPO_ROOT);
    res.json({ success: true, data: meta });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 3. Run Sandboxed EDA
app.post("/api/run-eda", async (req, res) => {
  try {
    const { datasetPath } = req.body;
    const pyScript = `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")
df = pd.read_csv("${datasetPath}")

# Plot 1: Target distribution by contract type if churn
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

# Plot 2: Correlation / Drivers
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
`;
    const result = await runPythonInSandbox(pyScript, REPO_ROOT);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 4. Benchmark ML Models
app.post("/api/benchmark-ml", async (req, res) => {
  try {
    const { datasetPath, targetColumn, taskType } = req.body;
    const absPath = path.isAbsolute(datasetPath) ? datasetPath : path.join(REPO_ROOT, datasetPath);
    const result = await benchmarkMLModels(absPath, targetColumn || "churned", taskType || "classification", REPO_ROOT);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 5. Generate Executive Brief
app.post("/api/generate-brief", (req, res) => {
  try {
    const briefInput = req.body;
    const result = generateExecutiveBrief(briefInput, REPO_ROOT);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`\n🚀 DataForge AI Executive Web Dashboard running at: http://localhost:${PORT}\n`);
});
