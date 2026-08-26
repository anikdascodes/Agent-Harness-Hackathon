import assert from "assert";
import path from "path";
import { fileURLToPath } from "url";
import { inspectDataset } from "../mcp-servers/dataforge-tools/dist/tools/dataset-inspector.js";
import { runPythonInSandbox } from "../mcp-servers/dataforge-tools/dist/tools/sandbox-runner.js";
import { benchmarkMLModels } from "../mcp-servers/dataforge-tools/dist/tools/model-evaluator.js";
import { generateExecutiveBrief } from "../mcp-servers/dataforge-tools/dist/tools/consultant-brief.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");

async function runTests() {
  console.log("🧪 Testing DataForge MCP Tools Integration...\n");

  // 1. Test Dataset Inspector
  console.log("1️⃣ Testing inspectDataset()...");
  const churnDataset = path.join(REPO_ROOT, "datasets", "saas_customer_churn.csv");
  const meta = await inspectDataset(churnDataset, REPO_ROOT);
  assert(meta.totalRows === 1200, "Expected 1200 rows in SaaS dataset");
  assert(meta.columns.length === 12, "Expected 12 columns in SaaS dataset");
  assert(meta.dataHygieneAlerts.length > 0, "Expected hygiene alerts for missing data");
  console.log(`✅ inspectDataset passed: ${meta.totalRows} rows, ${meta.columns.length} cols, ${meta.dataHygieneAlerts.length} hygiene alerts.\n`);

  // 2. Test Sandbox Python Runner
  console.log("2️⃣ Testing runPythonInSandbox()...");
  const pyCode = `
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({"category": ["A", "B", "C"], "val": [10, 25, 15]})
plt.figure()
plt.bar(df["category"], df["val"], color="teal")
plt.title("Sample Sandbox Plot")
plt.show()
print("PROCESSED_ROWS=" + str(len(df)))
`;
  const pyRes = await runPythonInSandbox(pyCode, REPO_ROOT);
  assert(pyRes.success === true, "Sandbox execution should succeed");
  assert(pyRes.stdout.includes("PROCESSED_ROWS=3"), "Should output processed rows");
  assert(pyRes.plots.length >= 1, "Should capture at least 1 plot");
  console.log(`✅ runPythonInSandbox passed: captured plot ${pyRes.plots[0].filename} in ${pyRes.execution_duration_seconds}s.\n`);

  // 3. Test ML Model Benchmarking
  console.log("3️⃣ Testing benchmarkMLModels()...");
  const mlRes = await benchmarkMLModels(churnDataset, "churned", "classification", REPO_ROOT);
  assert(mlRes.modelsEvaluated.length === 4, "Expected 4 models evaluated");
  assert(mlRes.bestModel.name !== "", "Expected best model selected");
  assert(mlRes.topFeatures.length > 0, "Expected top features extracted");
  console.log(`✅ benchmarkMLModels passed: Best Model = ${mlRes.bestModel.name} (${mlRes.bestModel.primaryMetric}: ${mlRes.bestModel.score}), Top Feature = ${mlRes.topFeatures[0].feature}.\n`);

  // 4. Test Executive Brief Generator
  console.log("4️⃣ Testing generateExecutiveBrief()...");
  const brief = generateExecutiveBrief({
    businessGoal: "Reduce SaaS Customer Churn",
    keyTakeaways: [
      "Customers on Month-to-Month contracts with >3 support tickets churn at 4.2x average rate.",
      "Random Forest model achieved 0.88+ ROC-AUC for early identification."
    ],
    findingsBySection: [
      {
        title: "Contract Type Impact",
        description: "Month-to-Month contracts represent 55% of all users but 78% of all churn events.",
        charts: mlRes.visualizations.map(v => v.filepath)
      }
    ],
    prescriptiveRecommendations: [
      {
        action: "Deploy automated proactive check-ins for Month-to-Month users after 2 support tickets.",
        expectedImpact: "Estimated 15-22% reduction in early churn.",
        priority: "HIGH"
      }
    ],
    dataQualityNotes: [
      "income_bracket had 20% missing values; median imputation used without data loss."
    ]
  }, REPO_ROOT);

  assert(brief.markdownContent.includes("Executive Data Science Brief"), "Should generate valid markdown");
  console.log(`✅ generateExecutiveBrief passed: saved report to ${brief.reportPath}.\n`);

  console.log("🎉 All DataForge MCP Tools Integration Tests Passed Successfully!");
}

runTests().catch((err) => {
  console.error("❌ Integration test failed:", err);
  process.exit(1);
});
