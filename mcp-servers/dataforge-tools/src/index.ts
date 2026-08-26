import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import * as path from "path";
import { fileURLToPath } from "url";
import { inspectDataset } from "./tools/dataset-inspector.js";
import { runPythonInSandbox } from "./tools/sandbox-runner.js";
import { benchmarkMLModels } from "./tools/model-evaluator.js";
import { generateExecutiveBrief } from "./tools/consultant-brief.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Root of the hackathon repository
const REPO_ROOT = path.resolve(__dirname, "../../..");

const server = new Server(
  {
    name: "dataforge-tools",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Register Tool Definitions
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "inspect_dataset",
        description:
          "Inspects a dataset file (CSV), returning metadata, column types, missing value percentages, statistical summaries, sample rows, and data hygiene alerts.",
        inputSchema: {
          type: "object",
          properties: {
            filePath: {
              type: "string",
              description: "Relative or absolute path to the dataset file (e.g. 'datasets/saas_customer_churn.csv')",
            },
          },
          required: ["filePath"],
        },
      },
      {
        name: "run_sandbox_python",
        description:
          "Executes Python data science & visualization code inside an isolated sandbox environment. Automatically captures stdout, stderr, and any matplotlib/seaborn plots.",
        inputSchema: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description: "The complete Python script to execute (using pandas, numpy, scikit-learn, matplotlib, seaborn).",
            },
            description: {
              type: "string",
              description: "Brief summary of what this code does (e.g. 'Plotting churn rate by contract tier').",
            },
          },
          required: ["code"],
        },
      },
      {
        name: "benchmark_ml_models",
        description:
          "Runs automated machine learning benchmarking in the sandbox (Random Forest, Gradient Boosting, Logistic/Linear Regression, Decision Trees), computes evaluation metrics, extracts top feature drivers, and generates visualization plots.",
        inputSchema: {
          type: "object",
          properties: {
            datasetPath: {
              type: "string",
              description: "Path to dataset file (e.g. 'datasets/saas_customer_churn.csv')",
            },
            targetColumn: {
              type: "string",
              description: "Name of target column to predict (e.g. 'churned')",
            },
            taskType: {
              type: "string",
              enum: ["classification", "regression"],
              description: "Whether the task is classification or regression (default: classification)",
            },
            missingStrategy: {
              type: "string",
              enum: ["impute", "drop"],
              description: "How to handle missing values: 'impute' (median/mode + indicator flags) or 'drop' (submodel on complete rows only). Default: 'impute'",
            },
          },
          required: ["datasetPath", "targetColumn"],
        },
      },
      {
        name: "generate_executive_brief",
        description:
          "Compiles an executive-grade Data Science brief in Markdown format with key findings, visual chart links, and prioritized business recommendations.",
        inputSchema: {
          type: "object",
          properties: {
            businessGoal: {
              type: "string",
              description: "The primary business goal (e.g. 'Reduce customer churn by 15%')",
            },
            keyTakeaways: {
              type: "array",
              items: { type: "string" },
              description: "3-4 concise high-level executive takeaways",
            },
            findingsBySection: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  title: { type: "string" },
                  description: { type: "string" },
                  charts: { type: "array", items: { type: "string" } },
                },
                required: ["title", "description"],
              },
            },
            prescriptiveRecommendations: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  action: { type: "string" },
                  expectedImpact: { type: "string" },
                  priority: { type: "string", enum: ["HIGH", "MEDIUM", "LOW"] },
                },
                required: ["action", "expectedImpact", "priority"],
              },
            },
            dataQualityNotes: {
              type: "array",
              items: { type: "string" },
              description: "Optional methodology or data quality remarks",
            },
          },
          required: ["businessGoal", "keyTakeaways", "findingsBySection", "prescriptiveRecommendations"],
        },
      },
    ],
  };
});

// Handle Tool Execution Requests
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    if (name === "inspect_dataset") {
      const filePath = String(args?.filePath);
      const absPath = path.isAbsolute(filePath) ? filePath : path.join(REPO_ROOT, filePath);
      const result = await inspectDataset(absPath, REPO_ROOT);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    }

    if (name === "run_sandbox_python") {
      const code = String(args?.code);
      const result = await runPythonInSandbox(code, REPO_ROOT);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    }

    if (name === "benchmark_ml_models") {
      const datasetPath = String(args?.datasetPath);
      const absDatasetPath = path.isAbsolute(datasetPath) ? datasetPath : path.join(REPO_ROOT, datasetPath);
      const targetColumn = String(args?.targetColumn);
      const taskType = (args?.taskType as "classification" | "regression") || "classification";
      const missingStrategy = (args?.missingStrategy as "impute" | "drop") || "impute";
      const result = await benchmarkMLModels(absDatasetPath, targetColumn, taskType, REPO_ROOT, missingStrategy);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    }

    if (name === "generate_executive_brief") {
      const result = generateExecutiveBrief(args as any, REPO_ROOT);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    }

    throw new Error(`Unknown tool: ${name}`);
  } catch (error: any) {
    return {
      isError: true,
      content: [
        {
          type: "text",
          text: `Error executing ${name}: ${error?.message || String(error)}`,
        },
      ],
    };
  }
});

async function run() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("DataForge Tools MCP Server running on stdio");
}

run().catch((err) => {
  console.error("Fatal server error:", err);
  process.exit(1);
});
