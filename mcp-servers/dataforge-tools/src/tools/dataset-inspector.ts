import { runPythonInSandbox } from "./sandbox-runner.js";

export interface DatasetInspectionResult {
  filePath: string;
  totalRows: number;
  totalColumns: number;
  columns: Array<{
    name: string;
    type: string;
    nullCount: number;
    nullPercentage: number;
    uniqueCount: number;
    sampleValues: any[];
  }>;
  numericSummary: Record<string, any>;
  dataHygieneAlerts: Array<{
    severity: "HIGH" | "MEDIUM" | "INFO";
    column?: string;
    message: string;
    recommendation: string;
  }>;
  previewRows: any[];
}

export async function inspectDataset(
  filePath: string,
  repoRoot: string
): Promise<DatasetInspectionResult> {
  const pythonScript = `
import json
import pandas as pd
import numpy as np

file_path = "${filePath}"
df = pd.read_csv(file_path)

total_rows, total_cols = df.shape
columns_info = []
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_summary = {}

if numeric_cols:
    desc = df[numeric_cols].describe().to_dict()
    for col, stats in desc.items():
        numeric_summary[col] = {
            "mean": round(float(stats.get("mean", 0)), 2) if not pd.isna(stats.get("mean")) else None,
            "std": round(float(stats.get("std", 0)), 2) if not pd.isna(stats.get("std")) else None,
            "min": round(float(stats.get("min", 0)), 2) if not pd.isna(stats.get("min")) else None,
            "median": round(float(df[col].median()), 2) if not pd.isna(df[col].median()) else None,
            "max": round(float(stats.get("max", 0)), 2) if not pd.isna(stats.get("max")) else None,
        }

alerts = []

for col in df.columns:
    null_count = int(df[col].isnull().sum())
    null_pct = round((null_count / total_rows) * 100, 2)
    unique_count = int(df[col].nunique(dropna=True))
    sample_vals = df[col].dropna().head(3).tolist()

    columns_info.append({
        "name": col,
        "type": str(df[col].dtype),
        "nullCount": null_count,
        "nullPercentage": null_pct,
        "uniqueCount": unique_count,
        "sampleValues": sample_vals
    })

    # Data Hygiene checks
    if null_pct > 20.0:
        alerts.append({
            "severity": "HIGH",
            "column": col,
            "message": f"Column '{col}' has {null_pct}% missing values.",
            "recommendation": "Consult with business owner: Impute with median/mode + missing indicator, or evaluate if missingness is informative (MNAR)."
        })
    elif null_pct > 0.0:
        alerts.append({
            "severity": "MEDIUM",
            "column": col,
            "message": f"Column '{col}' has {null_pct}% missing values ({null_count} rows).",
            "recommendation": "Standard imputation (median for skewed numeric, mode for categorical) is safe."
        })

    if unique_count == 1:
        alerts.append({
            "severity": "MEDIUM",
            "column": col,
            "message": f"Column '{col}' has only 1 unique value (zero variance).",
            "recommendation": "Drop this column as it provides zero predictive signal."
        })
    elif unique_count == total_rows and df[col].dtype == "object":
        alerts.append({
            "severity": "INFO",
            "column": col,
            "message": f"Column '{col}' appears to be a unique ID or hash identifier.",
            "recommendation": "Exclude from ML training to avoid memorization / overfitting."
        })

# Preview top 5 rows
preview = df.head(5).replace({np.nan: None}).to_dict(orient="records")

result = {
    "filePath": file_path,
    "totalRows": total_rows,
    "totalColumns": total_cols,
    "columns": columns_info,
    "numericSummary": numeric_summary,
    "dataHygieneAlerts": alerts,
    "previewRows": preview
}

print("__JSON_START__" + json.dumps(result) + "__JSON_END__")
`;

  const runRes = await runPythonInSandbox(pythonScript, repoRoot);
  if (!runRes.success && runRes.error) {
    throw new Error(`Failed to inspect dataset: ${runRes.error}\n${runRes.stderr}`);
  }

  const match = runRes.stdout.match(/__JSON_START__(.*?)__JSON_END__/s);
  if (!match) {
    throw new Error(`Could not parse dataset metadata output:\n${runRes.stdout}`);
  }

  return JSON.parse(match[1]);
}
