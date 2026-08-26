import { runPythonInSandbox } from "./sandbox-runner.js";

export interface ModelBenchmarkResult {
  targetColumn: string;
  taskType: "classification" | "regression";
  modelsEvaluated: Array<{
    name: string;
    metrics: Record<string, number>;
    rank: number;
  }>;
  bestModel: {
    name: string;
    primaryMetric: string;
    score: number;
    businessInterpretation: string;
  };
  topFeatures: Array<{
    feature: string;
    importance: number;
    directionOrImpact: string;
  }>;
  visualizations: Array<{
    filename: string;
    filepath: string;
    type: string;
  }>;
}

export async function benchmarkMLModels(
  datasetPath: string,
  targetColumn: string,
  taskType: "classification" | "regression" = "classification",
  repoRoot: string
): Promise<ModelBenchmarkResult> {
  const pythonScript = `
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

df = pd.read_csv("${datasetPath}")
target = "${targetColumn}"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")

# Drop ID columns or target missing rows
df = df.dropna(subset=[target])
X = df.drop(columns=[target])
y = df[target]

# Drop obvious ID columns (e.g., customer_id, id)
id_cols = [c for c in X.columns if 'id' in c.lower() or 'key' in c.lower() or X[c].nunique() == len(X)]
if id_cols:
    X = X.drop(columns=id_cols)

# Separate numeric and categorical
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

task = "${taskType}"
is_classif = task == "classification"

if is_classif:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
    }
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge()
    }

evaluated = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ('pre', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    fitted_pipelines[name] = pipe

    if is_classif:
        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        
        auc = None
        if hasattr(pipe.named_steps['model'], "predict_proba") and len(np.unique(y_test)) == 2:
            try:
                y_prob = pipe.predict_proba(X_test)[:, 1]
                auc = round(roc_auc_score(y_test, y_prob), 4)
            except:
                pass
        
        metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}
        if auc is not None:
            metrics["ROC-AUC"] = auc
        score_for_rank = auc if auc is not None else f1
    else:
        r2 = round(r2_score(y_test, y_pred), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
        mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
        metrics = {"R2 Score": r2, "RMSE": rmse, "MAE": mae}
        score_for_rank = r2

    evaluated.append({
        "name": name,
        "metrics": metrics,
        "score_for_rank": score_for_rank
    })

# Rank models
evaluated.sort(key=lambda x: x["score_for_rank"], reverse=True)
for i, item in enumerate(evaluated):
    item["rank"] = i + 1

best_entry = evaluated[0]
best_name = best_entry["name"]
best_pipe = fitted_pipelines[best_name]

# Extract feature importances from best model
feature_names = []
if num_cols:
    feature_names.extend(num_cols)
if cat_cols:
    cat_ohe = best_pipe.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
    feature_names.extend(cat_ohe.get_feature_names_out(cat_cols).tolist())

importances = []
model_obj = best_pipe.named_steps['model']
if hasattr(model_obj, "feature_importances_"):
    raw_imps = model_obj.feature_importances_
    for fn, imp in zip(feature_names, raw_imps):
        importances.append({"feature": fn, "importance": round(float(imp), 4)})
    importances.sort(key=lambda x: x["importance"], reverse=True)
elif hasattr(model_obj, "coef_"):
    coefs = np.abs(model_obj.coef_).ravel()
    for fn, c in zip(feature_names, coefs):
        importances.append({"feature": fn, "importance": round(float(c), 4)})
    importances.sort(key=lambda x: x["importance"], reverse=True)

top_features = importances[:8]
for tf in top_features:
    tf["directionOrImpact"] = "Primary driver with highest relative predictive weight."

# Generate Visualizations (Plots are automatically saved by executor hook)
# 1. Feature Importance Plot
plt.figure(figsize=(10, 5))
feat_df = pd.DataFrame(top_features)
if not feat_df.empty:
    sns.barplot(data=feat_df, x="importance", y="feature", palette="Blues_r")
    plt.title(f"Top Predictive Drivers ({best_name})", fontsize=14, pad=15)
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# 2. Confusion Matrix Plot (if classification)
if is_classif:
    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix ({best_name})", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

# Plain English interpretation for business owner
if is_classif:
    best_metric_val = best_entry['metrics'].get('ROC-AUC', best_entry['metrics'].get('F1 Score'))
    best_metric_str = "ROC-AUC: " + str(best_metric_val)
    interp = "The " + best_name + " model achieved highest performance (" + best_metric_str + "). Strongest drivers: " + ", ".join([f['feature'] for f in top_features[:3]])
else:
    interp = "The " + best_name + " model explained " + str(round(best_entry['metrics']['R2 Score'] * 100, 1)) + "% of variance with an average error of " + str(best_entry['metrics']['MAE'])

final_data = {
    "targetColumn": target,
    "taskType": task,
    "modelsEvaluated": [{
        "name": m["name"],
        "metrics": m["metrics"],
        "rank": m["rank"]
    } for m in evaluated],
    "bestModel": {
        "name": best_name,
        "primaryMetric": "ROC-AUC" if is_classif else "R2 Score",
        "score": best_entry["score_for_rank"],
        "businessInterpretation": interp
    },
    "topFeatures": top_features,
    "visualizations": []
}

print("__ML_JSON_START__" + json.dumps(final_data) + "__ML_JSON_END__")
`;

  const runRes = await runPythonInSandbox(pythonScript, repoRoot);
  if (!runRes.success && runRes.error) {
    throw new Error(`Failed to benchmark ML models: ${runRes.error}\n${runRes.stderr}`);
  }

  const match = runRes.stdout.match(/__ML_JSON_START__(.*?)__ML_JSON_END__/s);
  if (!match) {
    throw new Error(`Could not parse ML benchmark output:\n${runRes.stdout}`);
  }

  const parsed: ModelBenchmarkResult = JSON.parse(match[1]);
  parsed.visualizations = runRes.plots.map((p, idx) => ({
    filename: p.filename,
    filepath: p.filepath,
    type: idx === 0 ? "feature_importance" : "confusion_matrix",
  }));

  return parsed;
}
