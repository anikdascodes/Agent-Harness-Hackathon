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
  const safeDatasetLiteral = JSON.stringify(datasetPath);
  const safeTargetLiteral = JSON.stringify(targetColumn);
  const safeTaskLiteral = JSON.stringify(taskType);

  const pythonScript = `
import json
import re
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

dataset_path = ${safeDatasetLiteral}
target = ${safeTargetLiteral}
task = ${safeTaskLiteral}

df = pd.read_csv(dataset_path)

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")

# Drop missing target rows
df = df.dropna(subset=[target])
if df.empty:
    raise ValueError(f"Dataset has no valid rows after dropping missing target '{target}'.")

# Accurate Datetime Detection: only check object/string columns that match date format
date_cols = []
for c in df.select_dtypes(include=['object', 'string']).columns:
    if c != target:
        sample_vals = df[c].dropna().head(10).astype(str)
        # Check for ISO or standard date formats
        if sample_vals.str.match(r'^\d{4}[-/]\d{2}[-/]\d{2}').all() or 'date' in c.lower():
            try:
                parsed_dt = pd.to_datetime(df[c], errors='coerce')
                if parsed_dt.notnull().mean() > 0.8:
                    date_cols.append(c)
            except:
                pass

# If date column present, sort chronologically
if date_cols:
    primary_date_col = date_cols[0]
    df[primary_date_col] = pd.to_datetime(df[primary_date_col], errors='coerce')
    df = df.sort_values(by=primary_date_col).reset_index(drop=True)

X = df.drop(columns=[target])
y = df[target]

# Narrow ID column detection: only drop explicit non-predictive identifiers
id_cols_to_drop = []
for c in X.columns:
    is_id_name = bool(re.match(r'^(id|.*_id|.*_key|uuid)$', c, re.I))
    is_unique_object = X[c].dtype in ['object', 'string'] and X[c].nunique() == len(X)
    if is_id_name and (is_unique_object or X[c].nunique() > len(X) * 0.9):
        id_cols_to_drop.append(c)

if id_cols_to_drop:
    X = X.drop(columns=id_cols_to_drop)

# Expand identified date columns into calendar features with missingness indicator
for dc in date_cols:
    if dc in X.columns:
        dt_series = pd.to_datetime(X[dc], errors='coerce')
        X[f'{dc}_is_missing'] = dt_series.isna().astype(float)
        X[f'{dc}_year'] = dt_series.dt.year.astype(float)
        X[f'{dc}_month'] = dt_series.dt.month.astype(float)
        X[f'{dc}_dayofweek'] = dt_series.dt.dayofweek.astype(float)
        X = X.drop(columns=[dc])

if X.empty or X.shape[1] == 0:
    raise ValueError("No predictive features remaining after preprocessing.")

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

transformers = []
if num_cols:
    transformers.append(('num', num_pipe, num_cols))
if cat_cols:
    transformers.append(('cat', cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers=transformers)

is_classif = task == "classification"

if is_classif:
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"Classification target '{target}' must have at least 2 distinct classes, but found only {len(class_counts)} class: {list(class_counts.index)}")
    
    can_stratify = bool((class_counts >= 2).all())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
    }
else:
    # If date-ordered, preserve sorted chronological sequence; otherwise random split
    has_date_sort = bool(date_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=not has_date_sort
    )
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
            primary_metric_name = "ROC-AUC"
            score_for_rank = auc
        else:
            primary_metric_name = "Weighted F1"
            score_for_rank = f1
    else:
        r2 = round(r2_score(y_test, y_pred), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
        mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
        metrics = {"R2 Score": r2, "RMSE": rmse, "MAE": mae}
        primary_metric_name = "R2 Score"
        score_for_rank = r2

    evaluated.append({
        "name": name,
        "metrics": metrics,
        "primary_metric_name": primary_metric_name,
        "score_for_rank": score_for_rank
    })

# Rank models
evaluated.sort(key=lambda x: x["score_for_rank"], reverse=True)
for i, item in enumerate(evaluated):
    item["rank"] = i + 1

best_entry = evaluated[0]
best_name = best_entry["name"]
best_pipe = fitted_pipelines[best_name]

# Extract feature importances with proper multiclass support
feature_names = []
if num_cols:
    feature_names.extend(num_cols)
if cat_cols and 'cat' in best_pipe.named_steps['pre'].named_transformers_:
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
    raw_coefs = model_obj.coef_
    if raw_coefs.ndim == 2:
        agg_coefs = np.mean(np.abs(raw_coefs), axis=0)
    else:
        agg_coefs = np.abs(raw_coefs).ravel()
    
    for fn, c in zip(feature_names, agg_coefs):
        importances.append({"feature": fn, "importance": round(float(c), 4)})
    importances.sort(key=lambda x: x["importance"], reverse=True)

top_features = importances[:8]
for tf in top_features:
    tf["directionOrImpact"] = "Primary driver with highest relative predictive weight."

# Generate Visualizations
plt.figure(figsize=(10, 5))
feat_df = pd.DataFrame(top_features)
if not feat_df.empty:
    sns.barplot(data=feat_df, x="importance", y="feature", palette="Blues_r")
    plt.title(f"Top Predictive Drivers ({best_name})", fontsize=14, pad=15)
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

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

best_metric_name = best_entry["primary_metric_name"]
best_score = best_entry["score_for_rank"]

if is_classif:
    interp = f"The {best_name} model achieved highest performance ({best_metric_name}: {best_score}). Strongest drivers: " + ", ".join([f['feature'] for f in top_features[:3]])
else:
    interp = f"The {best_name} model explained " + str(round(best_entry['metrics']['R2 Score'] * 100, 1)) + "% of variance with an average prediction error of $" + str(best_entry["metrics"]["MAE"]) + "."

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
        "primaryMetric": best_metric_name,
        "score": best_score,
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
