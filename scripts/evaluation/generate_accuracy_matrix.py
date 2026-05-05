import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

models = ["base", "baseline_lora", "ensemble_lora", "custom_ensemble_lora"]
prompts = ["baseline", "ensemble", "custom_ensemble"]

# Initialize a dictionary to store accuracies
matrix = {m: {p: "N/A" for p in prompts} for m in models}

# Regex to find the accuracy line
acc_pattern = re.compile(r"Overall Top-1 Accuracy:\s+([\d\.]+)%")

print(f"Scanning reports directory: {REPORTS_DIR}\n")

for m in models:
    for p in prompts:
        report_path = os.path.join(REPORTS_DIR, m, f"{p}_prompts", "classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                content = f.read()
                match = acc_pattern.search(content)
                if match:
                    matrix[m][p] = f"{match.group(1)}%"

# --- Formatting the Output as a Markdown Table ---

col_width = 22
header = "| " + "Model \\ Prompt".ljust(col_width) + " |"
for p in prompts:
    header += f" {p.ljust(col_width)} |"

separator = "|" + "-" * (col_width + 2) + "|"
for _ in prompts:
    separator += "-" * (col_width + 2) + "|"

print("### Accuracy Matrix\n")
print(header)
print(separator)

for m in models:
    row = f"| {m.ljust(col_width)} |"
    for p in prompts:
        val = matrix[m][p]
        row += f" {val.ljust(col_width)} |"
    print(row)

print("\n(N/A means the report file was not found)")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Create a numeric matrix for seaborn
numeric_matrix = []
for m in models:
    row = []
    for p in prompts:
        val_str = matrix[m][p]
        if val_str != "N/A":
            row.append(float(val_str.strip('%')))
        else:
            row.append(np.nan)
    numeric_matrix.append(row)

numeric_matrix = np.array(numeric_matrix)

# Plot the heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    numeric_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    xticklabels=prompts,
    yticklabels=models,
    cbar_kws={'label': 'Accuracy (%)'},
    vmin=60.0,  # Setting minimum around base model accuracy
    vmax=100.0
)

# Append % to the annotations
for t in ax.texts:
    if t.get_text() != "nan":
        t.set_text(t.get_text() + "%")

plt.title("Model vs Prompt Type Accuracy Matrix", fontsize=16, pad=20)
plt.xlabel("Prompt Type", fontsize=12)
plt.ylabel("Model Type", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

save_path = os.path.join(REPORTS_DIR, "accuracy_matrix_heatmap.png")
plt.savefig(save_path, dpi=300)
print(f"Saved heatmap visualization to: {save_path}")
