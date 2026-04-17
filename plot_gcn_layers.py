import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_acc = {
    "Dropout": [1, 2, 3, 4, 5, 6, 7, 8],
    "REST16": [91.07, 90.91, 90.42, 90.26, 90.75, 90.75, 91.07, 76.14],
    "REST15": [83.39, 81.73, 80.81, 84.69, 83.95, 84.87, 82.10, 81.18],
    "REST14": [85.00, 85.45, 86.07, 85.80, 84.91, 85.09, 84.38, 85.00],
    "LAP14": [79.94, 78.53, 77.43, 76.80, 78.37, 78.06, 79.15, 57.68],
}

data_f1 = {
    "Dropout": [1, 2, 3, 4, 5, 6, 7, 8],
    "REST16": [71.21, 73.72, 75.15, 72.85, 76.39, 71.02, 73.64, 28.82],
    "REST15": [71.30, 55.50, 54.28, 72.32, 67.94, 72.63, 55.42, 54.83],
    "REST14": [78.49, 79.95, 80.59, 80.99, 78.00, 77.18, 76.18, 77.25],
    "LAP14": [76.18, 74.24, 74.09, 72.13, 74.12, 73.37, 74.43, 35.54],
}

# Convert to DataFrame
df_acc = pd.DataFrame(data_acc)
df_f1 = pd.DataFrame(data_f1)

melted_acc = df_acc.melt(
    id_vars=["Dropout"],
    var_name="Dataset",
    value_name="Accuracy"
)

# Melt the data for plotting
melted_f1 = df_f1.melt(
    id_vars=["Dropout"],
    var_name="Dataset",
    value_name="F1"
)

# Plot Accuracy
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=melted_acc[melted_acc["Dataset"] == "REST16"],
    x="Dropout", y="Accuracy", marker="s", label="REST16", color="#ed6363", markersize=10
)
sns.lineplot(
    data=melted_acc[melted_acc["Dataset"] == "REST15"],
    x="Dropout", y="Accuracy", marker="o", label="REST15", color="#4e93cc", markersize=10
)
sns.lineplot(
    data=melted_acc[melted_acc["Dataset"] == "REST14"],
    x="Dropout", y="Accuracy", marker="D", label="REST14_", color="#ffc740", markersize=10
)
sns.lineplot(
    data=melted_acc[melted_acc["Dataset"] == "LAP14"],
    x="Dropout", y="Accuracy", marker="^", label="LAP14", color="#76CF50", markersize=10
)
plt.title("Effect of the number of GCN Layers", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("GCN Layers", fontsize=12)
plt.legend(title="Dataset", loc = "lower right", bbox_to_anchor=(1.132, 0.751))
plt.grid(False)
plt.show()


# Plot F1
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=melted_f1[melted_f1["Dataset"] == "REST16"],
    x="Dropout", y="F1", marker="s", label="REST16", color="#ed6363", markersize=10
)
sns.lineplot(
    data=melted_f1[melted_f1["Dataset"] == "REST15"],
    x="Dropout", y="F1", marker="o", label="REST15", color="#4e93cc", markersize=10
)
sns.lineplot(
    data=melted_f1[melted_f1["Dataset"] == "REST14"],
    x="Dropout", y="F1", marker="D", label="REST14", color="#ffc740", markersize=10
)
sns.lineplot(
    data=melted_f1[melted_f1["Dataset"] == "LAP14"],
    x="Dropout", y="F1", marker="^", label="LAP14", color="#76CF50", markersize=10
)
plt.title("Effect of the number of GCN Layers", fontsize=14)
plt.ylabel("F1 (%)", fontsize=12)
plt.xlabel("GCN Layers", fontsize=12)
plt.legend(title="Dataset", loc = "lower right", bbox_to_anchor=(1.125, 0.751))
plt.grid(False)
plt.show()
