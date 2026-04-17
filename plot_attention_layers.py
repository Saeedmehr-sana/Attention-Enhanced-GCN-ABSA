
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data (replace with your actual data)
data_acc = {
    "Dropout": [1, 2, 3, 4, 5, 6, 7, 8],
    "REST16": [90.91, 91.40, 91.40, 90.75, 89.61, 91.40, 89.94, 89.77],
    "REST15": [81.73, 81.00, 83.76, 85.42, 83.58, 84.13, 84.13, 83.95],
    "REST14": [85.45, 84.91, 86.16, 85.18, 84.64, 84.73, 85.54, 84.82],
    "LAP14": [78.53, 78.37, 77.27, 77.74, 77.27, 77.43, 78.06, 78.21],
}

data_f1 = {
    "Dropout": [1, 2, 3, 4, 5, 6, 7, 8],
    "REST16": [73.72, 74.11, 75.55, 75.14, 71.35, 77.52, 73.05, 60.63],
    "REST15": [55.50, 54.55, 65.49, 68.64, 68.74, 72.89, 72.89, 67.88],
    "REST14": [79.95, 78.38, 78.70, 77.84, 77.63, 76.74, 77.69, 77.69],
    "LAP14": [74.24, 74.23, 73.82, 73.87, 73.16, 72.99, 72.29, 75.53],
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
plt.legend(title="Dataset")
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
plt.legend(title="Dataset")
plt.grid(False)
plt.show()
