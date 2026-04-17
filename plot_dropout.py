import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example data (replace with your actual data)
data_acc = {
    "Dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "REST16": [91.23, 91.23, 90.26, 90.10, 92.05, 91.23, 91.23, 90.42],
    "REST15": [85.98, 84.13, 83.76, 84.50, 84.13, 83.39, 84.13, 85.06],
    "REST14": [85.09, 86.16, 85.27, 85.45, 85.80, 86.34, 85.89, 86.43],
    "LAP14": [78.21, 77.59, 77.90, 79.31, 77.43, 79.62, 78.53, 77.12],
}

data_f1 = {
    "Dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "REST16": [77.06, 76.77, 69.09, 71.03, 74.11, 73.42, 76.76, 74.26],
    "REST15": [73.49, 64.90, 60.38, 67.62, 71.74, 59.82, 67.51, 74.11],
    "REST14": [78.91, 79.22, 79.44, 78.10, 80.08, 79.66, 79.13, 80.23],
    "LAP14": [74.73, 73.00, 74.04, 75.83, 72.75, 75.53, 75.15, 71.15],
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
    x="Dropout", y="Accuracy", marker="D", label="REST14", color="#ffc740", markersize=10
)
sns.lineplot(
    data=melted_acc[melted_acc["Dataset"] == "LAP14"],
    x="Dropout", y="Accuracy", marker="^", label="LAP14", color="#76CF50", markersize=10
)
plt.title("Effect of the parameter Dropout", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Dropout", fontsize=12)
plt.legend(title="Dataset", loc = "lower right", bbox_to_anchor=(1.125, 0.751))
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
plt.title("Effect of the parameter Dropout", fontsize=14)
plt.ylabel("F1 (%)", fontsize=12)
plt.xlabel("Dropout", fontsize=12)
plt.legend(title="Dataset", loc = "lower right", bbox_to_anchor=(1.125, 0.751))
plt.grid(False)
plt.show()