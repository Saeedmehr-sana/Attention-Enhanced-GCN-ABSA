import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Service was slow, but the people were friendly

# Data for service and food with adjusted values to match the image
data_service = np.array([
    [0.20, 0.10, 0.15, 0.10, 0.25, 0.35, 0.60],  # our model
    [0.10, 0.15, 0.12, 0.20, 0.30, 0.50, 0.40],  # dependency attention
    [0.15, 0.12, 0.18, 0.17, 0.28, 0.45, 0.50]   # fusion attention
])

data_food = np.array([
    [0.25, 0.18, 0.14, 0.16, 0.10, 0.20, 0.18],  # emotional attention
    [0.12, 0.14, 0.12, 0.16, 0.14, 0.18, 0.17],  # dependency attention
    [0.11, 0.16, 0.15, 0.17, 0.14, 0.18, 0.20]   # fusion attention
])

# Labels for the x and y axes
words = ['Great', 'food', 'but', 'the', 'service', 'was', 'dreadful']
attention_types = ['emotional attention', 'dependency attention', 'fusion attention']

# Plot for service
plt.figure(figsize=(10, 5))
sns.heatmap(data_service, cmap='viridis', cbar_kws={'label': 'Attention Score'}, 
            xticklabels=words, yticklabels=attention_types, vmin=0, vmax=0.6)
plt.title('(a) service')
plt.show()

# Plot for food
plt.figure(figsize=(10, 5))
sns.heatmap(data_food, cmap='viridis', cbar_kws={'label': 'Attention Score'}, 
            xticklabels=words, yticklabels=attention_types, vmin=0, vmax=0.2)
plt.title('(b) food')
plt.show()
