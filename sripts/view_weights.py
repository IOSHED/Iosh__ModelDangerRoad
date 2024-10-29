import torch
import seaborn as sns
import matplotlib.pyplot as plt

weights = torch.load('../output/weights.csv', weights_only=False)

layer_name = 'fc1.weight'
layer_weights = weights[layer_name].cpu().detach().numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(layer_weights, annot=False, cmap='coolwarm')
plt.title(f'Weights of {layer_name}')
plt.show()
