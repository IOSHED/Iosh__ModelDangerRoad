import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('../dataset/road_accidents_data_with_answer_4.csv')

output_dir = '../output/graph'
os.makedirs(output_dir, exist_ok=True)

sns.set(style="whitegrid")


plt.figure(figsize=(12, 8))

# График 1: Сравнение 'statistics_accidents_on_road' и 'coating_humidity'
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='statistics_accidents_on_road', y='coating_humidity', hue='accident_occurred')
plt.title('Accidents on Road vs. Coating Humidity')
plt.savefig(os.path.join(output_dir, 'scatter_accidents_vs_humidity.png'))

# График 2: Сравнение 'quality_coating' и 'traffic_congestion'
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='quality_coating', y='traffic_congestion', hue='accident_occurred')
plt.title('Quality Coating vs. Traffic Congestion')
plt.savefig(os.path.join(output_dir, 'scatter_quality_vs_congestion.png'))

# График 3: Сравнение 'width_road' и 'allowed_max_speed'
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='width_road', y='allowed_max_speed', hue='accident_occurred')
plt.title('Width of Road vs. Allowed Max Speed')
plt.savefig(os.path.join(output_dir, 'scatter_width_vs_speed.png'))

plt.tight_layout()
plt.show()
