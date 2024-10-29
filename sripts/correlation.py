import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('../dataset/road_accidents_data_with_answer_3.csv')

output_dir = '../output/graph'
os.makedirs(output_dir, exist_ok=True)

sns.set(style="whitegrid")

features = [
    'coating_humidity',
    'quality_coating',
    'traffic_congestion',
    'width_road',
    'allowed_max_speed',
    'allowed_min_speed',
    'turns_route',
    'tangles_turns_on_route',
    'has_marking_applied',
    'confidence_on_road',
    'driving_experience',
    'age',
    'total_traffic_violations',
    'total_offenses',
    'marginality_population',
    'wind_speed'
]

for feature in features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='statistics_accidents_on_road', y=feature, hue='accident_occurred')
    plt.title(f'Correlation between Statistics Accidents on Road and {feature}')
    plt.xlabel('Statistics Accidents on Road')
    plt.ylabel(feature)
    plt.savefig(os.path.join(output_dir, f'correlation_accidents_vs_{feature}.png'))
    plt.close()

