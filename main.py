import ast

import torch
import toml
import pandas as pd
from src.accident_model import AccidentPredictionModel


def load_toml_data(toml_file_path):
    data = toml.load(toml_file_path)
    p_data = {
        'statistics_accidents_on_road': data['statistics_accidents_on_road'],
        'coating_humidity': data['coating_humidity'],
        'quality_coating': data['quality_coating'],
        'traffic_congestion': data['traffic_congestion'],
        'width_road': data['width_road'],
        'allowed_max_speed': data['allowed_max_speed'],
        'allowed_min_speed': data['allowed_min_speed'],
        'turns_route': data['turns_route'],
        'tangles_turns_on_route':
            sum(ast.literal_eval(data['tangles_turns_on_route'])) /
            len(ast.literal_eval(data['tangles_turns_on_route'])) if data['tangles_turns_on_route'] else 0,
        'has_marking_applied': data['has_marking_applied'],
        'confidence_on_road': data['our_driver']['confidence_on_road'],
        'driving_experience': data['our_driver']['driving_experience'],
        'age': data['our_driver']['age'],
        'total_traffic_violations': data['our_driver']['total_traffic_violations'],
        'total_offenses': data['our_driver']['total_offenses'],
        'number_accidents': data['our_driver']['number_accidents'],
        'marginality_population': data['other_factors']['marginality_population'],
        'wind_direction': 1 if data['other_factors']['wind']['direction'] in ['left', 'right'] else 0,
        'wind_speed': data['other_factors']['wind']['speed'],
    }
    print(p_data)
    return pd.DataFrame([p_data])


if __name__ == '__main__':
    toml_file = 'input/characteristics_road.toml'
    weights_file = 'output/weights.csv'

    toml_data = load_toml_data(toml_file)
    toml_data = toml_data.astype(float)

    input_size = toml_data.shape[1]
    model = AccidentPredictionModel(input_size)

    model.load_state_dict(torch.load(weights_file, weights_only=True))

    input_tensor = torch.tensor(toml_data.values, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_class = prediction.round()
        print(f'Predicted class for TOML data: {predicted_class.item()}')
