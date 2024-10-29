import torch
import logging
import os
import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data_for_training(file_path):
    data = pd.read_csv(file_path)

    if 'tangles_turns_on_route' in data.columns:
        data['avg_turn_angle'] = data['tangles_turns_on_route'].apply(
            lambda x_column: sum(ast.literal_eval(x_column)) / len(ast.literal_eval(x_column)) if x_column else 0
        )
        data = data.drop(columns=['tangles_turns_on_route'])

    if 'wind_direction' in data.columns:
        data['wind_direction'] = data['wind_direction'].apply(
            lambda x_column: 1 if x_column in ['left', 'right'] else 0
        )

    x = data.drop(columns=['accident_occurred'])
    y = data['accident_occurred']
    return x, y


def preprocess_data(x, y, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    return x_train, x_test, y_train, y_test


def get_middle_value_from_column(column):
    sum(ast.literal_eval(column)) / len(ast.literal_eval(column)) if column else 0


def setup_logging(log_dir='../output/log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=f'{log_dir}/training.log',
                        filemode='a',
                        format='%(asctime)s %(levelname)s - %(message)s',
                        level=logging.INFO)


def log_message(message):
    # print(message)
    logging.info(message)
