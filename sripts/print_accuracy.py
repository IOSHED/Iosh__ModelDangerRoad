import torch
from src.accident_model import AccidentPredictionModel
from src.utils import load_data_for_training, preprocess_data


data_file = '../dataset/road_accidents_data_with_answer_1.csv'
weights_file = '../output/weights.csv'

x, y = load_data_for_training(data_file)
x_train, x_test, y_train, y_test = preprocess_data(x, y)

input_size = x_train.shape[1]
model = AccidentPredictionModel(input_size)

model.load_state_dict(torch.load(weights_file, weights_only=True))

model.eval()
with torch.no_grad():
    y_test_predict = model(x_test)
    y_test_predict_cls = y_test_predict.round()
    accuracy = (y_test_predict_cls == y_test).float().mean()

print(f'Accuracy with loaded weights: {accuracy.item():.4f}')
