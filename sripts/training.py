import csv
import torch
import torch.optim as optim
from torch.nn import BCELoss
from src.accident_model import AccidentPredictionModel
from src.utils import load_data_for_training, preprocess_data, setup_logging, log_message


epochs = 200
batch_size = 64
data_file = '../dataset/road_accidents_data_with_answer_4.csv'
output_dir = '../output'
weights_file = f'{output_dir}/weights.csv'
accuracy_file = f'{output_dir}/accuracy.csv'

setup_logging()

x, y = load_data_for_training(data_file)
x_train, x_test, y_train, y_test = preprocess_data(x, y)

input_size = x_train.shape[1]
model = AccidentPredictionModel(input_size)

criterion = BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss = None


for epoch in range(epochs):
    model.train()
    for i in range(0, x_train.size(0), batch_size):
        X_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Прямой проход
        y_predict = model(X_batch)

        # Потери
        loss = criterion(y_predict, y_batch)

        # Обратное распространение
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            y_test_predict = model(x_test)
            test_loss = criterion(y_test_predict, y_test)
            y_test_predict_cls = y_test_predict.round()
            accuracy = (y_test_predict_cls == y_test).float().mean()

        log_message(f''
                    f'Epoch {epoch + 1}/{epochs}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Test Loss: {test_loss.item():.4f}, '
                    f'Accuracy: {accuracy.item():.4f}'
                    )

        with open(accuracy_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, accuracy.item()])

weights = model.state_dict()
torch.save(weights, weights_file)
