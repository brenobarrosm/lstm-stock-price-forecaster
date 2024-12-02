import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


class StockPredictor:
    def __init__(self, symbol, start_date, end_date, seq_length=20, test_split=0.8):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.test_split = test_split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    def load_data(self):
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        self.data = df[['Close']].dropna()
        self.data['Close'] = self.scaler.fit_transform(self.data[['Close']])

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length])
        return np.array(X), np.array(y)

    def prepare_data(self):
        if self.data is None:
            raise ValueError("Error: data not loaded")
        X, y = self.create_sequences(self.data['Close'].values)
        split = int(self.test_split * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

    def get_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, criterion, optimizer, X_train, y_train, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluate_model(model, criterion, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions.squeeze(), y_test)
    print(f"Test Loss: {test_loss.item():.4f}")


def save_model(model, path):
    torch.save(model.state_dict(), path)


symbol = "AZUL4.SA"
start_date = "2018-01-01"
end_date = "2024-11-27"
seq_length = 15
num_epochs = 100
learning_rate = 0.001

predictor = StockPredictor(symbol, start_date, end_date, seq_length)
predictor.load_data()
predictor.prepare_data()
X_train, y_train, X_test, y_test = predictor.get_data()

input_size = 1
hidden_size = 64
num_layers = 3
output_size = 1

model = StockLSTM(input_size, hidden_size, num_layers, output_size).to(predictor.device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, criterion, optimizer, X_train, y_train, num_epochs, predictor.device)
evaluate_model(model, criterion, X_test, y_test)

model_path = "trained_models/lstm_stock_model.pth"
save_model(model, model_path)
