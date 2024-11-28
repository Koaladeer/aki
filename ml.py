import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import convert_change_percent


# Define the PyTorch model
class StockPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(StockPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class StockStudyPredictor:
    def __init__(self, stock_file, study_file):
        self.stock_file = stock_file
        self.study_file = study_file
        self.scaler = StandardScaler()
        self.model = None

    def load_and_prepare_data(self):
        # Load CSV files
        stocks_df = pd.read_csv(self.stock_file)
        studies_df = pd.read_csv(self.study_file)

        # Data Preparation
        stocks_df['Date'] = pd.to_datetime(stocks_df['Date'], format='%m/%d/%Y')
        stocks_df['Change %'] = stocks_df['Change %'].str.replace('%', '').astype(float) / 100.0
        stocks_df['Vol.'] = stocks_df['Vol.'].fillna(0)
        stocks_df['Vol.'] = stocks_df['Vol.'].apply(convert_change_percent)
        studies_df['Start Date'] = pd.to_datetime(studies_df['Start Date'], errors='coerce')
        studies_df['Primary Completion Date'] = pd.to_datetime(studies_df['Primary Completion Date'], errors='coerce')

        # Merge and select features
        merged_data = stocks_df.merge(studies_df, left_on='Date', right_on='Start Date', how='inner')
        features = merged_data[['Price', 'Open', 'High', 'Low', 'Vol.', 'Conditions', 'Interventions']]
        targets = merged_data['Change %']#todo:change target to price

        # One-hot encode categorical features
        features = pd.get_dummies(features, columns=['Conditions', 'Interventions'])

        # Normalize features
        X = self.scaler.fit_transform(features)
        y = targets.values
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train, X_test=None, y_test=None, epochs=1000, lr=0.001):
        """
        Trains the model and optionally evaluates it on test data after each epoch.

        Parameters:
        - X_train, y_train: Training data and labels
        - X_test, y_test: Optional test data and labels for evaluation
        - epochs: Number of training epochs
        - lr: Learning rate

        Returns:
        - loss_vals: List of training loss values for each epoch
        - test_loss_vals: List of test loss values for each epoch (if test data is provided)
        """
        input_size = X_train.shape[1]
        self.model = StockPredictionModel(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        if X_test is not None and y_test is not None:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        print(f"Target Range: Min={y_train.min()}, Max={y_train.max()}")

        # Track loss values
        loss_vals = []
        test_loss_vals = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())

            # Evaluation phase (if test data is provided)
            if X_test is not None and y_test is not None:
                self.model.eval()
                with torch.no_grad():  # Disable gradient computation
                    test_outputs = self.model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor)
                    test_loss_vals.append(test_loss.item())

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                if X_test is not None and y_test is not None:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
                else:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        return loss_vals, test_loss_vals


import matplotlib.pyplot as plt

def plot_predictions_vs_true(y_true, y_pred, title="Predictions vs True Values"):
    """
    Plots a scatter plot comparing true values and predictions.

    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted values
    - title: str, title of the plot
    """
    print(f"y_pred shape: {y_pred.shape}")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)  # Reference line
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    predictor = StockStudyPredictor("Data/stock_data.csv", "Data/studies_data.csv")

    # Load and prepare data
    X_train, X_test, y_train, y_test = predictor.load_and_prepare_data()

    # Train the model
    train_loss, test_loss = predictor.train_model(X_train, y_train, X_test,y_test)

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    plt.plot(train_loss, label='Training loss', color='red')
    print(test_loss)
    plt.plot(test_loss, label='Test loss', color='blue')
    plt.show()

    print('Loss Value Training: ' + str(train_loss[99]))
    print('Loss Value Test: ' + str(test_loss[99]))