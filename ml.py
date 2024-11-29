import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib

from LSTMAttention import AttentionLSTM
from LSTMBase import LSTMBase

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import convert_change_percent, encode_date_column, numeric_encode_column
from sklearn import preprocessing


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
    def predict(self, X):
        """
        Generates predictions for the given input features.

        Parameters:
        - X: Input features (numpy array or tensor).

        Returns:
        - Predictions: Predicted values (numpy array).
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please train the model before predicting.")

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Convert input to tensor if not already
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Disable gradient calculations
        with torch.no_grad():
            predictions = self.model(X_tensor)

        # Convert predictions to numpy array
        return predictions.numpy()

    def __init__(self, stock_file, study_file):
        self.stock_file = stock_file
        self.study_file = study_file
        self.scaler = preprocessing.MinMaxScaler()
        self.model = None

    def load_and_prepare_data_v2(self,sequence_length = 5):
        # Load CSV files
        stocks_df = pd.read_csv(self.stock_file)
        studies_df = pd.read_csv(self.study_file)

        # Data Preparation
        stocks_df['Date'] = pd.to_datetime(stocks_df['Date'], format='%m/%d/%Y')
        stocks_df['Change %'] = stocks_df['Change %'].str.replace('%', '').astype(float) / 100.0
        stocks_df['Vol.'] = stocks_df['Vol.'].fillna(0)
        stocks_df['Vol.'] = stocks_df['Vol.'].apply(convert_change_percent)

        #Dates
        # Convert date columns to datetime
        date_columns = ["Start Date", "Primary Completion Date", "Completion Date", "First Posted",
                        "Last Update Posted"]
        for col in date_columns:
            studies_df[col] = pd.to_datetime(studies_df[col], errors='coerce')
            # Numeric encoding for date columns
            encoded_date_dfs = []
            for col in date_columns:
                # Perform numeric encoding
                encoded_date_df = numeric_encode_column(studies_df[col])
                encoded_date_dfs.append(encoded_date_df)

            # Concatenate all encoded DataFrames with the original studies_df
            studies_df = pd.concat([studies_df] + encoded_date_dfs, axis=1)

        studies_df['Enrollment'] = studies_df['Enrollment'].fillna(studies_df['Enrollment'].mean())
        studies_df = studies_df[studies_df['Sponsor'] == 'Bayer']

        # Drop unnecessary columns
        studies_df = studies_df.drop(
            columns=[
                'Study Type', 'Collaborators', 'Results First Posted', 'Acronym',
                'Study URL', 'NCT Number', 'Study Title', 'Interventions', 'Study Design',
                'Sponsor'
            ]
        )


        # One-hot encode categorical columns
        studies_df = pd.get_dummies(studies_df, columns=['Sex', 'Phases', 'Age', 'Study Results', 'Study Status'])

        # Merge datasets on the date
        #merged_data = stocks_df.merge(studies_df, left_on='Date', right_on='Start Date', how='inner')
        merged_data = stocks_df.merge(studies_df, how='cross')  # Kreuzprodukt beider DataFrames
        merged_data = merged_data[
            (merged_data['Date'] >= merged_data['Start Date']) &
            (merged_data['Date'] <= merged_data['Start Date'] + pd.Timedelta(days=7))
            ]
        # Select features and target
        features = merged_data.drop(
            columns=['Change %',"Start Date", "Primary Completion Date", "Completion Date", "First Posted",
                        "Last Update Posted","Date" ]
        )
        targets = merged_data['Change %']

        # Save the DataFrame as a CSV file
        filepath = os.path.join("Data", "features.csv")
        features.to_csv(filepath, index = False)
        # Normalize features
        X = self.scaler.fit_transform(features)
        y = targets.values

        # Reshape for LSTM: (batch_size, sequence_length, input_size)
        X_sequences, y_sequences = self.create_sequences(X, y, sequence_length)
        return train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

    def create_sequences(self, X, y, sequence_length):
        """
        Create sequences from the input features and targets for LSTM training.

        Parameters:
        - X: Feature array of shape (num_samples, input_size)
        - y: Target array of shape (num_samples,)
        - sequence_length: Number of time steps in each sequence

        Returns:
        - X_sequences: Feature sequences of shape (num_sequences, sequence_length, input_size)
        - y_sequences: Corresponding targets for each sequence
        """
        X_sequences = []
        y_sequences = []
        for i in range(len(X) - sequence_length + 1):
            X_sequences.append(X[i:i + sequence_length])
            y_sequences.append(y[i + sequence_length - 1])

        return np.array(X_sequences), np.array(y_sequences)

    def train_model(self, X_train, y_train, X_test=None, y_test=None, epochs=100, lr=0.003):
        """
        Trains the AttentionLSTM model and optionally evaluates it on test data after each epoch.

        Parameters:
        - X_train, y_train: Training data and labels
        - X_test, y_test: Optional test data and labels for evaluation
        - epochs: Number of training epochs
        - lr: Learning rate

        Returns:
        - loss_vals: List of training loss values for each epoch
        - test_loss_vals: List of test loss values for each epoch (if test data is provided)
        """
        input_size = X_train.shape[2]  # Number of features
        hidden_size = 512  # Number of LSTM units
        output_size = 1  # Predicting a single value (e.g., stock price)
        self.model = AttentionLSTM(input_size, hidden_size, output_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        # Convert test data if available
        if X_test is not None and y_test is not None:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Track loss values
        loss_vals = []
        test_loss_vals = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            y_train_tensor = y_train_tensor.view(-1, 1)  # Reshape target to match output
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())

            # Evaluation phase (if test data is provided)
            if X_test is not None and y_test is not None:
                self.model.eval()
                with torch.no_grad():
                    test_outputs = self.model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor.view(-1, 1))
                    test_loss_vals.append(test_loss.item())

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                if X_test is not None and y_test is not None:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
                else:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        return loss_vals, test_loss_vals


import matplotlib.pyplot as plt

def plot_testloss_vs_trainloss(train_loss, test_loss, title="train_loss vs test_loss"):
    """
    Plots a scatter plot comparing true values and predictions.

    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted values
    - title: str, title of the plot
    """
    plt.plot(train_loss, label='Training loss', color='red')
    print(test_loss)
    plt.plot(test_loss, label='Test loss', color='blue')
    plt.show()


def evaluate_learning():
    print('Loss Value Training: ' + str(train_loss[99]))
    print('Loss Value Test: ' + str(test_loss[99]))

    # Example: assuming y_test and predictions are available
    predictions = predictor.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")
# Example usage
if __name__ == "__main__":
    predictor = StockStudyPredictor("Data/stock_data.csv", "Data/studies_data_v2.csv")

    # Load and prepare data
    X_train, X_test, y_train, y_test = predictor.load_and_prepare_data_v2(7)

    # Train the model
    epochs = 100
    train_loss, test_loss = predictor.train_model(X_train, y_train, X_test,y_test, epochs)
    plot_testloss_vs_trainloss(train_loss,test_loss, epochs)
    evaluate_learning()


