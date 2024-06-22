import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import models, losses


class Model:
    def __init__(self, model: models.Sequential, model_name: str, conf_model: dict):
        self.model = model
        self.model_name = model_name
        self.conf_model = conf_model

    def add_dataset(
        self, x_test, y_test, scaler_x: MinMaxScaler, scaler_y: MinMaxScaler
    ):
        self.x_test = x_test
        self.y_test = y_test
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def add_history(self, history: dict):
        self.history = history

    def __predict(self):
        self.y_pred = self.model.predict(self.x_test)
        self.unscaled_y_test = pd.Series(
            self.scaler_y.inverse_transform(self.y_test).flatten()
        )
        self.unscaled_y_pred = pd.Series(
            self.scaler_y.inverse_transform(self.y_pred).flatten()
        )

    def calculate_mae(self):
        """Calculate Mean Absolute Error (MAE)."""
        return np.mean(np.abs(self.unscaled_y_pred - self.unscaled_y_test))

    def calculate_mse(self):
        """Calculate Mean Squared Error (MSE)."""
        return np.mean((self.unscaled_y_pred - self.unscaled_y_test) ** 2)

    def calculate_rmse(self):
        """Calculate Root Mean Squared Error (RMSE)."""
        return np.sqrt(self.calculate_mse())

    def calculate_mape(self):
        """Calculate Mean Absolute Percentage Error (MAPE). Handles zero values in y_true."""
        mask = self.unscaled_y_test != 0
        return (
            np.mean(
                np.abs(
                    (self.unscaled_y_test[mask] - self.unscaled_y_pred[mask])
                    / self.unscaled_y_test[mask]
                )
            )
            * 100
        )

    def calculate_smape(self):
        """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
        denominator = (
            np.abs(self.unscaled_y_test) + np.abs(self.unscaled_y_pred)
        ) / 2.0
        diff = np.abs(self.unscaled_y_test - self.unscaled_y_pred) / denominator
        diff[denominator == 0] = 0
        return 100 * np.mean(diff)

    def calculate_huber_loss(self):
        """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
        return losses.huber(self.unscaled_y_test, self.unscaled_y_pred, delta=1.0)

    def calculate_mda(self):
        """Calculate Mean Directional Accuracy (MDA)."""
        direction = np.sign(
            self.unscaled_y_pred[1:] - self.unscaled_y_pred[:-1]
        ) == np.sign(self.unscaled_y_test[1:] - self.unscaled_y_test[:-1])
        return np.mean(direction) * 100

    def metrics(self):
        print(f"Computing metrics for model '{self.model_name}'")
        self.__predict()
        print(f"MSE = {self.calculate_rmse():.6f}")
        print(f"RMSE = {self.calculate_mse():.6f}")
        print(f"MAPE = {self.calculate_mape():.6f}")
        print(f"sMAPE = {self.calculate_smape():.6f}")
        print(f"MAE = {self.calculate_mae():.6f}")
        print(f"Huber loss = {self.calculate_huber_loss():.6f}")
