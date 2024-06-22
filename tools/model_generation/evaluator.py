import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.losses import huber
import matplotlib.dates as mdates


class ModelEvaluator:
    """
    ModelEvaluator Class

    This class is used to evaluate the performance of a model by calculating various evaluation metrics and generating plots.

    Methods:
        - calculate_mae(): Calculate the Mean Absolute Error (MAE).
        - calculate_mse(): Calculate the Mean Squared Error (MSE).
        - calculate_rmse(): Calculate the Root Mean Squared Error (RMSE).
        - calculate_mape(): Calculate the Mean Absolute Percentage Error (MAPE).
        - calculate_smape(): Calculate the Symmetric Mean Absolute Percentage Error (sMAPE).
        - calculate_mda(): Calculate the Mean Directional Accuracy (MDA).
        - plot_actual_vs_predicted(): Plot the actual vs predicted values.
        - plot_residuals(): Plot the residuals of predictions.
    """

    def __init__(self, y_true, y_pred):
        self.y_true = pd.Series(y_true)
        self.y_pred = pd.Series(y_pred)

    def calculate_mae(self):
        """Calculate Mean Absolute Error (MAE)."""
        return np.mean(np.abs(self.y_pred - self.y_true))

    def calculate_mse(self):
        """Calculate Mean Squared Error (MSE)."""
        return np.mean((self.y_pred - self.y_true) ** 2)

    def calculate_rmse(self):
        """Calculate Root Mean Squared Error (RMSE)."""
        return np.sqrt(self.calculate_mse())

    def calculate_mape(self):
        """Calculate Mean Absolute Percentage Error (MAPE). Handles zero values in y_true."""
        mask = self.y_true != 0
        return (
            np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
        )

    def calculate_smape(self):
        """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
        denominator = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2.0
        diff = np.abs(self.y_true - self.y_pred) / denominator
        diff[denominator == 0] = 0
        return 100 * np.mean(diff)

    def calculate_huber_loss(self):
        """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
        return huber(self.y_true, self.y_pred, delta=1.0)

    def calculate_mda(self):
        """Calculate Mean Directional Accuracy (MDA)."""
        direction = np.sign(self.y_pred[1:] - self.y_pred[:-1]) == np.sign(
            self.y_true[1:] - self.y_true[:-1]
        )
        return np.mean(direction) * 100

    def plot_actual_vs_predicted(self, extended_y_true, N_STEPS):
        """Plot actual vs predicted values."""
        fig, ax = plt.subplots(figsize=(10, 5))
        y_pred = self.y_pred
        ax.plot(
            extended_y_true["timestamp"],
            extended_y_true["GHI"],
            color="blue",
            label="Actual",
        )
        ax.plot(
            extended_y_true[N_STEPS:]["timestamp"],
            y_pred,
            color="green",
            label="Predicted",
        )
        date_format = mdates.DateFormatter("%d/%m/%Y %H:%M")
        ax.xaxis.set_major_formatter(date_format)

        # Rotate the x-axis labels
        plt.xticks(rotation=45)

        plt.title("Actual vs Predicted Values")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_residuals(self):
        """Plot residuals of predictions."""
        residuals = self.y_true - self.y_pred
        plt.figure(figsize=(10, 5))
        plt.plot(residuals, label="Residuals", color="purple")
        plt.title("Residuals of Predictions")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.axhline(0, color="black", linestyle="--")
        plt.legend()
        plt.grid(True)
        plt.show()

    def metrics(self):
        print(f"MSE = {self.calculate_rmse():.6f}")
        print(f"RMSE = {self.calculate_mse():.6f}")
        print(f"MAPE = {self.calculate_mape():.6f}")
        print(f"sMAPE = {self.calculate_smape():.6f}")
        print(f"MAE = {self.calculate_mae():.6f}")
        print(f"Huber loss = {self.calculate_huber_loss():.6f}")

    @staticmethod
    def plot_loss(history, loss_metric: str = "MSE"):
        loss = history["loss"]
        # Get number of epochs
        epochs = range(len(loss))
        plt.figure()
        # Plot training and validation loss per epoch
        plt.plot(epochs, loss, marker=".")
        plt.title(f"Training loss ({loss_metric})")
        plt.grid(linestyle=":")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.show()

    def performance_report(self, history_vars, flg_history:bool, model_name, extended_y_true, N_STEPS):
        if flg_history:
            print(f"--Generated metrics for model: {model_name}")
            self.metrics()
            training_metrics = [i for i in list(history_vars.keys()) if 'val' not in i]
            fig, ax = plt.subplots(1, len(training_metrics), figsize=(15,4))
            for index, loss_metric in enumerate(training_metrics):
                loss = history_vars[loss_metric]
                # Get number of epochs
                epochs = range(len(loss))
                # Plot training and validation loss per epoch
                ax[index].plot(epochs, loss, marker=".", label=loss_metric.upper())
                try:
                    val_loss = history_vars[f"val_{loss_metric}"]
                    ax[index].plot(epochs, val_loss, linestyle='--', marker=".", label=f"Validation {loss_metric.upper()}")
                except Exception as e:
                    print(f"There was not found a validation history for loss variable {loss_metric.upper()}")
                finally:
                    ax[index].set_title(f"Loss per epoch ({loss_metric.upper()})")
                    ax[index].grid(linestyle=":")
                    ax[index].legend()
                    ax[index].set_xlabel("Epoch")
                    ax[index].set_ylabel(loss_metric.upper())
            plt.show()
        self.plot_actual_vs_predicted(extended_y_true, N_STEPS)
        self.plot_residuals()