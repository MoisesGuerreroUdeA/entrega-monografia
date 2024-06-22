import os
import json
import logging
import numpy as np
import pandas as pd
import textwrap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.getLevelName(os.environ["LOG_LEVEL"]),
)


class Trainer:
    def train_model_params(self, config: dict, dataset: pd.DataFrame, save_dir: str):
        logging.info(f"Starting training for all available models...")
        x_train, y_train, x_test, y_test, scaler_x, scaler_y = self.__generate_dataset(
            train_params=config["model"]["training"], dataset=dataset
        )
        model_name = config["model"]["name"]
        callbacks_list = []
        if config["model"]["training"]["early_stopping"]:
            early_stop = callbacks.EarlyStopping(
                **config["model"]["training"]["early_stopping_params"]
            )
            callbacks_list.append(early_stop)
        logging.info(
            f"Model with name {model_name} will be saved on directory {save_dir}..."
        )
        model_checkpoint = callbacks.ModelCheckpoint(
            os.path.join(save_dir, f"{model_name}.keras"),
            monitor="val_loss",
            save_best_only=True,
        )
        callbacks_list.append(model_checkpoint)
        params = {
            "x": x_train,
            "y": y_train,
            "batch_size": config["model"]["training"]["batch_size"],
            "epochs": config["model"]["training"]["epochs"],
            "validation_data": (x_test, y_test),
            "callbacks": callbacks_list,
            "shuffle": False,
            "verbose": 1,
        }
        return params, scaler_x, scaler_y

    @staticmethod
    def save_history(model_name: str, history: dict, save_dir: str):
        logging.info(f"Saving model {model_name} history on directory {save_dir}")
        with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
            f.write(json.dumps(history, indent=2))
        logging.info(f"History file was successfully wrote for model {model_name}!")

    def __generate_dataset(self, train_params: dict, dataset: pd.DataFrame):
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        logging.info(f"Generating a scaled dataset by using MinMaxScaler...")
        logging.info(
            f"The dataset includes columns: {train_params['columns_for_training']}"
        )
        scaled_dataset = scaler_x.fit_transform(
            dataset[train_params["columns_for_training"]]
        )
        logging.debug(
            f"Generating output scaler by using column {train_params['columns_for_training'][0]}"
        )
        logging.warning("Output column should by the first column on dataset!")
        scaler_y.fit(
            dataset[train_params["columns_for_training"][:1]]
        )  # Output should be the first column
        logging.info(
            f"Splitting data into train and test datasets with test size: {train_params['test_size']}"
        )
        train_data, test_data = train_test_split(
            scaled_dataset, test_size=train_params["test_size"], shuffle=False
        )
        x_train, y_train = self.__create_sequences(
            train_data,
            n_past_steps=train_params["n_past_steps"],
            n_forecast_steps=train_params["n_forecast_steps"],
            n_overlay_steps=train_params["n_overlay_steps"],
            include_target_as_feature=train_params["include_target_as_feature"],
        )
        x_test, y_test = self.__create_sequences(
            test_data,
            n_past_steps=train_params["n_past_steps"],
            n_forecast_steps=train_params["n_forecast_steps"],
            n_overlay_steps=train_params["n_overlay_steps"],
            include_target_as_feature=train_params["include_target_as_feature"],
        )
        logging.info(
            textwrap.dedent(
                f"""
        -----------------------------
        X Train shape: {x_train.shape} 
        y Train shape: {y_train.shape} 
        X Test shape: {x_test.shape} 
        y Test shape: {y_test.shape} 
        -----------------------------
        """
            )
        )
        return x_train, y_train, x_test, y_test, scaler_x, scaler_y

    @staticmethod
    def __create_sequences(
        data,
        n_past_steps: int,
        n_forecast_steps: int,
        n_overlay_steps: int,
        include_target_as_feature: bool = True,
    ):
        x_sequence = []
        y_sequence = []
        start_idx = 0

        # Loop through the input data to create the sequences
        while True:
            # Determine the end of the sequence
            end_idx = start_idx + n_past_steps
            forecast_end_idx = end_idx + n_forecast_steps

            # Check if we have run out of data
            if forecast_end_idx > len(data):
                break

            # Get the sequences
            if include_target_as_feature:
                x = data[start_idx:end_idx]
            else:
                x = data[start_idx:end_idx, 1:]  # the target is the first column
            y = data[
                end_idx:forecast_end_idx, 0
            ]  # prediction target is 'is the first column
            x_sequence.append(x)
            y_sequence.append(y)

            # Move along to get the next sequence, by "n_overlay_steps" steps
            start_idx += n_overlay_steps

        return np.array(x_sequence), np.array(y_sequence)
