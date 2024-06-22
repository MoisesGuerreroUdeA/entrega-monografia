import os
import logging
from keras import Sequential, layers, losses, backend

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
)


class Generator:
    def generate_model(self, conf_model: dict):
        model = Sequential(name=conf_model["model"]["name"])
        for layer in conf_model["model"]["structure"]:
            model.add(self.__layers_mapper(layer))
        model.compile(
            loss=self.__loss_functions_mapper(
                conf_model["model"]["compile"]["loss"]
            ),
            optimizer=conf_model["model"]["compile"]["optimizer"],
            metrics=conf_model["model"]["compile"]["metrics"],
        )
        model.summary()
        return model

    @staticmethod
    def __loss_functions_mapper(loss: str):
        if loss == "huber":
            return losses.Huber()
        return loss

    @staticmethod
    def __layers_mapper(layer: dict):
        layer_params = {
            param: layer[param]
            for param in layer
            if param not in ["type", "bidirectional"]
        }
        logging.debug(f"Layer parameters: {layer_params}")
        flg_bidirectional = layer.get("bidirectional", False)
        layer_type = layer["type"]
        layer_name = layer.get("name", None)
        layer_mapping = {
            "input": layers.Input,
            "simple_rnn": layers.SimpleRNN,
            "lstm": layers.LSTM,
            "gru": layers.GRU,
            "dropout": layers.Dropout,
            "batch_normalization": layers.BatchNormalization,
            "dense": layers.Dense,
            "conv1d": layers.Conv1D,
            "conv2d": layers.Conv2D,
        }
        if layer_type not in layer_mapping:
            raise Exception(
                f"Configured layer type '{layer_type}' is not available in Generator class layers mapper"
            )

        logging.info(
            f"Adding a {layer_type.capitalize()} layer to the model with name {layer_name}"
        )
        if layer_type == "input":
            logging.info(f"Configuring input shape = {tuple(layer['shape'])}")

        layer_obj = layer_mapping[layer_type](**layer_params)

        return layers.Bidirectional(layer_obj) if flg_bidirectional else layer_obj

