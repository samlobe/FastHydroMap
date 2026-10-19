"""FastHydroMap v2 inference. Import Predictor only when PyTorch is needed."""


def __getattr__(name):
    if name == "Predictor":
        from .predictor import Predictor

        return Predictor
    raise AttributeError(name)
