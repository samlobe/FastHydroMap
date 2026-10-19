__all__ = ["Predictor", "selftest"]


def __getattr__(name):
    if name == "Predictor":
        from .v2.predictor import Predictor
        return Predictor
    raise AttributeError(name)


def selftest(verbose: bool = True) -> None:
    """Check the installed model files and load all ensemble members on CPU.

    This is a packaging check, not an end-to-end structural prediction test.
    """
    from .v2.predictor import Predictor
    predictor = Predictor()
    if verbose:
        print(f"FastHydroMap model check passed ({len(predictor.members)} ensemble members)")
