class NotFitedError(Exception):
    """
    Exception class to be thrown when an operation requires the model to be
    trained, but it is not trained.
    """
    pass


class InvalidDataError(Exception):
    """
    Exception class to be thrown when the input data is not in the correct format.
    """
    pass
