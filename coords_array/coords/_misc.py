class CoordinateError(RuntimeError):
    """This error is raised when axes is defined in a wrong way."""

class CoordinateWarning(UserWarning):
    """This warning is raised when axes underwent unexpected calculation."""