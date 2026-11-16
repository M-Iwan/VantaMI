import numpy as np
import numpy.typing as npt


def round_to_significant(value: float | int | npt.NDArray, n: int) -> float | npt.NDArray:
    """
    Round a numeric scalar or NumPy array to n significant digits.
    """

    if value is None or isinstance(value, str):
        return np.nan

    if n < 1:
        raise ValueError("n must be at least 1.")

    values = np.asarray(value)

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("Value must be a numeric scalar or numeric NumPy array.")

    scalar_input = values.ndim == 0
    values = np.atleast_1d(values).astype(float, copy=False)

    result = values.copy()
    finite_nonzero = np.isfinite(result) & (result != 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        decimal_places = n - np.floor(np.log10(np.abs(result[finite_nonzero]))).astype(int) - 1

    scale = np.power(10.0, decimal_places)
    result[finite_nonzero] = np.round(result[finite_nonzero] * scale) / scale

    result[result == 0] = 0.0

    return result[0].item() if scalar_input else result
