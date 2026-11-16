from __future__ import annotations

import numpy as np
import numpy.typing as npt


def round_to_significant(value: float | int | npt.NDArray, n: int) -> float | int | npt.NDArray:
    """
    Round a numeric scalar or NumPy array to `n` significant digits.

    Parameters
    ----------
    value: float | int | npt.NDArray
        A float, int, or NumPy array containing floats and/or integers.
    n: int
        Number of significant digits.

    Returns
    -------
    float | int | npt.NDArray
        Rounded value(s), retaining scalar input as a scalar and array input as a NumPy array.
    """

    if value is None:
        return np.nan

    if isinstance(value, str):
        return np.nan

    if n < 1:
        raise ValueError("n must be at least 1.")

    values = np.asarray(value)

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("Value must be a numeric scalar or numeric NumPy array.")

    nonzero = values != 0

    result = values.astype(float, copy=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        decimal_places = n - np.floor(np.log10(np.abs(result))).astype(int) - 1

    result[nonzero] = np.round(result[nonzero], decimal_places[nonzero])
    result[~nonzero] = 0.0

    return result.item() if np.isscalar(value) else result
