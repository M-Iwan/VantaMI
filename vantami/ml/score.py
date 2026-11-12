import numpy as np
import numpy.typing as npt


def ece_score(y_true: npt.NDArray, y_score: npt.NDArray, sample_weight: npt.NDArray | None = None, n_bins: int = 10):
    """
    Compute the Expected Calibration Error score.

    Parameters
    ----------
    y_true: npt.NDArray
        Array with true binary values
    y_score: npt.NDArray
        Predicted probabilities in range [0, 1]
    sample_weight: npt.NDArray | None
        Array of sample weights. Must be the same shape as y_true
    n_bins: int
        Number of bins to use. Default is 10.

    Returns
    -------
    ece: float
        Expected Calibration Error score
    """

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_score, bin_edges) -1, 0, n_bins - 1)

    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    bin_weights = np.zeros(n_bins)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx

        if np.sum(mask) == 0:
            continue

        weights = sample_weight[mask]
        bin_weights[bin_idx] = weights.sum()

        prob_true[bin_idx] = np.average(y_true[mask], weights=weights)
        prob_pred[bin_idx] = np.average(y_score[mask], weights=weights)

    total_weight = sample_weight.sum()
    ece = np.sum((bin_weights / total_weight) * np.abs(prob_true - prob_pred))

    return ece
