import numpy as np


def compute_metrics(targets, probabilities):
    from dlordinal.metrics import (
        accuracy_off1,
        amae,
        gmes,
        mes,
        minimum_sensitivity,
        mmae,
        ranked_probability_score,
    )
    from scipy.special import softmax
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        mean_absolute_error,
        recall_score,
    )

    targets = np.array(targets)
    probabilities = np.array(probabilities)

    if len(probabilities.shape) > 1:
        predictions = np.argmax(probabilities, axis=1)
    else:
        predictions = np.array(probabilities)

    metrics = {
        "QWK": cohen_kappa_score(targets, predictions, weights="quadratic"),
        "MAE": mean_absolute_error(targets, predictions),
        "1-off": accuracy_off1(targets, predictions),
        "CCR": accuracy_score(targets, predictions),
        "MS": minimum_sensitivity(targets, predictions),
        "BalancedAccuracy": balanced_accuracy_score(targets, predictions),
        "AMAE": amae(targets, predictions),
        "MMAE": mmae(targets, predictions),
        "RPS": ranked_probability_score(targets, softmax(probabilities, axis=1)),
        "MES": mes(targets, predictions),
        "GMES": gmes(targets, predictions),
    }

    # Compute sensitivities for each class
    sensitivities = np.array(recall_score(targets, predictions, average=None))

    for i, sens in enumerate(sensitivities):
        metrics[f"Sens{i}"] = sens

    return metrics
