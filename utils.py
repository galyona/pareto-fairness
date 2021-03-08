from sklearn.metrics import confusion_matrix
import numpy as np

# in the case of recidivism, the outcome Y = 1 is the bad one, so we obviously care
# about the imbalance in the fpr, or E[y_hat | y = 0], the average scores given to those that did not recidivate.


def tpr(y, y_hat, binary=False):
    # this function computes E[y_hat | y = 1].
    # if the binary flag is on, this applies a threshold of 0.5 to y_hat beforehand,
    # so this corresponds to the usual notion of TPR for the binary classifier y_hat>=0.5.
    # i.e., same as:
    # tn, fp, fn, tp = confusion_matrix(y >= 0.5, y_hat >= 0.5).ravel()
    # return tp / (tp + fn)
    # TODO: verify these are the same.

    if binary:
        y_hat = y_hat >= 0.5

    # keep only y = 1
    pos_idx = np.where(y == 1)
    return np.mean(y_hat[pos_idx])

def fpr(y, y_hat, binary=False):
    # this function computes E[y_hat | y = 1].
    # if the binary flag is on, this applies a threshold of 0.5 to y_hat beforehand,
    # so this corresponds to the usual notion of TPR for the binary classifier y_hat>=0.5.
    # i.e., same as:
    # tn, fp, fn, tp = confusion_matrix(y >= 0.5, y_hat >= 0.5).ravel()
    # return tp / (tp + fn)
    # TODO: verify these are the same.

    if binary:
        y_hat = y_hat >= 0.5

    # keep only y = 1
    pos_idx = np.where(y == 0)
    return np.mean(y_hat[pos_idx])
