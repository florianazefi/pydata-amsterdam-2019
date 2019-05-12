import numpy as np
import pandas as pd

import torch
import torch.nn as nn


def sticky_sequence(a, threshold=0.25):
    """Transform a sequence into a "sticky sequence" where the value is falling back to the previous sequence element
    if the relative change in the original sequence is falling below a "sticky threshold".

    Args:
        a (numpy.ndarray): A 1-d or 2-d array containing the sequence.
            The first dimension represents the evolution of the sequence,
            while the second (optional) dimension is the dimension of the sequence elements.
        threshold (double): The "sticky threshold".
            The sequence values will not change if the change in the original array falls below this threshold.

    Returns:
        a (numpy.ndarray): An array of the same dimension as the original array, just altered by the sticky property.
    """

    if len(a.shape) == 1:
        a = a[:, np.newaxis]
    if len(a.shape) > 2:
        raise ValueError("Input arrays of dim > 2 not supported.")

    diff = np.diff(a, prepend=0.0, axis=0)
    stick = np.abs(diff) < threshold
    stick[0] = False

    res = np.cumsum(np.where(stick, 0.0, diff), axis=0)
    b = np.where(~stick, np.cumsum(np.where(stick, diff, 0.0), axis=0), np.zeros_like(stick))
    for i in range(b.shape[1]):
        b[b[:, i] != 0.0, i] = np.diff(b[b[:, i] != 0.0, i], prepend=0.0)
    res += np.cumsum(b, axis=0)
    return res


def make_sequential(
    n_samples=100,
    n_sequences=10,
    n_features=3,
    n_informative=3,
    random_state=None,
    lstm_input_size=1,
    sticky_threshold=0.01,
):
    """Generate a random sequential dataset.

    Args:
        n_samples (int): The total number of entries.
        n_sequences (int): The number of sequences.
            Each sequence will on average be of length n_sequences/n_samples.
        n_features (int): The number of features.
        n_informatives (int): The number of informative features generated with the LSTM.
            The remaining features will be drawn from a normal distribution.
        random_state (int): The random seed.
        lstm_input_size (int): The input size of the LSTM.
            The higher the complicated the generate sequential pattern.

    Returns:
        pandas.DataFrame: The DataFrame containing the sequential values.
    """

    if n_informative != n_features:
        raise NotImplementedError

    if not random_state is None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    a = np.linspace(0, 1, n_samples)
    np.random.shuffle(a)
    df = pd.DataFrame({"sequence": np.cumsum(a < (n_sequences - 1) / n_samples)})

    rnn = nn.LSTM(lstm_input_size, n_features)

    def fill(df):
        n = len(df)
        in_tensor = torch.randn(n, 1, lstm_input_size)
        out_tensor, _ = rnn(in_tensor)
        out = out_tensor.detach().numpy().reshape(n, n_features)

        out = sticky_sequence(out, threshold=sticky_threshold)

        df["v1"] = out[:, 0]
        df["v2"] = out[:, 1]
        df["v3"] = out[:, 2]

        return df

    return df.groupby("sequence").apply(fill)
