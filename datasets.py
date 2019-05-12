# MIT License; see ./LICENSE


import numpy as np
import pandas as pd

import torch
import torch.nn as nn


def _split_n_in_m_parts(n, m):
    """ Split integer n in m integers which sum up to n again, even if m * (n//m) != n.
    """
    a = [n // m] * m
    a[0] = n - (m - 1) * a[0]
    return a


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
    n_classes=1,
    random_state=2019,
    lstm_input_size=1,
    sticky_threshold=0.01,
):
    """Generate a random sequential dataset.

    Args:
        n_samples (int): The total number of entries.
        n_sequences (int): The number of sequences.
            Each sequence will on average be of length n_sequences/n_samples.
        n_features (int): The number of features.
        n_classes (int): The number of different classes in the dataset.
            For each class, the sequences will be generated with a different LSTM.
        n_informatives (int) TODO: The number of informative features generated with the LSTM.
            The remaining features will be drawn from a normal distribution.
        random_state (int): The random seed.
        lstm_input_size (int): The input size of the LSTM.
            The higher the complicated the generate sequential pattern.
        sticky_threshold (double): When the difference from one element to the next is smaller than this threshold,
            the next element will just be set to the previous element.

    Returns:
        pandas.DataFrame: The DataFrame containing the sequential values.
    """

    if n_classes < 1:
        raise ValueError("n_classes should be an integer >= 1")

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if n_classes > 1:
        data_frames = []

        n_samples = _split_n_in_m_parts(n_samples, n_classes)
        n_sequences = _split_n_in_m_parts(n_sequences, n_classes)

        for i in range(n_classes):
            df = make_sequential(
                n_samples=n_samples[i],
                n_sequences=n_sequences[i],
                n_features=n_features,
                random_state=random_state + i,
                lstm_input_size=lstm_input_size,
                sticky_threshold=sticky_threshold,
            )
            df["class"] += i
            if data_frames:
                df["sequence"] += data_frames[-1]["sequence"].max() + 1
            data_frames.append(df)

        return pd.concat(data_frames, ignore_index=True)

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

        for i in range(n_features):
            df["v{0}".format(i)] = out[:, i]

        return df

    df["class"] = 0

    return df.groupby("sequence").apply(fill)


def make_bigram_toy_data(n_samples=10000, n_sequences=1000, n_features=3):
    """ Create sequential toy data with two classes and features representing
    the sign of the differences to the previous sequence elements.
    """

    variables = ["v{0}".format(i) for i in range(n_features)]

    variables_diffsign = [v + "_diffsign" for v in variables]

    df = make_sequential(
        n_samples=n_samples, n_sequences=n_sequences, n_features=n_features, n_classes=2, random_state=11
    )

    def diff(df):
        df[variables_diffsign] = np.sign(df[variables].diff())
        return df

    df = df.groupby("sequence").apply(diff)
    df[variables_diffsign] = df[variables_diffsign].fillna(1.0)
    df[variables_diffsign] = df[variables_diffsign].astype(np.int)

    return df
