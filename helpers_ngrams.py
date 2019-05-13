# MIT License; see ./LICENSE

# AUTHOR : Floriana ZEFI
# CONTACT : florianagjzefi@gmail.com or floriana.zefi@ing.com
# FIRST PRESENTATION : PyData - Amsterdam - 2019


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import datetime
import matplotlib.pylab as plt


def all_timedeltas_to_int(df):
    """
    Function to convert the timedelta to int
    Args:
        df: Pandas data frame
    Return:
        df: Pandas data frame with timedelta converted as int 
    """
    date_columns = df.select_dtypes(include=[np.timedelta64]).columns
    func = lambda series: series.dt.days
    df[date_columns] = df[date_columns].apply(func, axis=1)

    return df


def diff_and_first_to_one(df):
    """
    Function to return a data frame with 1, 0 and -1 
    based on the changes the client did
    Args:
        df: pandas data frame 
    Return:
        col_to_drop (list): list of columns which are not to be converted 
    """
    # We drop columns of dtype 'object' because they can't be subtracted.
    df = df.select_dtypes(exclude=[np.dtype("O")])
    first_row = df.iloc[0]
    df = df.diff()
    # For the first row, makes the difference with itself and returns zero.
    # This approach is "NaN-safe" in the way that if the first row contained
    # missing values, they will be still missing in the output of this function.

    df.iloc[0] = first_row - first_row
    df = all_timedeltas_to_int(df)
    # Now we set the first row to 1
    df.iloc[0] = df.iloc[0] + 1

    return df


def add_deltas_to_df(df, column_to_drop, suffix="_delta"):
    """
    Args:
        df: Pandas data frame
        column_to_drop (list): columns to drop in a list
        suffix (str) : suffix for the columns which will indicate the changes  
    Return:
        df (Pandas data frame): original data frame together with the changes and encoded state   
    Example:
        df = add_deltas_to_df(df, column_to_drop = ['col1', 'col2'], suffix = '_delta')
    """
    col_groupby = "exp"
    # apply function to encode the changes as -1, 0, 1
    df_deltas = df.groupby(col_groupby).apply(diff_and_first_to_one).drop(column_to_drop, axis=1)
    # df_deltas = df_deltas.drop(column_to_drop, axis = 1)
    df_deltas = df_deltas.add_suffix(suffix)
    df_deltas = np.clip(df_deltas, -1, 1).astype(int)
    df = pd.merge(df, df_deltas, how="left", left_index=True, right_index=True)

    return df


def encode_states(df, return_states_df=False, verbose=False):
    """
    Function to calculate encode different observations
    as observations * options. If we have 3 observations
    and 3 options will return 3*3*3 encoded values. Each 
    state is encoded by a unique number. 
    Args:
        df: Pandas data frame
        return_states_df: Pandas data frame 
    Return:
        states, states_df: Pandas data frame
    """
    from sklearn.preprocessing import LabelEncoder

    n, m = df.shape

    encoders = []
    n_classes = []
    factors = []

    states = np.zeros(len(df), dtype=int)

    for column, dtype in zip(df.columns, df.dtypes):
        enc = LabelEncoder()
        enc.fit(df[column])
        encoders.append(enc)
        factors.append(np.prod(list(n_classes), dtype=int))
        states = states + enc.transform(df[column]) * factors[-1]
        n_classes.append(len(enc.classes_))

        if verbose:
            print(("Value counts for column " + column))
            print((df[column].value_counts()))
            print("\n")

    n_states = np.prod(n_classes)

    combinations = np.zeros((n_states, m), dtype=int)
    encoded_combinations = np.arange(n_states)
    for i, f in enumerate(factors[::-1]):
        combinations[:, m - i - 1] = encoded_combinations // f
        encoded_combinations = encoded_combinations % f

    states_df = pd.DataFrame()
    states_df.index.name = "state"
    for i in range(m):
        states_df[df.columns[i]] = encoders[i].inverse_transform(combinations[:, i])

    if return_states_df:
        return states, states_df
    return states
