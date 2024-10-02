from collections import Counter
import numpy as np
import pandas as pd

def iqr_method(df: pd.DataFrame, n: int, features: list, drop_outliers: bool = True):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        n (int): _description_
        features (list): _description_
        drop_outliers (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        n = 1
        features = ["f1", "f2", "f3"]

        df = iqr_method(df=df, n=n, features=features, drop_outliers=True)
    """
    outlier_list = []

    for column in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determining a list of indices of outliers
        outlier_list_column = df[
            (df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)
        ].index
        # appending the list of outliers
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] < Q1 - outlier_step]
    df2 = df[df[column] > Q3 + outlier_step]

    print("Total number of outliers is:", df1.shape[0] + df2.shape[0])

    if drop_outliers:
        df = df.drop(multiple_outliers).reset_index(drop=True)
        return df