import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.testing import assert_array_almost_equal


def generate_data(seed=1):
    """ Generate random uniform data with a given seed """
    np.random.seed(seed)
    n_cols = 1000
    n_rows = 5
    random_data = np.random.uniform(size=n_cols * n_rows)
    random_data = random_data.reshape(n_rows, n_cols)
    return pd.DataFrame(random_data)


def ols_sklearn(row):
    model = LinearRegression()
    X = np.arange(row.shape[0]).reshape(-1, 1)
    model.fit(X, row.values)
    slope = model.coef_[0]
    return slope


def ols_lstsq(row):
    lenght_x = row.shape[0]
    X = np.arange(lenght_x)
    ones = np.ones(lenght_x)
    full_X = np.vstack((X, ones)).T
    slope, intercept = np.linalg.lstsq(full_X, row.values, rcond=-1)[0]
    return slope


if __name__ == '__main__':
    results_sklearn = df.apply(ols_sklearn, axis=1)
    results_lstsq = df.apply(ols_lstsq, axis=1)
    assert_array_almost_equal(results_sklearn, results_lstsq)

    # Pandas iloc and looping

    %%timeit
    ms = []
    for row_idx in range(df.shape[0]):
        row = df.iloc[row_idx]
        slope = ols_lstsq(row)
        ms.append(m)
    reults = pd.Series(ms)
    assert_array_almost_equal(


