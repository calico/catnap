import cvxpy
import numpy as np
import pandas as pd

ONE_DAY_IN_NANOSECONDS = (1e9 * 60 * 60 * 24)


def group_l1_trend_filter(x, y, lmbda):
    try:
        mask = ~np.isnan(y)
    except:
        print(y.dtype)
        raise ValueError()
    intervals = ((x[1:] - x[:-1]) / ONE_DAY_IN_NANOSECONDS).astype(float)
    interpolate = (1 / intervals[:-1]) + (1 / intervals[1:])
    n, num_cols = y.shape
    m = n - 2
    zeros = np.zeros((m, 1))
    D_mat = (
        np.hstack([np.diag(1 / intervals[:-1]), zeros, zeros])
        + np.hstack([zeros, np.diag(-interpolate), zeros])
        + np.hstack([zeros, zeros, np.diag(1 / intervals[1:])])
    )
    y_hat = cvxpy.Variable(shape=(n, num_cols))
    difference_in_differences = D_mat @ y_hat  # t-2 by num_cols
    objective = cvxpy.norm((y - y_hat)[mask], p=2)
    for i in range(m):
        objective += lmbda * cvxpy.norm(difference_in_differences[i, :], 4)
    prob = cvxpy.Problem(cvxpy.Minimize(objective))
    prob.solve(verbose=False, max_iters=200)
    return y_hat.value


def l1_trend_filter_agg(df, time_var, lmbda, vars_to_smooth=None):
    x = df[time_var].values
    if vars_to_smooth is None:
        vars_to_smooth = [col for col in df.columns if col != time_var]
    y = df[vars_to_smooth].values
    if y.shape[0] < 4:
        return pd.DataFrame(y, columns=vars_to_smooth, index=df.index)
    feature_std = df[vars_to_smooth].std()
    feature_mean = df[vars_to_smooth].mean()
    normalized_y = y - feature_mean[vars_to_smooth].values[None, :]
    normalized_y = normalized_y / feature_std[vars_to_smooth].values[None, :]
    smoothed = group_l1_trend_filter(x, normalized_y, lmbda)
    smoothed = smoothed * np.clip(
        feature_std[vars_to_smooth].values[None, :], 1e-4, np.inf)
    smoothed = smoothed + feature_mean[vars_to_smooth].values[None, :]
    return pd.DataFrame(smoothed, columns=vars_to_smooth, index=df.index)
