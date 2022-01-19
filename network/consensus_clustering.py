import datetime
from joblib import Parallel, delayed
import numpy as np
import time

import network.utils as utils
import data_processing.study_constants as constants


AGE_ORDER = constants.AGE_ORDER
META_COLS = ['mouse_id', 'age_group', 'run_number', 'age_in_months', 'age',
             'fraction_of_lifespan', 'days_to_death', 'trace_date']


def sample_covariances(df, n=10, col_subsample=0.9, row_subsample=1.0):
    """Subsample features and examples for consensus clustering.

    Takes in a dataframe of observations by features (with additional
    metadata columns that have to be removed).

    Subsamples `n` sub-matrices per age-group - `col_subsample` fraction
    of features (columns) are subsampled without replacement and
    `row_subsample` fraction of rows are subsampled with replacement.

    Covariance is estimated using the nonparanormal SKEPTIC estimator,
    and returned as a nested list / 2d array of n by number of age groups,
    along with an array of n lists of the features sampled.
    """
    all_sampled_covariances = []
    sampled_features = []
    feature_df = df.drop(META_COLS, axis=1)
    num_cols = feature_df.shape[1]

    for i in range(n):
        print('>>>> Sampling covariances -- %d' % i)
        sample_cols = np.random.choice(
            feature_df.columns, size=int(num_cols * col_subsample),
            replace=False)
        sampled_features.append(sample_cols)
        sampled_covariances = []
        for age_group in AGE_ORDER:
            print('>>>> Sampling covariances -- %d (age group: %s)' %
                  (i, age_group))
            mask = df['age_group'] == age_group
            subset_df = feature_df[mask]
            num_rows = subset_df.shape[0]
            sample_indices = np.random.choice(
                subset_df.index, size=int(num_rows * row_subsample),
                replace=True)
            sampled_data = subset_df.loc[sample_indices, sample_cols]
            cov = utils.estimate_covariance(sampled_data)
            sampled_covariances.append(cov)
        all_sampled_covariances.append(sampled_covariances)
    return all_sampled_covariances, sampled_features


def get_consensus(affinity, ks, features):
    """Takes in an affinity matrix and returns consensus matrix for k in `ks`.

    Repeatedly clusters the affinity matrix with num clusters in `ks` and
    returns the affinity matrix for each value of k. The affinity matrix is
    a matrix A_ij = 1 if feature i and feature j are in the same cluster
    and 0 otherwise.
    """
    agreement_counts = [np.zeros((len(features), len(features))) for _ in ks]
    mask = np.in1d(features, affinity.columns).astype(float)
    row_idx, col_idx = np.where(features[:, None] ==
                                affinity.columns.values[None, :])
    col_idx = row_idx[np.argsort(col_idx)]
    indicator_counts = mask[:, None] * mask[None, :]
    affinity = np.abs(affinity.values)
    for i, k in enumerate(ks):
        spec_cluster = utils.cluster_nodes(affinity, n_clusters=k)
        labels = spec_cluster.labels_
        tmp = agreement_counts[i][col_idx]
        tmp[:, col_idx] = labels[:, None] == labels[None, :]
        agreement_counts[i][col_idx] = tmp
    return agreement_counts, indicator_counts


def get_stability_curves(df, ks, n_samples=10, seed=0):
    """Bootstrap samples covariance matrices and computes consensus matrix.

    Main function of consensus clustering that:

    1. Bootstrap samples feature matrix, split by age-group and computes the
    nonparanormal SKEPTIC estimator for the empirical cov for each sample.
    2. Runs TVGL using the age-groups as the time covariate to infer the
    regularized precision matrix for each (age-group, sample).
    3. For each age-group, spectral clusters TVGL inferred precision matrix
    from each sample and gets consensus matrix for k in `ks`.
    """
    print('>>>> Starting seed %d ...' % seed)
    np.random.seed(seed=seed)
    consensus_matrices = {}
    indicator_matrices = {}
    all_features = df.drop(META_COLS, axis=1).columns.values
    sampled_covariances, sampled_features = sample_covariances(df, n=n_samples)
    all_thetas = []

    # Compute precision matrices using TVGL
    start_time = time.time()
    cov_and_features = zip(sampled_covariances, sampled_features)
    for i, (covariance_set, features) in enumerate(cov_and_features):
        print(">>>> Computing theta set %d" % i)
        all_thetas.append(utils.get_inferred_precision_matrices(
           covariance_set, features))
        elapsed_time = time.time() - start_time
        print(">>>> Computed theta set %d (%0.1f seconds)" % (i, elapsed_time))

    precision_matrices_by_age_group = zip(*all_thetas)

    age_group_and_precision = zip(AGE_ORDER, precision_matrices_by_age_group)
    for age_group, precision_matrices in age_group_and_precision:
        consensus_matrices[age_group] = {}
        print('>>>> Processing: %s' % age_group)
        counts = Parallel(n_jobs=16)(
            delayed(get_consensus)(precision_matrix, ks, all_features)
            for precision_matrix in precision_matrices)
        agreement_matrices, count_matrices = zip(*counts)
        indicator_counts = sum(count_matrices)
        cluster_agreement_matrices = zip(*agreement_matrices)
        indicator_matrices[age_group] = indicator_counts
        for k, cluster_agreement_matrix in zip(ks, cluster_agreement_matrices):
            cluster_agreement_matrix = sum(cluster_agreement_matrix)
            consensus_matrices[age_group][k] = cluster_agreement_matrix
        elapsed_time = time.time() - start_time
        now = datetime.datetime.now()
        current_time = now.strftime("%c")
        print('>>>> %s - Done with %s took %f seconds' %
              (current_time, age_group, elapsed_time))
    return consensus_matrices, indicator_matrices


def get_theta(df, lmbda=0.5, beta=1.0):
    """Estimate precision matrix using TVGL and SKEPTIC estimator."""
    covariances = []
    feature_df = df.drop(META_COLS, axis=1)
    for age_group in AGE_ORDER:
        print('>>>> Estimating covariance (age group: %s)' % (age_group))
        mask = df['age_group'] == age_group
        subset_df = feature_df[mask]
        cov = utils.estimate_covariance(subset_df)
        covariances.append(cov)

    # Compute precision matrices using TVGL
    start_time = time.time()
    print(">>>> Computing thetas")
    thetas = utils.get_inferred_precision_matrices(
            covariances, feature_df.columns, lmbda=lmbda, beta=beta)
    print(">>>> Computed thetas (%0.1f seconds)" % (time.time() - start_time))
    return thetas
