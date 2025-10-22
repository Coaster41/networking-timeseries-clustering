import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import feature_calculators
from typing import Optional, Union, Tuple, Iterable, Callable
from sklearn.mixture import GaussianMixture
import json
from warnings import warn
from datetime import timedelta, datetime
import matplotlib.dates as mdates


CENTRAL_FEATURES = ["mean", "truncated_mean", "quantile", "fft_aggregated"]

TSFRESH_SETTINGS = {
    "variation_coefficient": None, 
    "mean": None
}
VALUE_COLUMN = "rtt"
TIME_COLUMN = "timestamp"

def get_fc_settings(path="data/potential_features.json"):
    fc_settings = TSFRESH_SETTINGS

    with open(path, "r") as f:
        fc_settings = json.load(f)

    for f_or_function_name, args in list(fc_settings.items()):
        func = getattr(feature_calculators, f_or_function_name, None)
        if func is None:
            func = globals().get(f_or_function_name, None)
            if func is None:
                warn(f"Function {f_or_function_name} not found. Ignored.")
            else:
                del fc_settings[f_or_function_name]
                fc_settings[func] = args
    return fc_settings

def truncated_mean(x: Union[pd.Series, np.ndarray], q: float) -> float:
    """
    Args:
        x: The timeseries to extract features of
        q: The percentage of points higher and lower than p to ignore (0, 0.5)
    Returns:
        The truncated mean with the outlier values ignored
    """
    lower_bound = np.quantile(x, q)
    upper_bound = np.quantile(x, 1-q)
    valid_indices =  np.where((x > lower_bound) & (x < upper_bound))
    return np.mean(x[valid_indices])


def cluster(features: np.ndarray, n_clusters: int, max_starts: int = 75
        ) -> Tuple[Callable, np.ndarray, np.ndarray, float]:
    """
    Cluster the given features into n_clusters max_starts number of times, 
        and maximize the mean log expectation of the Gaussian Mixture Model
    
    Args:
        features: The features to cluster
        n_clusters: The number of clusters (mixture components to have)
        max_starts: The max number of random states to try
    Returns:
        A tuple of (The gmm model, 
            The predicted labels (array of shape (n_samples, )),
            The predicted scores of each sampler for each component (array of shape (n_samples, n_components)),
            The mean log expectation)
    """

    #scaled_features = scale(features, with_mean=True)
    scaled_features = features
    scores = []

    for i in range(max_starts):
        gmm = GaussianMixture(random_state=i, 
            n_components=n_clusters, n_init=1, init_params="k-means++", 
            covariance_type="diag", max_iter=300)
        labels = gmm.fit_predict(scaled_features)
        score = gmm.score(scaled_features)
        scores.append([gmm, labels, gmm.predict_proba(scaled_features), score])

    scores.sort(key= lambda x:x[-1])
    return scores[-1]


def _train_single_tree(seed, scaled_features, labels):
    """
    Helper function to train a single decision tree with a given seed.
    This function will be executed in parallel.
    
    Args:
        seed: Random seed for the decision tree
        scaled_features: Scaled feature matrix
        labels: Cluster labels from GMM
    
    Returns:
        Feature importances from the trained decision tree
    """
    from sklearn.tree import DecisionTreeClassifier
    
    dt = DecisionTreeClassifier(random_state=seed, class_weight="balanced")
    dt.fit(scaled_features, labels)
    return dt.feature_importances_

from joblib import Parallel, delayed


def calculate_important_features_joblib(features: pd.DataFrame, n_clusters: int, 
                                       n_seeds: int = 75, n_jobs: int = -1) -> np.ndarray:
    """
    Implementation using joblib - recommended for scikit-learn projects.
    Joblib is optimized for numpy arrays and scientific computing.
    
    Args:
        features: The features returned by extract_trace_features, a DataFrame of (n_traces, n_features)
        n_clusters: The number of clusters to cluster with
        n_seeds: The number of seeds of the decision tree to compute the mean of the scores over.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
    
    Returns:
        The mean importance, a np.ndarray of shape (n_features)
    """
    from sklearn.preprocessing import scale
    
    # Scale features and get cluster labels (sequential)
    #scaled_features = scale(features.values, with_mean=True)
    scaled_features = features.values # get only the values from the dataframe
    _, labels, *___ = cluster(scaled_features, n_clusters)
    
    # Parallel execution using joblib
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_train_single_tree)(seed, scaled_features, labels) 
        for seed in range(n_seeds)
    )
    
    return np.mean(np.array(scores), axis=0)


def filter_features(features: pd.DataFrame, features_to_keep: float = 0.5, steps: int = 5) -> pd.DataFrame:
    """
    Filter the extracted features by selecting the most important features while
        iteratively increasing the number of clusters and decreasing the number of clusters.
        This strategy allows us to keep a majority of the features when the clustering is poorer 
            (there are too many features to cluster over), and reduce them when the clustering is better.
        When filtering, the top features selected from 
            [features the describe the central tendancy, features that describe the spread] in a balanced way

    Args:
        features: The features to filter, a DataFrame returned by extrace_trace_features
        features_to_keep: The fraction (0, 1) of features to keep
        steps: The number of iterations to carry out
    Returns:
        The filtered version of the features
    """
    features_to_keep = int(features_to_keep * features.shape[1])
    features_to_keep = max(2, features_to_keep)

    n_clusters_counter = np.linspace(3, 7, 
        num=steps, endpoint=True, dtype=np.int64)
    n_features_counter = np.linspace(int(features.shape[1] * .975), features_to_keep, 
        num=steps, endpoint=True, dtype=np.int64)

    are_central_features = [feature.split("__")[1] in CENTRAL_FEATURES for feature in features.columns]
    are_central_features = np.array(are_central_features)
    central_features = features.columns[are_central_features]
    variance_features = features.columns[~are_central_features]
    
    filtered_features = features.copy()
    for n_clusters, n_features in zip(n_clusters_counter, n_features_counter):
        divided_n_features = np.array([(n_features + 1) // 2, n_features // 2], dtype=int) 
        all_scores = calculate_important_features_joblib(filtered_features, n_clusters = n_clusters, n_jobs=-1)

        divided_scores = [[], []]
        for score, feature_name in zip(all_scores, filtered_features.columns):
            for facet_i, facet in enumerate([central_features, variance_features]):
                if feature_name in facet:
                    divided_scores[facet_i].append([feature_name, score])
        relevant_features = []
        for faceted_scores, faceted_n_features in zip(divided_scores, divided_n_features):
            faceted_scores.sort(key=lambda x: x[-1], reverse=True)
            relevant_features.extend(
                [feature_name for feature_name, _ in faceted_scores[:faceted_n_features]])
        filtered_features = filtered_features.loc[:, relevant_features]
    
    return filtered_features

def _evaluate_single_cluster_count(n_clusters, scaled_features, min_cluster_size):
    """
    Helper function to evaluate clustering with a specific number of clusters.
    This function will be executed in parallel.
    
    Args:
        n_clusters: Number of clusters to test
        scaled_features: Scaled feature matrix
        min_cluster_size: Minimum cluster size threshold
    
    Returns:
        Tuple of (n_clusters, labels, dist, score) or (n_clusters, None, None, -1) if invalid
    """
    from sklearn.metrics import silhouette_score
    
    try:
        gmm_model, labels, dist, ___ = cluster(scaled_features, n_clusters=n_clusters)
        _, counts = np.unique(labels, return_counts=True)
        score = silhouette_score(scaled_features, labels, metric="euclidean")
        
        # Check minimum cluster size constraint
        if np.amin(counts) <= min_cluster_size:
            score = -1
            
        return (gmm_model, n_clusters, labels, dist, score)
    
    except Exception as e:
        # Handle any clustering failures
        print(f"Error clustering with {n_clusters} clusters: {e}")
        return (None, n_clusters, None, None, -1)


def determine_n_clusters_joblib(features: pd.DataFrame,
                              min_clusters: int = 6,
                              max_clusters: int = 15, 
                              min_cluster_size: float = 0.0001, 
                              return_labels: bool = False,
                              n_jobs: int = -1) -> Union[int, Tuple[int, np.ndarray]]:
    """
    Joblib implementation - recommended for scikit-learn compatibility.
    
    Args:
        features: The features returned by tsfresh of the traces
        min_clusters: The minimum number of clusters to try
        max_clusters: The maximum number of clusters to try
        min_cluster_size: If the smallest cluster is smaller than 
            this fraction (proportional to the number of traces), this clustering is ignored.
        return_labels: Whether or not to return the ideal cluster labels found
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
    
    Returns:
        The ideal number of clusters, if return_labels is False
        (number of clusters, labels, dist, scores), otherwise 
    """    
    from sklearn.preprocessing import scale
    
    # Scale features once (sequential)
    #scaled_features = scale(features.values, with_mean=True)
    scaled_features = features.values
    min_cluster_size_abs = min_cluster_size * features.shape[0]
    
    # Parallel evaluation of different cluster counts
    cluster_range = range(min_clusters, max_clusters + 1)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_single_cluster_count)(n_clusters, scaled_features, min_cluster_size_abs)
        for n_clusters in cluster_range
    )
    
    # Process results and find best clustering
    scores = []
    for gmm_model, n_clusters, labels, dist, score in results:
        scores.append([gmm_model, n_clusters, labels, dist, score])
    
    # Sort by score to find the best clustering
    sorted_scores = sorted(scores, key=lambda x: x[-1])
    print("Lowest score: " , sorted_scores[0][-1])
    print("Highest score: ", sorted_scores[-1][-1])
    best_gmm_model, best_n_clusters, best_labels, best_dist, score = sorted_scores[-1]
    
    if return_labels:
        return (best_gmm_model,best_n_clusters, best_labels, best_dist, scores, score)
    else:
        return best_n_clusters
    

def get_features(df_raw, fc_settings, top_feature_names=None):
    tsfresh_df = df_raw.rename(columns={VALUE_COLUMN: "value"})

    extracted_features = extract_features(tsfresh_df, column_id="id", 
        column_sort=TIME_COLUMN, default_fc_parameters=fc_settings, 
        disable_progressbar=False, impute_function=impute, n_jobs=16)
    extracted_features = extracted_features.sort_index()
    if top_feature_names == None:
        filtered_features = filter_features(extracted_features, features_to_keep=0.3, steps=5)
        top_feature_names = filtered_features.columns.tolist()
    else:
        filtered_features = extracted_features[top_feature_names]

    return filtered_features, top_feature_names


def cluster_features(filtered_features, df_raw, gmm=None):
    # cluster filtered features
    if gmm == None:
        gmm, cluster_number, cluster_labels, \
            dist, results, score = determine_n_clusters_joblib(filtered_features, return_labels=True)
    else:
        cluster_labels = gmm.predict(filtered_features)
    # print("Number of Clusters: ", cluster_number)
    # print(f"Cluster counts: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")
    id_list = filtered_features.index.unique()
    cluster_map = dict(zip(id_list, cluster_labels))

    # Create df_clustered with labeled cluster
    df_clustered = df_raw.copy()
    df_clustered["cluster"] = df_clustered["id"].map(cluster_map)
    return df_clustered, cluster_map, gmm

def save_clusters(save_dir, df_clustered, cluster_map):
    cluster_results = df_clustered.groupby(["id"])["timestamp"].min().dt.round("s").reset_index()
    cluster_results["cluster"] = cluster_results["id"].map(cluster_map)
    cluster_results.to_csv(f"{save_dir}/cluster_results.csv")

