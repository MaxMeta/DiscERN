import os
import numpy as np
from sklearn.metrics import pairwise_distances 
import warnings
import matplotlib.pyplot as plt
#matplotlib.use('PDF') 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from typing import Dict, Optional, Tuple, Any
import json
import csv

def make_dense_vectors(vectors,default_value=0):
    # Get a sorted list of all unique feature names across all vectors
    all_keys = sorted(list(set().union(*(v.keys() for v in vectors.values()))))
    # Convert sparse vectors to dense NumPy arrays
    dense_vectors = {
        name: np.array([v.get(key, default_value) for key in all_keys])
        for name, v in vectors.items()
    }
    return dense_vectors


def analyse_vector_collections(
    ref_names,
    all_vectors,
    k=None,
    x=1.1,
    outlier_std_dev_factor=2.0,
    distance_metric='cosine'):
    
    """
    find outliers in ref vector collection
    find other vectors that should (perhaps) be included in the ref collection
    """

    results = {
        'special_centroid': None,
        'special_distances': None,
        'special_mean_distance': None,
        'special_std_dev_distance': None,
        'special_max_distance': 0.0,
        'special_outliers_by_distance': [],
        'special_clustering_results': None,
        'other_inclusion_threshold': None,
        'other_distances': None,
        'close_other_vectors': [],
        'warnings': []
    }
    if len(ref_names) <2:
        raise ValueError("please provide at least two reference vectors.")

    all_dense_vectors=make_dense_vectors(all_vectors)

        
    ref_names=set(ref_names)
    
    special_vectors=[]
    special_names=[]
    
    other_vectors=[]
    other_names=[]

    for bgc in all_dense_vectors:
        if bgc in ref_names:
            special_vectors.append(all_dense_vectors[bgc])
            special_names.append(bgc)
        else:
            other_vectors.append(all_dense_vectors[bgc])
            other_names.append(bgc)
            

    special_vectors = np.asarray(special_vectors)
    
    n_special, dim_special = special_vectors.shape
    

        
    other_vecs_exist=False 
    
    if len(other_vectors) > 0:
        other_vecs_exist=True
        other_vectors = np.asarray(other_vectors)
        if other_vectors.shape[0]==1:
            other_vectors.reshape(1, -1)


    special_centroid = np.mean(special_vectors, axis=0)
    results['special_centroid'] = special_centroid
    centroid_reshaped = special_centroid.reshape(1, -1)
    special_distances = pairwise_distances(special_vectors, centroid_reshaped, 
                                           metric=distance_metric).flatten()
    
    results['special_distances'] = special_distances
    results['special_mean_distance'] = np.mean(special_distances)
    results['special_max_distance'] = np.max(special_distances)
    results['special_std_dev_distance'] = np.std(special_distances)


    #Identify Outliers by stdev from mean
    if results['special_std_dev_distance'] > 1e-9:
        outlier_distance_threshold = results['special_mean_distance'] + \
                                     outlier_std_dev_factor * results['special_std_dev_distance']
        outlier_indices = np.where(special_distances > outlier_distance_threshold)[0]
        results['special_outliers_by_distance'] = [
            (special_names[i], special_distances[i]) for i in outlier_indices
        ]
    else:
        results['warnings'].append("Outlier detection skipped: all vectors (near) identical")


    #Part 2: Analyze Other Vectors to find additional candidates for inclusion in ref-vecs
    if other_vecs_exist:

        results['other_distances'] = np.array([])
        results['other_inclusion_threshold'] = x * results['special_max_distance']

        # 2b. Calculate Distances of Other Vectors to Special Centroid
        
        other_distances = pairwise_distances(
            other_vectors, centroid_reshaped, metric=distance_metric
        ).flatten()
        results['other_distances'] = other_distances
            

        # 2c. Identify Close Other Vectors
        inclusion_threshold = results['other_inclusion_threshold']
        if inclusion_threshold is not None:
             close_indices = np.where(other_distances <= inclusion_threshold)[0]
             results['close_other_vectors'] = [
                 (other_names[i], other_distances[i]) for i in close_indices
             ]
    elif n_other == 0:
         results['warnings'].append("No other_vectors provided for comparison.")
    elif n_special == 0:
         results['warnings'].append("Comparison with other_vectors skipped: No special_vectors to define centroid.")


    return results



def plot_hclust_dendrogram(
    vector_dict: Dict[str, np.ndarray],
    distance_metric: str = 'cosine',
    linkage_method: str = 'average',
    color_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 7),
    orientation: str = 'right',
    leaf_font_size: Optional[int] = None, # Default will be set based on N
    title: str = 'Hierarchical Clustering Dendrogram',
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[Optional[plt.Axes], Optional[np.ndarray]]:
    """
    Performs hierarchical clustering on vectors stored in a dictionary
    and plots the resulting dendrogram.

    Args:
        vector_dict: Dictionary where keys are labels (str) and values
                     are dense NumPy arrays (vectors). All vectors must
                     have the same dimension.
        distance_metric: The distance metric to use for calculating pairwise
                         distances between vectors. See scipy.spatial.distance.pdist
                         documentation for options (e.g., 'euclidean', 'cosine',
                         'correlation', 'cityblock', etc.). Defaults to 'cosine'.
        linkage_method: The linkage algorithm to use. See
                        scipy.cluster.hierarchy.linkage documentation for options
                        (e.g., 'average', 'complete', 'single', 'ward').
                        'ward' requires Euclidean distance. Defaults to 'average'.
        color_threshold: Distance threshold for coloring clusters. Clusters below
                         this linkage distance will have distinct colors. If None,
                         all branches have the default color. Defaults to None.
        figsize: Tuple representing the figure size (width, height) in inches.
                 Defaults to (10, 7). Ignored if 'ax' is provided.
        orientation: The direction to plot the dendrogram ('top', 'bottom',
                     'left', 'right'). Defaults to 'top'.
        leaf_font_size: Font size for the leaf labels (dictionary keys).
                        If None, a default size (e.g., 8 or 10) is used, which
                        might need manual adjustment if labels overlap.
        title: Title for the plot. Defaults to 'Hierarchical Clustering Dendrogram'.
        xlabel: Label for the x-axis. Defaults based on orientation.
        ylabel: Label for the y-axis. Defaults based on orientation.
        ax: An existing Matplotlib Axes object to plot on. If None, a new
            figure and axes are created. Defaults to None.

    Returns:
        Tuple[Optional[plt.Axes], Optional[np.ndarray]]:
            - The Matplotlib Axes object containing the plot (or None if input invalid).
            - The linkage matrix Z generated by scipy.cluster.hierarchy.linkage
              (or None if input invalid).

    Raises:
        ValueError: If input dictionary is empty, vectors have inconsistent
                    dimensions, or linkage/metric combination is invalid (e.g., 'ward'
                    with 'cosine').
    """
    if not vector_dict:
        warnings.warn("Input vector_dict is empty. Cannot perform clustering.")
        return None, None

    labels = list(vector_dict.keys())
    vectors = list(vector_dict.values())

    if not all(isinstance(v, np.ndarray) for v in vectors):
        raise ValueError("All values in vector_dict must be NumPy arrays.")

    # Input Validation
    try:
        # Stack vectors into a 2D array for pdist
        vector_matrix = np.vstack(vectors)
    except ValueError as e:
        raise ValueError(f"Vectors have inconsistent dimensions: {e}") from e

    if vector_matrix.ndim != 2:
         # Should not happen with vstack if inputs are 1D arrays, but check anyway
         raise ValueError("Could not form a 2D matrix from input vectors.")

    n_vectors = vector_matrix.shape[0]
    if n_vectors < 2:
        warnings.warn("Need at least 2 vectors for clustering. Plotting skipped.")
        return None, None

    if linkage_method == 'ward' and distance_metric != 'euclidean':
        raise ValueError("Ward linkage method requires the 'euclidean' distance metric.")


    if leaf_font_size is None:
        # Heuristic: Smaller font for many leaves, larger for few. Caps at ~10-12.
        leaf_font_size = max(4, min(15, int(300 / n_vectors))) if n_vectors > 15 else 15

    default_ylabel = f'{distance_metric.capitalize()} Distance'
    default_xlabel = 'Sample Index / Cluster'
    if orientation in ('left', 'right'):
        default_xlabel, default_ylabel = default_ylabel, default_xlabel # Swap

    xlabel = xlabel if xlabel is not None else default_xlabel
    ylabel = ylabel if ylabel is not None else default_ylabel

    try:
        # Calculate condensed pairwise distance matrix
        condensed_dist_matrix = pdist(vector_matrix, metric=distance_metric)

        # Perform hierarchical/agglomerative clustering
        Z = linkage(condensed_dist_matrix, method=linkage_method, metric=distance_metric)
    except Exception as e:
        raise RuntimeError(f"Error during distance calculation or linkage: {e}") from e

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure # Get figure from axes

    try:
        dendrogram(
            Z,
            ax=ax,
            labels=labels,
            orientation=orientation,
            leaf_rotation=90 if orientation in ('top', 'bottom') else 0,  # Rotate labels if plotted top/bottom
            leaf_font_size=leaf_font_size,
            color_threshold=color_threshold,
        )
        ax.set_title(title, fontsize=plt.rcParams.get('axes.titlesize', 12))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()

    except Exception as e:
        warnings.warn(f"Error during dendrogram plotting: {e}")
        return None
    


    return ax, condensed_dist_matrix 