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

def average_condensed_dms(
    matrices: List[Union[np.ndarray, list]]
) -> np.ndarray:
    """
    Calculates the element-wise average of a collection of condensed distance matrices.

    A condensed distance matrix is a flat, 1D array representing the upper
    triangle of a square distance matrix, as produced by functions like
    `scipy.spatial.distance.pdist`.

    Args:
        matrices: A list or tuple of 1D NumPy arrays or lists. All matrices
                  in the collection must have the same length.

    Returns:
        A 1D NumPy array representing the average condensed distance matrix.

    Raises:
        ValueError: If the input list is empty, or if the matrices within
                    the list do not all have the same shape.
        TypeError: If elements of the list cannot be converted to NumPy arrays.
    """
    # 1. Input Validation
    if not isinstance(matrices, (list, tuple)) or len(matrices) == 0:
        raise ValueError("Input must be a non-empty list or tuple of distance matrices.")

    try:
        # Convert all elements to NumPy arrays for consistency and efficiency
        matrices_np = [np.asarray(m) for m in matrices]
    except Exception as e:
        raise TypeError(f"All elements in the input collection must be array-like. Error: {e}")

    first_shape = matrices_np[0].shape
    # Check that all matrices have the same shape and are 1D
    if any(m.shape != first_shape for m in matrices_np[1:]):
        raise ValueError("All distance matrices in the collection must have the same shape.")
    if len(first_shape) != 1:
        raise ValueError("Input matrices must be 1D condensed distance matrices.")

    # 2. Averaging using NumPy
    # Stack the 1D arrays into a 2D array where each row is a matrix.
    # For N matrices of length K, this creates an (N, K) array.
    stacked_matrices = np.stack(matrices_np)

    # Calculate the mean along axis 0 (i.e., down the columns). This computes
    # the average for each position across all matrices.
    average_matrix = np.mean(stacked_matrices, axis=0)

    return average_matrix

def build_newick_string(node, labels):
    """
    Recursively builds a Newick string from a SciPy ClusterNode object.
    This function correctly includes branch lengths.

    Args:
        node (scipy.cluster.hierarchy.ClusterNode): The current node in the tree.
        labels (List[str]): The list of leaf labels.

    Returns:
        str: The Newick formatted string for the subtree rooted at this node.
    """
    # If the node is a leaf, return its label. The branch length will be added by the parent call.
    if node.is_leaf():
        return labels[node.id]
    
    # If the node is not a leaf, it has children. Recursively build their strings.
    else:
        # Get the Newick strings for the left and right children
        left_child_str = build_newick_string(node.get_left(), labels)
        right_child_str = build_newick_string(node.get_right(), labels)
        
        # Calculate the branch length for each child.
        # It's the distance of the parent node minus the distance of the child node.
        # For a leaf, its own distance is 0.
        left_branch_length = node.dist - node.get_left().dist
        right_branch_length = node.dist - node.get_right().dist
        
        # Combine them into the Newick format: (left:len,right:len)
        return f"({left_child_str}:{left_branch_length:.6f},{right_child_str}:{right_branch_length:.6f})"


def generate_hclust_tree(
    condensed_dm: np.ndarray, 
    labels: List[str], 
    output_filepath_newick: str, 
    pdf_output_filepath: Optional[str] = None,
    method: str = 'average'
) -> None:
    """
    Performs hierarchical clustering, saves the tree to a Newick file,
    and optionally generates and saves a dendrogram plot to a PDF.

    Args:
        condensed_dm (np.ndarray): A 1D NumPy array representing the condensed
                                   distance matrix.
        labels (List[str]): A list of labels for the items being clustered.
        output_filepath_newick (str): The path to save the output Newick file.
        pdf_output_filepath (Optional[str], optional): The path to save the output
                                                     dendrogram PDF. If None, no
                                                     PDF is created. Defaults to None.
        method (str, optional): The linkage algorithm to use. Defaults to 'average'.

    Raises:
        ValueError: If the number of labels does not match the number of items.
    """
    #Validation
    num_items = int(round((1 + np.sqrt(1 + 8 * len(condensed_dm))) / 2))
    if len(labels) != num_items:
        raise ValueError(
            f"The number of labels ({len(labels)}) does not match the number of "
            f"items ({num_items}) inferred from the distance matrix."
        )

    #Perform hierarchical clustering
    print(f"Performing hierarchical clustering using the '{method}' method...")
    linkage_matrix = linkage(condensed_dm, method=method)

    if pdf_output_filepath:
        print(f"Generating dendrogram plot for PDF output...")
        try:
            # Dynamically adjust figure height based on the number of labels
            # This prevents labels from overlapping on large trees
            fig_height = max(8, len(labels) * 0.25)
            fig, ax = plt.subplots(figsize=(10, fig_height))

            dendrogram(
                linkage_matrix,
                labels=labels,
                orientation='right', # 'right' or 'left' is better for many labels
                leaf_font_size=8,
                ax=ax
            )
            
            ax.set_title(f"Hierarchical Clustering Dendrogram (Method: {method})")
            ax.set_xlabel("Distance")
            plt.tight_layout() # Adjust plot to ensure everything fits
            
            # Save the figure to a PDF
            plt.savefig(pdf_output_filepath, format='pdf')
            plt.close(fig) # Close the figure to free up memory
            print(f"Successfully saved dendrogram plot to: {pdf_output_filepath}")
        except Exception as e:
            print(f"Error generating or saving PDF: {e}")
    
    #Convert the linkage matrix to a root ClusterNode object
    print("Converting linkage matrix to a tree object using to_tree()...")
    tree = to_tree(linkage_matrix, rd=False)

    #Build the Newick string from the tree structure using our helper function
    print("Generating Newick tree string...")
    newick_string = build_newick_string(tree, labels) + ";"

    #Save the Newick string to a file
    try:
        with open(output_filepath_newick, 'w') as f:
            f.write(newick_string)
        print(f"Successfully saved Newick tree to: {output_filepath_newick}")
    except IOError as e:
        print(f"Error saving Newick file: {e}")

def get_vecs_for_trees(passing_hits_by_k,min_k=3):
    """
    get gbk paths (keys) to use in tree making
    
    """
    to_return=[]
    for k in (1,2,3,4):
        if k>=min_k and k in passing_hits_by_k:
            to_return.extend(passing_hits_by_k[k])
    return to_return

def split_on_second_to_last(path_str):
    """
    Splits a path at the second-to-last separator using os.path.split().
    Returns a tuple (head, tail).
    """
    temp_head, tail1 = os.path.split(path_str)
    head, tail2 = os.path.split(temp_head)
    tail = os.path.join(tail2, tail1)
    
    return head, tail

def make_trees(out_folder_path,
               vecs_to_use,
               refs,
               bs_vecs,
               cb_vecs,
               pol_vecs,
               mibig_vecs_bs,
               mibig_vecs_cb,
               mibig_vecs_pol,
               use_pol=True,
               use_bs=True,
               use_cb=True
              ):
    
    """
    make tree files from hits and references using list returned by
    get_vecs_for_trees
    """
    
    cb_ax=None
    cb_mat=None
    bs_ax=None
    bs_mat=None
    pol_mat=None
    pol_ax=None
    
    if use_pol:
        pol_subset={}
        #pol_subset_labels=[] #just use vecs_to_use + ref
        for vec in vecs_to_use:
            strain,bgc=os.path.split(vec)
            #print(strain,">>",bgc)
            json_name=os.path.split(strain)[-1][:-3]+'json'
            pv_key=os.path.join(strain,json_name)
            bgc_number=bgc.split(".")[-2].split('region')[-1].lstrip("0")
            contig_name=bgc.split(".region")[0]
            pv_key2=contig_name+"::"+bgc_number
            #print(pv_key,pv_key2)
            pol_subset[split_on_second_to_last(vec)[-1]]=pol_vecs[pv_key][pv_key2]

        for ref in refs:
            pol_subset[ref]=mibig_vecs_pol[ref]
            
        pol_mat,_,pol_ax=hierarchical_cluster_sets(list(pol_subset.values()),
                                           list(pol_subset.keys()))
        labels=list(pol_subset.keys())
        pol_ax.figure.savefig(os.path.join(out_folder_path,'pol_denrogram.pdf'))

    if use_bs:
        bs_subset={}
        #pol_subset_labels=[] #just use vecs_to_use + ref
        for vec in vecs_to_use:
            bs_subset[split_on_second_to_last(vec)[-1]]=bs_vecs[vec]

        for ref in refs:
            bs_subset[ref]=mibig_vecs_bs[ref]
           
        bs_subset=make_dense_vectors(bs_subset)
        bs_ax, bs_mat=plot_hclust_dendrogram(bs_subset)
        labels=list(bs_subset.keys())
        bs_ax.figure.savefig(os.path.join(out_folder_path,'bs_denrogram.pdf'))


    if use_cb:
        cb_subset={}
        #pol_subset_labels=[] #just use vecs_to_use + ref
        for vec in vecs_to_use:
            cb_subset[split_on_second_to_last(vec)[-1]]=cb_vecs[vec]

        for ref in refs:
            cb_subset[ref]=mibig_vecs_cb[ref]
            
        cb_subset=make_dense_vectors(cb_subset)
        cb_ax, cb_mat=plot_hclust_dendrogram(cb_subset)
        labels=list(cb_subset.keys())
        cb_ax.figure.savefig(os.path.join(out_folder_path,'cb_denrogram.pdf'))

            


    mean_matrix=average_condensed_dms([i for i in [cb_mat,bs_mat,pol_mat] if \
                                       type(i)==np.ndarray])
    newick_out_path=os.path.join(out_folder_path,"newick_tree.txt")
    pdf_out_path=os.path.join(out_folder_path,"combined_tree.pdf")
    generate_hclust_tree(mean_matrix,labels,newick_out_path,pdf_out_path)