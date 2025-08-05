import os
#import re
#import random
#import sys
import json
from Bio import SeqIO
import glob
from collections import defaultdict
from typing import List, Dict, Union, Optional, Set,Tuple, Any, Hashable
from numbers import Real
from scipy.spatial.distance import euclidean, cosine, cityblock, jaccard, braycurtis
#from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
#import seaborn as sns
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
from tqdm import tqdm
from io import StringIO
from pyhmmer import plan7, hmmer
from pyhmmer.easel import TextSequence, Alphabet
import argparse
import math
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
import subprocess

####################################
#-bigslice-vector-making-functions-#
####################################

#BGC vectorisation method developed by Satria A. Kautsar, JGI
#Copyright (C) 2022 Satria A. Kautsar

def extract_bgcs_from_regiongbks(path_list):
    #this function adapted from orignal written by Satria A. Kautsar, JGI
    #Copyright (C) 2022 Satria A. Kautsar

    # parse record and extract bgcs
    results = []
        
    for fp in path_list:
        for contig in SeqIO.parse(fp, "genbank"):
            contig_edge = False
            bgc = {
                "name": fp,
                "cds": []
            }
            for feature in contig.features:
                if feature.type == "CDS":
                    bgc["cds"].append(
                        TextSequence(name=bytes("{}-{}".format(
                            len(results),
                            len(bgc["cds"])
                        ), "utf-8"), sequence=feature.qualifiers["translation"][0]).digitize(Alphabet.amino())
                    )
                    
            results.append(bgc)
            
    return results

def extract_features(bgcs, hmmdb_folder, top_k=3, num_cpus=0):
    
    #this function adapted from orignal written by Satria A. Kautsar, JGI
    #Copyright (C) 2022 Satria A. Kautsar
    
    # store hmmdb model data
    biosyn_pfams = []
    with open(os.path.join(hmmdb_folder, "biosynthetic_pfams", "Pfam-A.biosynthetic.hmm"), "r") as ii:
        for line in ii:
            if line.startswith("NAME "):
                biosyn_pfams.append(line.split("NAME ")[-1].lstrip().rstrip())
    core_pfams = []
    sub_pfams = []
    for fp in glob.iglob(os.path.join(hmmdb_folder, "sub_pfams", "hmm", "*.subpfams.hmm")):
        hmm_acc = os.path.basename(fp).split(".subpfams.hmm")[0]
        core_pfams.append(hmm_acc)
        with open(fp, "r") as ii:
            for line in ii:
                if line.startswith("NAME "):
                    sub_pfams.append(line.split("NAME ")[-1].lstrip().rstrip())
    
    # perform biosyn_scan
    biosyn_hits = []
    subpfam_to_scan = {}
    biosyn_pfam_model = os.path.join(hmmdb_folder, "biosynthetic_pfams", "Pfam-A.biosynthetic.hmm")
    sequences = [x for y in [bgc["cds"] for bgc in bgcs] for x in y]
    with plan7.HMMFile(biosyn_pfam_model) as hmm_file:
        for top_hits in hmmer.hmmsearch(
            hmm_file, sequences,
            cpus=num_cpus,
            bit_cutoffs="gathering"
            ):
            for hit in top_hits:
                if hit.best_domain.score < top_hits.domT:
                    continue
                bgc_id, cds_id = list(map(int, hit.name.decode().split("-")))
                hmm_name = top_hits.query_accession.decode()
                alignment = hit.best_domain.alignment
                biosyn_hits.append((bgc_id, hmm_name, 255))
                
                # check if need subpfam scan
                if hmm_name in core_pfams:
                    if hmm_name not in subpfam_to_scan:
                        subpfam_to_scan[hmm_name] = []
                    subpfam_to_scan[hmm_name].append(TextSequence(name=bytes("{}-{}".format(
                        bgc_id,
                        len(subpfam_to_scan[hmm_name])
                    ), "utf-8"), sequence=alignment.target_sequence).digitize(Alphabet.amino()))
                
    # perform subpfam_scan
    subpfam_hits = []
    for hmm_name, sequences in subpfam_to_scan.items():
        sub_pfam_model = os.path.join(
            hmmdb_folder, "sub_pfams", "hmm", "{}.subpfams.hmm".format(
                hmm_name
            )
        )
        with plan7.HMMFile(sub_pfam_model) as hmm_file:
            parsed = {}
            for top_hits in hmmer.hmmsearch(
                hmm_file, sequences,
                cpus=num_cpus,
                T=20, domT=20
                ):
                for hit in top_hits:
                    if hit.best_domain.score < top_hits.domT:
                        continue
                    hsp_name = hit.name.decode()
                    hmm_name = top_hits.query_name.decode()
                    score = hit.best_domain.score
                    if hsp_name not in parsed:
                        parsed[hsp_name] = {}
                    if hmm_name in parsed[hsp_name]:
                        parsed[hsp_name][hmm_name] = max(
                            score, parsed[hsp_name][hmm_name]
                        )
                    else:
                        parsed[hsp_name][hmm_name] = score
                    
            for hsp_name, hits in parsed.items():
                k = 0
                for hmm_name, score in sorted(hits.items(), key=lambda n: n[1], reverse=True):
                    if k >= top_k:
                        break
                    bgc_id, hsp_id = list(map(int, hsp_name.split("-")))
                    subpfam_hits.append((bgc_id, hmm_name, 255 - int((255 / top_k) * k)))
                    k += 1
    
    df = pd.DataFrame(
        [*biosyn_hits, *subpfam_hits],
        columns = ["bgc_id", "hmm_name", "value"]
    ).sort_values("value", ascending=False).drop_duplicates(["bgc_id", "hmm_name"])
    
    df = pd.pivot(
        df,
        index="bgc_id", columns="hmm_name", values="value"
    ).reindex(
        [*biosyn_pfams, *sub_pfams], axis="columns"
    ).fillna(0).astype(int)
    
    df.index = df.index.map(lambda i: bgcs[i]["name"])
    
    return df

def dataframe_to_sparse_dict( 
    df: pd.DataFrame,
    fill_value: Union[int, float] = 0,
    include_nan: bool = False
) -> Dict[Hashable, Dict[str, Union[float, int, str]]]:
    """
    Converts a Pandas DataFrame to a nested sparse dictionary representation
    using the DataFrame's index. Handles column names that are not valid
    Python identifiers.

    The outer dictionary keys are taken from the DataFrame's index.
    The inner dictionaries map feature column names to their values for that index,
    but only include entries where the value is not equal to `fill_value`
    (and optionally not NaN).

    Args:
        df: The input Pandas DataFrame with a set index.
        fill_value: The value to treat as "sparse" or "default". Entries equal
                    to this value will be excluded from the inner dictionaries.
                    Typically 0 for numeric data.
        include_nan: If False (default), NaN values will also be excluded from
                     the inner dictionaries, similar to `fill_value`. If True,
                     NaN values will be included if they are not equal to
                     `fill_value`.

    Returns:
        A nested dictionary where outer keys are from `df.index` and inner
        dictionaries contain {feature_name: value} pairs for non-fill_value
        (and optionally non-NaN) entries in that row.

    Raises:
        TypeError: If values in `df.index` are not hashable (cannot be dict keys).
    """

    sparse_representation: Dict[Hashable, Dict[str, Any]] = {}

    # Check if index values are hashable before iterating
    try:
        _ = {item for item in df.index}
    except TypeError:
        first_non_hashable = next((type(item) for item in df.index if not isinstance(item, Hashable)), "Unknown")
        raise TypeError(f"Values in DataFrame index are not hashable (e.g., type {first_non_hashable} found). Cannot use as dictionary keys.")

    # Get original feature column names
    feature_cols: List[str] = list(df.columns)
    # Create a mapping from original column name to its position IN THE TUPLE yielded by itertuples
    # The tuple structure is (Index, col1_value, col2_value, ...)
    # So the column at df.columns[0] is at tuple index 1, df.columns[1] is at tuple index 2, etc.
    col_name_to_tuple_idx: Dict[str, int] = {
        col_name: i + 1 for i, col_name in enumerate(feature_cols)
    }

    # Iterate through rows efficiently using itertuples
    # index=True includes the index as the first element (at tuple index 0)
    for row_tuple in df.itertuples(index=True, name=None): # Use name=None for plain tuple
        key: Hashable = row_tuple[0] # Index is always at position 0
        inner_dict: Dict[str, Any] = {}

        # Iterate using the original column names
        for feature_name in feature_cols:
            # Get the value using its positional index in the tuple
            value_index = col_name_to_tuple_idx[feature_name]
            value: Any = row_tuple[value_index]

            # --- Condition for inclusion (same logic as before) ---
            is_nan = pd.isna(value)
            include_this = True

            if not include_nan and is_nan:
                include_this = False
            elif value == fill_value:
                 if pd.isna(fill_value):
                     if is_nan: # Both are NaN, so exclude
                        include_this = False
                 else: # fill_value is not NaN, simple comparison is enough
                     include_this = False

            if include_this:
                # Use the ORIGINAL feature name as the key in the inner dict
                inner_dict[feature_name] = value
            # --- End condition logic ---


        # Only add the key to the outer dictionary if the inner dict is not empty
        if inner_dict:
            if key in sparse_representation:
                 # Handle duplicate keys in index - overwrite with the last one found
                 print(f"Warning: Duplicate key '{key}' found in DataFrame index. Overwriting with last occurrence.")
            sparse_representation[key] = inner_dict

    return sparse_representation
    

def make_sparse_dict(antismash_folder,
                     hmmdb_folder, top_k=3, num_cpus=0,
                     glob_pattern=None,chunk_size=500):
    
    available_cpus = psutil.cpu_count(logical=False)
    if num_cpus == 0 or num_cpus > available_cpus:
        num_cpus = available_cpus

    print(f'making bigslice vectors with {num_cpus} cpus')
    
    if glob_pattern:
        
        gbk_files=glob.glob(os.path.join(antismash_folder,glob_pattern))
    else:
        gbk_files=glob.glob(os.path.join(antismash_folder,"*/*.region*.gbk"))


    if not gbk_files:
        print("No gbk files found matching the pattern. Exiting.")
        exit()

    total_files = len(gbk_files)
    print(f"Found {total_files} files to process.")
    
    # Calculate the number of chunks
    num_chunks = math.ceil(total_files /chunk_size)
    print(f"Dividing gbk list into {num_chunks} chunks of up to {chunk_size} files each.")
    
    collated_data = {}
    
    # Iterate through the list in chunks using slicing
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        # The slice automatically handles the end boundary correctly for the last chunk
        current_chunk_filenames = gbk_files[start_index:end_index]
    
        chunk_num = i + 1
        print(f"Processing chunk {chunk_num}/{num_chunks} ({len(current_chunk_filenames)} files)...")
    
        # Process the current list of filenames

        chunk_seqs=extract_bgcs_from_regiongbks(current_chunk_filenames)
        chunk_df=extract_features(chunk_seqs,hmmdb_folder,num_cpus=num_cpus)
        chunk_dict=dataframe_to_sparse_dict(chunk_df)
        
    
        # Update the main dictionary
        collated_data.update(chunk_dict)
    

    
    print(f"Finished processing {num_chunks} chunks.")
    return collated_data

##############################
#-feature-counting-functions-#
##############################

def make_counts(gbk_file: str,
                   search_criteria: Dict[str, Dict[str, List[str]]]) -> Dict[str, Union[int, Dict[str, int]]]:
    """
    Counts occurrences of specified domain patterns within feature types/qualifiers.

    Args:
        gbk_file: Path to the GenBank file.
        search_criteria: A dictionary defining feature types, qualifiers,
                         and domain patterns.  Structure:
                         {
                             feature_type_1: {
                                 feature_qualifier_1: [domain_pattern_1, ...],
                                 ...
                             },
                             ...
                         }

    Returns:
        A nested dictionary of counts, simplified if only one
        feature/qualifier combo exists.
    """

    if not search_criteria:
        raise ValueError("search_criteria cannot be empty.")

    try:  # Added try-except block to handle file parsing errors.
        parsed_gbk = SeqIO.parse(gbk_file, 'genbank')
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for record in parsed_gbk:
            for feature in record.features:
                feature_type = feature.type
                if feature_type in search_criteria:
                    for qualifier, patterns in search_criteria[feature_type].items():
                        if qualifier in feature.qualifiers:
                            for domain_string in feature.qualifiers.get(qualifier, []):
                                for pattern in patterns:
                                    if pattern in domain_string:
                                        counts[feature_type][qualifier][pattern] += 1
    except Exception as e:
        print(f"Error parsing file {gbk_file}: {e}")
        return {}  # Return an empty dictionary on error

    # Convert defaultdict to regular dict (important for JSON serialisation)
    final_counts = {
        ftype: {q: dict(domain_counts) for q, domain_counts in qualifier_counts.items()}
        for ftype, qualifier_counts in counts.items()
    }

    # Simplify if only one feature/qualifier combination exists
    if len(final_counts) == 1:
        first_feature = list(final_counts.keys())[0]
        if len(final_counts[first_feature]) == 1:
            first_qualifier = list(final_counts[first_feature].keys())[0]
            return final_counts[first_feature][first_qualifier]

    return final_counts


def process_file_chunk(file_chunk,feature_dict):
    """Processes a chunk of files, returning a dictionary of results."""
    results = {}
    for file_path in file_chunk:
        results[file_path] = make_counts(file_path, feature_dict)
    return results


def make_all_feature_counts(as_dir,num_cpus, feature_dict, glob_pattern=None):

    if not glob_pattern:
    
        gbk_files=glob.glob(os.path.join(as_dir,'*','*region*.gbk'))

    #assumes glob_pattern is valid for os/file system
    else:
        gbk_files=glob.glob(os.path.join(as_dir,glob_pattern))
        


    if not gbk_files:
        print("No GBK files found.")
        return

    available_cpus = psutil.cpu_count(logical=False)
    if num_cpus == 0 or num_cpus > available_cpus:
        num_cpus = available_cpus

    if num_cpus <= 0:
        num_cpus = 1

    print(f'processing with {num_cpus} cpus')


    chunk_size = (len(gbk_files) + num_cpus - 1) // num_cpus
    file_chunks = [gbk_files[i:i + chunk_size] for i in range(0, len(gbk_files), chunk_size)]

    count_dict = {}  # Use a regular dictionary

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(process_file_chunk, chunk, feature_dict) for chunk in file_chunks]

        for future in as_completed(futures):
            try:
                chunk_result = future.result()
                count_dict.update(chunk_result)  # Update the main dictionary


            except Exception as e:
                print(f"An error occurred in a worker process: {e}")


    return count_dict


########################
#-CB-parsing-functions-#
########################


def parse_bgc_file(file_path):
    """
    Parses an antismash clusterblast file with repeating BGC chunks, extracts 
    relevant data, and calculates the sum of highest BLAST scores for each BGC.

    Args:
        filepath: Path to the input file.

    Returns:
        A dictionary where keys are BGC names and
        values are the corresponding cumulative blast score of
        the query BGC against that BGC
    """

    bgc_results = {}

    with open(file_path, 'r') as f:
        content = f.read()

    # Split the file content into chunks based on the ">>" delimiter
    chunks = content.split(">>\n")
    
    #The first chunk is empty/doesn't contain what we want, so we can skip it.
    for chunk in chunks[1:]:
        bgc_name = chunk.split(". ")[1].split("\n")[0]
        table_chunk=chunk.split('e-value):')[1].strip()
        df = pd.read_csv(StringIO(table_chunk), sep='\t')
        df.columns=['query gene', 'subject gene', '%identity', 'blast score', '%coverage', 'e-value']
        if 'blast score' not in df.columns:
             print("WARNING: 'blast score' column not found")
             continue


        # Get the highest 'blast score' for each 'query gene'
        max_blast_scores = df.groupby('query gene')['blast score'].max()

        # Sum the highest blast scores and store in the dictionary
        bgc_results[bgc_name] =int(max_blast_scores.sum())
        

    return bgc_results

def process_file(file_path: str, mibig_mode: bool) -> Tuple[str, Dict[str, float]]:
    """
    Helper function to parse a single file and return the BGC number and data.
    This is designed for use with multiprocessing.  Handles MIBIG mode.

    Args:
        file_path:  The path to the BGC file.
        mibig_mode:  Boolean, whether to use MIBIG-style naming.

    Returns:
        A tuple: (BGC number, dictionary of BGC IDs and BLAST scores).
    """
    if mibig_mode: #only used when compiling MiBig ref json
        bgc_number = file_path.split("/")[-1].split("_")[0]
    else:
        bgc_number = file_path  # Use the whole file path as identifier
    bgc_data = parse_bgc_file(file_path)
    return bgc_number, bgc_data

def parse_cb_outputs(target_dir: str, glob_pattern: str = None, 
                     num_cpus: int = 0, mibig_mode: bool = False):
    
    """
    Main function to parse BGC files.

    Args:
        target_dir: The base directory to search.
        glob_pattern:  Glob pattern to match files.
        output_file: Output JSON file (optional).
        num_cpus: Number of CPUs (optional).
        mibig_mode: Use MIBIG naming convention.
    """
    print("compiling file list from target directory")


    if not glob_pattern:
        cb_files = glob.glob(os.path.join(target_dir, "*",'knownclusterblast','*.txt'))
    else:
        cb_files = glob.glob(os.path.join(target_dir, glob_pattern))
        #assumes glob pattern is a string correctly formatted fro file system

    if not cb_files:
        print(f"No files found matching pattern '{glob_pattern}' in '{target_dir}'.")
        return

    # --- CPU Handling ---
    
    available_cpus = psutil.cpu_count(logical=False)
    if num_cpus == 0 or num_cpus > available_cpus:
        num_cpus = available_cpus
        print(f"Using all available physical cores: {num_cpus}")
    else:
        print(f"Using {num_cpus} physical cores.")



    all_vectors = {}

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit tasks with mibig_mode
        futures = {executor.submit(process_file, file_path, mibig_mode): file_path for file_path in cb_files}

        for future in tqdm(as_completed(futures), 
                           total=len(cb_files), desc="Processing files", dynamic_ncols=True):
            file_path = futures[future]
            try:
                bgc_number, bgc_data = future.result()
                all_vectors[bgc_number] = bgc_data
                #tqdm.write(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


    print("Processing complete.")
    
    return all_vectors 


############################
#-feature filter functions-#
############################


def find_conserved_features(feature_counts_list: List[Dict]) -> Dict:
    """
    Identifies conserved features and their minimum counts across a list of
    feature count dictionaries, returning a sparse dictionary.

    Args:
        feature_counts_list: A list of dictionaries, each representing feature counts.
            Structure:
            {
                feature_type_1: {
                    feature_qualifier_1: {domain_pattern_1: count, ...},
                    ...
                },
                ...
            }

    Returns:
        A sparse dictionary representing conserved features and their minimum counts.
        Only features present in *all* input dictionaries, with counts > 0,
        are included.  Empty dictionaries at any level are removed.  Returns
        an empty dictionary if the input list is empty or if there are no
        conserved features with counts > 0.
    """

    if not feature_counts_list:
        return {}

    # Collect all possible feature types, qualifiers, and domain patterns.
    all_feature_types = set()
    all_qualifiers = defaultdict(set)
    all_domains = defaultdict(lambda: defaultdict(set))

    for feature_counts in feature_counts_list:
        for feature_type, qualifier_data in feature_counts.items():
            all_feature_types.add(feature_type)
            for qualifier, domain_data in qualifier_data.items():
                all_qualifiers[feature_type].add(qualifier)
                for domain in domain_data:
                    all_domains[feature_type][qualifier].add(domain)

    # Initialise a dictionary to store the minimum counts. Start with infinity.
    min_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float('inf'))))

    # Iterate through the input dictionaries and update the minimum counts.
    for feature_counts in feature_counts_list:
        for feature_type in all_feature_types:
            if feature_type in feature_counts:
                for qualifier in all_qualifiers[feature_type]:
                    if qualifier in feature_counts[feature_type]:
                        for domain in all_domains[feature_type][qualifier]:
                            count = feature_counts[feature_type][qualifier].get(domain, 0)
                            min_counts[feature_type][qualifier][domain] = min(
                                min_counts[feature_type][qualifier][domain], count
                            )

    # Filter for conserved features (present in all) AND count > 0 (sparse).
    conserved_counts = {}
    for feature_type in all_feature_types:
        conserved_counts[feature_type] = {}
        for qualifier in all_qualifiers[feature_type]:
            conserved_counts[feature_type][qualifier] = {}
            for domain in all_domains[feature_type][qualifier]:
                min_count = min_counts[feature_type][qualifier][domain]
                # Only include if present in all (min_count != inf) AND count > 0
                if min_count != float('inf') and min_count > 0:
                    conserved_counts[feature_type][qualifier][domain] = min_count

            # Remove empty qualifier dicts (SPARSITY)
            if conserved_counts[feature_type][qualifier]:
                conserved_counts[feature_type][qualifier] = dict(conserved_counts[feature_type][qualifier])
            else:
                del conserved_counts[feature_type][qualifier]

        # Remove empty feature type dicts (SPARSITY)
        if conserved_counts[feature_type]:
            conserved_counts[feature_type] = dict(conserved_counts[feature_type])
        else:
            del conserved_counts[feature_type]

    return dict(conserved_counts)
    

def analyse_mibig_set(ref_set: Union[List[str], Set[str]],
                        mibig_counts: Dict) -> Dict:
    """
    Finds conserved features in a specified set of mibig BGCs

    Args:
        ref_set:
                The names a set or list of strings specifying the names of the mibig
                BGCs to analyse for conserved features
        mibig_counts:
                Dictionary of feature counts for all mibig BGCs

    Returns:
        A sparse dictionary of conserved features and their minimum counts across all
        BGCs.  Returns an empty dictionary if no BGCs specified 
        or if no conserved features with counts > 0 are found.

    """

    feature_counts_list = []
    for bgc in ref_set:
        try:
            feature_counts_list.append(mibig_counts[bgc])
        except Exception as e:
            print(f"Error processing {bgc}: {e}")
    conserved_features = find_conserved_features(feature_counts_list)
    return conserved_features

def analyse_genbank_set(gbk_set: Union[List[str], Set[str]],
                        search_criteria: Dict[str, Dict[str, List[str]]]) -> Dict:
    """
    Analyzes a directory of GenBank files, calculates feature counts, and finds conserved features.

    Args:
        gbk_list: a list containing paths forthe GenBank (.gbk) files to be analysed
        search_criteria:  The dictionary defining feature types, qualifiers, and domains
                         to search for (same structure as for make_counts).

    Returns:
        A sparse dictionary of conserved features and their minimum counts across all
        GenBank files in the directory.  Returns an empty dictionary if no .gbk files
        are found or if no conserved features with counts > 0 are found.

    Raises:
        FileNotFoundError: If the provided directory_path does not exist.
        ValueError: If the search_criteria is empty.
    """


    if not search_criteria:
        raise ValueError("search_criteria cannot be empty.")

    feature_counts_list = []
    for filepath in gbk_set:
        try:
            counts = make_counts(filepath, search_criteria)
            feature_counts_list.append(counts)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    conserved_features = find_conserved_features(feature_counts_list)
    return conserved_features


def make_counts(gbk_file: str,
                   search_criteria: Dict[str, Dict[str, List[str]]]) -> Dict[str, Union[int, Dict[str, int]]]:
    """
    Counts the occurrences of specified domain patterns within given feature types and qualifiers
    in an antiSMASH output GenBank file, based on a nested dictionary of search criteria.

    Args:
        gbk_file: Path to the GenBank file.
        search_criteria: A dictionary defining the feature types, qualifiers, and domain
                         patterns to search for.  Structure:
                         {
                             feature_type_1: {
                                 feature_qualifier_1: [domain_pattern_1, domain_pattern_2, ...],
                                 feature_qualifier_2: [domain_pattern_3, ...],
                                 ...
                             },
                             feature_type_2: {
                                 ...
                             },
                             ...
                         }

    Returns:
        A nested dictionary. The structure is:
        {
            feature_type_1: {
                feature_qualifier_1: {domain_pattern_1: count, domain_pattern_2: count, ...},
                feature_qualifier_2: {domain_pattern_3: count, ...},
                ...
            },
            feature_type_2: {
                ...
            },
            ...
        }
       If, within search_criteria, there is only one feature type and one qualifier, and no
       other features, the output is simplified to {domain_pattern_1: count, ...}.

    Raises:
        ValueError: if search_criteria is empty
        KeyError: If a feature type or qualifier encountered in the GenBank file is not
                  present in the `search_criteria` dictionary.
    """

    if not search_criteria:
        raise ValueError("search_criteria cannot be empty.")

    parsed_gbk = SeqIO.parse(gbk_file, 'genbank')
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for record in parsed_gbk:
        for feature in record.features:
            feature_type = feature.type
            if feature_type in search_criteria:
                for qualifier, patterns in search_criteria[feature_type].items():
                    if qualifier in feature.qualifiers:
                         for domain_string in feature.qualifiers.get(qualifier, []):
                            for pattern in patterns:
                                if pattern in domain_string:
                                    counts[feature_type][qualifier][pattern] += 1

    # Convert defaultdict to regular dict
    final_counts = {
        ftype: {q: dict(domain_counts) for q, domain_counts in qualifier_counts.items()}
        for ftype, qualifier_counts in counts.items()
    }

    # Simplify if only one feature/qualifier combination exists in search_criteria
    if len(final_counts) == 1:
        first_feature = list(final_counts.keys())[0]
        if len(final_counts[first_feature]) == 1:
            first_qualifier = list(final_counts[first_feature].keys())[0]
            return final_counts[first_feature][first_qualifier]

    return final_counts
    
    
def filter_by_domains(cb_hits,count_dict,domain_cutoffs):
    """
    compares feature counts for a list of gbk file paths to minium counts in domain_cutoffs
    returns a list of gbk file paths that had >= minimum count for each feature
    """
    passing=[]
    for key in cb_hits:
        add=False
        if key in count_dict:
            add=True
            for domain in domain_cutoffs:
                if not domain in count_dict[key]:
                    add=False
                elif domain_cutoffs[domain]>count_dict[key][domain]:
                    add=False
        if add:
            passing.append(key)
    return passing


####################
#-vector functions-#
####################


def calculate_center_and_distances(
    vectors: Dict[str, Dict[str, float]], distance_metric: str, default_value: float = 0.0
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    Calculates the centroid of a dictionary of sparse vectors, the maximum
    distance between any vector and the centroid, and a dictionary of
    vector-centroid distances.  Handles zero vectors for cosine distance.

    Args:
        vectors: A dictionary of sparse vectors represented as dictionaries
                 (keys are vector identifiers, values are the vectors).
        distance_metric: The distance metric ('euclidean', 'cosine', etc.).
        default_value: The default value for missing features.

    Returns:
        A tuple containing:
        - The centroid as a dictionary.
        - The maximum vector-centroid distance.
        - A dictionary of vector-centroid distances (keys are vector IDs).
    """
    if not vectors:
        return {}, 0.0, {}  # Return empty dicts and 0 if no vectors

    # Get a sorted list of all unique feature names across all vectors
    all_keys = sorted(list(set().union(*(v.keys() for v in vectors.values()))))

    # Convert sparse vectors to dense NumPy arrays
    dense_vectors = {
        name: np.array([v.get(key, default_value) for key in all_keys])
        for name, v in vectors.items()
    }

    # Calculate the centroid (mean vector) as a dense NumPy array
    centroid_vector = np.mean(list(dense_vectors.values()), axis=0)

    # Convert the centroid back to a sparse dictionary
    center_dict = {key: value for key, value in zip(all_keys, centroid_vector)}

    # Get the appropriate distance function
    distance_functions = {
        "euclidean": euclidean,
        "cosine": cosine,
        "manhattan": cityblock,
        "jaccard": jaccard,
        "braycurtis": braycurtis,
    }
    distance_func = distance_functions[distance_metric]

    max_distance = 0.0
    distances = {}  # Dictionary to store distances

    for name, dense_vector in dense_vectors.items():
        # Zero-Vector Handling for Cosine Distance
        if distance_metric == "cosine" and (
            not np.any(dense_vector) or not np.any(centroid_vector)
        ):
            distances[name] = float('nan')  # Use NaN for undefined cosine dist.
            continue

        dist = distance_func(dense_vector, centroid_vector)
        distances[name] = dist
        max_distance = max(max_distance, dist)

    return center_dict, max_distance, distances

def calculate_metrics(
    all_names: List[str],
    true_positive_names: List[str],
    predicted_positive_names: List[str],
    beta: float = 1.0,
    score_metric: str = "mcc",
) -> Tuple[float, float, float]:
    """
    Calculates either the F_beta score, precision, and recall, OR the MCC.

    Args:
        all_names: A list of names for all data points.
        true_positive_names: A list of names for data points that are actually positive.
        predicted_positive_names: A list of names for data points classified as positive.
        beta: Beta value for F_beta score (default: 1.0, i.e., F1 score).
        metric:  Which metric to calculate and return. Either "fbeta" (default) or "mcc".

    Returns:
        A tuple containing: (metric_value, precision, recall).  Returns (0.0, 0.0, 0.0)
        if there are no true positives and no predicted positives, or if an invalid
        metric is specified.  If MCC is selected, the 'metric_value' will be the MCC,
        and precision/recall will still be calculated and returned.

    Raises:
        ValueError: If `metric` is not "fbeta" or "mcc".
        ValueError: If true/predicted positive names are not a subset of all names.
    """

    # Convert lists to sets for efficient set operations
    all_names_set: Set[str] = set(all_names)
    true_positives_set: Set[str] = set(true_positive_names)
    predicted_positives_set: Set[str] = set(predicted_positive_names)

    # --- Input Validation ---
    if not true_positives_set.issubset(all_names_set):
        raise ValueError("True positive names must be a subset of all names.")
    if not predicted_positives_set.issubset(all_names_set):
        raise ValueError("Predicted positive names must be a subset of all names.")
    if score_metric not in ("fbeta", "mcc"):
        raise ValueError("Invalid metric.  Must be 'fbeta' or 'mcc'.")

    # Calculate True Positives (TP): Intersection of true positives and predicted positives
    tp = len(true_positives_set.intersection(predicted_positives_set))

    # Calculate False Positives (FP): Predicted positives that are not true positives
    fp = len(predicted_positives_set - true_positives_set)

    # Calculate False Negatives (FN): True positives that are not predicted positives
    fn = len(true_positives_set - predicted_positives_set)
    
    # Calculate True Negatives (TN): All names - (TP + FP + FN).  This is needed for MCC.
    tn = len(all_names_set) - (tp + fp + fn)


    # --- Calculate Precision and Recall ---
    if tp + fp == 0:
        precision = 0.0  # Avoid division by zero
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0  # Avoid division by zero
    else:
        recall = tp / (tp + fn)

    # --- Calculate Chosen Metric ---
    if score_metric == "fbeta":
        if precision + recall == 0:
            f_beta = 0.0  # Avoid division by zero
        else:
            f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        return f_beta, precision, recall

    elif score_metric == "mcc":
        # Calculate Matthews Correlation Coefficient (MCC)
        numerator = (tp * tn) - (fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5
        if denominator == 0:
            mcc = 0.0  # Avoid division by zero (undefined MCC)
        else:
            mcc = numerator / denominator

        return mcc, precision, recall

def find_nearby_vectors_multiple_thresholds(
    vectors: Dict[str, Dict[str, float]],
    centroid: Dict[str, float],
    thresholds: List[float],
    metric='cosine',
    default_value=0.0) -> Dict[float, List[str]]:
    
    """
    Identifies vectors within given cutoff distances from a centroid for multiple thresholds.

    Args:
        vectors: Dictionary of sparse vectors {'name': {vector}}.
        centroid: The centroid vector (sparse dictionary).
        thresholds: A list of distance thresholds.
        metric: Distance metric ('euclidean', 'cosine', etc.).
        default_value: Default value for missing features.

    Returns:
        A dictionary where keys are thresholds and values are lists of
        vector names that are within the corresponding threshold distance
        from the centroid.
    """

    # Get all unique feature names and create dense centroid vector
    all_keys = sorted(list(set().union(*(v.keys() for v in vectors.values()))))
    dense_centroid = np.array([centroid.get(key, default_value) for key in all_keys])

    # Create dense vectors for all input vectors (do this only once)
    dense_vectors = {
        name: np.array([vector.get(key, default_value) for key in all_keys])
        for name, vector in vectors.items()
    }

    # Get the appropriate distance function
    # Only cosine used, keeping others for possible future updates
    distance_functions = {
        "euclidean": euclidean,
        "cosine": cosine,
        "manhattan": cityblock,
        "jaccard": jaccard,
        "braycurtis": braycurtis,
    }
    distance_func = distance_functions[metric]

    # Initialise the dictionary to store results
    nearby_vectors_by_threshold: Dict[float, List[str]] = {}

    # 5. Iterate through each threshold
    for threshold in thresholds:
        nearby_vectors_by_threshold[threshold] = []  # Initialise list for this threshold
        # 6. Iterate through each vector
        for name, dense_vector in dense_vectors.items():

            # 7. Handle zero vectors for cosine distance
            if metric == "cosine" and (
                not np.any(dense_vector) or not np.any(dense_centroid)
            ):
                #  Skip distance calculation, vector is not "nearby"
                continue

            # 8. Calculate the distance
            distance = distance_func(dense_vector, dense_centroid)

            # 9. Check if the distance is within the current threshold
            if distance <= threshold:
                nearby_vectors_by_threshold[threshold].append(name)

    return nearby_vectors_by_threshold


def find_best_score(
    all_vectors: Dict[str, Dict[str, float]],
    true_positive_names: List[str],
    threshold_dict: Dict[float, List[str]],
    beta: float = 1.0,
    verbose: bool = False,
    score_metric='mcc',
) -> Tuple[float, float, float, float]:
    """
    Finds the best distance threshold based on the F-beta score.

    Iterates through a dictionary of thresholds and predicted positives,
    calculates the F-beta score for each, and returns the threshold
    with the highest F-beta score.

    Args:
        all_vectors: A dictionary of all vectors (used only to get all names).
        true_positive_names: A list of names of the true positive vectors.
        threshold_dict: A dictionary where keys are distance thresholds (floats)
            and values are lists of vector names predicted to be positive
            at that threshold.
        beta: The beta parameter for the F-beta score (default: 1.0, the F1-score).
        verbose: If True, print the F-beta score, precision, and recall for
            each threshold.

    Returns:
        A tuple: (best_threshold, best_f_beta, best_precision, best_recall),
        where:
            best_threshold: The threshold with the highest F-beta score.
            best_f_beta: The highest F-beta score.
            best_precision: The precision at the best threshold.
            best_recall: The recall at the best threshold.
    """
    all_names = [key for key in all_vectors]
    best_f = -1.0
    best_cutoff = -1.0  # Initialise with a value that will be overridden
    best_precision = -1.0 #initialise
    best_recall = -1.0 #initialise

    for threshold, predicted_positives in threshold_dict.items():
        f_beta, precision, recall = calculate_metrics(
            all_names, true_positive_names, predicted_positives, beta=beta,score_metric=score_metric
        )
        if verbose:
            print(
                f"For cutoff {threshold} {score_metric} score was: {f_beta:.4f}, "
                f"precision was: {precision:.4f}, recall was: {recall:.4f}"
            )
        if f_beta >= best_f:#changed from > to >= so that the largest cutoff will be returned if multiple have same F_beta
            best_f = f_beta
            best_cutoff = threshold
            best_precision = precision
            best_recall = recall

    print(
        f"The best cutoff was {best_cutoff:.4f}, {score_metric} score was: {best_f:.4f}, "
        f"precision was: {best_precision:.4f}, recall was: {best_recall:.4f}"
    )

    return best_cutoff, best_f, best_precision, best_recall

def optimise_distance_threshold(
    all_vectors: Dict[str, Dict[str, float]],
    special_vector_names: List[str],
    distance_metric: str = "cosine",
    lower: float = 0.01,
    upper: float = 2.0,
    n_steps: int = 500,
    default_value: float = 0.0,
    verbose: bool = False,
    beta: float = 1.0,
    score_metric='mcc',
) -> Tuple[float, float, float, float]:
    """
    Optimises the distance threshold for classifying special vectors based on F-beta score.

    Calculates the centroid of the special vectors, generates a range of
    distance thresholds, and finds the threshold that yields the highest
    F-beta score when classifying vectors as "special" based on their
    distance to the centroid.

    Args:
        all_vectors: A dictionary of all vectors {'name': {vector}}.
        special_vector_names: A list of names of the "special" vectors.
        distance_metric: The distance metric to use ('euclidean', 'cosine', etc.).
        lower: The lower bound for the threshold range, as a fraction of the
            maximum distance between special vectors and their centroid.
        upper: The upper bound for the threshold range, as a multiple of the
            maximum distance.
        n_steps: The number of thresholds to test within the range.
        default_value: The default value for missing features in vectors.
        verbose: If True, print verbose output during the optimisation process.
        beta: The beta parameter for the F-beta score.

    Returns:
        A tuple: (best_cutoff, best_f_beta, best_precision, best_recall)
    """
    # Input validation
    if not (0 < lower < 1):
        raise ValueError("Lower bound must be between 0 and 1 (exclusive).")
    if not (1 < upper):
        raise ValueError("Upper bound must be greater than 1.")
    if n_steps <= 0:
        raise ValueError("Number of steps must be positive.")


    # 1. Get the actual vector data for the special vectors
    special_vectors = {name: all_vectors[name] for name in special_vector_names}

    # 2. Calculate the centroid and maximum distance
    center_dict, max_distance, _ = calculate_center_and_distances(
        special_vectors, distance_metric, default_value
    )

    # 3. Handle the edge case where the maximum distance is 0
    if max_distance == 0:
        return 0.0, 0.0, 1.0, 1.0  # Return 0 threshold, 0 F-beta, 1 precision, 1 recall.

    # 4. Calculate the lower and upper bounds for the thresholds
    lower_bound = lower * max_distance
    upper_bound = upper * max_distance
    
    if distance_metric=='cosine':
        upper_bound=min(1.0,upper_bound)

    # 5. Generate an array of thresholds
    thresholds = np.linspace(lower_bound, upper_bound, n_steps)

    # 6. Find nearby vectors for each threshold
    nearby_vector_dict = find_nearby_vectors_multiple_thresholds(
        all_vectors,
        center_dict,
        thresholds,
        metric=distance_metric,
        default_value=default_value,
    )

    # 7. Find the best threshold based on F-beta score
    best_cutoff, best_f, best_precision, best_recall = find_best_score(
        all_vectors, special_vector_names, nearby_vector_dict, beta=beta, verbose=verbose,score_metric=score_metric
    )

    return best_cutoff, best_f, best_precision, best_recall


def find_hits(
    unkown_vectors: Dict[str, Dict[str, float]],
    known_vectors: Dict[str, Dict[str, float]],
    special_vector_names: List[str],
    custom=False,
    cutoff: float = None,
    distance_metric: str = "cosine",
    default_value=0.0,beta=None):

    if beta:
        score_metric='fbeta'
    else:
        score_metric='mcc'

    reference_vecs={bgc:known_vectors[bgc] for bgc in special_vector_names}

    centre_vec,_,_=calculate_center_and_distances(reference_vecs,distance_metric,default_value=default_value)

    if not cutoff:
        cutoff,_,_,_=optimise_distance_threshold(known_vectors,special_vector_names,distance_metric=distance_metric,
                                                 beta=beta,score_metric=score_metric)
        
    hits=find_nearby_vectors_multiple_thresholds(unkown_vectors,
                                                 centre_vec,
                                                 [cutoff],
                                                 metric=distance_metric,
                                                 default_value=default_value)
    
    return(hits[cutoff],cutoff)
    

##########################
#-intersection functions-#
##########################

def find_index_of_first_max(numbers: List[int]) -> int:
    """
    Finds the index of the first occurrence of the maximum value in a list of integers.

    Args:
        numbers: A list of integers.

    Returns:
        The index (0-based) of the first occurrence of the maximum value in the list.
        Returns -1 if the list is empty.

    Raises:
        TypeError: If the input is not a list or if the list contains non-integer elements.
    """

    if not numbers:
        return -1  # Handle empty list case

    max_value = numbers[0]
    max_index = 0

    for i, num in enumerate(numbers):
        if num > max_value:
            max_value = num
            max_index = i

    return max_index

def make_cb_lists(d,reference_bgcs=None,skip_self=True,return_type='list'):

    return_dict=False

    if not reference_bgcs:
        return_dict=True
        reference_bgcs=list(d.keys())
    items={key1:sorted(d[key1].items(),key=lambda x: x[1],reverse=True) for key1 in d}
    cb_lists={}
    
    for key1 in items:
        cb_lists[key1] = [i[0] for i in items[key1]]#list of just the names
        if skip_self:
            cb_lists[key1]=cb_lists[key1][1:]
    if return_dict:
        return cb_lists

    return [cb_lists[bgc][1:] for bgc in reference_bgcs]#skip the first hit as this is self


def min_intersection_size_with_reference(lists: List, reference_set: Set) -> List:
    
    """
    Calculates the minimum intersection size with a reference set for increasing prefixes of lists.

    For each k from 1 to L (length of the shortest list), this function:
    1. Takes the first k elements from each list in `lists`.
    2. Converts each prefix sublist to a set.
    3. Calculates the size of the intersection between each set and the `reference_set`.
    4. Returns the minimum of these intersection sizes.

    Args:
        lists: A list of lists.  The inner lists can contain any hashable items.
        reference_set: A set.  The items in this set should be of the same type as the items in the lists.

    Returns:
        A list of integers.  The i-th element is the minimum intersection size
        (across all input lists) when considering the first (i+1) elements of each list.
        Returns an empty list if `lists` is empty or if any of the inner lists are empty.
        Returns an empty list if reference_set is None.

    Raises:
        TypeError: If 'lists' is not a list or if any element of 'lists' is not a list, or if reference_set is not a set.

    """


    if not lists or reference_set is None:
        return []

    min_len = min(len(sublist) for sublist in lists)
    if min_len == 0:
        return []  # Handle cases with empty sublists

    min_intersection_sizes = []
    for k in range(1, min_len + 1):
        intersection_sizes = []
        for sublist in lists:
            current_set = set(sublist[:k])
            intersection_size = len(current_set.intersection(reference_set))
            intersection_sizes.append(intersection_size)
        min_intersection_sizes.append(min(intersection_sizes))

    return min_intersection_sizes

def find_hits_intersection(mibig_cb_dict,
                           all_cb_dict,
                           reference_bgcs,
                           parse=True,
                           stringent=True,
                           wiggle=1.0,
                           skip_self=False):
    #tp=0
    reference_set=set(reference_bgcs)
    #get k in n for refs
    ref_lists=make_cb_lists(mibig_cb_dict,reference_bgcs,skip_self=skip_self)
    intersect_list=min_intersection_size_with_reference(ref_lists,reference_set)
    index=find_index_of_first_max(intersect_list)
    value=int(wiggle*(intersect_list[index]))

    #skip self only set to True for leave one out analysis
    
    if parse:
        all_lists=make_cb_lists(all_cb_dict, reference_bgcs=None,skip_self=skip_self)
        print(len(all_lists))

    else:
        all_lists=all_cb_dict

    hits = []

    for key in all_lists:
        
        is_hit=False
        if len(all_lists[key])<index+1:
            continue
        to_check=set(all_lists[key][:index])
        if len(to_check&reference_set)>=value:
            is_hit=True
        if stringent and not all_lists[key][0] in reference_set:
            is_hit=False #top hit must be in refefence BGCs
            
        if is_hit:
            hits.append(key)
            
    return hits

def flatten_nested_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary, keeping only the final key-value pairs for each unique path.

    Args:
        nested_dict: The nested dictionary to flatten.
        parent_key:  The string key of the parent dictionary (used in recursive calls).
        sep: The separator to use between keys in the flattened dictionary.

    Returns:
        A flattened dictionary.

    Raises:
        TypeError: If the input is not a dictionary.
    """
    if not isinstance(nested_dict, dict):
        raise TypeError("Input must be a dictionary.")

    items: list[tuple[str, Any]] = []
    for key, value in nested_dict.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_nested_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


    
###################
#-other functions-#
###################

def change_file_paths(file_paths):
    formatted=[]
    for fp in file_paths:
        items=fp.split('knownclusterblast/')

        if len(items)>1:
            pre_path=items[0]
            name=items[1]
            items_2=name.split("_c")
            num=items_2[-1].split(".")[0]
            new_name=items_2[0]
            zeros=(3-len(num))*"0"
            formatted.append(pre_path + new_name+ ".region" + zeros + num + ".gbk")
        else:
            (formatted.append(fp))
    return formatted
        

def tsv_to_sparse_dict(file_path: str, delimiter: str = "\t",
                       has_header :bool = True, has_row_names :bool = True) -> Dict[str, Dict[str, float]]:
    """
    Converts a TSV file to a sparse dictionary representation.

    Args:
        file_path: Path to the TSV file.
        delimiter: The delimiter character (default: tab).
        has_header: True if the first row contains feature names.
        has_row_names: True if the first column contains data point names.

    Returns:
        A dictionary where keys are data point names and values are
        dictionaries representing the sparse vectors (feature: value).
        Returns an empty dictionary if the file is empty or an error occurs.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: if the file is poorly formatted
    """
    try:
        with open(file_path, 'r', newline='') as tsvfile:  # Use newline='' for correct CSV handling
            reader = csv.reader(tsvfile, delimiter=delimiter)

            if has_header:
                header = next(reader)  # Read the header row (feature names)
                if has_row_names:
                    feature_names = header[1:]  # Exclude the first element (row name header)
                else:
                    feature_names = header
            else:
                feature_names = None #will be generated later.

            sparse_data = {}

            for row_num, row in enumerate(reader):
                if not row:  # Skip empty rows
                    continue
                if has_row_names:
                    data_point_name = row[0]
                    row_values = row[1:]
                else:
                    data_point_name = f"row_{row_num+1}" # +1 for 1-based indexing.
                    row_values = row

                if feature_names is None: #if no header, name them.
                    feature_names = [f"col_{i+1}" for i in range(len(row_values))] #+1 for 1 based indexing

                if len(row_values) != len(feature_names):
                    raise ValueError(
                        f"Row {row_num + 2 if has_header else row_num +1} has " #+2 is to skip header.
                        f"{len(row_values)} values, expected {len(feature_names)}."
                    )

                row_dict = {}
                for i, value_str in enumerate(row_values):
                    try:
                        value = float(value_str)
                        if value != 0.0:  # Only store non-zero values for sparsity
                            row_dict[feature_names[i]] = value
                    except ValueError:
                        if value_str.strip() != "":  # Ignore empty strings, but raise for other errors
                            raise ValueError(
                                f"Invalid numeric value '{value_str}' at row "
                                f"{row_num + 2 if has_header else row_num + 1}, column {i + 2 if has_row_names else i +1}"
                            ) from None

                sparse_data[data_point_name] = row_dict

            return sparse_data

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}") from None
    except Exception as e:
        raise ValueError(f"Error reading or parsing TSV file: {e}") from None

def calculate_venn_labels(sets: Tuple[set, ...]) -> List[int]:
    """Calculates the size of each region for 2 or 3 sets."""
    num_sets = len(sets)
    if num_sets == 2:
        s1, s2 = sets
        return [
            len(s1 - s2),        # 10
            len(s2 - s1),        # 01
            len(s1 & s2),        # 11
        ]
    elif num_sets == 3:
        s1, s2, s3 = sets
        return [
            len(s1 - s2 - s3),   # 100
            len(s2 - s1 - s3),   # 010
            len((s1 & s2) - s3), # 110
            len(s3 - s1 - s2),   # 001
            len((s1 & s3) - s2), # 101
            len((s2 & s3) - s1), # 011
            len(s1 & s2 & s3),   # 111
        ]
    else:
        # Should be caught by the main function, but added for safety
        raise ValueError("Label calculation only implemented for 2 or 3 sets.")

def create_venn_new(sets: Tuple[set, ...],
                labels: Tuple[str, ...],
                title: str = "",
                out_path: Optional[str] = None) -> None:
    """
    Creates a Venn diagram for 2 or 3 sets, labels each section with its size,
    and saves it to a file.

    Note: This function currently relies on matplotlib_venn, which primarily
    supports visually distinct Venn diagrams for 2 or 3 sets only. For more
    than 3 sets, consider alternative visualisation libraries or methods
    (e.g., upsetplot).

    Args:
        sets: A tuple of 2 or 3 sets.
        labels: A tuple of labels for the sets. Must match the number of sets.
        title: (Optional) A title for the diagram. Defaults to "".
        out_path: (Optional) The path (including filename and extension,
            e.g., 'diagram.png', 'venn.pdf') where the diagram will be saved.
            If None, the plot will be shown instead of saved.
            Defaults to 'venn.pdf'.

    Raises:
        ValueError: If the number of sets is not 2 or 3, or if the number
            of labels doesn't match the number of sets.
        ImportError: If matplotlib or matplotlib_venn is not installed.
    """
    num_sets = len(sets)
    if num_sets not in (2, 3):
        raise ValueError(
            f"This function currently supports only 2 or 3 sets due to "
            f"matplotlib_venn limitations. Received {num_sets} sets."
            " Consider using libraries like 'upsetplot' for more sets."
        )
    if len(labels) != num_sets:
        raise ValueError(f"Number of labels ({len(labels)}) must match "
                         f"number of sets ({num_sets}).")

    # --- Plotting Setup ---
    # Create a new figure and axes for each call to avoid overlap if called multiple times
    fig, ax = plt.subplots(figsize=(8, 8)) # Using axes object is generally preferred

    # --- Select Venn Function and Draw Base Diagram ---
    if num_sets == 2:
        venn_func = venn2
        venn_circles_func = venn2_circles
        # IDs for regions in venn2: '10', '01', '11'
        region_ids = ['10', '01', '11']
    else: # num_sets == 3
        venn_func = venn3
        venn_circles_func = venn3_circles
        # IDs for regions in venn3: '100', '010', '110', '001', '101', '011', '111'
        region_ids = ['100', '010', '110', '001', '101', '011', '111']

    # Draw the venn diagram structure
    # Pass ax to the venn function
    v = venn_func(subsets=sets, set_labels=labels, ax=ax)

    # --- Calculate and Apply Region Labels (Counts) ---
    region_counts = calculate_venn_labels(sets)

    for region_id, count in zip(region_ids, region_counts):
        label_obj = v.get_label_by_id(region_id)
        if label_obj:  # Check if the region exists graphically (might be None if count is 0 and default labeling hid it)
            label_obj.set_text(str(count)) # Set text to the actual count
        # Optional: Add label even if region was initially hidden by venn function
        # elif count > 0:
            # Manually add text - requires knowing coordinates, more complex.
            # Sticking to modifying existing labels is safer.
            # pass

    # --- Customise Appearance ---
    # Draw circle outlines
    # Pass ax to the circles function
    venn_circles_func(subsets=sets, linestyle='solid', linewidth=0.8, color="black", ax=ax)

    # Set title on the axes
    ax.set_title(title)
    # Optional: Turn off axis lines/ticks if they appear (usually not needed for venn)
    ax.set_axis_on() # Or ax.set_axis_off() if you see axes lines

    # --- Save or Show Figure ---
    if out_path:
        try:
            # Use bbox_inches='tight' to prevent labels/title from being cut off
            fig.savefig(out_path, bbox_inches='tight', format=out_path.split('.')[-1])
            print(f"Venn diagram saved to: {out_path}")
        except Exception as e:
            print(f"Error saving file to {out_path}: {e}")
            # Optionally re-raise the error: raise e
        finally:
            # Close the figure to free memory, especially important if calling in a loop
             plt.close(fig)
    else:
        # If no out_path provided, show the plot interactively
        plt.show()


# not used in current release, consider using in future, works fine for >= 3 sets
def create_venn(sets: Tuple[set, ...], labels: Tuple[str, ...], title: str = "", out_path: str = 'venn.pdf') -> None:
    """
    Creates a Venn diagram from sets, labeling each section with its size.

    Args:
        sets: A tuple of sets (maximum 3 sets).
        labels: A tuple of labels for the sets.
        title:  (Optional) A title for the diagram.

    Raises:
        ValueError: If the number of sets is not 2 or 3, or if the number
            of labels doesn't match the number of sets.
    """

    num_sets = len(sets)
    if num_sets not in (2, 3):
        raise ValueError("This function supports only 2 or 3 sets.")
    if len(labels) != num_sets:
        raise ValueError("The number of labels must match the number of sets.")

    if num_sets == 2:
        venn_func = venn2
        venn_circles_func = venn2_circles
    else:  # num_sets == 3
        venn_func = venn3
        venn_circles_func = venn3_circles

    plt.figure(figsize=(8, 8))  # Adjust figure size as needed
    v = venn_func(subsets=sets, set_labels=labels)

    # Label each section with the number of elements
    if num_sets == 2:
        # Calculate the sizes of each region for 2-set Venn diagram
        labels_ = [
            len(sets[0] - sets[1]),
            len(sets[1] - sets[0]),
            len(sets[0] & sets[1]),
        ]
        for text, label in zip(v.subset_labels, labels_):
          if text:
            text.set_text(label)
    else: # num_sets == 3:
       # Calculate sizes of all seven regions for 3-set Venn diagram
        labels_ = [
            len(sets[0] - sets[1] - sets[2]),
            len(sets[1] - sets[0] - sets[2]),
            len((sets[0] & sets[1]) - sets[2]),
            len(sets[2] - sets[0] - sets[1]),
            len((sets[0] & sets[2]) - sets[1]),
            len((sets[1] & sets[2]) - sets[0]),
            len(sets[0] & sets[1] & sets[2]),
        ]

        for text, label in zip(v.subset_labels, labels_):
            if text: #handle case where section is empty
                text.set_text(label)

    # Optional: Customise circle outlines
    venn_circles_func(subsets=sets, linestyle='solid', linewidth=0.5, color="black")
    plt.title(title)
    
    plt.savefig(out_path)

def combine_genbank_files(gbk_file_paths, output_file="combined.gbk"):
    """Combines multiple GenBank files into a single output file.

    Args:
        gbk_file_paths: A list of file paths to the GenBank files.
        output_file: The path to the output combined GenBank file.
    """

    try:
        with open(output_file, "w") as outfile:
            for gbk_file in gbk_file_paths:
                try:  # Handle individual file errors
                    for record in SeqIO.parse(gbk_file, "genbank"):
                        SeqIO.write(record, outfile, "genbank")
                except Exception as e:
                    print(f"Error processing {gbk_file}: {e}")


    except Exception as e:  # Handle overall file writing errors
        print(f"Error writing to output file: {e}")

def resmash_hits(hit_gbk, output_dir=None,num_cpus=0):
    available_cpus = psutil.cpu_count(logical=False)
    if num_cpus == 0 or num_cpus > available_cpus:
        num_cpus = available_cpus

    if num_cpus <= 0:
        num_cpus = 1
        
    cmd=['antismash', 
         hit_gbk,
         '--cb-knownclusters', 
         '--clusterhmmer', 
         '-c', str(num_cpus), 
         '--output-dir', output_dir]
    try:
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        print(f"an error occured when trying to antismash your hits files {e}")