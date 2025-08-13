# DiscERN: Discoverer of Evolutionarily Related Natural products

<p align="center">
  <img alt="DiscERN Logo" src="./assets/logo.png" width=400>
</p>

## Table of Contents
*   [Overview](#overview)
*   [Installation](#install-instructions)    
*   [How DiscERN Works](#how-discern-works)
    *   [Core Algorithmic Approaches](#core-algorithmic-approaches)
    *   [The DiscERN Workflow](#the-discern-workflow)
*   [Usage](#usage)
    *   [Command-Line Arguments](#command-line-arguments)
*   [Example Walkthrough](#example-walkthrough)
*   [Interpreting the Output](#interpreting-the-output)
    *   [Primary Hit Files](#primary-hit-files)
    *   [Analysis and Visualization Files](#analysis-and-visualization-files)
    *   [Intermediate Data Files](#intermediate-data-files)

## Overview
DiscERN is an automated genome mining tool for the targeted discovery of bacterial natural products. It expands user-defined families of Biosynthetic Gene Clusters (BGCs) by identifying evolutionarily related BGCs from large genomic datasets.

To achieve this, DiscERN uses a flexible ensemble method that integrates four complementary algorithms. These algorithms classify BGCs based on three distinct measures of relatedness: Pfam domain content, direct protein sequence similarity, and predicted final product structure. This multi-modal approach allows DiscERN to strategically balance discovery sensitivity with predictive precision, providing a reliable path from genomic data to a prioritized list of candidate BGCs.


## Installation

DiscERN is designed to be installed and run within a Conda environment to manage its complex dependencies. The following instructions will guide you through the process.

**Prerequisites:**
*   You must have [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) installed on your system.

---

### Step 1: Get the DiscERN Source Code


```bash
git clone https://github.com/MaxMeta/DiscERN.git
cd DiscERN/discern
```

**From this point on, all commands should be run from within the `DiscERN/` directory.**

### Step 2: Download and Set Up BiG-SLiCE Models

DiscERN uses models from the BiG-SLiCE tool for its `pfam-vec` algorithm. These need to be downloaded and placed in the correct directory.

If you don't have wget, install it:
```bash
conda install wget -y
```

Create the directory for the models and change into it
```bash
mkdir bigslice_models
cd bigslice_models
```

Download the model archive
```bash
wget https://github.com/medema-group/bigslice/releases/download/v2.0.0rc/bigslice-models.2022-11-30.tar.gz
```
Check md5. This should be ```aaabde911ec107d08e5c24f68aaf31d1```
```bash
md5 bigslice-models.2022-11-30.tar.gz
```

Extract the models and clean up the archive file
```bash
tar -xvzf bigslice-models.2022-11-30.tar.gz
rm bigslice-models.2022-11-30.tar.gz
```

Return to the main DiscERN project directory
```
cd ../..
```

### Step 3: Create and Activate the Conda Environment


For Linux use the following command
```bash
conda create -n discern antismash==8.0.2 
```

For Mac osX use the following command
```bash
conda create -n discern antismash==8.0.2 --platform osx-64
```

Activate your environment
```bash
conda activate discern
```

Download the antiSMASH databases, this will take a few minutes
```bash
antismash-download-databases
```

### Step 4: Install DiscERN

Finally, with the `discern` Conda environment activated, install the DiscERN package itself using pip. The `-e` flag installs the package in "editable" mode, which links the installation to the source code directory.

```bash
pip install -e .
```

You can verify that the installation was successful by running:
```bash
discern --help
```
This should display the help menu with all available command-line options.  You are now ready to use DiscERN. It might take a few minutes the first time you run it


Verify antismash installed correctly. This should display the help menu with all available command-line options.
```bash
antismash --help
```


## How DiscERN Works

### Core Algorithmic Approaches
DiscERN's strength lies in its ensemble approach, which combines the outputs of four distinct classification methods. A given BGC is assigned a confidence score from k=1 to k=4 based on how many algorithms independently identify it as a "hit".

1.  **Pfam-Vec (`pfam-vec` / `bs_hits`)**: This method leverages the BGC vectorization technique from BiG-SLiCE. It converts each BGC into a numerical vector representing the presence and abundance of its Pfam domains. This provides a high-level "fingerprint" of the BGC's architecture. A centroid vector is calculated for the reference family, and query BGCs are classified based on their cosine distance to this centroid.
2.  **BLAST-Vec (`blast-vec` / `cb_hits`)**: This method generates a "blast-vector" for each query BGC. The vector is created by treating the cumulative BLAST score of the query BGC's proteins against each individual reference BGC as a dimension. Like the Pfam-Vec method, it uses a centroid-based classification approach to identify new family members.
3.  **BLAST-Rank (`blast-rank` / `ol_hits`)**: This rank-based method offers improved robustness against fragmented or concatenated BGCs. It first determines a signature for the reference family based on how many in-family BLAST hits (`n`) are typically found within a certain rank (`k`) for all known members. A query BGC is then classified as a hit if its top BLAST hit is a member of the reference family and it has at least `n` in-family hits within the top `k` results.
4.  **Structural K-mer Intersection (`k-mer` / `pol_hits`)**: Designed specifically for NRPS and PKS clusters, this method compares BGCs based on their predicted final product. It parses antiSMASH outputs to predict the sequence of monomers (amino acids or acyl-CoAs) for each BGC. These sequences are then broken down into overlapping k-mers (e.g., monomers, dimers, trimers). A query BGC is assigned to the family with which its k-mer set has the highest overlap, provided the score exceeds a pre-optimized threshold.

### The DiscERN Workflow
DiscERN automates the process of identifying new BGC family members from a collection of antiSMASH outputs.

1.  **Input**: The user provides two main inputs:
    *   A directory containing the output folders from a collection of antiSMASH runs.
    *   A defined "family" of reference BGCs, which can be specified using MIBiG accession numbers or by providing a directory of custom reference files.

2.  **Vectorization & Feature Extraction**: For every BGC in the user-provided antiSMASH directory, DiscERN:
    *   Calculates its Pfam-Vec, BLAST-Vec, and BLAST-Rank scores relative to the reference family.
    *   (Optional) Predicts the polymer structure and generates structural k-mers if `--poly_search` is enabled.
    *   Counts the occurrences of specific Pfam domains for later filtering.

3.  **Classification & Scoring**: Each of the four algorithms classifies the query BGCs. DiscERN collates these results, giving each potential hit a score (`k`) from 1 to 4, representing the number of algorithms that provided support.

4.  **Filtering**: DiscERN automatically identifies Pfam domains that are conserved across all members of the input reference family. It then uses this profile to segregate the hits into two categories:
    *   **Filtered Hits**: BGCs that contain the core, conserved domains. These are the highest-confidence candidates.
    *   **Unfiltered Hits**: BGCs that were identified by at least one algorithm but are missing one or more of the core domains. These may represent more distant relatives or fragmented clusters.

5.  **Output**: The primary output is a set of GenBank files containing the hit BGCs, organized into folders based on their confidence score (`k`) and whether they passed the filtering step. DiscERN also generates several analysis files, including dendrograms and Newick trees, to help visualize the relationships between the newly discovered hits and the original reference family.

## Usage
DiscERN is a command-line tool. The basic command structure is as follows:
```bash
discern -a <antismash_directory> -o <output_directory> -r <reference_bgcs> [OPTIONS]
```

### Command-Line Arguments
#### **Required Arguments:**
*   `-a`, `--antismash_dir`: Path to the base directory containing your antiSMASH output folders. DiscERN will recursively search this directory for `*.gbk` and `*.json` files.
*   `-o`, `--output_dir`: Path where the output folder will be created. If the folder does not exist, it will be created.
*   `-r`, `--reference_bgcs`: Defines the BGC family to expand. This can be provided in three ways:
    1.  **MIBiG IDs (string)**: A space-separated string of MIBiG accession numbers (e.g., `"BGC0000306 BGC0000330"`).
    2.  **MIBiG IDs (file)**: Path to a text file containing one MIBiG accession number per line.
    3.  **Custom Directory**: Path to a directory containing your own reference BGCs, formatted as antiSMASH outputs (including `.gbk`, `.json`, and the `knownclusterblast` text files). This is for using non-MIBiG BGCs as a reference.

#### **Common Options:**
*   `-c`, `--num_cpus`: Number of CPUs to use for parallel processing. (Default: all available physical cores).
*   `-p`, `--poly_search`: Activates the "Structural K-mer Intersection" method. This is highly recommended for families of NRPS, PKS, or hybrid NRPS/PKS BGCs, especially those with 5 or more modules.
*   `-u`, `--reuse`: Path to a previous DiscERN output directory. If specified, DiscERN will reuse the intermediate vector and count files (`bs_vecs.json`, `cb_vecs.json`, etc.) from that run, saving significant computation time. The `--antismash_dir` must point to the same input directory used for the original run.
*   `-s`, `--resmash`: If specified, DiscERN will automatically run antiSMASH on the final combined hit `.gbk` files. This is useful for re-annotating the hits with Cluster-BLAST information relative to each other.
*   `--as6`: Flag to indicate that the antiSMASH outputs were generated with antiSMASH version 6. This adjusts the parsing logic for polymer predictions.

#### **Advanced Options:**
*   `-k`, `--min_k`: The minimum algorithm support score (`k`) required for a hit to be included in the output hierarchical clustering trees. (Default: 3).
*   `-b`, `--beta`: Use the F-beta score instead of the Matthews Correlation Coefficient (MCC) for training the classification models. You must provide a beta value (e.g., `-b 1.0` for the F1-score).
*   `-x`, `--mibig_exclude`: Exclude specific MIBiG BGCs from the background reference database during analysis. Can be a space-separated string of IDs, a file with one ID per line, or the keyword `self` to automatically exclude the reference BGCs themselves from the background set.
*   `--bigslice_cutoff`, `--clusterblast_cutoff`: Manually set the cosine distance cutoffs for the `pfam-vec` and `blast-vec` methods, respectively. This overrides the automatically determined thresholds.
*   `--vec_check`: Perform a check on the reference BGC collection for internal consistency and identify other MIBiG BGCs that are closely related. (Default: True).
*   `--hclust`: Generate hierarchical clustering dendrograms and a combined Newick tree for high-confidence hits. (Default: True).


---

### Input Directory Structure

The `--antismash_dir` (or `-a`) flag should point to a top-level directory that contains 
the output folders from your various antiSMASH runs. Each subdirectory should correspond 
to the complete antiSMASH output for a single genome.

The antiSMASH runs need to have been completed with known cluster blast activated using 
the flag ```--cb-knownclusters``` 
Make sure you enable this when generating antismash outputs for use with DiscERN

DiscERN will automatically scan these subdirectories to find the necessary files (e.g., `.gbk`, `.json`, and `knownclusterblast/` files) for its analysis.

Here is an example of a correctly formatted input directory:

```bash
# Your main input directory
antismash_runs/
│
├── Streptomyces_coelicolor_A3(2)/
│   ├── knownclusterblast/
│   │   ├── NC_003888.3_c1.txt
│   │   ├── NC_003888.3_c2.txt
│   │   └── ...
│   ├── index.html
│   ├── NC_003888.3.gbk
│   ├── NC_003888.3.json
│   ├── NC_003888.3.region001.gbk
│   ├── NC_003888.3.region002.gbk
│   └── ... (and all other antiSMASH output files)
│
├── Streptomyces_griseus_DSM_40236/
│   ├── knownclusterblast/
│   │   └── ...
│   ├── index.html
│   ├── CP002472.1.gbk
│   ├── CP002472.1.json
│   ├── CP002472.1.region001.gbk
│   └── ... (all antiSMASH files for this genome)
│
└── Nocardia_brasiliensis_ATCC_700358/
    ├── knownclusterblast/
    │   └── ...
    ├── index.html
    ├── CP003885.1.gbk
    ├── CP003885.1.json
    ├── CP003885.1.region001.gbk
    └── ... (all antiSMASH files for this genome)
```
## Example Walkthrough
Let's find new members of the calcium-dependent lipopeptide (CDA) family from a set of actinomycete genomes that have already been run through antiSMASH.

**1. Prepare Inputs**

In this example, you will download some test data and analyse this. 
First, make an input directory, then change into this directory and download the example data
I this example, we will make the input and output directories in your home dir.
You can change this if you want.
```bash
mkdir ~/discern_test/
cd ~/discern_test/
wget <add_url>
```

Unzip the example data folder and remove the compressed file
```bash
tar -xvzf <file_name>
rm <file_name>
```

Run DiscERN with a string specifying the CDA family as reference BGCs.  
Since CDAs are NRPSs, we will use the `--poly_search` flag.
We will also enable antiSMASH analysis of the compiled results
```bash
discern -a ~/discern_test/inputs \
-o ~/discern_test/cda_output \
-r "BGC0000315 BGC0001370 BGC0001968 BGC0001984 BGC0000291 BGC0000336 BGC0000379 BGC0000354 BGC0001448 BGC0002430 BGC0000439"  \
--poly_search
-s
```



**3. Analyze the Results**
Once the run is complete, navigate to the `cda_discovery_run/` directory.

*   **Prioritize Hits**: Start by examining the antiSMASH outputs in the directories called `filtered_hits_k4.gbk` and `filtered_hits_k3.gbk` files. These contain the BGCs that were identified by 4 and 3 algorithms, respectively, and which also possess the core conserved domains of the CDA family. These are your most promising candidates.
*   **Visualize Relationships**: Open the `combined_tree.pdf` and `pol_denrogram.pdf` files. These dendrograms show how your new hits (labeled by their genome and region number) cluster with the known CDA references from MIBiG. Outliers or distinct sub-clusters may represent particularly novel structural variants. You can use the `newick_tree.txt` file with a viewer like iTOL for more advanced visualization.
*   **Refine your Family (Optional)**: Open `blast_vec_check.json` and `bigslice_vec_check.json`. The `outliers` field will tell you if any of your reference BGCs are very different from the others. The `other_close_mibig_bgcs` field may reveal other MIBiG BGCs that you might want to consider adding to your reference set for future runs.

## Interpreting the Output
DiscERN creates a structured output directory to facilitate hit prioritization.

### Primary Hit Files
The most important outputs are the combined GenBank files containing the identified BGCs.

*   `filtered_hits_k{k}.gbk`: These files contain BGCs that passed the conserved domain filter. The `{k}` indicates the number of algorithms that supported the hit. **These are your highest-priority targets.** A higher `k` value corresponds to higher confidence. Based on our benchmarking, `k≥3` yields very high precision (>95%).
*   `unfiltered_hits_k{k}.gbk`: These files contain BGCs that were identified but are missing one or more of the core domains found in the reference family. These could be interesting but more divergent family members, or simply fragmented BGCs.
*   if antiSMASH analysis of compiled results is enabled, you will also see directories with the same name as the gbk files. These contain antiSMASH outputs 

### Analysis and Visualization Files
*   `combined_tree.pdf` & `newick_tree.txt`: A hierarchical clustering tree created by averaging the distance metrics from all three underlying vector types (Pfam, BLAST, and polymer k-mers). Use this to visualize the overall relatedness of your high-confidence hits (`k >= min_k`) and the references.
*   `bs_denrogram.pdf`, `cb_denrogram.pdf`, `pol_denrogram.pdf`: Individual dendrograms for each of the distance metrics.
*   `blast_vec_check.json`, `bigslice_vec_check.json`: JSON files containing an analysis of your reference set's coherence. Useful for refining the set for future runs.
*   `polymer_matches.json`: If using `--poly_search`, this file contains detailed statistics on the similarity between the predicted polymer structures of your hits and their best-matching reference BGC.

### Intermediate Data Files
These files are generated during the run and are primarily for internal use, but they are crucial for the `--reuse` functionality.

*   `bs_vecs.json`: The calculated Pfam-Vecs for all input BGCs.
*   `cb_vecs.json`: The calculated BLAST-Vecs for all input BGCs.
*   `feature_counts.json`: The per-BGC counts of key Pfam domains.
*   `poly_dict.pkl`: A pickled Python dictionary containing the structural k-mer sets for all NRPS/PKS BGCs.
*   `bs_hits.json`, `cb_hits.json`, `ol_hits.json`, `polymer_hits.json`: JSON files listing the raw hits from each individual algorithm.
