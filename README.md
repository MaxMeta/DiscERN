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
    *   [Custom Reference BGCs](#custom-reference-bgcs)
    *   [Native Homology Search (`--cb_source diamond`)](#native-homology-search---cb_source-diamond)
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
download-antismash-databases
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

### Step 5 (optional): Set Up MIBiG for the Native Homology Search

Only needed for `--cb_source diamond` (see [Native Homology Search](#native-homology-search---cb_source-diamond)). This downloads MIBiG 4.0's protein sequences (md5-checked), builds a DIAMOND database and searches MIBiG against itself. Searching MIBiG against itself took about 35 minutes with 8 CPUs in our tests, and the set needs about 90 MB in `discern/mibig_db/` (about 250 MB with the legacy set below). To put it elsewhere, use `discern-setup-mibig --db_dir <folder>` and `discern --mibig_db_dir <folder>`, or set the `DISCERN_MIBIG_DB` environment variable for both.
```bash
discern-setup-mibig -c 8
```
To also use the entries that MIBiG dropped before release 4.0 (see [MIBiG releases](#mibig-releases-and-dropped-entries)):
```bash
discern-setup-mibig --set 4.0_legacy -c 8
```


## How DiscERN Works

### Core Algorithmic Approaches
DiscERN's strength lies in its ensemble approach, which combines the outputs of four distinct classification methods. A given BGC is assigned a confidence score from k=1 to k=4 based on how many algorithms independently identify it as a "hit".

1.  **Pfam-Vec (`pfam-vec` / `bs_hits`)**: This method leverages the BGC vectorization technique from BiG-SLiCE. It converts each BGC into a numerical vector representing the presence and abundance of its Pfam domains. This provides a high-level "fingerprint" of the BGC's architecture. A centroid vector is calculated for the reference family, and query BGCs are classified based on their cosine distance to this centroid.
2.  **BLAST-Vec (`blast-vec` / `cb_hits`)**: This method generates a "blast-vector" for each query BGC. The vector is created by treating the cumulative BLAST score of the query BGC's proteins against each individual reference BGC as a dimension. Like the Pfam-Vec method, it uses a centroid-based classification approach to identify new family members.
3.  **BLAST-Rank (`blast-rank` / `ol_hits`)**: This rank-based method offers improved robustness against fragmented or concatenated BGCs. It first determines a signature for the reference family based on how many in-family BLAST hits (`n`) are typically found within a certain rank (`k`) for all known members. A query BGC is then classified as a hit if its top BLAST hit is a member of the reference family and it has at least `n` in-family hits within the top `k` results.
4.  **Structural K-mer Intersection (`k-mer` / `pol_hits`)**: Designed specifically for NRPS and PKS clusters, this method compares BGCs based on their predicted final product. It parses antiSMASH outputs to predict the sequence of monomers (amino acids or acyl-CoAs) for each BGC. These sequences are then broken down into overlapping k-mers (e.g., monomers, dimers, trimers). A query BGC is assigned to the family with which its k-mer set has the highest overlap, provided the score reaches a pre-optimized threshold.

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
    3.  **Custom Directory**: Path to a directory of your own reference BGCs (`<name>.gbk`, `<name>.txt` and, for `--poly_search`, `<name>.json`), for using non-MIBiG BGCs as references. See [Custom Reference BGCs](#custom-reference-bgcs) for how to prepare it and what this mode can't do.

    To combine MIBiG and custom references, give the MIBiG IDs with `-r` and the directory with `--custom_refs`.

#### **Common Options:**
*   `-c`, `--num_cpus`: Number of CPUs to use for parallel processing. (Default: all available physical cores).
*   `-p`, `--poly_search`: Activates the "Structural K-mer Intersection" method. This is highly recommended for families of NRPS, PKS, or hybrid NRPS/PKS BGCs, especially those with 5 or more modules.
*   `-u`, `--reuse`: Path to a previous DiscERN output directory. If specified, DiscERN will reuse the intermediate vector and count files (`bs_vecs.json`, `cb_vecs.json`, etc.) from that run, saving significant computation time. The `--antismash_dir` must point to the same input directory used for the original run.
*   `-s`, `--resmash`: If specified, DiscERN will automatically run antiSMASH on the final combined hit `.gbk` files. 
*   `--as6`: Flag to indicate that the antiSMASH outputs were generated with antiSMASH version 6. This adjusts the parsing logic for polymer predictions.
*   `--custom_refs`: A directory of custom reference BGCs to use together with the MIBiG references given with `-r`.
*   `--cb_source`: Where the BLAST-derived vectors for `blast-vec` and `blast-rank` come from: `antismash` (default) parses antiSMASH KnownClusterBlast output and uses the bundled MIBiG data, the setup used for the published results; `diamond` uses DiscERN's own DIAMOND search. See [Native Homology Search](#native-homology-search---cb_source-diamond).
*   `--mibig_set`: With `--cb_source diamond`, which MIBiG set to use: `4.0` (default, MIBiG 4.0 as published) or `4.0_legacy` (4.0 plus the entries dropped before 4.0).

#### **Advanced Options:**
*   `-k`, `--min_k`: The minimum algorithm support score (`k`) required for a hit to be included in the output hierarchical clustering trees. (Default: 3). If this is above the highest `k` a hit can reach in the run (3, or 4 with `--poly_search`; with `--cb_source antismash` and custom references, 2, or 3 with `--poly_search`), that maximum is used instead.
*   `-b`, `--beta`: Use the F-beta score instead of the Matthews Correlation Coefficient (MCC) for training the classification models. You must provide a beta value (e.g., `-b 1.0` for the F1-score).
*   `-x`, `--mibig_exclude`: Exclude specific MIBiG BGCs from the background reference database during analysis. Can be a space-separated string of IDs, a file with one ID per line, or the keyword `self` to automatically exclude the reference BGCs themselves from the background set (`self` applies to the MIBiG references given with `-r`, and is ignored when there are none).
*   `--bigslice_cutoff`, `--clusterblast_cutoff`: Manually set the cosine distance cutoffs for the `pfam-vec` and `blast-vec` methods, respectively. This overrides the automatically determined thresholds.
*   `--vec_check`: Perform a check on the reference BGC collection for internal consistency and identify other MIBiG BGCs that are closely related. (Default: True).
*   `--hclust`: Generate hierarchical clustering dendrograms and a combined Newick tree for high-confidence hits. (Default: True).
*   `--mibig_db_dir`: With `--cb_source diamond`, where `discern-setup-mibig` built the MIBiG sets (default: `discern/mibig_db`, or the `DISCERN_MIBIG_DB` environment variable).
*   `--cb_cap`: With `--cb_source diamond`, how many subject BGCs each blast vector keeps, after MIBiG and custom hits are merged (default: 50, as antiSMASH KnownClusterBlast; 0 keeps all).


---

### Input Directory Structure

The `--antismash_dir` (or `-a`) flag should point to a top-level directory that contains 
the output folders from your various antiSMASH runs. Each subdirectory should correspond 
to the complete antiSMASH output for a single genome.

The antiSMASH runs need to have been completed with known cluster blast activated using 
the flag ```--cb-knownclusters``` 
Make sure you enable this when generating antismash outputs for use with DiscERN. (With `--cb_source diamond` this is not needed: DiscERN runs its own search on the region `.gbk` files.)

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

### Custom Reference BGCs

Instead of MIBiG accessions, `-r` can point to a directory of your own reference BGCs, for example family members that are not in MIBiG.

Use `-r <directory>` for custom references only, or `-r "<MIBiG IDs>" --custom_refs <directory>` to mix them with MIBiG references.

**What each reference needs.** AntiSMASH output files that share one base name, `<name>`:

| File | Source | Needed |
|---|---|---|
| `<name>.gbk` | The antiSMASH **region** GenBank file, `<record>.region001.gbk`. Not the whole-record `<record>.gbk`. | Always |
| `<name>.txt` | The KnownClusterBlast output for that region, `knownclusterblast/<record>_c1.txt`. antiSMASH only writes this when run with `--cb-knownclusters`. | With `--cb_source antismash` (the default) |
| `<name>.json` | The antiSMASH JSON, `<record>.json`. | With `--poly_search` |

With `--cb_source diamond`, names must not contain `|` or spaces, or be the same as a MIBiG ID.

If a file is missing or the names don't match, DiscERN stops before the analysis and lists the unmatched files.

**How to make them.** Run antiSMASH with `--cb-knownclusters` on the GenBank file of each reference BGC, then copy and rename the outputs. Use one reference BGC per GenBank file, so that each JSON describes one BGC (if a JSON contains several NRPS/PKS regions, DiscERN merges them into one reference and prints a warning).

```bash
antismash --cb-knownclusters --output-dir as_ref1 ref1.gbk

mkdir -p my_refs
cp as_ref1/<record>.region001.gbk            my_refs/ref1.gbk
cp as_ref1/knownclusterblast/<record>_c1.txt my_refs/ref1.txt
cp as_ref1/<record>.json                     my_refs/ref1.json   # only needed for --poly_search
```

Here `<record>` is the record ID antiSMASH gives the input sequence. If antiSMASH finds more than one region, use the `regionNNN.gbk` and `_cN.txt` files of the BGC you want.

**Limits.**
*   With `--cb_source antismash`, BLAST-rank (`ol_hits`) is unavailable, because custom references never appear in each other's KnownClusterBlast results. The maximum support is therefore k=2 (pfam-vec and blast-vec), or k=3 with `--poly_search`, and `--min_k` is capped to that maximum. DiscERN prints a notice saying this at the start of the run. With `--cb_source diamond`, custom references are part of the search, so all four methods run and the maximum is k=4, as with MIBiG references.
*   The k≥3 precision figure under [Primary Hit Files](#primary-hit-files) was benchmarked with MIBiG references and does not apply here.
*   Use at least two references. With a single reference the trained distance thresholds collapse to 0 and `--vec_check` is skipped.

**Exclusions.** If the family already has members in MIBiG, they stay in the background set and count as negatives when the pfam-vec, blast-vec and polymer thresholds are trained, which makes the thresholds tighter. Exclude them with `-x` and explicit MIBiG IDs. To find candidates, check `other_close_mibig_bgcs` in `blast_vec_check.json` and `bigslice_vec_check.json`; DiscERN also prints these BGCs at the end of a custom-reference run. `-x self` only excludes MIBiG references given with `-r`. With `--cb_source antismash`, excluded BGCs are removed from the MIBiG and reference blast vectors but stay as dimensions of the query vectors; with `--cb_source diamond` they are removed from every vector, as if they were not in MIBiG.

**Example.** Using four calcium-dependent antibiotic (CDA) BGCs re-run through antiSMASH as custom references, and excluding their MIBiG entries from the background:

```bash
discern -a ~/discern_test/inputs \
-o ~/discern_test/custom_cda_output \
-r ~/my_cda_refs \
-x "BGC0000315 BGC0000336 BGC0000354 BGC0000439" \
--poly_search
```

### Native Homology Search (`--cb_source diamond`)

By default, `blast-vec` and `blast-rank` use antiSMASH's KnownClusterBlast output, whose subjects are MIBiG BGCs from the KnownClusterBlast database of your antiSMASH version, compared against MIBiG vectors bundled with DiscERN. With `--cb_source diamond`, DiscERN instead runs its own DIAMOND search of the query regions against MIBiG proteins and your custom references, scored the same way (antiSMASH KnownClusterBlast settings: e-value 1e-5, identity > 45 %, query coverage > 40 %, hits from at least two query genes on at least two reference genes; score = sum over query genes of the best bitscore; the top 50 references kept per BGC). This means:

*   custom references are subjects of the search, so they get `blast-vec` dimensions and can appear in `blast-rank` hit lists: all four methods run with MIBiG, custom or mixed references;
*   antiSMASH no longer needs `--cb-knownclusters`, and results don't depend on antiSMASH's KnownClusterBlast database version;
*   the MIBiG background is built by the same code as the query vectors, from one MIBiG release (and the Pfam-vec vectors are rebuilt from that release too).

Run `discern-setup-mibig` once first (see [Step 5](#step-5-optional-set-up-mibig-for-the-native-homology-search)). Then add `--cb_source diamond`:

```bash
discern -a ~/discern_test/inputs \
-o ~/discern_test/cda_diamond_output \
-r "BGC0000315 BGC0001370 BGC0001968 BGC0001984" \
--custom_refs ~/my_cda_refs \
--cb_source diamond \
--poly_search
```

The query hits against MIBiG are saved in the output folder (`cb_raw_mibig-<set>.json.gz`), so `-u` reuses them when you run another reference family on the same inputs. Each run writes `run_info.json` with the DiscERN version and commit, the blast source, MIBiG set and releases, DIAMOND version, search settings and the MIBiG and custom references used.

`--cb_source diamond` is new and has not yet been re-benchmarked against the published results, which were produced with `--cb_source antismash` in DiscERN v1.0.0. Bug fixes since then change some results with either source (see [CHANGELOG.md](CHANGELOG.md)); use the `v1.0.0` tag to reproduce the published numbers exactly.

#### MIBiG releases and dropped entries

The native search uses MIBiG 4.0 as published by default (`--mibig_set 4.0`). MIBiG has dropped entries over time: 244 BGCs that were in releases 1.4–3.1 are not in 4.0. Where MIBiG recorded a reason, it is mostly that the entry duplicated another one (79), failed validation (28), was split over several contigs or loci (27) or contained a single gene (16); 78 have no recorded reason. Some of these are still useful references. To use them, build and select the `4.0_legacy` set, which adds each dropped entry back from the most recent release that still contained it:

```bash
discern-setup-mibig --set 4.0_legacy -c 8
discern ... --cb_source diamond --mibig_set 4.0_legacy -r "BGC0000025 ..."
```

[docs/mibig_dropped_entries.md](docs/mibig_dropped_entries.md) lists every dropped entry with MIBiG's reason and its closest MIBiG 4.0 entry. Some are near-duplicates of a current entry; in the legacy set they are part of the background too, so if a near-duplicate's current twin is in your reference family, add the dropped entry to your references or exclude it with `-x`. If you give a dropped entry as a reference without the legacy set, DiscERN stops and says so. Dropped entries have no bundled feature counts or polymer predictions, so the conserved-domain filter and the polymer method leave them out.

## Example Walkthrough
Let's find new members of the calcium-dependent lipopeptide (CDA) family from a set of actinomycete genomes that have already been run through antiSMASH.

**1. Prepare Inputs**

In this example, you will download some test data and analyse this. 
```bash
cd ~
wget https://zenodo.org/records/16899260/files/discern_test.tar.gz
```

Unzip the example data folder and remove the compressed file
```bash
tar -xvzf discern_test.tar.gz 
rm discern_test.tar.gz 
```

Run DiscERN with a string specifying the CDA family as reference BGCs.  
Since CDAs are NRPSs, we will use the `--poly_search` flag.
We will also enable antiSMASH analysis of the compiled results

```bash
discern -a ~/discern_test/inputs \
-o ~/discern_test/cda_output \
-r "BGC0000315 BGC0001370 BGC0001968 BGC0001984 BGC0000291 BGC0000336 BGC0000379 BGC0000354 BGC0001448 BGC0002430 BGC0000439"  \
--poly_search \
-s
```

Now we will use the glycopeptide family as reference BGC.
We can reuse the processed data from our first run with CDAs by using `-u` 
This will save a lot of computation. Especially with large inputs.

```bash
discern -a ~/discern_test/inputs \
-o ~/discern_test/glyco_output \
-u ~/discern_test/cda_output \
-r "BGC0000289 BGC0000290 BGC0000311 BGC0000322 BGC0000326 BGC0000418 BGC0000419 BGC0000440 BGC0000441 BGC0000455 BGC0001178 BGC0001459 BGC0001460 BGC0001461 BGC0001462 BGC0001635 BGC0001955 BGC0002314 BGC0002344 BGC0002637 BGC0002638 BGC0002702 BGC0002992 BGC0003168"  \
--poly_search \
-s
```



## Interpreting the Output
DiscERN creates a structured output directory to facilitate hit prioritization.

### Primary Hit Files
The most important outputs are the combined GenBank files containing the identified BGCs and the corresponding antiSMASH output folders (if this option was enabled)

*   `filtered_hits_k{k}.gbk`: These files contain BGCs that passed the conserved domain filter. The `{k}` indicates the number of algorithms that supported the hit. **These are your highest-priority targets.** A higher `k` value corresponds to higher confidence. Based on our benchmarking, `k≥3` yields very high precision (>95%). (Benchmarked with DiscERN v1.0.0 and MIBiG references; the fixes listed in [CHANGELOG.md](CHANGELOG.md) change some results.)
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
*   `run_info.json`: How the run was set up: DiscERN version and commit, blast source, MIBiG set, DIAMOND version and search settings, and the MIBiG and custom references.
*   `cb_raw_mibig-<set>.json.gz`, `cb_vecs_diamond_mibig-<set>.json` (`--cb_source diamond` only): The uncapped DIAMOND scores of every query region against MIBiG (reused by `-u`), and the final query blast vectors.
