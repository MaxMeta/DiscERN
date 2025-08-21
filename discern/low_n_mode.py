import json
import os
from pathlib import Path
from tqdm import tqdm

# Import reusable functions from your existing modules
from .polymer_tools import get_polymer_matches, polymer_dict_key_to_gbk_path
from .funcs_2 import combine_genbank_files, change_file_paths

def _find_top_blast_hit(blast_vector: dict) -> str:
    """Finds the MIBiG ID with the highest score in a BLAST vector."""
    if not blast_vector:
        return ""
    # The key with the maximum value in the dictionary
    return max(blast_vector, key=blast_vector.get)

def run_low_n_search(
    cb_vecs: dict,
    poly_dict: dict,
    mibig_vecs_pol: dict,
    ref_set: set,
    output_dir: str
):
    """
    Runs the low-N discovery mode and saves the results.

    A hit is defined as a BGC where BOTH the top BLAST hit AND the top
    polymer structure hit are members of the user-defined reference set.
    """
    print("Starting Low-N Mode search for BGC family expansion.")
    
    #Find the top BLAST hit for every query BGC
    print("Determining top BLAST hit for all query BGCs...")
    top_blast_hits = {}
    for cb_path, blast_vector in tqdm(cb_vecs.items(), desc="Processing BLAST vectors"):
        # Convert the clusterblast text file path to a standard gbk path
        gbk_path = change_file_paths([cb_path])[0]
        top_blast_hits[gbk_path] = _find_top_blast_hit(blast_vector)
        print(cb_path,">>",top_blast_hits[gbk_path])

    #Find the top polymer hit for every query BGC ---
    print("Determining top polymer hit for all query BGCs...")
    top_polymer_matches_raw = get_polymer_matches(poly_dict, mibig_vecs_pol)
    
    top_polymer_hits = {}
    for polymer_key, (score, top_hit) in top_polymer_matches_raw.items():
        gbk_path = polymer_dict_key_to_gbk_path(polymer_key)
        top_polymer_hits[gbk_path] = top_hit
        #print("-->",top_polymer_hits[gbk_path])

    #Find the intersection of hits satisfying both criteria
    print("Identifying BGCs with consensus top hits in the reference family...")
    
    # Find all BGCs whose top BLAST hit is in the reference set
    blast_hit_candidates = {gbk for gbk, top_hit in top_blast_hits.items() if top_hit in ref_set}
    print(blast_hit_candidates)
    
    # Find all BGCs whose top Polymer hit is in the reference set
    polymer_hit_candidates = {gbk for gbk, top_hit in top_polymer_hits.items() if top_hit in ref_set}
    print(polymer_hit_candidates)
    
    # The final hits are the intersection of these two sets
    final_hits = list(blast_hit_candidates.intersection(polymer_hit_candidates))
    
    print(f"Found {len(final_hits)} high-confidence hits using Low-N Mode.")

    # Write the output files ---
    if not final_hits:
        print("No hits found in Low-N Mode. Exiting.")
        return

    # Create a summary report
    summary_data = []
    for hit_path in final_hits:
        summary_data.append({
            "hit_gbk_path": os.path.basename(hit_path),
            "genome_source": Path(hit_path).parent.name,
            "top_blast_hit": top_blast_hits.get(hit_path, "N/A"),
            "top_polymer_hit": top_polymer_hits.get(hit_path, "N/A"),
        })

    summary_file = Path(output_dir) / "low_n_mode_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Detailed summary of hits saved to: {summary_file}")

    # Combine the hit BGCs into a single GenBank file
    output_gbk = Path(output_dir) / "low_n_mode_hits.gbk"
    combine_genbank_files(final_hits, str(output_gbk))
    print(f"Combined GenBank file of hits saved to: {output_gbk}")