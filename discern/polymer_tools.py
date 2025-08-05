import json
import pickle
import os
from typing import Dict, List, Tuple, Optional, Set, Any
import re
from collections import defaultdict, Counter
import glob
import numpy as np 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def get_contigs(parsed_json):
    """
    makes a dictionary of data for individual contigs
    contig names are keys
    values are the corresponding record from the antismash json
    """
    contigs={}
    i=0
    for record in parsed_json['records']:
        i+=1
        try:
            contigs[record['id']]=record['modules']
        except:
            #print(f'No NRPS/PKS genes found in {record['id']}')
            pass
    return contigs


def get_domain_structure_for_orfs(orfs,to_parse):
    """
    takes a list of orf names in a BGC and the corresponding
    'antismash.detection.nrps_pks_domains' output from an antismash json
    returns a dictionary of domain structure for each module in he orf list
    """
    domain_structure={}
    for orf in orfs:
        module_list=[]
        modules=to_parse['cds_results'][orf]['modules']
        for module in modules:
            domains=[]
            components=module['components']
            for component in components:
                if 'domain'in component:
                    domains.append(component['domain']['hit_id'])
            module_list.append(domains)
                    
        domain_structure[orf]=module_list
    return domain_structure

def format_orfs(modules,consensus,stachels,as6=False):
    """
    Args:
        modules: output from get_domain_structure_for_orfs 
        consensus: consensus predcitions for NRPS/PKS orfs from A.S. json
        stachels: stachelhaus and minnowa predictions or NRPS/PKS orfs from A.S. json
    Returns: 
        A dictionary of predicted substrates for each ORFrepresented in modules
        {orf_number:[substrate1, substrate2 ... etc],...}

    Notes:
    * Where a consensus prediction for a module exists, this is used. 
    * If no consensus A-domains default to the best stachelhaus match 
    * If no consensus AT-domains default to the best Minnowa prediction.
    * For ORFs that have a C-starter domain, the substrate list starts with "FA" (fatty acid)
    * For ORFs containing a TE- or TD- domain, the substrate list ends with "TE" or "TD"
    * If a module contains an epimerisation domain, or Dual-C, "D-" is prepended to the substrate
    * If a module contains an NMe domain, "NMe-" is prepended to the substrate
    * If a module contains a CMe domain, "CMe-" is prepended to the substrate
    * If a module contains an OMe domain, "OMe-" is prepended to the substrate
    * If a module contains an A-Ox domain, "Oxo-" is prepended to the substrate
    * If a module contains a heterocyclisation domain, "Cyc-" is prepended to the substrate
    * If a module contains a glycopeptide like X-domain, "Xdom-" is prepended to the substrate
    """
    
    minowa_dict={'Methoxymalonyl-CoA':'meoxmal',
                 'Malonyl-CoA':'mal',
                 'Methylmalonyl-CoA':'mmal',
                 'Isobutyryl-CoA':'ibu',
                 'inactive':'pk',
                 'Acetyl-CoA':'ac',
                 '2-Methylbutyryl-CoA':'2mebu',
                 'Ethylmalonyl-CoA':'emal',
                 'fatty_acid':'fa',
                 'Benzoyl-CoA':'bz',
                 'Propionyl-CoA':'prop',
                 '3-Methylbutyryl-CoA':'3mebu',
                 'CHC-CoA':'chc',
                 'trans-1,2-CPDA':'cpda'}

    output={}
    
    for orf_name in modules:
        substrates=[]
        ad_count=0
        at_count=0
        aox_count=0
        for module in modules[orf_name]:
            consensus_name=None

            module=set(module)
    
            is_d=False
            is_cyclic=False
            is_nme=False
            is_ome=False
            is_cme=False
            is_aox=False
            is_te=False
            is_td=False
            is_xd=False
            #not implimented yet. Use for 'pk' ?
            is_ks=False
            is_kr=False
            is_dh=False
            is_er=False
            # to incorporate
            is_ech=False
            is_amt1_2=False
            is_amt3=False
            is_amt4=False
            is_amt5=False
            is_beta=False
            prefix=''
            substrate=''
            
            if 'Condensation_Starter' in module:
                substrates.append('FA')

            if 'CAL_domain' in module:
                substrates.append('CAL')

            if 'GNAT'  in module:
                substrates.append('gnat_acetyl')
                
            if 'AMP-binding' in module:
                ad_count+=1
                consensus_name="nrpspksdomains_"+orf_name+"_AMP-binding."+str(ad_count)
                
            elif 'PKS_AT' in module:
                at_count+=1
                consensus_name="nrpspksdomains_"+orf_name+"_PKS_AT."+str(at_count)
                #calculate degree of reduction
                if len({"PKS_DH", "PKS_DHt", "PKS_DH2"} & module) > 0:
                    is_dh=True
                if 'PKS_KR' in module:
                    is_kr=True
                if 'PKS_ER' in module:
                    is_er=True

                if is_kr and is_dh and is_er:
                    prefix='C-C_'
                
                elif is_kr and is_dh:
                    prefix='C=C_'
                    
                elif is_kr:
                    prefix='OH_'

            elif 'PKS_KS' in module and len({"PKS_DH", "PKS_DHt", 
                                "PKS_DH2",'PKS_KR','PKS_ER','ACP', 
                                'ACP_beta', 'cMT', 'nMT' ,'oMT', 'PCP', 
                                'PKS_PP','Tra_KS','Trans-AT_docking'} & module) > 0:
                # deal with trans-AT polyketides
                substrate="tr_pk"
                
                if len({"PKS_DH", "PKS_DHt", "PKS_DH2"} & module) > 0:
                    is_dh=True
                if 'PKS_KR' in module:
                    is_kr=True
                if 'PKS_ER' in module:
                    is_er=True

                if is_kr and is_dh and is_er:
                    prefix='C-C_'
                
                elif is_kr and is_dh:
                    prefix='C=C_'
                    
                elif is_kr:
                    prefix='OH_'
                
            if 'Epimerization' in module or 'Condensation_Dual' in module:
                is_d=True
            if 'Thioesterase' in module:
                is_te=True
            if 'TD' in module:
                is_td=True
            if "Heterocyclization" in module:
                is_cyclic=True
            if "nMT" in module:
                is_nme=True
            if "oMT" in module:
                is_ome=True
            if "cMT" in module:
                is_cme=True        
            if "A-OX" in module:
                is_aox=True
                aox_count+=1
                consensus_name="nrpspksdomains_"+orf_name+"_A-OX."+str(aox_count)
                
            if 'ECH' in module:
                is_ech=True               
            if 'Aminotran_1_2' in module:
                is_amt1_2=True
            if 'Aminotran_3' in module:
                is_amt3=True
            if 'Aminotran_4' in module:
                is_amt4=True               
            if 'Aminotran_5' in module:
                is_amt5=True            
                is_gnat=False
            if 'ACP_beta' in module:
                is_beta=True


            if consensus_name:
                substrate=consensus[consensus_name]
        
                if 'X' in substrate:
                    #print("is X")
                    try:
                        if as6:
                            #as6 version
                            substrate=stachels[consensus_name]['NRPSPredictor2']\
                            ['stachelhaus_predictions'][0]
                            
                        else:
                            #as7/8_version
                            substrate=stachels[consensus_name]['nrpys']\
                            ['stachelhaus_matches'][0]['substrates'][0]['short']

                    except Exception as e:
                        print(f"failed to parse Stachelhaus prediction for {consensus_name} exception :{e}")
                        pass #keep X if it fails for some reason

                if substrate=='pk':
                    #print("is pk")
                    try:                        
                        
                        substrate=stachels[consensus_name]['minowa_at']['predictions'][0][0]
                        substrate=minowa_dict[substrate]
                    except Exception as e:
                        print(f"failed to parse Minowa prediction for {consensus_name} exception :{e}")
                        pass #keep pk if it fails for some reason
            
            if substrate:      
                substrates.append("".join([prefix,is_d*"D-",is_cyclic*"Cyc-",is_nme*"NMe-",
                                           is_ome*"OMe-",is_cme*"CMe-",is_aox*"Oxo-",is_xd*"Xdom-",
                                           is_ech*'ECH-',is_amt1_2*'AM*1*2-',is_amt3*'AM*3-',
                                           is_amt4*'AM*4-',is_amt5*'AM*5-',is_beta*'BBr-',substrate]))
                if is_te:
                    substrates.append("TE")
                if is_td:
                    substrates.append("TD")
                    
                output[orf_name]=substrates
        
    return output


def make_predictions(json_file,as6=False):
    """
    Args:
        json_file: the path to an antismash json file output
        
    Returns:
        A dictionary of predicted substrates for each BGC
        {BGC_name:{orf_number:{[substrate1, substrate2] ... etc]...}...}
    Calls:
        get_contigs <get dict of contig data from as json>
        get_domain_structure_for_orfs <get module compositions for ORFs>
        format_orfs <main logic to derive susbtrates from modules/domains in each ORF>
    """
    
    with open(json_file) as F:
        parsed_json=json.load(F)
        
    bgcs={}
    
    contigs=get_contigs(parsed_json)
    
    for contig_name in contigs:
        #print(contig_name)
        try:
            consensus=contigs[contig_name]['antismash.modules.nrps_pks']['consensus']
            predictions=contigs[contig_name]['antismash.modules.nrps_pks']\
            ['region_predictions']
            stachels=contigs[contig_name]['antismash.modules.nrps_pks']\
            ['domain_predictions']
            
            for bgc_number in predictions:
                try:          
                    bgc_name=contig_name+"::"+str(bgc_number)           
                    bgc=predictions[bgc_number][0] #always get 0, longest cluster]
                    orfs=bgc['ordering']
                    to_parse=contigs[contig_name]['antismash.detection.nrps_pks_domains']
                    domain_structure=get_domain_structure_for_orfs(orfs,to_parse)
                    substrates=format_orfs(domain_structure,consensus,stachels,as6=as6)
                    
                    if substrates:
                        bgcs[bgc_name]=substrates
                        
                except Exception as e:
                    #print(e)
                    pass # skip bgc if substrate passing fails. won't halt.
        except:
            #print(f"contig name {contig_name} had no NRPS/PKS")
            pass
    return bgcs


def generate_kmers_from_monomer_list(monomer_list: List[str], max_k: int) -> List[str]:
    """
    Generates all sequential k-mers (1 to max_k) from a list of monomers.
    K-mers are joined by " - ".
    """
    kmers = []
    n = len(monomer_list)
    for k_val in range(1, max_k + 1): # Iterate for k from 1 to max_k
        if k_val > n: # Cannot form k-mers longer than the list itself
            break
        for i in range(n - k_val + 1):
            kmer_monomers = monomer_list[i : i + k_val]
            kmers.append(" - ".join(kmer_monomers))
    return kmers


def polymer_predictions_to_kmer_vectors(
    prediction_dict: Dict[str, str],
    max_kmer_size: int
) -> Dict[str, Dict[str, int]]:
    """
    Converts a dictionary of polymer predictions into a dictionary of
    k-mer count vectors.

    Args:
        prediction_dict: A dictionary where keys are identifiers (e.g., GenBank filenames)
                         and values are the polymer prediction strings.
                         Example: {"file1.gbk": "(A - B) + (C)"}
        max_kmer_size: The maximum length of k-mers to consider (e.g., 3 for
                       monomers, dimers, and trimers).

    Returns:
        A dictionary where keys are the same identifiers from prediction_dict,
        and values are dictionaries representing k-mer counts for that polymer.
        Example: {"file1.gbk": {"A": 1, "B": 1, "C": 1, "A - B": 1}}
    """
    if max_kmer_size < 1:
        raise ValueError("max_kmer_size must be at least 1.")

    kmer_vectors = {}

    for identifier, prediction_string in prediction_dict.items():
        kmer_counts_for_polymer = defaultdict(int)

        # 1. Extract all groups of monomers
        #skip this as we have list now, need to conver to list o list from dict
        #monomer_groups = extract_monomers_from_prediction(prediction_string)
        # Example: "(X) + (A - B - C) + (D)" -> [['X'], ['A', 'B', 'C'], ['D']]
        monomer_groups=[]
        for key in prediction_string:
            monomer_groups.append(prediction_string[key])

        # 2. For each group, generate and count k-mers
        for group in monomer_groups:
            # group is a list like ['A', 'B', 'C']
            kmers_in_group = generate_kmers_from_monomer_list(group, max_kmer_size)
            # Example: for ['A', 'B', 'C'] and max_k=2 -> ['A', 'B', 'C', 'A - B', 'B - C']
            for kmer in kmers_in_group:
                kmer_counts_for_polymer[kmer] += 1

        kmer_vectors[identifier] = dict(kmer_counts_for_polymer)

    return kmer_vectors


def flatten_polymer_dicts(polymer_dict):
    flattened_dict={}
    for key_1 in polymer_dict:
        to_add=set({})
        for key_2 in polymer_dict[key_1]:
            for i in range(polymer_dict[key_1][key_2]):
                flattened = str(i+1)+"_"+key_2
                to_add.add(flattened)
        flattened_dict[key_1]=to_add
    return flattened_dict


def get_polymer_stats(mibig_flat,ref_set):
    """
    Args:
        mibig_flat: A flattened dictionary of substrate kmers from all mibig NRP/PK BGCs
        ref_set: family of reference BGCs for which model is to be created
        
    Proceedure:
        * Performs all against all determination of overlap in substrate kmer sets for mibig NRP/PK BGCs
        * Identifies all BGCs for which the best non-self match is in the ref-set
        * These are either TP or FP:
            > for TPs, a member of the ref_set has a best non-self match that is also in ref_set
            > for FPs, a mibig BGC that is not in the ref_set that has a best non-self match in the ref_set
        * Prints F1 score, Number of TP, FP, TN, FN
    Returns:
        a dictionary of set overlap sizes for TP and FP {"TP":[score1, score2...], FP:[score1, score2...]}    
    """
    TP=[]
    FP=[]
    for bgc in mibig_flat:
        max_score=0
        best_match='initial'
        s1a=mibig_flat[bgc]
        for ref in mibig_flat:
            if not ref == bgc:
                s2a=mibig_flat[ref]
                score=len(s1a&s2a)
                if score >= max_score:
                    best_match=ref
                    max_score=score
        if best_match in ref_set:
            if bgc in ref_set:
                TP.append(max_score)
            else:
                FP.append(max_score)
    
    TN=len(mibig_flat) -(len(TP) + len (FP))
    FN=len(ref_set)-len(TP)
    F1_score=len(TP)/(len(TP)+0.5*(len(FP)+FN))
    print(f"TP: {len(TP)}, FP: {len(FP)}, TN: {TN}, FN: {FN}, F1_score: {F1_score}")
    
    return{"TP":TP,"FP":FP}


def calculate_f1(tp: int, fp: int, fn: int) -> float:
    """Calculates the F1-score."""
    if tp == 0: # If no true positives are identified, precision and recall are 0
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def find_best_f1_threshold(
    tp_scores: List[float],
    fp_scores: List[float],
    total_true_positives: Optional[int] = None,
    stringent: bool = False
) -> Tuple[Optional[float], float, float, float, int, int, int]:
    """
    Iterates through possible thresholds to find the one that maximizes F1-score.

    Args:
        tp_scores: A list of scores for items known to be true positives.
        fp_scores: A list of scores for items known to be true negatives
                   (these become false positives if they exceed the threshold).
        total_true_positives: The total number of actual positive items in the
                              entire dataset. If None, it's assumed to be len(tp_scores),
                              meaning all true positives received a score.

    Returns:
        A tuple containing:
        - best_threshold (float or None if no valid threshold found)
        - max_f1_score (float)
        - best_precision (float)
        - best_recall (float)
        - tp_at_best_threshold (int)
        - fp_at_best_threshold (int)
        - fn_at_best_threshold (int)
        
    Subsequent use for classification:
        An unknown BGC is classified as a member of the ref_set if:
            1) its best match among all mibig vectors is in the ref_set
            2) the size of its overlap with the substrate kmer set of this match exceets the best threshold
    """
    if total_true_positives is None:
        # Assume all true positives are in tp_scores if not specified
        total_true_positives = len(tp_scores)
    elif total_true_positives < len(tp_scores):
        raise ValueError("total_true_positives cannot be less than the number of provided tp_scores.")

    all_scores = sorted(list(set(tp_scores + fp_scores)))
    if not all_scores: # No scores to threshold on
        if total_true_positives > 0: # All are false negatives
            return None, 0.0, 0.0, 0.0, 0, 0, total_true_positives
        else: # No positives, no scores, no FPs.
            return None, 0.0, 0.0, 0.0, 0, 0, 0

    potential_thresholds = set()
    if all_scores:
        potential_thresholds.add(all_scores[0] - 1e-6) # Threshold below min score
        potential_thresholds.add(all_scores[-1] + 1e-6) # Threshold above max score
    for score in all_scores:
        potential_thresholds.add(score - 1e-6) # Just below a score
        potential_thresholds.add(score + 1e-6) # Just above a score
        potential_thresholds.add(score)        # Exactly at the score

    sorted_thresholds = sorted(list(potential_thresholds))

    # If no scores, we might still have FNs if total_true_positives > 0
    if not sorted_thresholds and total_true_positives > 0:
        return None, 0.0, 0.0, 0.0, 0, 0, total_true_positives
    elif not sorted_thresholds and total_true_positives == 0: # No scores, no positives
        return None, 0.0, 0.0, 0.0, 0, 0, 0


    best_threshold: Optional[float] = None
    max_f1 = -1.0  # F1 is between 0 and 1
    best_precision = 0.0
    best_recall = 0.0
    tp_at_best = 0
    fp_at_best = 0
    fn_at_best = total_true_positives # Initialise with all TPs as FNs

    np_tp_scores = np.array(tp_scores)
    np_fp_scores = np.array(fp_scores)

    for threshold in sorted_thresholds:
        # Items with score >= threshold are predicted positive
        tp_at_threshold = np.sum(np_tp_scores >= threshold)
        fp_at_threshold = np.sum(np_fp_scores >= threshold)

        fn_at_threshold = total_true_positives - tp_at_threshold

        current_f1 = calculate_f1(tp_at_threshold, fp_at_threshold, fn_at_threshold)


        if current_f1 >= max_f1:#more stringent
 
            if current_f1 > max_f1:
                max_f1 = current_f1
                best_threshold = threshold
                tp_at_best = tp_at_threshold
                fp_at_best = fp_at_threshold
                fn_at_best = fn_at_threshold
                # Recalculate precision and recall for the best F1
                if tp_at_best == 0:
                    best_precision = 0.0
                    best_recall = 0.0
                else:
                    best_precision = tp_at_best / (tp_at_best + fp_at_best)
                    best_recall = tp_at_best / (tp_at_best + fn_at_best) # which is tp_at_best / total_true_positives
            elif current_f1 == max_f1:
                # Tie-breaking: e.g., prefer higher recall, or lower threshold
                # For simplicity, let's just update if the threshold is "better" (e.g., lower for more sensitivity)
                # Or pick the one that maximizes recall as a secondary criterion
                current_recall_for_tie_break = 0.0
                if tp_at_best == 0: pass
                else: current_recall_for_tie_break = tp_at_threshold / (tp_at_threshold + fn_at_threshold)

                if current_recall_for_tie_break > best_recall: # Prefer higher recall at same F1
                    max_f1 = current_f1 # Redundant but clear
                    best_threshold = threshold
                    tp_at_best = tp_at_threshold
                    fp_at_best = fp_at_threshold
                    fn_at_best = fn_at_threshold
                    best_precision = tp_at_best / (tp_at_best + fp_at_best) if (tp_at_best + fp_at_best) > 0 else 0.0
                    best_recall = current_recall_for_tie_break
        min_tp=min(tp_scores)

    best_threshold=int(np.round(best_threshold))

    if best_threshold < min_tp:
        print(min_tp, best_threshold, max_f1)
        if stringent:
            best_threshold=min_tp
        else:
            best_threshold=int(np.round((best_threshold+min_tp)/2))

        

        tp_at_best = np.sum(np_tp_scores >= best_threshold)
        fp_at_best = np.sum(np_fp_scores >= best_threshold)
        fn_at_best = total_true_positives - tp_at_best
        max_f1 = calculate_f1(tp_at_best, fp_at_best, fn_at_best)
        

    
    return best_threshold, max_f1, best_precision, best_recall, tp_at_best, fp_at_best, fn_at_best


def make_polymer_dict(antismash_folder,glob_pattern=None,k=5,as6=False):

    polymer_dict={}
    if glob_pattern:
        
        json_files=glob.glob(os.path.join(antismash_folder,glob_pattern))
    else:
        json_files=glob.glob(os.path.join(antismash_folder,"*/*.json"))

    for json_file in json_files:
        #print(json_file.split("/")[-1])
        try:
            d1=make_predictions(json_file,as6=as6)
            d2=polymer_predictions_to_kmer_vectors(d1,k)
            d3=flatten_polymer_dicts(d2)
            polymer_dict[json_file]=d3
        except Exception as e:
            print(e)
            print(f"no NRPS/PKS found in {json_file}")
            pass

        

    return polymer_dict  


def get_polymer_matches(polymer_dict,mibig_vecs):
    """
    searches a dictionary of extracted polymer kmer vectors for bgcs to classify against ref_vecs
    returns a dictionary:
    {genome::bgc_name:(score,ref_hit),...}
    score is len({query_kmers}&{ref_kmers}) for best hit in ref_vecs
    """
    top_matches={}
    for genome in polymer_dict:
        for bgc in polymer_dict[genome]:
            best_score=0
            best_hit="initial"
            for ref in mibig_vecs:
                current_score=len(polymer_dict[genome][bgc]&mibig_vecs[ref])
                if current_score>best_score:
                    best_score=current_score
                    best_hit=ref
            top_matches[genome+"::"+bgc]=(best_score,best_hit)
    return top_matches


def filter_polymer_matches(top_matches,ref_set,cutoff):
    """
    Filters polymer matches by:
        1) finding queries whose top match is in ref_set
        2) keeping subset of (1) where  len({query_kmers}&{ref_kmers}) >= cutoff
    
    cutoff is previously determined by training on ref set with:
        1) get_polymer_stats(mibig_flat,ref_set)
        2) find_best_f1_threshold(polymer_stats['TP'], polymer_stats['FP'])
    """
    filtered_matches={}
    ref_set=set(ref_set)#incase its a list
    for match in top_matches:
        if top_matches[match][1] in ref_set and top_matches[match][0]>=cutoff:
            print(match,top_matches[match])
            filtered_matches[match]=top_matches[match]
    return filtered_matches


def polymer_dict_key_to_gbk_path(polymer_dict_key):
    """
    converts a key from polymer_dict into the path to the corresponding gbk file
    use to get paths for combining gbks
    """
    head, tail = os.path.split(polymer_dict_key)
    strain, contig, region_number=tail.split("::")
    return os.path.join(head,contig+".region"+region_number.zfill(3)+".gbk")



def print_match_scores(matches,mibig_vecs,polymer_dict):   
    for key in matches:
        bgc=matches[key][1]
        score=matches[key][0]
        ref_len=len(mibig_vecs[bgc])
        pd_k1,contig,bgc_name=key.split("::")
        pd_k2="::".join([contig,bgc_name])
        
        query_len=len(polymer_dict[pd_k1][pd_k2])
        print(query_len)
        print()
        print(key)
        print(f"maches {bgc}")
        print(f"{score/ref_len*100} % kmers in {bgc} match the query")
        print(f"{score/query_len*100} % kmers the query match {bgc}")
        print("~~~~~~~~~~~~~~~~~~~~~")


def get_match_stats(matches,mibig_vecs,polymer_dict):
    match_stats={}
    for key in matches:
        bgc=matches[key][1]
        score=matches[key][0]
        ref_len=len(mibig_vecs[bgc])
        pd_k1,contig,bgc_name=key.split("::")
        pd_k2="::".join([contig,bgc_name])        
        query_len=len(polymer_dict[pd_k1][pd_k2])
        ref2query=score/ref_len*100
        query2ref=score/query_len*100
        match_stats[key]={"ref_match":bgc, "ref2query":ref2query, "query2ref":query2ref }
    return match_stats
        

def find_elements_in_n_of_k_sets(list_of_sets: List[Set[Any]]) -> Dict[int, Set[Any]]:
    """
    Given a list of k sets, finds elements that are present in exactly
    n of these k sets, for n from 2 to k.

    Args:
        list_of_sets (List[Set[Any]]): A list containing k set objects.

    Returns:
        Dict[int, Set[Any]]: A dictionary where:
            - Keys are integers 'n' (from 2 up to k, or the max count found).
            - Values are sets of elements found in exactly 'n' of the input sets.
            Example: {
                2: {elements in exactly 2 sets},
                3: {elements in exactly 3 sets},
                ...
                k: {elements in all k sets} # This is the intersection
            }
    """
    if not list_of_sets:
        return {}
    if not all(isinstance(s, set) for s in list_of_sets):
        raise TypeError("Input must be a list of sets.")

    k = len(list_of_sets)
    element_counts = Counter()

    for s in list_of_sets:
        element_counts.update(s) 

    results: Dict[int, Set[Any]] = {i: set() for i in range(1, k + 1)}


    for element, count in element_counts.items():
        if count >= 1: # Only consider elements appearing in 2 or more sets
            if count in results: # Check if this specific count (n) is a key we care about
                results[count].add(element)

    return results