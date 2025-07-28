
import argparse
import pathlib
import pickle
from .funcs_2 import *         
from .vector_tools import * 
from .polymer_tools import *

_THIS_DIR = pathlib.Path(__file__).parent.resolve()
_DEFAULT_DATA_DIR = _THIS_DIR / "data"
_DEFAULT_DB_DIR = _THIS_DIR / "bigslice_models"

# --- Define Default Paths to Bundled Data ---
_DEFAULT_MIBIG_CB_FILE = _DEFAULT_DATA_DIR / "mibig_cb_dict.json"
_DEFAULT_MIBIG_BS_FILE = _DEFAULT_DATA_DIR / "mibig_bs_vecs.tsv"
_DEFAULT_FEATURE_DICT_FILE = _DEFAULT_DATA_DIR / "feature_dict.json"
_DEFAULT_MIBIG_COUNT_FILE = _DEFAULT_DATA_DIR / "mibig_counts_rf.json"
_DEFAULT_MIBIG_POL_FILE = _DEFAULT_DATA_DIR / "mibig_pol_dict.pkl"

def main():
    mibig_cb_file=_DEFAULT_MIBIG_CB_FILE
    mibig_bs_file=_DEFAULT_MIBIG_BS_FILE
    feature_dict_file=_DEFAULT_FEATURE_DICT_FILE
    mibig_count_file=_DEFAULT_MIBIG_COUNT_FILE
    mibig_pol_dict_file=_DEFAULT_MIBIG_POL_FILE
    db_dir=_DEFAULT_DB_DIR

    mibig_cb_file = str(mibig_cb_file)
    mibig_bs_file = str(mibig_bs_file)
    feature_dict_file = str(feature_dict_file)
    mibig_count_file = str(mibig_count_file)
    mibig_pol_dict_file = str(mibig_pol_dict_file)

    parser = argparse.ArgumentParser(description="arguments for main script")
    
    parser.add_argument("-a", "--antismash_dir", help="Base directory containing antismash outputs.")

    parser.add_argument("-o", "--output_dir", default=None, help="path for output folder to create")
    parser.add_argument("-c", "--num_cpus", type=int, default=0,
                        help="Number of CPUs (default: all available physical cores).")

    parser.add_argument("-r", "--reference_bgcs", default=None, 
                        help="reference bgcs defining family to be expanded")

    parser.add_argument("-b", "--beta", default=None, 
                        help="use f-beta score instead of mcc, define value of beta here")

    parser.add_argument("-u","--reuse", default=None,
                          help="path to previous output directory to reuse. -a should be input that was used to make the output")
    
    parser.add_argument("-s","--resmash", action='store_true',default=False,
                              help="run antismash analysis on combined gbk outputs")

    parser.add_argument("-x","--mibig_exclude", default=None,
                              help="file or space seperated string of MiBig BGCs to ignore during analysis")

    parser.add_argument("--bigslice_cutoff", default=None,
                              help="cutoff to use for inclusion with bigslice vectors")
    
    parser.add_argument("--clusterblast_cutoff", default=None,
                              help="cutoff to use for inclusion with clusterblast vectors")

    parser.add_argument("--hclust", default=True, 
                        help="conduct hierachical clustering of reference vectors")

    parser.add_argument("--vec_check", default=True, 
                        help="check ref collection for outliers, and possible mibig entrires that should be added")

    args=parser.parse_args()

    antismash_dir=args.antismash_dir
    output_dir=args.output_dir
    num_cpus=args.num_cpus
    ref_set=args.reference_bgcs
    beta=args.beta
    mibig_exclude=args.mibig_exclude
    reuse=args.reuse
    resmash=args.resmash
    hclust=args.hclust
    vec_check=args.vec_check

    if args.clusterblast_cutoff:
        ccb=float(args.clusterblast_cutoff)
    else:
        ccb=None
    if args.bigslice_cutoff:
        cbs=float(args.bigslice_cutoff)
    else:
        cbs=None

    ref_dir=False

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if os.path.isdir(ref_set):
        
        ref_set_bs=glob.glob(os.path.join(ref_set,"*.gbk"))     
        ref_set_cb=glob.glob(os.path.join(ref_set,"*.txt"))    
        ref_bs_vecs=make_sparse_dict(ref_set,db_dir,num_cpus=num_cpus,
                                     glob_pattern="*.gbk")
        
        ref_set_pol=glob.glob(os.path.join(ref_set,"*.json"))
        ref_bs_vecs_rf={key.split(":")[0]:ref_bs_vecs[key] for key in ref_bs_vecs}
        ref_bs_vecs=ref_bs_vecs_rf
        ref_cb_vecs=parse_cb_outputs(ref_set,glob_pattern="*.txt")
        ref_pol_vecs=make_polymer_dict(ref_set,glob_pattern="*.json")
        if mibig_exclude:
            for key in ref_cb_vecs:#delete  excluded BGCs from ref cb vecs
                for bgc in mibig_exclude:
                    if bgc in ref_cb_vecs[key]:
                        del ref_cb_vecs[key][bgc]
        ref_dir=True

    elif os.path.isfile(ref_set):
        lines=set({})
        with open(ref_set) as F:
            for line in F:
                if line.strip():
                    lines.add(line.strip())
        ref_set=lines
    
    else:
        ref_set = set(ref_set.split())
    
    print('loading data...')

    with open(feature_dict_file) as F:
        feature_dict=json.load(F)
        
    with open(mibig_count_file) as F:
        mibig_counts=json.load(F)
    
    mibig_vecs_bs=tsv_to_sparse_dict(mibig_bs_file)
    
    mibig_vecs_rf={}
    for key in mibig_vecs_bs:
        bgc=os.path.split(key)[-1].split(".")[0]#use BGC number as key
        mibig_vecs_rf[bgc]=mibig_vecs_bs[key]
    mibig_vecs_bs=mibig_vecs_rf

    with open(mibig_cb_file) as F:
        mibig_vecs_cb = json.load(F)

    with open(mibig_pol_dict_file,'rb') as F:
        mibig_vecs_pol=pickle.load(F)
        

    if mibig_exclude:
        if os.path.isfile(mibig_exclude):
            lines=set({})
            with open(mibig_exclude) as F:
                for line in F:
                    if line.strip():
                        lines.add(line.strip())
            mibig_exclude=lines
        
        elif mibig_exclude=='self':
            if not ref_dir:
                mibig_exclude=ref_set
            else:
                print("ignoring mibig_exclude=self argument as this is not compatible with external ref files")
            #print(mibig_exclude)
        else:
            mibig_exclude=mibig_exclude.split()
            
        new_cb={key:mibig_vecs_cb[key] for key in mibig_vecs_cb if not key in mibig_exclude}
        for key in new_cb:
            for bgc in mibig_exclude:
                if bgc in new_cb[key]:
                    #print(bgc,new_cb[key][bgc])
                    del new_cb[key][bgc]
        new_bs={key:mibig_vecs_bs[key] for key in mibig_vecs_bs if not key in mibig_exclude}
        new_pol={key:mibig_vecs_pol[key] for key in mibig_vecs_pol if not key in mibig_exclude}
        mibig_vecs_cb=new_cb
        mibig_vecs_bs=new_bs
        mibig_vecs_pol=new_pol
    
    if ref_dir:
        #add the externally defined reference vectors to mibig
        mibig_vecs_bs.update(ref_bs_vecs)
        mibig_vecs_cb.update(ref_cb_vecs)
        mibig_vecs_pol.update(ref_pol_vecs)
        
    if reuse:
        print("loading previously constructed big-slice vectors from file")
        with open(os.path.join(reuse,"bs_vecs.json")) as F:
            bs_vecs=json.load(F)

    else:

        print("making big-slice vectors")
        bs_vecs = make_sparse_dict(antismash_dir,db_dir)
        with open(os.path.join(output_dir,"bs_vecs.json"),'w') as F:
            json.dump(bs_vecs,F,indent=4)

    if reuse:
        print("loading previously constructed clusterblast vectors from file")
        with open(os.path.join(reuse,"cb_vecs.json")) as F:
            cb_vecs=json.load(F)
    else:
        print("making clusterblast vectors")   
        cb_vecs = parse_cb_outputs(antismash_dir,num_cpus=num_cpus)
        with open(os.path.join(output_dir,"cb_vecs.json"),'w') as F:
            json.dump(cb_vecs,F,indent=4)
            
    if reuse:
        print("loading previously constructed feature counts from file")
        with open(os.path.join(reuse,"feature_counts.json")) as F:
            all_counts=json.load(F)
    else:
        print("making feature counts")  
        all_counts = make_all_feature_counts(antismash_dir,num_cpus,feature_dict)
        all_counts_rf={}
        for key in all_counts:
            all_counts_rf[key]=flatten_nested_dict(all_counts[key])
        all_counts=all_counts_rf
        with open(os.path.join(output_dir,"feature_counts.json"),'w') as F:
            json.dump(all_counts,F,indent=4)
    if reuse:
        print("loading previously constructed polymer dict from file")
        with open(os.path.join(reuse,"poly_dict.pkl")) as F:
            poly_dict=json.load(F)
    else:
        print("making polymer dict")
        poly_dict=make_polymer_dict(antismash_dir)
        with open(os.path.join(output_dir,"poly_dict.pkl"),'wb') as F:
            pickle.dump(poly_dict,F)    

    if ref_dir:
        print('finding conserved features from genbank files')
        conserved_features=flatten_nested_dict(analyse_genbank_set(ref_set_bs, feature_dict))
        
    else:
        print('finding conserved features from mibig vecs')
        conserved_features=flatten_nested_dict(analyse_mibig_set(ref_set, mibig_counts))
    print(conserved_features)
    
    print('finding bs vec hits')
    
    if ref_dir:
        bs_hits=find_hits(bs_vecs,mibig_vecs_bs,ref_set_bs,beta=beta,cutoff=cbs)
    else:
        bs_hits=find_hits(bs_vecs,mibig_vecs_bs,ref_set,beta=beta,cutoff=cbs)
       
    print('finding cb vec hits')


    if ref_dir:
        cb_hits=find_hits(cb_vecs,mibig_vecs_cb,ref_set_cb,beta=beta,cutoff=ccb)
    else:  
        cb_hits=find_hits(cb_vecs,mibig_vecs_cb,ref_set,beta=beta,cutoff=ccb) 

    #add run_polymer=True clause
    print('finding polymer matches')    
    pol_matches=get_polymer_matches(poly_dict,mibig_vecs_pol)

    
    if ref_dir:        
        stats=get_polymer_stats(mibig_vecs_pol,ref_set_pol)#updated version with added ref_dir polymers
        pol_threshold, pol_f1, pol_precision, pol_recall, pol_tp,\
        pol_fp, pol_fn = find_best_f1_threshold(stats['TP'],stats['FP'],stringent=True)
        pol_hits=filter_polymer_matches(pol_matches,ref_set_pol,pol_threshold)
        
    else:  
        stats=get_polymer_stats(mibig_vecs_pol,ref_set)
        pol_threshold, pol_f1, pol_precision, pol_recall, pol_tp,\
        pol_fp, pol_fn = find_best_f1_threshold(stats['TP'],stats['FP'],stringent=True)
        pol_hit_dict=filter_polymer_matches(pol_matches,ref_set,pol_threshold)

    pol_hits=set([polymer_dict_key_to_gbk_path(key) for key in pol_hit_dict])

    print_match_scores(pol_hit_dict,mibig_vecs_pol,poly_dict)
            

    if not ref_dir:
        print('finding ol hits')
        ol_hits=find_hits_intersection(mibig_vecs_cb,cb_vecs,ref_set)
        ol_hits=set(change_file_paths(ol_hits))

    cb_hits=set(change_file_paths(cb_hits))
    bs_hits=set(bs_hits)

    print('filtering by feature counts')

    if ref_dir:
        hits_by_k=find_elements_in_n_of_k_sets([cb_hits,bs_hits,pol_hits])


    else:
        hits_by_k=find_elements_in_n_of_k_sets([cb_hits,bs_hits,pol_hits,ol_hits])

    passing_hits_by_k={}
    failing_hits_by_k={}


    for k in hits_by_k:
        passing_hits_by_k[k]=filter_by_domains(hits_by_k[k],all_counts,conserved_features)
        failing_hits_by_k[k]=hits_by_k[k].difference(passing_hits_by_k[k])
    
    print('writing output files')

    for k in passing_hits_by_k:
        output_file=os.path.join(output_dir,f'filtered_hits_k{k}.gbk')
        if len(passing_hits_by_k[k])>0:
            combine_genbank_files(passing_hits_by_k[k],output_file)
            
    for k in passing_hits_by_k:
        output_file=os.path.join(output_dir,f'unfiltered_hits_k{k}.gbk')
        if len(failing_hits_by_k[k])>0:
            combine_genbank_files(failing_hits_by_k[k],output_file)
        

    if resmash:
        to_smash=glob.glob(os.path.join(output_dir,'*_hits_k*.gbk'))
        for gbk_file in to_smash:
            out_name=gbk_file.split(".gbk")[0]
            resmash_hits(ogbk_file, output_dir=os.path.join(output_dir,out_name),num_cpus=num_cpus)

    if hclust:
        if ref_dir:
            dense_vecs_cb=make_dense_vectors({os.path.split(path)[-1]:ref_cb_vecs[path] for path in ref_cb_vecs})
            dense_vecs_bs=make_dense_vectors({os.path.split(path)[-1]:ref_bs_vecs[path] for path in ref_bs_vecs})
        else:
            dense_vecs_cb=make_dense_vectors({bgc:mibig_vecs_cb[bgc] for bgc in ref_set})
            dense_vecs_bs=make_dense_vectors({bgc:mibig_vecs_bs[bgc] for bgc in ref_set})
                                              
        
        ax_cb=plot_hclust_dendrogram(dense_vecs_cb)
        ax_bs=plot_hclust_dendrogram(dense_vecs_bs)

        if ax_cb:
            try:
                ax_cb.figure.savefig(os.path.join(output_dir,'cb_denrogram.pdf'))
                
            except Exception as e:
                print(f'failed to save cb_dendrogram figure: {e}')
        else:
            print('no cb ax!')
            
        if ax_cb:
            try:
                ax_bs.figure.savefig(os.path.join(output_dir,'bs_denrogram.pdf'))
            except Exception as e:
                print(f'failed to save cb_dendrogram figure: {e}')
        else:
            print('no cb ax!')
        

    if vec_check:
        if ref_dir:
            rf_vecs_cb={os.path.split(path)[-1]:ref_cb_vecs[path] for path in ref_cb_vecs}|mibig_vecs_cb
            rf_vecs_bs={os.path.split(path)[-1]:ref_bs_vecs[path] for path in ref_bs_vecs}|mibig_vecs_bs
            cb_names=[os.path.split(path)[-1] for path in ref_cb_vecs]
            bs_names=[os.path.split(path)[-1] for path in ref_bs_vecs]
            chk_cb=analyse_vector_collections(cb_names,rf_vecs_cb)
            chk_bs=analyse_vector_collections(bs_names,rf_vecs_bs)
            
            

        else:
            chk_cb=analyse_vector_collections(ref_set,mibig_vecs_cb)
            chk_bs=analyse_vector_collections(ref_set,mibig_vecs_cb)

        print(chk_cb)
        print(chk_bs)
        
    

    #make venn diagram
    #write summary text file.



#from CB parse script
if __name__ == "__main__":
    
    print('starting DiscERN')

    
    main()
